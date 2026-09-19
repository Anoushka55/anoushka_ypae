# Architecture Report — The Invisible Planet

**Phase 0 deliverable.** Repository audit, technical risk assessment, and recommended
target architecture. No production code is written in this phase.

---

## 1. Repository audit (state at time of audit)

Repository root: `anoushka_ypae` (not yet a git repository at audit time).

| Path | Size | Description | Relation to this project |
|---|---|---|---|
| `sim.py` | 36.6 KB | Standalone real-time Milky Way / Andromeda galaxy-collision visualiser. Taichi GGUI, restricted N-body ("macro-particle skeleton" + massless tracer stars), leapfrog integration, orbit camera. | **Unrelated prototype.** Different science (galactic dynamics), different scale, different goals. Recommend preserving under `prototypes/` rather than deleting. |
| `imgui.ini` | 127 B | Auto-generated Taichi GGUI window-layout cache written by `sim.py`. | Build artifact. Should be git-ignored. |
| `data/raw/*` | 261 KB | NASA Exoplanet Archive snapshot pulled during this audit (see §2). | Phase 1 input. |

**Findings:**

- There is **no existing physics engine, integrator, test suite, data layer, or
  inference code** for The Invisible Planet. This is a greenfield build.
- The only reusable knowledge from `sim.py` is *practical Taichi experience*:
  GGUI window/scene/camera usage, leapfrog structure, and the confirmed fact that
  this machine's Taichi GPU backend is **Vulkan** (no CUDA — `nvcuda.dll` not found).
  That last fact is a critical numerical constraint; see §3.1.
- No dependency manifest, no tests, no CI, no documentation, no license.
- Python on this machine: system 3.11.0rc2 / 3.14.3. Taichi supports 3.10–3.12, so
  the project must pin an interpreter (`uv run --python 3.12`).

**Nothing was modified in this phase** except the addition of `data/raw/` archive
snapshots, which are read-only provenance inputs required by Phase 1.

---

## 2. Data reconnaissance (performed to de-risk Phase 1)

The NASA Exoplanet Archive TAP service is reachable from this environment.

A non-obvious and important discovery: **the archive does not index this system under
the hostname `Kepler-90`.** It is filed under **`KOI-351`** (= Kepler-90 = KIC 11442793).
A naive `hostname='Kepler-90'` query returns zero rows, and `hostname LIKE 'Kepler-90%'`
silently returns the *wrong systems* (Kepler-900 … Kepler-909). Any future query code
must use `hostname='KOI-351'`.

81 published parameter rows exist across 8 planets. Coverage by reference:

| Reference | Planets | P | a | e | M | R | i | T0 |
|---|---|---|---|---|---|---|---|---|
| Weiss et al. 2024 | b c d e f g h **i** | 8 | 8 | 8 (all fixed 0) | 2 (M sin i) | 8 | – | – |
| Cabrera et al. 2014 | b c d e f g h | 7 | 7 | – | – | 7 | 7 | 7 |
| Shallue & Vanderburg 2018 | i | 1 | – | – | – | 1 | 1 | 1 |
| Liang et al. 2021 | g h | 2 | – | 2 | **2 (TTV dynamical)** | – | 2 | – |
| Shaw et al. 2025 | g h | 2 | – | 2 | **2 (TTV dynamical)** | – | – | 2 |
| Fulton & Petigura 2018 | (stellar) | – | – | – | – | – | – | – |

**Consequences for Phase 1 that the architecture must accommodate:**

1. **No single reference covers every parameter.** Cross-source combination is
   unavoidable, so the data layer must record source *per parameter*, not per planet.
2. **Only two of eight planets have measured masses** (g and h, from transit timing
   variations). The remaining six have *no* published mass. They must be **derived**
   from a cited mass–radius relation and flagged as model-dependent — never presented
   as observed.
3. **Eccentricities are adopted, not measured**, for b–f. Weiss et al. 2024 fixes
   e = 0 for all eight planets (no error bars). Only g and h have measured
   eccentricities.
4. **The real system genuinely exhibits TTVs** — the adopted masses of g and h were
   *derived from transit timing variations* (Liang et al. 2021). This grounds the
   project's premise (infer a mass from timing residuals) in a method that has been applied
   to this very system, although other published masses for g and h disagree
   (`docs/KEPLER90_DATA.md`, CHECK 6).
5. Stellar mass disagrees across sources (0.967 → 1.242 M☉). A single self-consistent
   choice must be made and justified, because **semi-major axis depends on it**
   through Kepler's third law.

A consistency check performed against the archive snapshot (reproduced by
`scripts/build_reference_dataset.py`, CHECK 1 and 1b): for planet h, Kepler's third law with
the IAU value of GM☉, M★ = 1.108 M☉ and P = 331.601108 d gives a = 0.970193 au against the
published 0.970206 au (relative difference 1.3×10⁻⁵, identical for all eight planets). That
uniform offset is not a disagreement about the system: the published axes equal
(M★ P_yr²)^(1/3) to 7×10⁻¹⁰, i.e. they were computed with the 4π² au³/yr² shortcut and
the stellar mass alone (`docs/UNITS.md`). Weiss et al. 2024 adopt the same M★ = 1.108 M☉ as
Fulton & Petigura 2018, so the two can be paired without mixing stellar-mass scales.

---

## 3. Technical risks

### 3.1 CRITICAL — floating-point precision (f32 is disqualifying)

Taichi defaults to `f32`. This project measures **transit timing residuals of order
seconds to minutes** over integration baselines of **thousands of days**.

- f32 carries ≈ 7 significant decimal digits. At t ≈ 10⁴ days, the representable
  time resolution is ≈ 10⁻³ days ≈ **86 seconds**.
- The TTV signals we must resolve are of the same order or smaller.
- f32 would therefore manufacture numerical "TTVs" indistinguishable from physical ones,
  which would silently invalidate every scientific claim in the project.

**Mitigation (mandatory):** `ti.init(default_fp=ti.f64)` for all physics. This machine's
GPU backend is **Vulkan, which commonly lacks f64 support**. Physics must therefore run
on `arch=ti.cpu` (f64-safe), with rendering free to use f32.

This is not a performance problem, because of §3.2.

### 3.2 Where Taichi actually helps (and where it does not)

The Kepler-90 system has ~10 bodies. A pairwise force loop over 10 bodies is ~90
interactions — utterly trivial. **Parallelising over bodies is pointless here.**

The real computational load is the **inverse problem** (Phases 8, 10, 11): thousands of
forward N-body integrations over candidate Planet X parameters. The correct design is
to **parallelise over the candidate/ensemble dimension** — integrate `K` independent
candidate systems simultaneously, one Taichi thread per candidate, each running the same
10-body integration. This is embarrassingly parallel and is where Taichi earns its place.

**Architectural consequence:** the N-body state must be laid out with a leading
*ensemble* index from the start (`shape=(K, N)`), with `K = 1` for the baseline
simulation. Retrofitting this later would require rewriting every kernel.

### 3.3 Integrator and secular drift

Master rules forbid Euler and require velocity Verlet / leapfrog. Additional concerns:

- Kepler-90 is **extremely compact** (P from 7 d to 331 d, a from 0.074 to 0.97 AU).
  The innermost planet sets the timestep; the outermost sets the integration length.
  The period ratio is ~47×, so a fixed shared timestep is wasteful but simple and
  symplectic-safe. Adaptive or individual timesteps break symplecticity and must not be
  introduced casually.
- Fixed-step leapfrog is symplectic **only at constant dt**. dt must never be changed
  mid-run, and must never be tied to frame rate.
- The rendering loop must never drive the physics timestep.

### 3.4 Transit timing precision vs. integrator step

A transit centre must be located far more precisely than dt (dt will be ~0.01–0.1 d,
but we need seconds). Detecting a transit by "the step where the projected separation is
smallest" quantises transit times to dt and would swamp the signal. **Root-finding /
interpolation on the projected-separation derivative is required**, not nearest-step
sampling.

### 3.5 Ground-truth leakage in the inference stage

Phases 9–11 require the inference module to have *no* access to Planet X's true state.
In a single Python process this is easy to violate accidentally (shared globals, an
imported config, a cached object). **Mitigation:** the observer dataset must be
serialised to disk as the only channel between the generator and the solver, and
inference must import from `observations/` only — enforced by a test.

### 3.6 Reproducibility

Randomness enters via noise models and search initialisation. All must take explicit
seeds routed through one experiment-config object.

### 3.7 Scientific-claim risk

The project name ("The Invisible Planet") and the Kepler-90 setting make it easy to
imply a real discovery. Every artefact — UI, README, plots, demo narration — must state
that Planet X is synthetic. This is a documentation/UI requirement, tracked as a risk
because it is a *correctness* issue for a science audience, not a cosmetic one.

---

## 4. Recommended target architecture

Separation of concerns is a hard requirement (master rule 20). Physics never imports
rendering; inference never imports ground truth.

```
invisible_planet/
  constants.py         Units + physical constants. SINGLE source of truth (Phase 2).
  physics/
    state.py           Ensemble body state (K, N) containers; Taichi field allocation.
    gravity.py         Newtonian pairwise acceleration kernels (Taichi, f64).
    integrators.py     Velocity Verlet / leapfrog. Fixed dt. Symplectic.
    diagnostics.py     KE, PE, E_tot, linear p, angular L, centre of mass.
    kepler.py          Orbital elements <-> Cartesian. Kepler equation solver.
  systems/
    reference.py       Loads data/kepler90_reference.json -> validated dataclasses.
    kepler90.py        Builds the Cartesian initial state for the real system.
    planet_x.py        Synthetic Planet X parameterisation (Phase 7).
  observations/
    transits.py        Transit geometry + refined transit-centre finding (Phase 6).
    ttv.py             Linear ephemeris fit, O-C residuals.
    noise.py           Measurement noise model (explicit seed).
    dataset.py         Observer dataset writer/reader. THE ONLY inference input.
  inference/
    forward.py         theta -> simulated transit times (batched over candidates).
    loss.py            Objective function(s).
    search.py          Coarse-to-fine search. No ML. No ground-truth access.
    identifiability.py Degeneracy / correlation analysis (Phase 11).
  render/
    scene.py           Taichi GGUI 3D view. Consumes state; never writes it.
    panels.py          O-C diagram, residual plots, diagnostics readouts.
  app/
    story.py           Autonomous demonstration mode (Phase 14).
    investigate.py     Interactive investigation mode (Phase 15).
    main.py            Entry point.
  experiments/
    config.py          Seeds, timestep, durations, Planet X params. Serialisable.

tests/                 pytest. One command runs the whole scientific validation suite.
data/
  raw/                 Immutable archive snapshots + access date.
  sources_registry.json, kepler90_reference.json/.csv
docs/                  ARCHITECTURE, KEPLER90_DATA, NUMERICAL_METHODS, VALIDATION, ...
scripts/               Reproducible data-build scripts.
prototypes/            Pre-existing unrelated work (sim.py).
```

**Dependency direction (enforced, one-way):**

```
constants  <-  physics  <-  systems  <-  observations  <-  inference
                                   \                            /
                                    \-> render / app <---------/
```

`physics` must not import `render`, `app`, or `inference`.
`inference` must not import `systems.planet_x` ground-truth values.

---

## 5. Phase-by-phase mapping to modules

| Phase | Primary modules | Gate |
|---|---|---|
| 1 Data | `scripts/build_reference_dataset.py`, `data/` | Kepler III consistency, all sources recorded |
| 2 Units | `constants.py` | Representative magnitudes documented |
| 3 Two-body | `physics/`, `tests/test_twobody.py` | Kepler III, vis-viva, E, L within tolerance |
| 4 N-body | `physics/gravity.py`, `diagnostics.py` | Conservation tests pass |
| 5 Baseline | `systems/kepler90.py` | Stable over the observation baseline |
| 6 Transits | `observations/` | Transit times match period; O-C ≈ 0 for isolated orbit |
| 7 Planet X | `systems/planet_x.py` | m_X → 0 removes the signal (causality test) |
| 8 Sweep | `inference/forward.py` batched | Stability + detectability metrics recorded |
| 9 Observations | `observations/dataset.py` | No ground-truth leakage (tested) |
| 10 Inference | `inference/search.py` | Blind recovery reported honestly |
| 11 Identifiability | `inference/identifiability.py` | Degeneracies quantified, not hidden |
| 12 Validation | `tests/` | One command, machine-readable report |
| 13–16 UI | `render/`, `app/` | Rendering never alters physics |

---

## 6. Recommended immediate decisions (for review)

1. **Primary orbital solution: Weiss et al. 2024** — the only all-8-planet source, and
   internally self-consistent with its own stellar mass via Kepler's third law.
2. **Stellar parameters: Fulton & Petigura 2018** (M★ = 1.108 ± 0.035 M☉,
   R★ = 1.185 ± 0.031 R☉) — the same M★ Weiss et al. 2024 adopt, so no scale mixing.
3. **Masses of g and h: Liang et al. 2021** (TTV dynamical), cross-checked against the
   independent Shaw et al. 2025 analysis.
4. **Masses of b, c, i, d, e, f: derived** from the Chen & Kipping (2017) probabilistic
   mass–radius relation, flagged `derived` with its scatter documented. No other option
   exists without inventing values.
5. **Precision: f64 everywhere in physics; CPU backend for physics** (Vulkan f64 is not
   available on this machine).
6. **Ensemble-leading array layout `(K, N)` from day one**, so the inference phases do
   not require a rewrite.
7. Preserve `sim.py` under `prototypes/`; it is unrelated but is the user's work.

---

## 7. Speculative claims explicitly avoided

This report makes no claim about the existence, number, or properties of any unconfirmed
body in the real Kepler-90 system. The planet to be injected in later phases is
**synthetic** and exists only inside this numerical experiment.
