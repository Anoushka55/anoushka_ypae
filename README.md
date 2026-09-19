# The Invisible Planet

**An unseen body bends the transits of the planets we can see — and from those
bent timings we can work out where it is and roughly how heavy it is.**

A scientifically rigorous, interactive astrophysics experiment built on real
Kepler-90 data, Newtonian N-body dynamics, and a transparent inverse-problem
solver. Python + Taichi.

> ## ⚠ Scientific boundary
>
> **Planet X is synthetic.** It is a body invented for a controlled numerical
> experiment. **Nothing in this project is a claim that the real Kepler-90 system
> contains a ninth planet**, and no output here should ever be presented as a
> discovery. The *method* is real — it is how Neptune was found, and it is how
> the masses of the real Kepler-90 g and h were measured — but this particular
> planet is not.

---

## 1. The scientific question

Can an unseen gravitating body be detected, and its properties constrained,
purely from the timing of transits of *other* planets?

## 2. Real-world relevance

Yes — and this is established science, not speculation:

- **Neptune (1846)** was predicted from anomalies in Uranus's orbit before anyone saw it.
- **Kepler-19c (2011)** was identified purely from transit timing variations induced
  in Kepler-19b. It does not transit. It was invisible.
- **Kepler-90 g and h** — two of the eight real planets in *this* system — have masses
  known only because they perturb each other's transit times (Liang et al. 2021;
  Shaw et al. 2025).

## 3. Kepler-90 background

Kepler-90 (also **KOI-351**, KIC 11442793) is a Sun-like star with **eight known
planets** — the joint-largest planetary system known. Its architecture is compact:
periods from 7.0 to 331.6 days, semi-major axes from 0.074 to 0.970 au.

Four adjacent pairs lie near low-order mean-motion resonances (b–c near 5:4,
c–i near 5:3, e–f near 4:3, f–g near 5:3), which is exactly why the system shows
strong transit timing variations.

## 4. Real data vs. synthetic experiment

| Real, from the literature | Synthetic, invented here |
|---|---|
| The star's mass and radius | Planet X's mass |
| The eight planets' periods, radii, transit epochs, impact parameters | Planet X's orbit |
| Measured masses of planets g and h (from TTVs) | — |
| Published transit-timing uncertainties (used as the noise model) | — |

Every real value carries a source, an uncertainty and a status flag
(`observed` / `inferred` / `adopted-model` / `derived` / `synthetic` / `numerical`)
in `data/sources_registry.json`. See **[docs/KEPLER90_DATA.md](docs/KEPLER90_DATA.md)**.

**Six of the eight planets have no published mass at all.** Their masses are derived
from the Weiss & Marcy (2014) mass–radius relation and are flagged `derived`
everywhere, with the relation's large intrinsic scatter (RMS 4.3 M⊕) carried through
as the uncertainty. They are never presented as measurements.

## 5. Physics

**Newtonian gravity, no softening, no fudge terms.** For body *i*:

```
a_i = G · Σ_{j≠i}  m_j (r_j − r_i) / |r_j − r_i|³
```

Every body obeys this, **including Planet X**. There is no special "hidden planet
force", and no trajectory anywhere in the codebase is scripted — every position is
one the integrator produced.

Softening is deliberately absent: it would alter the very force law the inference
rests on. Close approaches are *detected and reported* instead.

## 6. Numerical integration

**Velocity Verlet (kick–drift–kick), fixed timestep, symplectic.**
Measured convergence order: **2.000** in position, velocity and energy.

`dt = P_innermost / 1000 ≈ 0.00701 d ≈ 10 minutes`, justified by the convergence
study rather than by how smooth the animation looks — see
**[docs/NUMERICAL_METHODS.md](docs/NUMERICAL_METHODS.md)**.

All physics runs in **f64**. In f32 the representable time resolution at t ~ 10⁴ days
is ~86 s — the same order as the signals being measured — so f32 would manufacture
"TTVs" indistinguishable from real ones.

## 7. Units

**au / day / M☉**, with **G = 2.959122081921 × 10⁻⁴ au³ M☉⁻¹ day⁻²** derived from the
IAU 2015 nominal GM☉ and cross-checked against the Gaussian constant *k*² to 3 × 10⁻¹⁰.

The textbook shortcut G M☉ = 4π² au³ yr⁻² is **rejected**: it is wrong by 3.8 × 10⁻⁵,
which is ~8 minutes of coherent timing drift on Kepler-90 h — right on top of the
signal. See **[docs/UNITS.md](docs/UNITS.md)**.

## 8. Transits and O−C

A transit centre is the instant of minimum sky-projected star–planet separation with
the planet in front. It is located by **cubic Hermite root-finding** on the derivative
of the projected separation, using position, velocity and acceleration at the
bracketing steps — not by taking the nearest timestep, which would quantise transit
times to 10 minutes.

A linear ephemeris `T_n = T₀ + nP` is fitted and subtracted; the residual is the
**O−C**, and its structure is the TTV signal.

## 9. Validation

**85 tests, 18/18 contract requirements PASS.** One command:

```bash
uv run --python 3.12 --extra dev python scripts/run_validation.py
```

Nothing is validated against another simulation. Every check is against an analytic
result, an independent implementation, or a theoretical scaling law:

| Quantity | Result |
|---|---|
| Convergence order | **2.000** (theoretical: 2) |
| Angular momentum drift | ~5 × 10⁻¹⁴, independent of dt |
| Linear momentum | ~10⁻¹⁹ |
| Energy error (8-planet, 1500 d) | 4 × 10⁻⁸, bounded and non-trending |
| O−C residual, isolated planet | **< 1 s** |
| Physical TTV signal | ~106 s RMS |
| Best published timing precision | 71 s |

See **[VALIDATION_RESULTS.md](VALIDATION_RESULTS.md)** and
**[docs/VALIDATION.md](docs/VALIDATION.md)**.

## 10. The inverse problem

The inference sees **only** a JSON file of noisy transit times with uncertainties.
It never sees Planet X. This is enforced structurally, not by discipline:

- `ObserverDataset` has **no field** capable of holding Planet X's parameters.
- The dataset is written to disk; the solver reads that file.
- Two guards scan the serialised dataset — one for forbidden identifiers, one for
  the hidden body's **parameter values** — and a test verifies the guards actually fire.
- A test parses the inference modules' **AST** to prove they never import the
  experiment configuration that holds the ground truth.

**No machine learning. No LLM.** Every candidate is scored by an explicit χ² against
a full N-body integration.

## 11. Results

Blind recovery, from noisy transit times alone:

| Parameter | True | Recovered | Error |
|---|---|---|---|
| mass | 40.0 M⊕ | **35.0 M⊕** | −5.0 (truth inside 1σ: 32.5–40.0) |
| semi-major axis | 0.2200 au | **0.2202 au** | +0.0002 |
| orbital phase | 2.0900 rad | **2.0944 rad** | +0.0044 |
| period | 35.805 d | **35.842 d** | +0.037 |

**Δχ² = 50.9 over the no-planet hypothesis ≈ 7.1σ**, reduced χ² = 1.03.

## 12. What went wrong, and what it taught us

Reported because it is the most scientifically interesting part of the project.

**A coarse grid search cannot solve this problem.** The first search returned
mass 8.8 M⊕ and a = 0.162 au. Rather than tune it, χ² was evaluated *at the true
parameters* — and the truth scored **better** than anything the search had found.
So the optimiser was broken, not the physics.

The reason: the χ² minimum is **razor-sharp in period**.

```
a = 0.2195  →  χ² = 627.6
a = 0.2200  →  χ² = 516.8    ← true value
a = 0.2205  →  χ² = 564.7
```

A half-width of ~0.0005 au. The grid spacing was 0.03 au — **60× too coarse**. The
TTV signal is quasi-periodic, so a period error decorrelates the model across the
baseline. The remedy is a **periodogram**: scan period densely, solve the other
parameters analytically.

**And the fast approximate model must not be trusted to rank candidates.** The
linearised periodogram's *best* peak was wrong:

```
peak 1: a=0.2565  linearised χ²=519.8  →  nonlinear χ²= 695.6   (spurious)
peak 2: a=0.2198  linearised χ²=523.8  →  nonlinear χ²= 518.8   ← TRUTH
peak 5: a=0.2670  linearised χ²=531.8  →  nonlinear χ²=2150.7   (catastrophic)
```

Ranking by the fast model would have confidently reported a planet that does not
exist. Only re-scoring every shortlisted peak with full N-body physics found the
real one. The reference mass used to build the linear basis also matters: at
40 M⊕ the basis simulations were themselves unstable at some trial periods, which
manufactured spurious peaks. Dropping it to 5 M⊕ removed them.

## 13. Limits of what this shows

From **[docs/IDENTIFIABILITY.md](docs/IDENTIFIABILITY.md)**:

- **Period is sharply determined** (< 0.2%), and — importantly — it is recovered
  correctly *even when the solver assumes the wrong inclination*.
- **Mass is degenerate with inclination.** The solver assumes a near-coplanar orbit.
  Varying that assumption over 85°–89° changes the inferred mass from 20 to 50 M⊕,
  and the best χ² occurs at 87°, not at the true 88°. Once inclination is treated as
  unknown, **mass is constrained only to within a factor of ~2** — not the ~20%
  implied by a fixed-inclination χ² profile. Mass should be quoted as
  *"tens of Earth masses"*.
- **The adopted noise level is optimistic.** It uses published transit-*epoch*
  uncertainties, which are smaller than single-transit timing errors by roughly √N.
  At 2× the adopted noise the detection falls to 4.2σ.
- **The detection rests mainly on one planet.** Only planet d's induced signal
  (415 s RMS) exceeds its own timing uncertainty (363 s).
- **Significance is not monotonic in baseline** (3.5σ at 400 d, 3.3σ at 700 d,
  7.3σ at 1400 d) because the TTV signal has its own long super-period.
- At **twice** the real timing uncertainties the detection drops to ~4σ.

## 14. Assumptions and approximations

| Assumption | Status |
|---|---|
| Eccentricities of all planets = 0 | `adopted-model`; measured values exist only for g and h |
| Longitude of ascending node = 0 for all | unconstrained by transit photometry |
| Masses of b, c, i, d, e, f | `derived` from a mass–radius relation; **no published masses exist** |
| Timing noise is independent Gaussian | real errors share systematics |
| Point masses; no GR, no tides, no oblateness | not justified at this precision for this purpose |
| Inclination derived from impact parameter | see below |

Two data problems were found and fixed rather than absorbed:

1. **Kepler-90 h's published inclination (89.60°) implies b = 1.23 — it would never
   transit**, contradicting its discovery by transit. The same paper's impact
   parameter (0.36) is self-consistent. Inclination is now derived from *b*.
2. **The catalogue's semi-major axes omit the planet mass** (they use `a³ = GM★P²/4π²`).
   For h that distorts the period by 2.2 hours per orbit. `a` is now derived from the
   measured period using the correct total mass.

## 15. How to run

See **[RUN.md](RUN.md)**. Short version:

```bash
uv run --python 3.12 --extra dev python -m invisible_planet.app.main      # watch it
uv run --python 3.12 --extra dev python scripts/run_validation.py         # verify it
uv run --python 3.12 --extra dev python scripts/run_experiment.py         # reproduce it
```

## 16. Reproducibility

Every random element is seeded through one config object
(`invisible_planet/experiments/config.py`). The data layer is rebuilt from an
immutable archive snapshot in `data/raw/` with a recorded access date:

```bash
uv run --python 3.12 --extra dev python scripts/build_reference_dataset.py
```

## 17. Data sources

| Source | Used for |
|---|---|
| NASA Exoplanet Archive (`ps` table, TAP) | all catalogue values |
| Weiss et al. (2024), ApJS 270, 8 | periods, radii — the only all-8-planet solution |
| Cabrera et al. (2014), ApJ 781, 18 | transit epochs, impact parameters (b–h) |
| Shallue & Vanderburg (2018), AJ 155, 94 | Kepler-90 i discovery parameters |
| Liang et al. (2021), AJ 161, 202 | TTV dynamical masses of g and h |
| Shaw et al. (2025), AJ 170, 146 | independent TTV cross-check |
| Fulton & Petigura (2018), AJ 156, 264 | stellar mass and radius |
| Weiss & Marcy (2014), ApJ 783, L6 | mass–radius relation for the massless planets |

⚠ The archive indexes this system as **`KOI-351`**, not `Kepler-90`. Querying
`hostname='Kepler-90'` returns nothing, and `LIKE 'Kepler-90%'` silently returns
*Kepler-900 … 909* — the wrong systems with plausible-looking data.

## 18. Scientific claim

This project demonstrates, in a controlled numerical experiment, that **an unseen
gravitating body leaves measurable dynamical signatures in observable bodies, and
that those signatures constrain its properties** — its **orbital period tightly and
robustly**, its **mass only loosely**, to within a factor of ~2 once the orbital
inclination is treated as unknown.

A skeptical review of this work, including the objections that survived, is in
**[docs/HOSTILE_REVIEW.md](docs/HOSTILE_REVIEW.md)**.

**It does not claim:** that Kepler-90 has a ninth planet; that the mass is precisely
measured; that the adopted noise level reflects what real single-transit timing
errors would be; that this is a faithful
end-to-end replica of a real discovery pipeline; that the inferred parameters are
uniquely recoverable under all conditions; or that the simplified model captures
every effect present in real exoplanet observations.

> **The equations create the motion. The motion creates the evidence.
> The evidence reveals the planet.**
