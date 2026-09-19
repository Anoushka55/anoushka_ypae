# Final Architecture

**Phase 22 deliverable.** What was built, where the physics lives, and how a judge
can inspect it quickly.

## Where to look first

| Question | File |
|---|---|
| **What is the force law?** | `invisible_planet/physics/engine.py` — one kernel, ~25 lines |
| **What is the integrator?** | same file, `_integrate` |
| **Where are the constants?** | `invisible_planet/constants.py` — the only place any exist |
| **Where does the data come from?** | `scripts/build_reference_dataset.py` → `data/sources_registry.json` |
| **How is a transit measured?** | `engine.py`, transit-detection block inside `_integrate` |
| **How is the planet inferred?** | `invisible_planet/inference/search.py` |
| **Can the solver see the answer?** | `invisible_planet/observations/dataset.py` + `tests/test_inference.py` |
| **What is the scientific contract?** | `PHYSICS_LOCK.md` |

## Layout

```
invisible_planet/
  constants.py          units + physical constants (SINGLE source of truth)
  physics/
    backend.py          Taichi init: f64, CPU (idempotent)
    engine.py           N-body kernel, velocity Verlet, transit detection
    kepler.py           orbital elements <-> Cartesian, Kepler solver
    diagnostics.py      osculating elements, Hill separations
  systems/
    build.py            elements -> barycentric Cartesian initial conditions
    kepler90.py         the real system, from the reference dataset
    planet_x.py         the SYNTHETIC body + control/mystery pair
  observations/
    transits.py         run a system, collect transit times (single + batched)
    ttv.py              linear ephemeris fit, O-C residuals
    noise.py            measurement noise, from published uncertainties
    dataset.py          the observer dataset + leakage guards
  inference/
    forward.py          candidate parameters -> predicted transit times
    search.py           periodogram, peak verification, chi-squared
  render/scene.py       Taichi GGUI view (consumes state, never writes it)
  app/
    story.py            autonomous demonstration (Phase 14)
    investigate.py      interactive hypothesis testing (Phase 15)
    main.py             entry point
  experiments/config.py seeds, timestep, baseline, ground truth

scripts/                reproducible pipeline (data build, sweep, experiment,
                        identifiability, validation)
tests/                  111 tests
data/raw/               immutable archive snapshot + access date
docs/                   architecture, units, numerical methods, data, validation,
                        Planet X design, identifiability, hostile review
prototypes/             unrelated prior work, quarantined
```

## Dependency direction (one-way, enforced by test)

```
constants <- physics <- systems <- observations <- inference
                              \                        /
                               \--> render / app <----/
```

`physics` never imports `render`, `app` or `inference`.
`inference` never imports `experiments.config` — verified by parsing its AST.

## Key design decisions, and why

| Decision | Reason |
|---|---|
| **f64, CPU backend** | f32 resolves time to only ~86 s at t~10⁴ d — the same size as the signal |
| **Ensemble-leading arrays `(K, N)`** | parallelism that matters is across *candidate systems*, not bodies (N~10) |
| **Time loop inside the kernel** | per-step kernel launches made the suite ~25× slower |
| **Transit detection inside the integrator** | sub-second transit centres need the bracketing states; sampling outside would quantise to 10 min |
| **No softening** | it would alter the force law the entire inference rests on |
| **Periodogram, not grid search** | the χ² minimum is ~0.0005 au wide; a coarse grid finds false answers |
| **Nonlinear re-scoring of peaks** | the linearised model's *best* peak was demonstrably wrong |
| **Small (5 M⊕) linearisation reference mass** | a large one destabilises trial systems and manufactures spurious peaks |
| **Observer dataset on disk** | makes ground-truth isolation structural rather than a matter of care |

## Status

- **111 tests pass**; **18/18** contract requirements PASS (`VALIDATION_RESULTS.md`).
- Blind recovery works: period to 0.1%, mass to within a factor of ~2.
- Known limitations are enumerated in `PHYSICS_LOCK.md` §12 and
  `docs/HOSTILE_REVIEW.md`.
