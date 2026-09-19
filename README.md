# The Invisible Planet

> **Infer an unseen gravitational perturber from the motion and transit timing of a real planetary system.**

An interactive Kepler-90 experiment built with **Python + Taichi**. A synthetic Planet X is placed in the N-body simulation, but the inference engine receives only noisy transit timings from the visible planets. Its gravitational fingerprint is then used to recover the planet's orbit.

> ### ⚠ Scientific boundary
> **Planet X is synthetic.** It is a body invented for a controlled numerical experiment.
> **Nothing here is a claim that the real Kepler-90 system contains a ninth planet**, and no
> output should be presented as a discovery. The *method* is real — it is how Neptune was
> predicted, and how the masses of the real Kepler-90 g and h were measured — but this planet
> is not.

## What it does

**Planet X → gravitational perturbation → transit-timing variation → O−C signal → inverse N-body fit → recovered planet**

The autonomous demo walks through this experiment from a normal Kepler-90 system to the final Planet X reveal. Planet X is present in the physics, and on screen, from *t* = 0; what the demo withholds is not the body but the *conclusion*. The inference never sees it — only a file of noisy transit times.

## Physics

- Newtonian N-body gravity, no softening and no fudge terms
- Fixed-step **velocity Verlet** (kick–drift–kick), symplectic, measured convergence order **2.000**
- Working units AU, days, solar masses, with *G* derived from the IAU 2015 nominal *GM*☉
- Geometric transit-centre detection by cubic Hermite root-finding, not timestep quantisation
- Transit Timing Variations (TTV) and O−C analysis
- Period search + nonlinear full N-body verification + posterior sampling
- No ML and no scripted planetary trajectories

## Results

The current synthetic experiment (`data/experiment_result.json`; regenerate with `scripts/run_experiment.py`):

| Quantity | True | Recovered |
|---|---:|---:|
| Mass | 100 M⊕ | 112 M⊕ *(posterior median; 68 %: 88–152)* |
| Semi-major axis | 0.2060 AU | 0.2058 AU *(68 %: 0.2054–0.2061)* |
| Period | 32.439 d | 32.389 d *(68 %: 32.306–32.454)* |

| Fit quality | |
|---|---:|
| Visible-only χ² | 123.0 *(28 transits, 24 dof)* |
| Best full N-body χ² | 31.4 |
| — planet d | 101.0 → 11.0 |
| — planet e | 22.0 → 20.4 |
| Best-fit period error | −0.059 % |
| False-alarm probability | **< 2.0 × 10⁻³** |

The orbital period is recovered tightly. Mass is less tightly constrained because mass and mutual inclination are degenerate in timing data alone.

**Two honest caveats on this table.**

1. The false-alarm probability is an *upper bound*, not a measurement: no noise-only trial out of 500 reached the observed statistic, so the exact look-elsewhere-corrected value is simply ≤ 1/501. A longer null run pushes the bound lower.
2. The posterior chain in the committed run is only ~9 autocorrelation times long, so **the credible intervals above are provisional** — the result file says so itself, in `inference.notes`. The truth falls inside the 95 % interval for mass, semi-major axis and period; it does not for eccentricity, because the true value (e = 0) sits exactly on the boundary of the sampler's parameterisation (e = k² + h² ≥ 0), which cannot place probability below it.

## Approximations and limitations

- Newtonian gravity; no GR, tides, or stellar oblateness.
- The model is 3D in the orbital state, but uses simplified adopted orbital assumptions, including mostly circular orbits and fixed nodes.
- Bodies are treated as point masses for gravity; collisions are not modeled.
- Transit detection is geometric; this is not a full photometric light-curve simulator.
- Timing noise is modeled as independent Gaussian noise using published timing uncertainties.
- Six of the eight real planets have no published mass; theirs are derived from a mass–radius relation and are flagged `derived` everywhere, never presented as measurements.
- The visible-only model does not reproduce the real g and h timing; planet g requires an explicit jitter term.
- The selected Planet X is *not* consistent with the real Kepler-90 d and e timing data. It was chosen for recoverability under pre-declared criteria, not for agreement with the real system.
- Planet X is **synthetic**. This is not a claim that Kepler-90 has a ninth planet.
- The experiment is a controlled numerical demonstration, not a real exoplanet discovery pipeline.

## Validation

**136 test functions across 12 files.** Nothing is validated against another run of this simulation: every check is against an analytic result, an independent implementation (REBOUND's IAS15), or a theoretical scaling law. Coverage:

- Newtonian force law, including that the perturbation vectors the UI draws are exactly one term of the same force sum
- two-body and Keplerian dynamics
- conservation of energy, momentum and angular momentum
- timestep convergence (order 2.000) and independent IAS15 comparison
- N-body interactions
- transit detection and O−C calculations
- Planet X causality (the signal vanishes as *m*→0 and scales as first-order theory requires)
- blind recovery
- noise reproducibility and the real observing pattern
- stability
- ground-truth leakage checks, including an AST check that the inference never imports the configuration holding the truth

Things this project got wrong, and how each was caught, are logged in [`docs/CORRECTIONS.md`](docs/CORRECTIONS.md).

## Run

Requires **Python 3.10–3.12**.

```bash
uv run --python 3.12 --extra dev python -m invisible_planet.app.main
```

Investigation mode — every control is a real model parameter (observation window, noise scale, candidate mass/axis/eccentricity/phase/tilt, run hypothesis):

```bash
uv run --python 3.12 --extra dev python -m invisible_planet.app.main --investigate
```

Validate:

```bash
uv run --python 3.12 --extra dev python scripts/run_validation.py
```

Reproduce the experiment:

```bash
uv run --python 3.12 --extra dev python scripts/run_experiment.py
```

> **Known issue:** the GGUI panel layer currently raises an ImGui `WithinFrameScope` assertion on
> this machine, so the two interactive modes above may fail to start. The physics, the inference
> and every script are unaffected and run headless.

## Data

The Kepler-90 reference dataset is sourced from the NASA Exoplanet Archive and cited literature. Per-transit timings come from Holczer et al. (2016) for planets d and e and Liang et al. (2021) for g and h; the published timing uncertainties are the noise model. Source provenance, uncertainties and a status flag (`observed` / `inferred` / `adopted-model` / `derived` / `synthetic` / `numerical`) for every value are recorded in `data/sources_registry.json` and [`docs/KEPLER90_DATA.md`](docs/KEPLER90_DATA.md).

## Documentation

| | |
|---|---|
| [`docs/UNITS.md`](docs/UNITS.md) | units, *G*, and the 4π² shortcut embedded in published semi-major axes *(generated)* |
| [`docs/NUMERICAL_METHODS.md`](docs/NUMERICAL_METHODS.md) | integrator, timestep choice, convergence study |
| [`docs/KEPLER90_DATA.md`](docs/KEPLER90_DATA.md) | every real value, its source and its uncertainty |
| [`docs/OBSERVING_PATTERN.md`](docs/OBSERVING_PATTERN.md) | the real observing cadence and per-transit uncertainties |
| [`docs/PLANET_X_DESIGN.md`](docs/PLANET_X_DESIGN.md) | the pre-declared criteria that selected Planet X |
| [`docs/IDENTIFIABILITY.md`](docs/IDENTIFIABILITY.md) | degeneracies and what the data can and cannot constrain |
| [`docs/VALIDATION.md`](docs/VALIDATION.md) | validation strategy and tolerances |
| [`docs/CORRECTIONS.md`](docs/CORRECTIONS.md) | errors found and fixed, and claims that were too strong |
| [`docs/HOSTILE_REVIEW.md`](docs/HOSTILE_REVIEW.md) | adversarial audit of the physics and the claims |
| [`PHYSICS_LOCK.md`](PHYSICS_LOCK.md) | the authoritative scientific contract |

> **The equations create the motion. The motion creates the evidence. The evidence reveals the planet.**
