# Validation

**Phase 4.2 deliverable**, extended by later phases. Every claim here is produced by
the test suite; reproduce with:

```
uv run --python 3.12 --extra dev python -m pytest tests/ -s
```

## Status

| Phase | Suite | Tests | Status |
|---|---|---|---|
| 3.1 Two-body | `tests/test_twobody.py` | 33 | **PASS** |
| 3.2 Convergence | `tests/test_convergence.py` | 8 | **PASS** |
| 4.2 N-body conservation | `tests/test_conservation.py` | 9 | **PASS** |
| | **total** | **50** | **PASS** |

## What is validated, and against what

Nothing in the suite compares the simulation against another simulation. Every check
is against an analytic result, an independent implementation, or a theoretical scaling
law.

| Check | Compared against |
|---|---|
| Force law | Closed-form `G m / r²` |
| Newton's third law | Exact cancellation of an isolated pair |
| Kepler's third law | Period **measured from the integration** by zero-crossing refinement |
| Vis-viva | Closed form `v² = GM(2/r − 1/a)` at 200 points along the orbit |
| Full trajectory | **Exact analytic Kepler propagation** after 10 orbits |
| Element conversions | Round-trip, plus Kepler's equation residual |
| Transit geometry | Independent geometric requirement (planet in front, sky separation ≈ 0) |
| Convergence order | Theoretical order of velocity Verlet (2) |
| Diagnostics kernel | **Independent NumPy implementation** of energy, momentum, angular momentum |
| G | Classical Gaussian constant k², an independent determination |

## Conservation results — 5-planet system, full Kepler baseline

Configuration: M★ = 1.108 M☉, five planets (2.7–203 M⊕, a = 0.08–0.51 au),
mutual gravity between all bodies, **dt = 0.007852 d (1000 steps per inner orbit)**,
integrated for **1499.7 days**.

```
    quantity          max          rms        final
      energy    4.069e-08    2.342e-08    3.119e-09      (bounded oscillation)
  energy osc.   7.911e-08    3.719e-08    3.119e-09
      angmom    4.765e-14    2.453e-14    4.544e-14
    momentum    1.898e-19    1.302e-19    1.694e-19
   com_drift    1.580e-16    7.588e-17    1.580e-16
  min sep [au]     0.0424
```

Interpretation:

- **Linear momentum ~10⁻¹⁹ and centre-of-mass drift ~10⁻¹⁶ au** — exact conservation
  up to round-off, as expected: every pairwise force enters the sum twice with
  opposite sign, and the system is built in the barycentric frame.
- **Angular momentum ~5 × 10⁻¹⁴ relative** — round-off level, with no dependence on
  dt. Velocity Verlet conserves r × v identically for central forces.
- **Energy error ~4 × 10⁻⁸ relative, bounded and non-trending.** Its two halves are
  statistically indistinguishable, which is the symplectic signature.
- **Minimum pairwise separation 0.042 au** — no close approach occurred, so the
  unsoftened force law was never pushed beyond what the fixed timestep can resolve.

### An honest caveat about the energy bound

In the two-body case the bounded O(dt²) oscillation can be removed by sampling
stroboscopically at the orbital period, isolating genuine secular drift (measured at
~10⁻¹² and falling as dt², see `docs/NUMERICAL_METHODS.md`). **A multi-planet system
admits no such sampling phase** — each planet oscillates at its own incommensurate
frequency — so the residual ~4 × 10⁻⁸ is a superposition of bounded oscillations, not
drift.

The suite therefore asserts what the data supports (boundedness, no trend) rather
than a tighter number it cannot justify. The decisive question — whether any of this
affects *transit timing* — is not settled by a proxy argument about energy and is
measured directly in Phase 6 as the post-fit O−C residual of an isolated planet.

## Errors deliberately left visible

- The along-track phase error of ~10⁻³ a after five orbits at 1000 steps/orbit is
  **real and reported**, not hidden. It is a constant frequency offset absorbed by the
  linear ephemeris fit; `test_phase_error_grows_linearly_in_time` verifies that the
  offset is constant (quadratic term < 10⁻³ of the linear term), which is the property
  that makes it harmless.
- Element recovery for a **circular** orbit is accurate only to ~10⁻⁷ rad, because the
  eccentricity vector is ill-conditioned at e = 0. This is a property of the
  parameterisation, not the integrator, and is documented in the test.

## Guards against the project's stated failure modes

| Prohibited failure | Guard |
|---|---|
| Euler integration | `test_integrator_is_not_first_order` fails if measured order < 1.5 |
| Hidden numerical drift | Conservation report prints max/RMS/final for every quantity |
| Kernel and test wrong in the same way | Independent NumPy diagnostics cross-check |
| Softening sneaking into the force law | `test_acceleration_matches_newtons_law` to 10⁻¹⁴ |
| Silent near-collision | `test_no_unphysical_close_approach_occurred` |
| Scripted trajectories | Trajectory compared against analytic Kepler propagation |
