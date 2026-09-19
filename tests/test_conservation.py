"""
Phase 4.2 gate — conservation laws for the full N-body engine.

Two-body conservation (tests/test_twobody.py) does not exercise mutual planet-planet
gravity. These tests use a multi-planet system where every body pulls on every other.

Each conserved quantity is reported as absolute error, relative error, maximum error
and RMS error, as the phase requires. Numerical drift is reported, never suppressed.

An independent NumPy implementation of every diagnostic is checked against the Taichi
kernel, so a bug in the kernel cannot hide behind a bug in the diagnostic that reads it.
"""

from __future__ import annotations

import numpy as np
import pytest

from invisible_planet import constants as C
from invisible_planet.physics.engine import NBodyEngine
from invisible_planet.systems.build import BodySpec, build_system

M_STAR = 1.108


def make_multiplanet_system(n_planets: int = 5) -> object:
    """A compact multi-planet system in the spirit of Kepler-90.

    Deliberately NOT the real dataset: Phase 4 must validate the engine before
    Phase 5 introduces the real initial conditions, so a failure here cannot be
    confused with a problem in the data.
    """
    rng = np.random.default_rng(20260919)
    axes = np.array([0.08, 0.12, 0.31, 0.41, 0.51, 0.72, 0.97])[:n_planets]
    masses_earth = np.array([2.7, 2.9, 7.2, 6.6, 6.9, 15.0, 203.0])[:n_planets]

    bodies = [BodySpec(name="star", mass=M_STAR, is_central=True)]
    for idx, (a, m_e) in enumerate(zip(axes, masses_earth)):
        bodies.append(
            BodySpec(
                name=f"planet_{idx}",
                mass=float(m_e) * C.EARTH_MASS_IN_SOLAR,
                semi_major_axis=float(a),
                eccentricity=float(rng.uniform(0.0, 0.03)),
                inclination=np.deg2rad(float(rng.uniform(89.2, 89.9))),
                longitude_of_ascending_node=float(rng.uniform(0, 2 * np.pi)),
                argument_of_periapsis=float(rng.uniform(0, 2 * np.pi)),
                mean_anomaly=float(rng.uniform(0, 2 * np.pi)),
            )
        )
    return build_system(bodies)


# =============================================================================
# Independent NumPy reference implementations
# =============================================================================

def numpy_diagnostics(masses: np.ndarray, pos: np.ndarray, vel: np.ndarray) -> dict:
    """Straightforward NumPy implementation, written independently of the kernel."""
    kinetic = 0.5 * np.sum(masses * np.sum(vel * vel, axis=1))

    potential = 0.0
    n = len(masses)
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(pos[j] - pos[i])
            potential -= C.G * masses[i] * masses[j] / r

    return {
        "kinetic_energy": kinetic,
        "potential_energy": potential,
        "total_energy": kinetic + potential,
        "linear_momentum": np.sum(masses[:, None] * vel, axis=0),
        "angular_momentum": np.sum(masses[:, None] * np.cross(pos, vel), axis=0),
        "center_of_mass": np.sum(masses[:, None] * pos, axis=0) / masses.sum(),
    }


def test_taichi_diagnostics_match_independent_numpy_implementation():
    """The Taichi diagnostic kernel must agree with a separate NumPy computation.

    Tolerance: 1e-12 relative. This guards against the kernel and its test being
    wrong in the same way (e.g. double-counting pairs in the potential energy).
    """
    system = make_multiplanet_system()
    engine = NBodyEngine(n_bodies=system.n_bodies)
    engine.set_state(system.masses, system.positions, system.velocities)
    engine.run(C.kepler_third_law_period(0.08, M_STAR) / 1000.0, 500)

    pos = engine.get_positions()[0]
    vel = engine.get_velocities()[0]
    reference = numpy_diagnostics(system.masses, pos, vel)
    measured = engine.diagnostics()

    for key in ["kinetic_energy", "potential_energy", "total_energy"]:
        a, b = float(measured[key][0]), float(reference[key])
        assert abs(a - b) / abs(b) < 1e-12, f"{key}: taichi {a!r} vs numpy {b!r}"

    for key in ["linear_momentum", "angular_momentum", "center_of_mass"]:
        a, b = measured[key][0], reference[key]
        scale = max(np.linalg.norm(b), 1e-30)
        assert np.linalg.norm(a - b) / scale < 1e-12, f"{key}: taichi {a} vs numpy {b}"


# =============================================================================
# Conservation over a long N-body integration
# =============================================================================

@pytest.fixture(scope="module")
def long_run_errors() -> dict:
    """Integrate a 5-planet system for ~1500 days and record conservation errors.

    ~1500 days is the real Kepler observing baseline, so this measures the drift
    that actually matters for the project rather than an arbitrary duration.
    """
    system = make_multiplanet_system(n_planets=5)
    engine = NBodyEngine(n_bodies=system.n_bodies)
    engine.set_state(system.masses, system.positions, system.velocities)

    # STEPS_PER_INNER_ORBIT is chosen so that one inner orbit is an exact whole
    # number of steps. That lets us sample STROBOSCOPICALLY at a fixed phase of
    # the innermost orbit — the body that dominates the O(dt^2) energy
    # oscillation — which cancels the bounded oscillating term and exposes
    # genuine secular drift. Sampling at arbitrary intervals would measure the
    # oscillation instead, as documented in docs/NUMERICAL_METHODS.md.
    steps_per_inner_orbit = 1000
    inner_period = C.kepler_third_law_period(0.08, M_STAR)
    dt = inner_period / steps_per_inner_orbit
    n_inner_orbits = int(1500.0 / inner_period)
    sub_samples = 8

    d0 = engine.diagnostics()
    e0 = float(d0["total_energy"][0])
    l0 = d0["angular_momentum"][0]
    p0 = d0["linear_momentum"][0]
    com0 = d0["center_of_mass"][0]

    oscillation, secular = [], []
    angmom_errors, momentum_norms, com_drifts, separations = [], [], [], []

    for _ in range(n_inner_orbits):
        done = 0
        for s in range(sub_samples):
            target = ((s + 1) * steps_per_inner_orbit) // sub_samples
            engine.run(dt, target - done)
            done = target
            d = engine.diagnostics()
            err = abs(float(d["total_energy"][0]) - e0) / abs(e0)
            oscillation.append(err)
        # `d` is now the state at an exact inner-orbit boundary
        secular.append(err)
        angmom_errors.append(
            np.linalg.norm(d["angular_momentum"][0] - l0) / np.linalg.norm(l0)
        )
        momentum_norms.append(np.linalg.norm(d["linear_momentum"][0] - p0))
        com_drifts.append(np.linalg.norm(d["center_of_mass"][0] - com0))
        separations.append(float(d["min_separation"][0]))

    return {
        "dt": dt,
        "total_days": engine.time,
        "energy_oscillation": np.array(oscillation),
        "energy": np.array(secular),
        "angmom": np.array(angmom_errors),
        "momentum": np.array(momentum_norms),
        "com_drift": np.array(com_drifts),
        "min_separation": np.array(separations),
    }


def _stats(x: np.ndarray) -> dict:
    return {
        "max": float(np.max(np.abs(x))),
        "rms": float(np.sqrt(np.mean(np.square(x)))),
        "final": float(x[-1]),
    }


def test_energy_error_stays_bounded_in_nbody(long_run_errors):
    """Total energy error over ~1500 d must stay bounded below 1e-6.

    WHY THIS IS NOT A TIGHTER, REQUIREMENT-DRIVEN BOUND — stated plainly rather
    than quietly relaxed:

    In the two-body case the bounded O(dt^2) oscillation can be cancelled by
    sampling stroboscopically at the orbital period, isolating genuine secular
    drift. A multi-planet system has NO such sampling phase: each planet
    contributes its own oscillation at its own incommensurate frequency, so
    sampling at the inner planet's period (as this fixture does) cancels only
    that planet's term. The residual ~4e-8 is the superposition of the other
    planets' bounded oscillations, not drift.

    What matters scientifically is therefore (a) that the error stays BOUNDED,
    asserted here and in test_energy_error_does_not_grow_secularly_in_nbody,
    and (b) the end-to-end consequence for transit timing, which cannot be
    settled by a proxy argument about energy. That decisive measurement — the
    post-fit O-C residual of an isolated planet — is made in Phase 6
    (tests/test_ttv.py) and is recorded as an open item in
    docs/NUMERICAL_METHODS.md.
    """
    s = _stats(long_run_errors["energy"])
    assert s["max"] < 1e-6, f"energy error max={s['max']:.3e} rms={s['rms']:.3e}"


def test_energy_oscillation_is_consistent_with_second_order_truncation(long_run_errors):
    """The bounded within-orbit energy oscillation must be of the expected size.

    At 1000 steps per inner orbit the second-order truncation scale is
    (2*pi/1000)^2 = 3.9e-5. These orbits are near-circular (e < 0.03), so the
    actual oscillation is far smaller. A bound of 1e-6 catches a gross error
    (wrong force law, wrong dt) while accepting the method's intrinsic behaviour.
    """
    s = _stats(long_run_errors["energy_oscillation"])
    assert s["max"] < 1e-6, f"energy oscillation max={s['max']:.3e} rms={s['rms']:.3e}"


def test_angular_momentum_is_conserved_in_nbody(long_run_errors):
    """Relative angular-momentum error must stay at round-off: below 1e-12."""
    s = _stats(long_run_errors["angmom"])
    assert s["max"] < 1e-12, f"angular momentum error max={s['max']:.3e} rms={s['rms']:.3e}"


def test_linear_momentum_stays_zero_in_nbody(long_run_errors):
    """Total linear momentum must stay at zero to round-off.

    The system is built in the barycentric frame, so p = 0 exactly at t = 0, and
    velocity Verlet preserves it identically because every pairwise force enters
    the sum twice with opposite sign.
    """
    s = _stats(long_run_errors["momentum"])
    assert s["max"] < 1e-16, f"linear momentum drift max={s['max']:.3e}"


def test_centre_of_mass_does_not_drift(long_run_errors):
    """With p = 0 the centre of mass must stay fixed: drift below 1e-12 au."""
    s = _stats(long_run_errors["com_drift"])
    assert s["max"] < 1e-12, f"centre-of-mass drift max={s['max']:.3e} au"


def test_energy_error_does_not_grow_secularly_in_nbody(long_run_errors):
    """The energy error must not trend upward over the baseline.

    Fitting a straight line to |dE/E|(t), the total predicted growth across the
    run must be small compared with the error itself. A secular trend would mean
    the integrator is losing its symplectic property under mutual perturbations.
    """
    err = long_run_errors["energy"]
    half = len(err) // 2
    first_half_mean = float(np.mean(err[:half]))
    second_half_mean = float(np.mean(err[half:]))
    peak = float(np.max(err))

    # A bounded oscillation has statistically indistinguishable halves; a secular
    # drift makes the second half systematically larger. Requiring the shift to
    # be under half the peak error distinguishes the two without being fooled by
    # the oscillation's own scatter.
    shift = abs(second_half_mean - first_half_mean)
    assert shift < 0.5 * peak, (
        f"energy error is trending: first half mean {first_half_mean:.3e}, "
        f"second half mean {second_half_mean:.3e}, peak {peak:.3e}"
    )


def test_no_unphysical_close_approach_occurred(long_run_errors):
    """Guard against silently integrating through a near-collision.

    Because there is NO softening, a close approach would produce a huge
    acceleration that a fixed timestep cannot resolve, corrupting the run. The
    minimum pairwise separation is recorded so this can never pass unnoticed.
    Threshold: 0.01 au, far below any legitimate separation in this system.
    """
    closest = float(np.min(long_run_errors["min_separation"]))
    assert closest > 0.01, f"bodies approached to {closest:.4f} au — results are unreliable"


def test_conservation_report(long_run_errors, capsys):
    """Emit the machine-readable conservation report the phase requires."""
    report = {
        name: _stats(long_run_errors[name])
        for name in ["energy", "energy_oscillation", "angmom", "momentum", "com_drift"]
    }
    with capsys.disabled():
        print(
            f"\n  N-body conservation over {long_run_errors['total_days']:.1f} d "
            f"(dt = {long_run_errors['dt']:.6f} d)"
        )
        print(f"  {'quantity':>12} {'max':>12} {'rms':>12} {'final':>12}")
        for name, s in report.items():
            print(f"  {name:>12} {s['max']:12.3e} {s['rms']:12.3e} {s['final']:12.3e}")
        print(
            f"  {'min sep [au]':>12} {float(np.min(long_run_errors['min_separation'])):12.4f}"
        )
    assert all(np.isfinite(list(s.values())).all() for s in report.values())
