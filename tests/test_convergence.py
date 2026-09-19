"""
Phase 3.2 gate — timestep convergence and the timestep policy.

This is the test that actually establishes the integrator is correct. Absolute
error thresholds are always somewhat arbitrary; a demonstrated ORDER OF
ACCURACY is not. Velocity Verlet is second order, so halving dt must reduce the
truncation error by a factor of ~4. If the measured order were 1, the integrator
would be (or would have degenerated to) Euler — the failure mode the project
rules single out.

The measured convergence data also DERIVES the production timestep, rather than
it being chosen because the animation looked smooth.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from invisible_planet import constants as C
from invisible_planet.physics.engine import NBodyEngine
from invisible_planet.physics.kepler import elements_to_state, state_to_elements
from invisible_planet.systems.build import two_body_system

M_STAR = 1.108
M_PLANET = 17.0 * C.EARTH_MASS_IN_SOLAR
A_TEST = 0.4
ECC_TEST = 0.2

# Steps per orbit to scan. Each is a factor of 2 finer than the previous, so a
# second-order method should show error ratios of ~4 between neighbours.
STEPS_PER_ORBIT_SCAN = [250, 500, 1000, 2000, 4000]


def _run_one(steps_per_orbit: int, n_orbits: int = 5) -> dict:
    """Integrate a known two-body orbit and measure error against the analytic solution."""
    mu = C.G * (M_STAR + M_PLANET)
    period = C.kepler_third_law_period(A_TEST, M_STAR + M_PLANET)
    dt = period / steps_per_orbit
    n_steps = steps_per_orbit * n_orbits

    system = two_body_system(M_STAR, M_PLANET, A_TEST, eccentricity=ECC_TEST)
    engine = NBodyEngine(n_bodies=2)
    engine.set_state(system.masses, system.positions, system.velocities)

    energy_0 = float(engine.total_energy()[0])
    angmom_0 = engine.diagnostics()["angular_momentum"][0]

    # TWO DISTINCT ENERGY ERROR METRICS — they behave differently and conflating
    # them is a real trap:
    #
    #   oscillation amplitude : energy sampled WITHIN the orbit. A symplectic
    #       integrator's energy error oscillates around the orbit with amplitude
    #       O(dt^2). This is bounded and does not accumulate.
    #
    #   secular drift : energy sampled STROBOSCOPICALLY, once per orbit at the
    #       same orbital phase. The oscillating term is phase-locked and cancels,
    #       exposing any genuine long-term drift. For a symplectic method this
    #       must stay at round-off level forever; for Euler it would grow without
    #       bound. THIS is the quantity that threatens a long integration.
    samples_per_orbit = 16
    oscillation_amplitude = 0.0
    secular_drift = 0.0
    for _ in range(n_orbits):
        done = 0
        for s in range(samples_per_orbit):
            # chunk boundaries chosen so the chunks sum EXACTLY to one orbit
            target = ((s + 1) * steps_per_orbit) // samples_per_orbit
            engine.run(dt, target - done)
            done = target
            err = abs(float(engine.total_energy()[0]) - energy_0) / abs(energy_0)
            oscillation_amplitude = max(oscillation_amplitude, err)
        secular_drift = max(secular_drift, err)  # value at the orbit boundary
    worst_energy = oscillation_amplitude

    pos = engine.get_positions()[0]
    vel = engine.get_velocities()[0]
    r_num = pos[1] - pos[0]
    v_num = vel[1] - vel[0]

    elapsed = n_steps * dt
    mean_motion = 2.0 * np.pi / period
    r_exact, v_exact = elements_to_state(
        mu, A_TEST, ECC_TEST, 0.0, 0.0, 0.0, mean_motion * elapsed
    )

    angmom_1 = engine.diagnostics()["angular_momentum"][0]
    elements = state_to_elements(mu, r_num, v_num)

    return {
        "steps_per_orbit": steps_per_orbit,
        "dt": dt,
        "position_error": float(np.linalg.norm(r_num - r_exact) / A_TEST),
        "velocity_error": float(
            np.linalg.norm(v_num - v_exact) / np.linalg.norm(v_exact)
        ),
        "period_error": abs(elements["period"] - period) / period,
        "energy_drift": worst_energy,
        "energy_secular_drift": secular_drift,
        "angmom_drift": float(
            np.linalg.norm(angmom_1 - angmom_0) / np.linalg.norm(angmom_0)
        ),
    }


@pytest.fixture(scope="module")
def convergence_table() -> list[dict]:
    return [_run_one(s) for s in STEPS_PER_ORBIT_SCAN]


def _measured_order(table: list[dict], key: str) -> float:
    """Fit  log(error) = order * log(dt) + c  and return the measured order."""
    dt = np.array([row["dt"] for row in table])
    err = np.array([row[key] for row in table])
    good = err > 0
    slope, _ = np.polyfit(np.log(dt[good]), np.log(err[good]), 1)
    return float(slope)


def test_position_error_is_second_order(convergence_table):
    """Position error must scale as dt^2. Measured order in [1.7, 2.3]."""
    order = _measured_order(convergence_table, "position_error")
    assert 1.7 < order < 2.3, f"measured convergence order {order:.2f}, expected ~2"


def test_velocity_error_is_second_order(convergence_table):
    """Velocity error must scale as dt^2. Measured order in [1.7, 2.3]."""
    order = _measured_order(convergence_table, "velocity_error")
    assert 1.7 < order < 2.3, f"measured convergence order {order:.2f}, expected ~2"


def test_energy_oscillation_is_second_order(convergence_table):
    """The bounded energy oscillation must scale as dt^2. Order in [1.7, 2.3]."""
    order = _measured_order(convergence_table, "energy_drift")
    assert 1.7 < order < 2.3, f"measured convergence order {order:.2f}, expected ~2"


def test_secular_energy_drift_is_far_below_the_oscillation(convergence_table):
    """Symplectic signature: no long-term energy drift.

    Sampled at a fixed orbital phase, the energy error must be orders of
    magnitude smaller than the within-orbit oscillation — the oscillation is a
    bounded artefact of the method, whereas a secular drift would accumulate
    over the thousands of orbits this project integrates. Requirement: the
    stroboscopic drift is below 1% of the oscillation amplitude.
    """
    for row in convergence_table:
        ratio = row["energy_secular_drift"] / row["energy_drift"]
        assert ratio < 1e-2, (
            f"secular energy drift is {ratio:.2%} of the oscillation amplitude at "
            f"{row['steps_per_orbit']} steps/orbit — integrator may not be symplectic"
        )


def test_integrator_is_not_first_order(convergence_table):
    """Explicit guard against the single most common failure mode.

    Forward Euler would show order ~1 and a secularly growing energy error. This
    assertion exists so that silently swapping the integrator cannot pass CI.
    """
    order = _measured_order(convergence_table, "position_error")
    assert order > 1.5, f"integrator behaves as order {order:.2f} — Euler-like, not symplectic"


def test_angular_momentum_is_conserved_to_machine_precision(convergence_table):
    """Angular momentum is conserved EXACTLY by velocity Verlet, up to round-off.

    Unlike energy, angular momentum is preserved by the kick-drift-kick map
    identically (each sub-step is a shear that preserves r x v for a central
    force). Its drift should therefore stay at round-off level (< 1e-13) at
    EVERY timestep, showing no dt dependence at all.
    """
    for row in convergence_table:
        assert row["angmom_drift"] < 1e-13, (
            f"angular momentum drifted by {row['angmom_drift']:.3e} at "
            f"{row['steps_per_orbit']} steps/orbit"
        )


def test_phase_error_grows_linearly_in_time():
    """The along-track error must be a CONSTANT frequency offset, not a drift.

    This is the most important numerical result for a transit-timing project,
    and it deserves stating carefully.

    A symplectic integrator does not reproduce the exact Kepler frequency: it
    integrates a nearby "shadow" Hamiltonian whose orbital frequency differs by
    O(dt^2). That produces an along-track position error which, at
    1000 steps/orbit, reaches ~1e-3 of the semi-major axis after only a few
    orbits — naively alarming for measurements that need second-level accuracy.

    It is harmless PROVIDED the offset is constant, because every transit-timing
    analysis (this project's and every real one) fits a linear ephemeris
    T_n = T_0 + n*P to the transit times and works with the residuals. A
    constant period offset is absorbed exactly by the fitted P and cancels out
    of the O-C residuals. What would NOT be absorbed is an accelerating or
    erratic phase error.

    So we test the property we actually rely on: phase error must be linear in
    time. We fit phase_error(t) = c1*t + c2*t^2 and require the quadratic term
    to be negligible compared with the linear one.

    The end-to-end confirmation — that an isolated planet yields O-C residuals
    of order milliseconds after ephemeris fitting — is in tests/test_ttv.py.
    """
    steps_per_orbit = 1000
    n_orbits = 40
    mu = C.G * (M_STAR + M_PLANET)
    period = C.kepler_third_law_period(A_TEST, M_STAR + M_PLANET)
    dt = period / steps_per_orbit
    mean_motion = 2.0 * np.pi / period

    system = two_body_system(M_STAR, M_PLANET, A_TEST, eccentricity=ECC_TEST)
    engine = NBodyEngine(n_bodies=2)
    engine.set_state(system.masses, system.positions, system.velocities)

    times, phase_errors = [], []
    for orbit in range(1, n_orbits + 1):
        engine.run(dt, steps_per_orbit)
        pos = engine.get_positions()[0]
        r_num = pos[1] - pos[0]
        elapsed = orbit * steps_per_orbit * dt
        r_exact, _ = elements_to_state(
            mu, A_TEST, ECC_TEST, 0.0, 0.0, 0.0, mean_motion * elapsed
        )
        # signed angle between the numerical and exact position vectors
        cross_z = r_exact[0] * r_num[1] - r_exact[1] * r_num[0]
        dot = float(np.dot(r_exact[:2], r_num[:2]))
        times.append(elapsed)
        phase_errors.append(math.atan2(cross_z, dot))

    t = np.array(times)
    phi = np.unwrap(np.array(phase_errors))
    quad, lin, _ = np.polyfit(t, phi, 2)

    # Compare the two terms' contributions over the span of the run.
    span = t[-1]
    linear_contribution = abs(lin * span)
    quadratic_contribution = abs(quad * span**2)
    assert linear_contribution > 0, "no measurable phase error; test is not exercising anything"
    assert quadratic_contribution / linear_contribution < 1e-3, (
        f"phase error is not a constant frequency offset: quadratic term is "
        f"{quadratic_contribution / linear_contribution:.2e} of the linear term, so it "
        "would NOT be absorbed by a linear ephemeris fit"
    )


def test_convergence_table_is_monotonic(convergence_table):
    """Refining the timestep must never make the answer worse."""
    errors = [row["position_error"] for row in convergence_table]
    for coarse, fine in zip(errors, errors[1:]):
        assert fine < coarse, "error did not decrease when the timestep was refined"


if __name__ == "__main__":
    # Print the convergence study for docs/NUMERICAL_METHODS.md
    table = [_run_one(s) for s in STEPS_PER_ORBIT_SCAN]
    print(f"Two-body convergence study: a = {A_TEST} au, e = {ECC_TEST}, 5 orbits")
    print(
        f"{'steps/orbit':>12} {'dt [d]':>12} {'pos err':>12} {'vel err':>12} "
        f"{'E oscill.':>12} {'E secular':>12} {'ang.mom.':>12}"
    )
    for row in table:
        print(
            f"{row['steps_per_orbit']:12d} {row['dt']:12.6f} {row['position_error']:12.3e} "
            f"{row['velocity_error']:12.3e} {row['energy_drift']:12.3e} "
            f"{row['energy_secular_drift']:12.3e} {row['angmom_drift']:12.3e}"
        )
    for key in ["position_error", "velocity_error", "energy_drift"]:
        print(f"measured order, {key:16s}: {_measured_order(table, key):.3f}")
