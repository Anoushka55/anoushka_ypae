"""
Phase 3 gate — two-body validation.

Every test states a PHYSICAL EXPECTATION, a NUMERICAL METRIC, and an explicit
TOLERANCE. Nothing here compares the simulation against another simulation: each
check is against an analytic result of Newtonian gravity.

If any test in this file fails, the project does not proceed to N-body work.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from invisible_planet import constants as C
from invisible_planet.physics.engine import NBodyEngine
from invisible_planet.physics.kepler import (
    elements_to_state,
    solve_kepler_equation,
    state_to_elements,
    transit_mean_anomaly,
)
from invisible_planet.systems.build import two_body_system

# A representative Kepler-90-like test orbit: solar-mass star, Neptune-mass planet.
M_STAR = 1.108
M_PLANET = 17.0 * C.EARTH_MASS_IN_SOLAR
A_TEST = 0.4  # au


def truncation_scale(steps_per_orbit: int, n_orbits: float = 1.0) -> float:
    """Expected size of velocity-Verlet truncation error, as a fraction.

    Velocity Verlet is second order, so the local error per step scales as
    (omega*dt)^2 where omega*dt = 2*pi / steps_per_orbit. Errors of this size are
    a PROPERTY OF THE METHOD, not a bug, and tolerances must be derived from it
    rather than guessed. Phase/along-track errors accumulate linearly in the
    number of orbits, which is why n_orbits appears.

    Tests below multiply this by an explicit, stated safety factor. The real
    proof of correctness is not any single threshold but the demonstrated
    second-order CONVERGENCE in tests/test_convergence.py.
    """
    return n_orbits * (2.0 * np.pi / steps_per_orbit) ** 2


# =============================================================================
# Unit system and constants
# =============================================================================

def test_gravitational_constant_matches_gaussian_constant():
    """G derived from the IAU nominal GM_sun must equal the classical Gaussian k^2.

    Two independent determinations; tolerance 1e-9 relative.
    """
    assert abs(C.G - C.GAUSS_K**2) / C.G < 1e-9


def test_four_pi_squared_shortcut_is_an_approximation_with_a_known_origin():
    """G M_sun = 4 pi^2 au^3/yr^2 is inexact, and the discrepancy has a known cause.

    It corresponds to a period of exactly 365.25 d at 1 au, whereas the Gaussian
    constant implies 2*pi/k = 365.2569 d. The relative error in GM is therefore
    (2*pi/k / 365.25)^2 - 1 = 3.777e-5; this test verifies that to 1e-3 of itself.
    Used consistently the shortcut would not alter any transit time; the project
    avoids it because published semi-major axes embed it (see docs/UNITS.md).
    """
    shortcut = 4.0 * math.pi**2 / C.DAYS_PER_YEAR_JULIAN**2
    relative_error = abs(shortcut - C.G) / C.G
    explained = (2.0 * math.pi / C.GAUSS_K / C.DAYS_PER_YEAR_JULIAN) ** 2 - 1.0
    assert abs(relative_error - explained) / explained < 1e-3


def test_catalogue_semi_major_axes_embed_the_shortcut():
    """The Weiss et al. (2024) axes must equal (M_star P_yr^2)^(1/3) to 1e-8.

    This is the evidence that the catalogue used the 4 pi^2 shortcut with the stellar
    mass alone, which is why the project derives `a` from the period instead.
    """
    import json
    from pathlib import Path

    ref = json.loads(
        (Path(__file__).resolve().parents[1] / "data" / "kepler90_reference.json").read_text(
            encoding="utf-8"
        )
    )
    m_star = ref["star"]["mass_solar"]["value"]
    for letter, planet in ref["planets"].items():
        period = planet["orbital_period_days"]["value"]
        catalogue = planet["semi_major_axis_au"]["value"]
        shortcut_axis = (m_star * (period / C.DAYS_PER_YEAR_JULIAN) ** 2) ** (1.0 / 3.0)
        assert abs(shortcut_axis - catalogue) / catalogue < 1e-8, letter


# =============================================================================
# Kepler solver
# =============================================================================

@pytest.mark.parametrize("eccentricity", [0.0, 0.01, 0.1, 0.3, 0.6, 0.9])
def test_kepler_equation_solution_satisfies_its_own_equation(eccentricity):
    """E - e sin E must reproduce M to machine precision. Tolerance 1e-12."""
    mean_anomaly = np.linspace(-np.pi, np.pi, 37)
    ecc_anom = solve_kepler_equation(mean_anomaly, eccentricity)
    residual = ecc_anom - eccentricity * np.sin(ecc_anom) - mean_anomaly
    assert np.max(np.abs(residual)) < 1e-12


@pytest.mark.parametrize("eccentricity", [0.0, 0.05, 0.2, 0.7])
def test_elements_state_roundtrip(eccentricity):
    """elements -> Cartesian -> elements must return the original elements.

    Tolerance: 1e-10 on a, 1e-10 on e, 1e-9 rad on angles.
    """
    mu = C.G * (M_STAR + M_PLANET)
    elements = dict(
        semi_major_axis=A_TEST,
        eccentricity=eccentricity,
        inclination=np.deg2rad(89.7),
        longitude_of_ascending_node=np.deg2rad(31.0),
        argument_of_periapsis=np.deg2rad(57.0),
        mean_anomaly=np.deg2rad(123.0),
    )
    r, v = elements_to_state(mu, **elements)
    back = state_to_elements(mu, r, v)

    assert abs(back["semi_major_axis"] - elements["semi_major_axis"]) < 1e-10
    assert abs(back["eccentricity"] - elements["eccentricity"]) < 1e-10
    assert abs(back["inclination"] - elements["inclination"]) < 1e-9

    if eccentricity > 1e-12:
        assert abs(back["argument_of_periapsis"] - elements["argument_of_periapsis"]) < 1e-9
        assert abs(back["mean_anomaly"] - elements["mean_anomaly"]) < 1e-9
    else:
        # A circular orbit has no periapsis, so omega and M are individually
        # UNDEFINED — only their sum (the argument of latitude) is physical.
        # Asserting them separately would be testing an arbitrary convention
        # rather than the physics.
        def wrap(x):
            return np.mod(x, 2.0 * np.pi)

        original = wrap(elements["argument_of_periapsis"] + elements["mean_anomaly"])
        recovered = wrap(back["argument_of_periapsis"] + back["mean_anomaly"])
        # Tolerance 1e-7 rather than 1e-9: recovering elements from a circular
        # state is ILL-CONDITIONED. The eccentricity vector is formed as
        # cross(v,h)/mu - r/|r|, a difference of two nearly equal vectors, so at
        # e = 0 it suffers catastrophic cancellation and its direction is only
        # determined to ~sqrt(machine epsilon) ~ 1.5e-8. This is a property of
        # the parameterisation, not of the integrator; the orbit itself is
        # represented exactly in Cartesian form.
        assert abs(wrap(recovered - original + np.pi) - np.pi) < 1e-7


def test_transit_geometry_convention():
    """At the transit mean anomaly the planet must lie in FRONT of the star.

    With +Z towards the observer, 'in front' means z > 0, and the sky-projected
    separation must be near zero for an edge-on orbit.
    """
    mu = C.G * (M_STAR + M_PLANET)
    for argp_deg in [0.0, 45.0, 90.0, 180.0, 270.0]:
        argp = np.deg2rad(argp_deg)
        for ecc in [0.0, 0.1]:
            m_transit = transit_mean_anomaly(ecc, argp)
            r, _ = elements_to_state(
                mu, A_TEST, ecc, np.pi / 2, 0.0, argp, m_transit
            )
            sky_separation = math.hypot(r[0], r[1])
            assert r[2] > 0.0, f"planet behind the star at argp={argp_deg}, e={ecc}"
            assert sky_separation < 1e-9 * A_TEST, "not aligned with the line of sight"


# =============================================================================
# Newtonian force law
# =============================================================================

def test_acceleration_matches_newtons_law():
    """The engine's acceleration must equal G*m/r^2 towards the other body.

    Tolerance 1e-14 relative — this is a direct formula check, so it should be
    correct to machine precision.
    """
    separation = 0.37
    m1, m2 = M_STAR, M_PLANET
    engine = NBodyEngine(n_bodies=2)
    engine.set_state(
        masses=np.array([m1, m2]),
        positions=np.array([[0.0, 0.0, 0.0], [separation, 0.0, 0.0]]),
        velocities=np.zeros((2, 3)),
    )
    acc = engine.acc.to_numpy()[0]

    expected_on_1 = C.G * m2 / separation**2
    expected_on_2 = C.G * m1 / separation**2

    assert abs(acc[0, 0] - expected_on_1) / expected_on_1 < 1e-14
    assert abs(acc[1, 0] + expected_on_2) / expected_on_2 < 1e-14
    assert np.allclose(acc[:, 1:], 0.0, atol=1e-18)


def test_newtons_third_law():
    """Total force on an isolated pair must vanish: m1*a1 + m2*a2 = 0."""
    engine = NBodyEngine(n_bodies=2)
    masses = np.array([M_STAR, M_PLANET])
    engine.set_state(
        masses=masses,
        positions=np.array([[0.1, -0.2, 0.05], [0.4, 0.3, -0.1]]),
        velocities=np.zeros((2, 3)),
    )
    acc = engine.acc.to_numpy()[0]
    net_force = (masses[:, None] * acc).sum(axis=0)
    scale = masses[1] * np.linalg.norm(acc[1])
    assert np.max(np.abs(net_force)) / scale < 1e-13


# =============================================================================
# Circular orbits
# =============================================================================

def test_circular_orbit_maintains_radius():
    """A circular orbit must keep a constant star-planet separation.

    Metric: peak-to-peak variation of |r_rel| over 20 orbits, relative to a.
    Tolerance: 2x the second-order truncation scale at 2000 steps/orbit
    (= 2.0e-5). The residual wobble is the integrator's O(dt^2) error, not a
    physical eccentricity; test_convergence.py proves it shrinks as dt^2.
    """
    steps_per_orbit = 2000
    system = two_body_system(M_STAR, M_PLANET, A_TEST, eccentricity=0.0)
    period = C.kepler_third_law_period(A_TEST, M_STAR + M_PLANET)
    dt = period / steps_per_orbit
    n_steps = int(round(20 * period / dt))
    tolerance = 2.0 * truncation_scale(steps_per_orbit)

    engine = NBodyEngine(n_bodies=2)
    engine.set_state(system.masses, system.positions, system.velocities)

    # sample in chunks: reading state back every single step would dominate the
    # run time, and the separation varies far too smoothly for that to matter
    chunk = 20
    separations = []
    for _ in range(n_steps // chunk):
        engine.run(dt, chunk)
        p = engine.get_positions()[0]
        separations.append(np.linalg.norm(p[1] - p[0]))

    separations = np.array(separations)
    variation = (separations.max() - separations.min()) / A_TEST
    assert variation < tolerance, (
        f"circular orbit radius varied by {variation:.3e}, tolerance {tolerance:.3e}"
    )


def test_circular_speed_matches_analytic_value():
    """Orbital speed of a circular orbit must equal sqrt(G*M_total/a).

    Tolerance 1e-12 relative.
    """
    system = two_body_system(M_STAR, M_PLANET, A_TEST, eccentricity=0.0)
    v_rel = np.linalg.norm(system.velocities[1] - system.velocities[0])
    expected = C.circular_speed(A_TEST, M_STAR + M_PLANET)
    assert abs(v_rel - expected) / expected < 1e-12


# =============================================================================
# Kepler's third law, measured from the integration itself
# =============================================================================

@pytest.mark.parametrize("a", [0.0742, 0.12, 0.4125, 0.9702])
def test_keplers_third_law_from_measured_period(a):
    """The NUMERICALLY MEASURED period must satisfy P^2 = 4 pi^2 a^3 / (G M).

    The period is measured by detecting a sign change of the y coordinate of the
    relative orbit and refining by linear interpolation — it is never assumed.
    Tolerance: 1e-6 relative.
    """
    system = two_body_system(M_STAR, M_PLANET, a, eccentricity=0.0)
    expected_period = C.kepler_third_law_period(a, M_STAR + M_PLANET)
    dt = expected_period / 5000.0

    engine = NBodyEngine(n_bodies=2)
    engine.set_state(system.masses, system.positions, system.velocities)

    def rel_y() -> float:
        p = engine.get_positions()[0]
        return float(p[1, 1] - p[0, 1])

    # start at y = 0 moving +y; integrate until y returns to zero from below
    y_prev, t_prev = rel_y(), engine.time
    measured = None
    for _ in range(int(round(1.5 * expected_period / dt))):
        engine.step(dt)
        y_now, t_now = rel_y(), engine.time
        if t_now > 0.5 * expected_period and y_prev < 0.0 <= y_now:
            # linear interpolation of the zero crossing
            measured = t_prev + (t_now - t_prev) * (-y_prev) / (y_now - y_prev)
            break
        y_prev, t_prev = y_now, t_now

    assert measured is not None, "orbit never completed"
    assert abs(measured - expected_period) / expected_period < 1e-6


# =============================================================================
# Vis-viva
# =============================================================================

@pytest.mark.parametrize("eccentricity", [0.0, 0.1, 0.4])
def test_vis_viva_holds_along_the_orbit(eccentricity):
    """At every point of the orbit, v^2 = G*M*(2/r - 1/a).

    'a' here is the INITIAL semi-major axis, so this simultaneously checks that
    the numerical orbit does not wander off its energy surface. Tolerance: 10x
    the second-order truncation scale at 20000 steps/orbit (= 9.9e-7). The
    safety factor covers the growth of truncation error with eccentricity
    (faster periapsis passage means a larger effective omega*dt).
    """
    steps_per_orbit = 20000
    system = two_body_system(M_STAR, M_PLANET, A_TEST, eccentricity=eccentricity)
    mu = C.G * (M_STAR + M_PLANET)
    period = C.kepler_third_law_period(A_TEST, M_STAR + M_PLANET)
    dt = period / steps_per_orbit
    tolerance = 10.0 * truncation_scale(steps_per_orbit)

    engine = NBodyEngine(n_bodies=2)
    engine.set_state(system.masses, system.positions, system.velocities)

    worst = 0.0
    for i in range(200):
        engine.run(dt, 200)
        p = engine.get_positions()[0]
        v = engine.get_velocities()[0]
        r_vec = p[1] - p[0]
        v_vec = v[1] - v[0]
        r = np.linalg.norm(r_vec)
        v2 = float(np.dot(v_vec, v_vec))
        expected = mu * (2.0 / r - 1.0 / A_TEST)
        worst = max(worst, abs(v2 - expected) / expected)

    assert worst < tolerance, f"vis-viva violated by {worst:.3e}, tolerance {tolerance:.3e}"


# =============================================================================
# Comparison against the exact analytic Kepler solution
# =============================================================================

@pytest.mark.parametrize("eccentricity", [0.0, 0.2])
def test_trajectory_matches_analytic_kepler_propagation(eccentricity):
    """The integrated trajectory must match the exact two-body solution.

    This is the strongest possible two-body test: the analytic Kepler
    propagation is an independent, closed-form answer, not another integration.

    Tolerance: 5x the accumulated second-order truncation scale over 10 orbits
    at 10000 steps/orbit (= 2.0e-5 of a). The dominant term is a small
    systematic frequency shift, characteristic of symplectic integrators, which
    produces an along-track phase error growing linearly with orbit count.
    """
    steps_per_orbit = 10000
    n_orbits = 10
    mu = C.G * (M_STAR + M_PLANET)
    period = C.kepler_third_law_period(A_TEST, M_STAR + M_PLANET)
    system = two_body_system(M_STAR, M_PLANET, A_TEST, eccentricity=eccentricity)

    dt = period / steps_per_orbit
    n_steps = int(round(n_orbits * period / dt))
    tolerance = 5.0 * truncation_scale(steps_per_orbit, n_orbits)

    engine = NBodyEngine(n_bodies=2)
    engine.set_state(system.masses, system.positions, system.velocities)
    engine.run(dt, n_steps)

    p = engine.get_positions()[0]
    numerical_rel = p[1] - p[0]

    elapsed = n_steps * dt
    mean_motion = 2.0 * np.pi / period
    analytic_rel, _ = elements_to_state(
        mu, A_TEST, eccentricity, 0.0, 0.0, 0.0, mean_motion * elapsed
    )

    error = np.linalg.norm(numerical_rel - analytic_rel) / A_TEST
    assert error < tolerance, (
        f"trajectory diverged from analytic solution by {error:.3e} a, "
        f"tolerance {tolerance:.3e}"
    )


# =============================================================================
# Conservation laws
# =============================================================================

@pytest.mark.parametrize("eccentricity", [0.0, 0.3])
def test_secular_energy_drift_is_negligible(eccentricity):
    """Energy sampled at a FIXED ORBITAL PHASE must not drift: below 1e-9 over 100 orbits.

    Measuring stroboscopically (once per orbit, same phase) is deliberate. A
    symplectic integrator carries a bounded energy oscillation of amplitude
    O(dt^2) around each orbit — ~1e-5 at 1000 steps/orbit — which is a harmless
    property of the method and is characterised in tests/test_convergence.py.
    Sampling at a fixed phase cancels that oscillation and exposes the quantity
    that actually threatens a long integration: genuine secular drift.

    TOLERANCE RATIONALE (requirement-driven, not fitted to the observed value):
    since P ~ E^(-3/2), a relative energy drift eps produces a relative period
    error ~1.5*eps. Over the ~1500-day observing baseline, eps = 1e-9 gives an
    accumulated timing error of ~1500 d * 1.5e-9 ~ 2e-6 d ~ 0.2 s — below the
    ~1 s transit-timing precision this project targets, and far below the
    minutes-scale TTV signals being measured.
    """
    system = two_body_system(M_STAR, M_PLANET, A_TEST, eccentricity=eccentricity)
    period = C.kepler_third_law_period(A_TEST, M_STAR + M_PLANET)
    dt = period / 2000.0

    engine = NBodyEngine(n_bodies=2)
    engine.set_state(system.masses, system.positions, system.velocities)

    e0 = float(engine.total_energy()[0])
    worst = 0.0
    for _ in range(100):
        engine.run(dt, 2000)
        worst = max(worst, abs(float(engine.total_energy()[0]) - e0) / abs(e0))

    assert worst < 1e-9, f"energy drifted by {worst:.3e}"


def test_energy_error_is_bounded_not_secular():
    """Symplectic signature: the energy error must not grow with time.

    We compare the worst error in the first half of the run with the worst in
    the second half; a secular (non-symplectic) drift would make the second much
    larger. Tolerance: second half < 10x first half.
    """
    system = two_body_system(M_STAR, M_PLANET, A_TEST, eccentricity=0.3)
    period = C.kepler_third_law_period(A_TEST, M_STAR + M_PLANET)
    dt = period / 1000.0

    engine = NBodyEngine(n_bodies=2)
    engine.set_state(system.masses, system.positions, system.velocities)
    e0 = float(engine.total_energy()[0])

    first_half, second_half = [], []
    for i in range(200):
        engine.run(dt, 1000)
        err = abs(float(engine.total_energy()[0]) - e0) / abs(e0)
        (first_half if i < 100 else second_half).append(err)

    assert max(second_half) < 10.0 * max(first_half), "energy error is growing secularly"


@pytest.mark.parametrize("eccentricity", [0.0, 0.3])
def test_angular_momentum_is_conserved(eccentricity):
    """Relative drift of total angular momentum must stay below 1e-12."""
    system = two_body_system(
        M_STAR, M_PLANET, A_TEST, eccentricity=eccentricity, inclination=np.deg2rad(89.7)
    )
    period = C.kepler_third_law_period(A_TEST, M_STAR + M_PLANET)
    dt = period / 2000.0

    engine = NBodyEngine(n_bodies=2)
    engine.set_state(system.masses, system.positions, system.velocities)

    l0 = engine.diagnostics()["angular_momentum"][0]
    l0_mag = np.linalg.norm(l0)

    worst = 0.0
    for _ in range(50):
        engine.run(dt, 2000)
        l = engine.diagnostics()["angular_momentum"][0]
        worst = max(worst, np.linalg.norm(l - l0) / l0_mag)

    assert worst < 1e-12, f"angular momentum drifted by {worst:.3e}"


def test_linear_momentum_and_centre_of_mass_are_conserved():
    """An isolated system built in the barycentric frame must keep p = 0 and a
    stationary centre of mass. Tolerance: |p| < 1e-18 au*M_sun/day, COM drift
    < 1e-12 au over 50 orbits.
    """
    system = two_body_system(M_STAR, M_PLANET, A_TEST, eccentricity=0.25)
    period = C.kepler_third_law_period(A_TEST, M_STAR + M_PLANET)
    dt = period / 1000.0

    engine = NBodyEngine(n_bodies=2)
    engine.set_state(system.masses, system.positions, system.velocities)

    d0 = engine.diagnostics()
    assert np.max(np.abs(d0["linear_momentum"][0])) < 1e-18

    engine.run(dt, 50 * 1000)
    d1 = engine.diagnostics()
    assert np.max(np.abs(d1["linear_momentum"][0])) < 1e-18
    assert np.max(np.abs(d1["center_of_mass"][0] - d0["center_of_mass"][0])) < 1e-12


# =============================================================================
# Ensemble machinery (the layout the inference phases depend on)
# =============================================================================

def test_ensemble_members_are_independent():
    """K independent systems integrated together must each reproduce the result
    they would have had alone. Tolerance: exact agreement to 1e-15.
    """
    axes = [0.2, 0.4, 0.6]
    period = C.kepler_third_law_period(max(axes), M_STAR + M_PLANET)
    dt = period / 5000.0
    n_steps = 3000

    # integrated together
    k = len(axes)
    masses = np.zeros((k, 2))
    positions = np.zeros((k, 2, 3))
    velocities = np.zeros((k, 2, 3))
    for i, a in enumerate(axes):
        s = two_body_system(M_STAR, M_PLANET, a, eccentricity=0.1)
        masses[i] = s.masses
        positions[i] = s.positions
        velocities[i] = s.velocities

    ensemble = NBodyEngine(n_bodies=2, n_ensembles=k)
    ensemble.set_state(masses, positions, velocities)
    ensemble.run(dt, n_steps)
    together = ensemble.get_positions()

    # integrated separately
    for i, a in enumerate(axes):
        s = two_body_system(M_STAR, M_PLANET, a, eccentricity=0.1)
        single = NBodyEngine(n_bodies=2)
        single.set_state(s.masses, s.positions, s.velocities)
        single.run(dt, n_steps)
        assert np.max(np.abs(single.get_positions()[0] - together[i])) < 1e-15
