"""
Phase 7 gate — the synthetic Planet X and the control-vs-mystery experiment.

The experiment is only meaningful if the two systems differ by EXACTLY ONE
THING. These tests establish that, and then establish causality: the signal must
vanish when Planet X's mass goes to zero, and must scale with that mass.

REMINDER: Planet X is synthetic. Nothing here is a claim about the real
Kepler-90 system.
"""

from __future__ import annotations

import numpy as np
import pytest

from invisible_planet import constants as C
from invisible_planet.observations.transits import observe_transits
from invisible_planet.observations.ttv import compare_ephemerides, fit_linear_ephemeris
from invisible_planet.physics.engine import NBodyEngine
from invisible_planet.systems.kepler90 import (
    build_kepler90,
    load_reference,
    recommended_timestep,
)
from invisible_planet.systems.planet_x import (
    PLANET_X_NAME,
    PlanetXParameters,
    build_system_with_planet_x,
    control_and_mystery,
)

# Working candidate, placed in the wide gap between planets i (0.120 au) and
# d (0.309 au). Phase 8 searches this space properly; this is the fixture the
# Phase 7 machinery is validated with.
CANDIDATE = PlanetXParameters(
    mass_earth=20.0,
    semi_major_axis_au=0.19,
    eccentricity=0.0,
    mean_anomaly=1.0,
)

BASELINE_DAYS = 600.0


@pytest.fixture(scope="module")
def reference():
    return load_reference()


# =============================================================================
# The experiment is controlled
# =============================================================================

def test_control_system_has_no_synthetic_body(reference):
    control, mystery = control_and_mystery(CANDIDATE, reference=reference)
    assert control.metadata["contains_synthetic_body"] is False
    assert PLANET_X_NAME not in control.names
    assert mystery.metadata["contains_synthetic_body"] is True
    assert mystery.names[-1] == PLANET_X_NAME
    assert mystery.n_bodies == control.n_bodies + 1


def test_visible_planets_are_identical_relative_to_the_star(reference):
    """Star-relative initial states of the visible planets must be bit-identical.

    This is the foundation of the whole experiment. If the visible planets
    started even slightly differently, any later timing difference could be
    attributed to the initial conditions rather than to Planet X's gravity.
    """
    control, mystery = control_and_mystery(CANDIDATE, reference=reference)
    n_visible = control.n_bodies

    control_rel_pos = control.positions[1:n_visible] - control.positions[0]
    mystery_rel_pos = mystery.positions[1:n_visible] - mystery.positions[0]
    control_rel_vel = control.velocities[1:n_visible] - control.velocities[0]
    mystery_rel_vel = mystery.velocities[1:n_visible] - mystery.velocities[0]

    # Tolerances are RELATIVE to the magnitude of the quantity. The barycentric
    # re-shift performs a subtraction in floating point, so agreement can only
    # be expected at the level of machine epsilon times the scale of the values,
    # not at an absolute threshold.
    position_scale = float(np.max(np.abs(control_rel_pos)))
    velocity_scale = float(np.max(np.abs(control_rel_vel)))

    assert np.max(np.abs(control_rel_pos - mystery_rel_pos)) < 1e-14 * position_scale
    assert np.max(np.abs(control_rel_vel - mystery_rel_vel)) < 1e-14 * velocity_scale
    assert np.array_equal(control.masses, mystery.masses[:n_visible])


def test_barycentric_difference_is_a_pure_galilean_shift(reference):
    """The frame offset between the two systems must be a uniform shift.

    Adding a mass moves the barycentre, so barycentric coordinates differ. That
    is physically inert only if the shift is IDENTICAL for every body; a
    body-dependent offset would be a real difference in the initial conditions.
    """
    control, mystery = control_and_mystery(CANDIDATE, reference=reference)
    n_visible = control.n_bodies

    position_shift = mystery.positions[:n_visible] - control.positions[:n_visible]
    velocity_shift = mystery.velocities[:n_visible] - control.velocities[:n_visible]

    # relative to the scale of the state, for the round-off reason above
    position_scale = float(np.max(np.abs(control.positions)))
    velocity_scale = float(np.max(np.abs(control.velocities)))

    assert np.max(np.abs(position_shift - position_shift[0])) < 1e-14 * position_scale
    assert np.max(np.abs(velocity_shift - velocity_shift[0])) < 1e-14 * velocity_scale


def test_both_systems_are_barycentric(reference):
    control, mystery = control_and_mystery(CANDIDATE, reference=reference)
    for system in (control, mystery):
        momentum = (system.masses[:, None] * system.velocities).sum(axis=0)
        scale = float(np.max(np.abs(system.masses[:, None] * system.velocities)))
        assert np.max(np.abs(momentum)) / scale < 1e-14


def test_planet_x_obeys_the_same_force_law_as_every_other_body(reference):
    """There must be no special 'hidden planet force'.

    The acceleration the engine computes for a visible planet must equal the
    plain Newtonian sum over ALL bodies including Planet X, computed
    independently here. If Planet X were given any special treatment, this
    comparison would fail.
    """
    mystery = build_system_with_planet_x(CANDIDATE, reference=reference)
    engine = NBodyEngine(n_bodies=mystery.n_bodies)
    engine.set_state(mystery.masses, mystery.positions, mystery.velocities)
    engine_acc = engine.acc.to_numpy()[0]

    for i in range(mystery.n_bodies):
        expected = np.zeros(3)
        for j in range(mystery.n_bodies):
            if i == j:
                continue
            d = mystery.positions[j] - mystery.positions[i]
            expected += C.G * mystery.masses[j] * d / np.linalg.norm(d) ** 3
        assert np.linalg.norm(engine_acc[i] - expected) / np.linalg.norm(expected) < 1e-13


# =============================================================================
# Causality
# =============================================================================

def test_zero_mass_planet_x_leaves_transit_times_unchanged(reference):
    """CAUSALITY: as m_X -> 0 the extra perturbation must disappear entirely.

    A massless Planet X still orbits — it is a test particle — but it exerts no
    force, so the visible planets must reproduce the control transit times to
    numerical precision. This is the single most important check that the
    measured signal is caused by Planet X's GRAVITY and is not an artefact of
    adding a tenth body to the integration.
    """
    dt = recommended_timestep(reference)
    massless = CANDIDATE.with_mass(0.0)

    control = build_kepler90(reference=reference)
    mystery = build_system_with_planet_x(massless, reference=reference)

    control_series = observe_transits(control, BASELINE_DAYS, dt)
    mystery_series = observe_transits(
        mystery, BASELINE_DAYS, dt, tracked=control.names[1:]
    )

    for name in control.names[1:]:
        a = control_series.for_planet(name)
        b = mystery_series.for_planet(name)
        n = min(len(a), len(b))
        difference = np.abs(a[:n] - b[:n]) * C.SECONDS_PER_DAY
        assert difference.max() < 1e-3, (
            f"{name}: massless Planet X shifted transit times by up to "
            f"{difference.max():.2e} s — it is not gravitationally inert"
        )


def test_planet_x_produces_a_measurable_timing_signature(reference):
    """The mystery system must differ measurably from the control.

    Without this, there would be nothing to infer. The difference is quoted
    after removing a best-fit line, because a constant offset or a slightly
    different mean period is absorbed by any real ephemeris fit — the STRUCTURE
    is the signal.
    """
    dt = recommended_timestep(reference)
    control, mystery = control_and_mystery(CANDIDATE, reference=reference)

    control_series = observe_transits(control, BASELINE_DAYS, dt)
    mystery_series = observe_transits(
        mystery, BASELINE_DAYS, dt, tracked=control.names[1:]
    )

    detected = _signature_by_planet(control_series, mystery_series, control.names[1:])
    assert detected, "no planet had enough transits to measure O-C"

    strongest = max(detected.values())
    assert strongest > 1.0, (
        f"Planet X produced no measurable signature; largest detrended O-C "
        f"difference was {strongest:.4f} minutes. All planets: {detected}"
    )


def _signature_by_planet(control_series, mystery_series, names) -> dict:
    """Detrended O-C difference per planet, skipping planets with too few transits.

    An O-C residual needs at least three transits: two define the linear
    ephemeris exactly and leave identically zero residuals. Over a 600-day
    window the outer planets g (P = 211 d) and h (P = 332 d) transit only twice,
    so they carry no timing information on this baseline and are excluded rather
    than silently contributing zeros.
    """
    out = {}
    for name in names:
        control_times = control_series.for_planet(name)
        mystery_times = mystery_series.for_planet(name)
        if len(control_times) < 3 or len(mystery_times) < 3:
            continue
        out[name] = compare_ephemerides(control_times, mystery_times)[
            "detrended_peak_to_peak_minutes"
        ]
    return out


def test_signal_grows_monotonically_with_planet_x_mass(reference):
    """The perturbation must increase monotonically with the perturber's mass.

    A signal that did not grow with the perturbing mass would not be
    gravitational in origin.
    """
    dt = recommended_timestep(reference)
    control = build_kepler90(reference=reference)
    control_series = observe_transits(control, BASELINE_DAYS, dt)

    signal_by_mass = {}
    for mass_earth in [5.0, 20.0, 80.0]:
        mystery = build_system_with_planet_x(
            CANDIDATE.with_mass(mass_earth), reference=reference
        )
        mystery_series = observe_transits(
            mystery, BASELINE_DAYS, dt, tracked=control.names[1:]
        )
        detected = _signature_by_planet(control_series, mystery_series, control.names[1:])
        signal_by_mass[mass_earth] = max(detected.values())

    masses = sorted(signal_by_mass)
    signals = [signal_by_mass[m] for m in masses]
    assert signals == sorted(signals), (
        f"signal is not monotonic in Planet X mass: {signal_by_mass}"
    )
    assert signals[-1] > 3.0 * signals[0], (
        f"signal barely responded to a 16x mass increase: {signal_by_mass}"
    )


def test_planet_x_parameters_are_labelled_synthetic():
    """Scientific-integrity check: the synthetic label must be carried everywhere."""
    record = CANDIDATE.to_dict()
    assert record["status"] == "synthetic"
    assert "not real" in record["warning"].lower()

    mystery = build_system_with_planet_x(CANDIDATE)
    assert mystery.metadata["planet_x"]["status"] == "synthetic"
    assert "synthetic" in PLANET_X_NAME.lower()
