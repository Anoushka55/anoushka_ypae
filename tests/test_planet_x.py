"""
Phase 7 gate - the synthetic Planet X and the control-vs-mystery experiment.

The experiment is only meaningful if the two systems differ by EXACTLY ONE THING. These tests
establish that, then establish causality: the signal must vanish when Planet X's mass goes to
zero and must scale with that mass as first-order perturbation theory requires.

They use a test-local candidate, independent of the candidate finally selected for the experiment.

REMINDER: Planet X is synthetic. Nothing here is a claim about the real Kepler-90 system.
"""

from __future__ import annotations

import numpy as np
import pytest

from invisible_planet import constants as C
from invisible_planet.experiments.sweep import (
    integrate_with_diagnostics,
    non_transiting_inclination_deg,
    orbital_residuals,
    template_dataset,
)
from invisible_planet.inference.forward import ForwardModel
from invisible_planet.inference.likelihood import TimingLikelihood
from invisible_planet.physics.engine import NBodyEngine
from invisible_planet.systems.kepler90 import (
    DEFAULT_DURATION_DAYS,
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

R_STAR = 0.005511


def candidate(mass=25.0, a=0.165, phase=1.0):
    return PlanetXParameters(mass_earth=mass, semi_major_axis_au=a, eccentricity=0.0,
                             mean_anomaly=phase,
                             inclination_deg=non_transiting_inclination_deg(a, R_STAR))


@pytest.fixture(scope="module")
def reference():
    return load_reference()


# =============================================================================
# The experiment is controlled
# =============================================================================

def test_control_system_has_no_synthetic_body(reference):
    control, mystery = control_and_mystery(candidate(), reference=reference)
    assert control.metadata["contains_synthetic_body"] is False
    assert PLANET_X_NAME not in control.names
    assert mystery.metadata["contains_synthetic_body"] is True
    assert mystery.names[-1] == PLANET_X_NAME
    assert mystery.n_bodies == control.n_bodies + 1


def test_visible_planets_are_identical_relative_to_the_star(reference):
    """Star-relative initial states of the visible planets must be identical to round-off.
    Tolerance 1e-14 of the scale of the state (the barycentric re-shift subtracts in f64)."""
    control, mystery = control_and_mystery(candidate(), reference=reference)
    n = control.n_bodies
    rel_pos_c = control.positions[1:n] - control.positions[0]
    rel_pos_m = mystery.positions[1:n] - mystery.positions[0]
    rel_vel_c = control.velocities[1:n] - control.velocities[0]
    rel_vel_m = mystery.velocities[1:n] - mystery.velocities[0]
    assert np.max(np.abs(rel_pos_c - rel_pos_m)) < 1e-14 * np.max(np.abs(rel_pos_c))
    assert np.max(np.abs(rel_vel_c - rel_vel_m)) < 1e-14 * np.max(np.abs(rel_vel_c))
    assert np.array_equal(control.masses, mystery.masses[:n])


def test_barycentric_difference_is_a_pure_galilean_shift(reference):
    """The frame offset between the two systems must be a uniform shift of every body."""
    control, mystery = control_and_mystery(candidate(), reference=reference)
    n = control.n_bodies
    dp = mystery.positions[:n] - control.positions[:n]
    dv = mystery.velocities[:n] - control.velocities[:n]
    assert np.max(np.abs(dp - dp[0])) < 1e-14 * np.max(np.abs(control.positions))
    assert np.max(np.abs(dv - dv[0])) < 1e-14 * np.max(np.abs(control.velocities))


def test_both_systems_are_barycentric(reference):
    for system in control_and_mystery(candidate(), reference=reference):
        momentum = (system.masses[:, None] * system.velocities).sum(axis=0)
        scale = float(np.max(np.abs(system.masses[:, None] * system.velocities)))
        assert np.max(np.abs(momentum)) / scale < 1e-14


def test_planet_x_obeys_the_same_force_law_as_every_other_body(reference):
    """No 'hidden planet force': the engine's accelerations must equal an independent Newtonian
    sum over ALL bodies, Planet X included. Tolerance 1e-13 relative."""
    mystery = build_system_with_planet_x(candidate(), reference=reference)
    engine = NBodyEngine(n_bodies=mystery.n_bodies)
    engine.set_state(mystery.masses, mystery.positions, mystery.velocities)
    acc = engine.acc.to_numpy()[0]
    for i in range(mystery.n_bodies):
        expected = np.zeros(3)
        for j in range(mystery.n_bodies):
            if i != j:
                d = mystery.positions[j] - mystery.positions[i]
                expected += C.G * mystery.masses[j] * d / np.linalg.norm(d) ** 3
        assert np.linalg.norm(acc[i] - expected) / np.linalg.norm(expected) < 1e-13


# =============================================================================
# Delta r(t), Delta v(t) and the timing signal
# =============================================================================

@pytest.fixture(scope="module")
def runs(reference):
    """Control plus mystery systems at four masses, integrated once."""
    dt = recommended_timestep(reference)
    control = build_kepler90(reference=reference)
    masses = [0.0, 5.0, 10.0, 20.0, 40.0]
    systems = [build_system_with_planet_x(candidate(m), reference=reference) for m in masses]
    names = ["Kepler-90 d", "Kepler-90 e"]
    control_run = integrate_with_diagnostics([control], names, DEFAULT_DURATION_DAYS, dt, n_samples=37)
    mystery_run = integrate_with_diagnostics(systems, names, DEFAULT_DURATION_DAYS, dt, n_samples=37)
    return {"masses": masses, "control": control_run, "mystery": mystery_run}


def _residuals(runs, i):
    return orbital_residuals(runs["mystery"]["rel_r"][:, i], runs["mystery"]["rel_v"][:, i],
                             runs["control"]["rel_r"][:, 0], runs["control"]["rel_v"][:, 0])


def test_zero_mass_planet_x_leaves_every_visible_orbit_unchanged(runs):
    """CAUSALITY: as m_X -> 0 the extra perturbation must vanish. A massless X is a test particle
    that exerts no force, so Delta r and Delta v of every visible planet must be zero to
    round-off.

    The scale of round-off is not a constant: the two systems differ only by a Galilean re-shift
    (rounded in f64), and a tight resonant chain amplifies that 1e-16 seed by ~1e2-1e3 over the
    window (measured: ~1e-11 au, i.e. millimetres). The tolerance is therefore stated RELATIVE TO THE
    SIGNAL: the massless residual must be below 1e-6 of the residual a 5 M_earth planet produces
    (measured ratio ~2e-8), and below 1e-9 au absolutely."""
    r = _residuals(runs, 0)
    signal = _residuals(runs, runs["masses"].index(5.0))
    assert max(r["max_dr_au"]) < 1e-9
    assert max(r["max_dr_au"]) < 1e-6 * max(signal["max_dr_au"])
    assert max(r["max_dv_au_per_day"]) < 1e-6 * max(signal["max_dv_au_per_day"])


def test_zero_mass_planet_x_leaves_transit_times_unchanged(runs, reference):
    """The same statement for the observable: transit times must not move by more than 1 ms."""
    forward = ForwardModel(template_dataset(), reference=reference)
    control = forward._at_dataset_epochs(runs["control"]["series"][0])
    massless = forward._at_dataset_epochs(runs["mystery"]["series"][0])
    for k in control:
        assert np.max(np.abs(control[k] - massless[k])) * 86400.0 < 1e-3


def test_orbital_perturbation_grows_linearly_with_the_perturbers_mass(runs):
    """First-order perturbation theory: the displacement of a visible planet scales as m_X.
    Doubling the mass (5 -> 10 -> 20 M_earth) must double max |Delta r| within 25%."""
    dr = {m: max(_residuals(runs, i)["max_dr_au"]) for i, m in enumerate(runs["masses"])}
    assert 1.6 < dr[10.0] / dr[5.0] < 2.5
    assert 1.6 < dr[20.0] / dr[10.0] < 2.5


def test_timing_signal_grows_quadratically_with_mass_at_small_mass(runs, reference):
    """The expected delta chi2 is |signal|^2 and the signal is linear in m_X, so doubling a SMALL
    mass must multiply it by ~4 (within 35%; the deviation at larger mass is genuine
    nonlinearity, not error)."""
    template = template_dataset()
    forward = ForwardModel(template, reference=reference)
    likelihood = TimingLikelihood(template)
    control = forward._at_dataset_epochs(runs["control"]["series"][0])
    delta = {}
    for i, m in enumerate(runs["masses"]):
        pred = forward._at_dataset_epochs(runs["mystery"]["series"][i])
        v = likelihood.model_vector(pred, control)
        delta[m] = float(v @ v)
    assert delta[0.0] < 1e-12
    assert delta[5.0] < delta[10.0] < delta[20.0] < delta[40.0]
    assert 2.6 < delta[10.0] / delta[5.0] < 5.4


def test_visible_planets_share_the_star_relative_frame_for_delta_r(runs):
    """Delta r is defined star-relative so that the barycentric shift cancels: it must be
    identical to zero for the massless case even though the barycentric frames differ."""
    assert max(_residuals(runs, 0)["max_dr_au"]) < 1e-9


def test_planet_x_parameters_are_labelled_synthetic():
    record = candidate().to_dict()
    assert record["status"] == "synthetic"
    assert "not real" in record["warning"].lower()
    mystery = build_system_with_planet_x(candidate())
    assert mystery.metadata["planet_x"]["status"] == "synthetic"
    assert "synthetic" in PLANET_X_NAME.lower()
