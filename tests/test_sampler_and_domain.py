"""
Tests of the search domain, the orbital-plane parametrisation, the ensemble sampler and the
exact null distribution of the periodogram statistic.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from invisible_planet import constants as C
from invisible_planet.inference import domain, sampler
from invisible_planet.inference.likelihood import TimingLikelihood
from invisible_planet.inference.periodogram import (
    PeriodogramBasis,
    false_alarm_probability,
    null_distribution,
    scan,
)
from invisible_planet.systems.kepler90 import load_reference
from invisible_planet.systems.planet_x import PlanetXParameters


@pytest.fixture(scope="module")
def reference():
    return load_reference()


# =============================================================================
# The orbital-plane parametrisation
# =============================================================================

def _normal(inclination_deg, node_deg):
    i, o = np.deg2rad(inclination_deg), np.deg2rad(node_deg)
    return np.array([np.sin(i) * np.sin(o), -np.sin(i) * np.cos(o), np.cos(i)])


@pytest.mark.parametrize("tx,ty", [(0.0, 0.0), (1.8, 0.0), (0.0, 2.5), (-3.0, 1.5), (4.0, -4.0)])
def test_tilt_vector_equals_the_mutual_inclination(tx, ty):
    """The angle between the candidate's orbit normal and the visible planets' normal must equal
    the magnitude of the tilt vector, whatever its direction. Tolerance 1e-9 degrees."""
    inclination, node = sampler.tilt_to_inclination_node(tx, ty)
    visible = _normal(sampler.VISIBLE_PLANE_INCLINATION_DEG, 0.0)
    candidate = _normal(inclination, node)
    angle = np.degrees(np.arccos(np.clip(visible @ candidate, -1, 1)))
    assert angle == pytest.approx(np.hypot(tx, ty), abs=1e-6)


def test_zero_tilt_reproduces_the_visible_plane():
    inclination, node = sampler.tilt_to_inclination_node(0.0, 0.0)
    assert inclination == pytest.approx(sampler.VISIBLE_PLANE_INCLINATION_DEG, abs=1e-9)
    assert node == pytest.approx(0.0, abs=1e-9)


def test_edge_on_orbits_with_different_nodes_are_NOT_coplanar():
    """Why the tilt parametrisation exists: two exactly edge-on orbits whose nodes differ by
    30 degrees are inclined to each other by 30 degrees."""
    a, b = _normal(90.0, 0.0), _normal(90.0, 30.0)
    assert np.degrees(np.arccos(a @ b)) == pytest.approx(30.0, abs=1e-9)


def test_parameter_vector_round_trip_for_the_coplanar_case():
    p = PlanetXParameters(mass_earth=37.0, semi_major_axis_au=0.213, eccentricity=0.04,
                          mean_anomaly=2.0, argument_of_periapsis=1.1)
    x = sampler.parameters_to_vector(p)
    q = sampler.vector_to_parameters(x)
    assert q.mass_earth == pytest.approx(p.mass_earth, rel=1e-12)
    assert q.eccentricity == pytest.approx(p.eccentricity, rel=1e-9)
    assert q.argument_of_periapsis == pytest.approx(p.argument_of_periapsis, abs=1e-9)


# =============================================================================
# Domain and priors
# =============================================================================

def test_hill_separation_matches_the_textbook_definition(reference):
    """Delta = (a2 - a1) / R_H with R_H = ((m1+m2)/(3M))^(1/3) (a1+a2)/2, checked by hand."""
    m_star = reference["star"]["mass_solar"]["value"]
    planets = domain.visible_planets(reference)
    a_d = next(p for p in planets if p[0] == "d")
    a_x, m_x = 0.2, 20.0 * C.EARTH_MASS_IN_SOLAR
    d_in, d_out = domain.hill_separations(a_x, m_x, reference)
    inner = max((p for p in planets if p[1] < a_x), key=lambda p: p[1])
    r_h_out = ((m_x + a_d[2]) / (3.0 * m_star)) ** (1 / 3) * 0.5 * (a_x + a_d[1])
    assert d_out == pytest.approx((a_d[1] - a_x) / r_h_out, rel=1e-12)
    r_h_in = ((m_x + inner[2]) / (3.0 * m_star)) ** (1 / 3) * 0.5 * (a_x + inner[1])
    assert d_in == pytest.approx((a_x - inner[1]) / r_h_in, rel=1e-12)


def test_search_zones_exclude_the_neighbourhoods_of_known_planets(reference):
    zones = domain.stable_zones(reference)
    for _, a, _ in domain.visible_planets(reference):
        assert not any(lo <= a <= hi for lo, hi in zones), f"a zone contains a known planet at {a}"
    assert any(lo < 0.25 < hi for lo, hi in zones)     # the wide gap between i and d


def test_axis_step_resolves_the_chi2_valley():
    """The measured valley half-width for a 36-day perturber over 1400 days is ~0.0005 au; the
    sampling step must be several times finer."""
    step = domain.axis_step(0.22, 1400.0, 1.108)
    assert 0.5e-4 < step < 3e-4


def test_transiting_candidates_are_rejected_by_the_prior(reference):
    radius = 0.005511
    transiting = PlanetXParameters(mass_earth=20, semi_major_axis_au=0.2, inclination_deg=89.9)
    hidden = PlanetXParameters(mass_earth=20, semi_major_axis_au=0.2, inclination_deg=87.5)
    assert not domain.is_plausible(transiting, reference, radius)
    assert domain.is_plausible(hidden, reference, radius)


def test_crowding_a_neighbour_is_rejected_by_the_prior(reference):
    radius = 0.005511
    crowded = PlanetXParameters(mass_earth=150, semi_major_axis_au=0.29, inclination_deg=87.0)
    assert not domain.is_plausible(crowded, reference, radius)


# =============================================================================
# The ensemble sampler
# =============================================================================

def test_stretch_move_sampler_recovers_a_correlated_gaussian():
    """Goodman-Weare on a strongly correlated 4-D Gaussian: recovered mean and covariance
    must match. This is exactly the situation (a narrow ridge) the sampler is chosen for."""
    rng = np.random.default_rng(0)
    dim = 4
    # a valid (positive-definite) covariance with a thin ridge: eigenvalues spanning 4 decades,
    # rotated by a random orthogonal matrix so that the ridge is not axis-aligned
    q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
    cov = q @ np.diag([1e-3, 0.4, 1.5, 4.0]) @ q.T
    mean = np.array([1.0, -2.0, 0.5, 3.0])
    inverse = np.linalg.inv(cov)

    def log_prob(xs):
        d = xs - mean
        return -0.5 * np.einsum("ni,ij,nj->n", d, inverse, d)

    start = mean + rng.standard_normal((24, dim)) * 0.1
    result = sampler.run_ensemble(log_prob, start, n_steps=3000, seed=1, verbose=False)
    flat = result["chain"][800:].reshape(-1, dim)
    assert np.allclose(flat.mean(axis=0), mean, atol=0.15)
    assert np.allclose(np.cov(flat.T), cov, rtol=0.3, atol=0.15)
    assert 0.2 < result["acceptance"] < 0.8


def test_autocorrelation_time_of_white_noise_is_about_one():
    rng = np.random.default_rng(1)
    chain = rng.standard_normal((4000, 16, 3))
    tau = sampler.integrated_autocorr_time(chain)
    assert np.all(np.abs(tau - 1.0) < 0.25)


def test_autocorrelation_time_of_an_ar1_process_matches_theory():
    """For x_t = phi x_{t-1} + noise, tau = (1 + phi) / (1 - phi)."""
    rng = np.random.default_rng(2)
    phi, n_steps, n_walkers = 0.9, 20000, 8
    x = np.zeros((n_steps, n_walkers, 1))
    noise = rng.standard_normal((n_steps, n_walkers, 1))
    for t in range(1, n_steps):
        x[t] = phi * x[t - 1] + noise[t]
    expected = (1 + phi) / (1 - phi)
    assert sampler.integrated_autocorr_time(x[1000:])[0] == pytest.approx(expected, rel=0.25)


# =============================================================================
# The exact null distribution of the periodogram statistic
# =============================================================================

def _toy_setup(seed=0, n_axes=30):
    rng = np.random.default_rng(seed)
    epochs = {"d": np.array([0, 1, 2, 4, 5, 7, 8, 9, 11]), "e": np.array([0, 1, 3, 4, 6, 7])}
    sigma = {k: rng.uniform(0.01, 0.03, len(v)) for k, v in epochs.items()}
    ds = types.SimpleNamespace(
        planet_letters=["d", "e"], epochs=epochs,
        transit_times_bjd={k: 100.0 + 50.0 * epochs[k] for k in epochs},
        timing_uncertainty_days=sigma)
    lk = TimingLikelihood(ds)
    basis = np.zeros((n_axes, 2, lk.n_points))
    for i in range(n_axes):
        for j in range(2):
            v = rng.standard_normal(lk.n_points)
            off = 0
            for k in lk.letters:
                m = len(lk.epochs[k])
                v[off:off + m] = lk._projector[k] @ v[off:off + m]
                off += m
            basis[i, j] = v
    pb = PeriodogramBasis(np.linspace(0.15, 0.3, n_axes), basis, 5.0, 88.0, "toy")
    return lk, pb


def test_null_distribution_matches_a_brute_force_calculation():
    """The vectorised null distribution must equal an explicit trial-by-trial least squares
    with the same random numbers."""
    lk, pb = _toy_setup()
    n_trials, seed = 60, 11
    fast = null_distribution(lk, pb, n_trials=n_trials, seed=seed)

    rng = np.random.default_rng(seed)
    xi = rng.standard_normal((lk.n_points, n_trials))
    projector = np.zeros((lk.n_points, lk.n_points))
    off = 0
    for k in lk.letters:
        m = len(lk.epochs[k])
        projector[off:off + m, off:off + m] = lk._projector[k]
        off += m
    brute = []
    for t in range(n_trials):
        d = projector @ xi[:, t]
        brute.append(scan(pb, d)["delta_chi2"].max())
    assert np.allclose(fast, brute, rtol=1e-8)


def test_false_alarm_probability_is_calibrated_on_pure_noise():
    """On noise-only data the false-alarm probability must be ~uniform: the fraction of
    trials with FAP < 0.1 must be ~10% (within sampling error)."""
    lk, pb = _toy_setup()
    null = null_distribution(lk, pb, n_trials=4000, seed=3)
    independent = null_distribution(lk, pb, n_trials=1000, seed=99)
    faps = np.array([false_alarm_probability(s, null) for s in independent])
    assert np.mean(faps < 0.1) == pytest.approx(0.1, abs=0.035)
    assert np.mean(faps < 0.5) == pytest.approx(0.5, abs=0.05)


def test_periodogram_finds_an_injected_signal_in_the_toy_basis():
    lk, pb = _toy_setup()
    rng = np.random.default_rng(5)
    truth = 17
    signal = 3.0 * pb.basis[truth, 0] + 1.5 * pb.basis[truth, 1]
    data = signal + rng.standard_normal(lk.n_points) * 0.05
    result = scan(pb, data)
    assert int(np.argmax(result["delta_chi2"])) == truth
