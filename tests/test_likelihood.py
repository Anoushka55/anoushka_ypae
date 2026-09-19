"""
The timing likelihood: chi-squared with each planet's linear ephemeris profiled out.

These tests check the statistic against an independent brute-force minimisation, and check
the properties that make it the right statistic for gappy data with unequal uncertainties.
"""

from __future__ import annotations

import types

import numpy as np
import pytest
from scipy.optimize import minimize

from invisible_planet.inference.likelihood import INVALID_CHI2, TimingLikelihood
from invisible_planet.observations.ttv import fit_linear_ephemeris, profile_chi2


def make_dataset(seed=0, gaps=True):
    rng = np.random.default_rng(seed)
    epochs = {"d": np.array([0, 1, 2, 3, 5, 6, 8, 11, 12, 15]),
              "e": np.array([0, 2, 3, 4, 7, 9, 10])}
    sigma = {k: rng.uniform(0.005, 0.02, size=len(v)) for k, v in epochs.items()}
    period = {"d": 59.7, "e": 91.9}
    data = {k: 100.0 + period[k] * epochs[k] + rng.normal(0, 1, len(epochs[k])) * sigma[k]
            for k in epochs}
    return types.SimpleNamespace(
        planet_letters=["d", "e"], epochs=epochs, transit_times_bjd=data,
        timing_uncertainty_days=sigma)


def brute_force_chi2(t, m, n, s):
    def f(p):
        return float(np.sum(((t - m - p[0] - p[1] * n) / s) ** 2))

    return minimize(f, [0.0, 0.0], method="Nelder-Mead",
                    options={"xatol": 1e-12, "fatol": 1e-12, "maxiter": 20000}).fun


def test_profile_chi2_equals_brute_force_minimisation():
    """profile_chi2 must equal a direct numerical minimisation over (a, b), with gaps and
    unequal weights. Tolerance 1e-6 relative."""
    ds = make_dataset()
    rng = np.random.default_rng(1)
    for k in ds.planet_letters:
        n, s, t = ds.epochs[k].astype(float), ds.timing_uncertainty_days[k], ds.transit_times_bjd[k]
        model = t + rng.normal(0, 0.02, len(t))
        analytic, dof = profile_chi2(t, model, s, ds.epochs[k])
        assert dof == 2
        brute = brute_force_chi2(t, model, n, s)
        assert abs(analytic - brute) < 1e-6 * max(1.0, brute)


def test_likelihood_class_matches_profile_chi2_summed_over_planets():
    ds = make_dataset()
    lk = TimingLikelihood(ds)
    rng = np.random.default_rng(2)
    model = {k: ds.transit_times_bjd[k] + rng.normal(0, 0.02, len(ds.epochs[k])) for k in ds.planet_letters}
    expected = sum(profile_chi2(ds.transit_times_bjd[k], model[k], ds.timing_uncertainty_days[k], ds.epochs[k])[0]
                   for k in ds.planet_letters)
    assert lk.chi2(model) == pytest.approx(expected, rel=1e-10)
    assert sum(lk.chi2_per_planet(model).values()) == pytest.approx(expected, rel=1e-10)


def test_chi2_is_invariant_to_a_linear_ephemeris_shift_of_the_model():
    """Adding a + b*n to the model must not change chi2: that freedom is profiled out."""
    ds = make_dataset()
    lk = TimingLikelihood(ds)
    rng = np.random.default_rng(3)
    model = {k: ds.transit_times_bjd[k] + rng.normal(0, 0.02, len(ds.epochs[k])) for k in ds.planet_letters}
    shifted = {k: model[k] + 0.37 + 0.0021 * ds.epochs[k] for k in ds.planet_letters}
    assert lk.chi2(model) == pytest.approx(lk.chi2(shifted), rel=1e-9)


def test_chi2_of_the_data_against_itself_is_zero():
    ds = make_dataset()
    assert TimingLikelihood(ds).chi2(ds.transit_times_bjd) == pytest.approx(0.0, abs=1e-12)


def test_chi2_has_the_right_number_of_degrees_of_freedom():
    """For pure noise about a straight line, E[chi2] = N - 2 per planet. Averaged over many
    draws the mean must be within 3% of n_points - 2 * n_planets."""
    ds = make_dataset()
    lk = TimingLikelihood(ds)
    rng = np.random.default_rng(4)
    values = []
    for _ in range(4000):
        noisy = {k: 5.0 + 0.1 * ds.epochs[k] + rng.normal(0, 1, len(ds.epochs[k])) * ds.timing_uncertainty_days[k]
                 for k in ds.planet_letters}
        values.append(lk.chi2({k: 5.0 + 0.1 * ds.epochs[k] for k in ds.planet_letters}) if False else
                      sum(profile_chi2(noisy[k], 5.0 + 0.1 * ds.epochs[k], ds.timing_uncertainty_days[k],
                                       ds.epochs[k])[0] for k in ds.planet_letters))
    assert np.mean(values) == pytest.approx(lk.dof_data, rel=0.03)


def test_missing_model_epoch_is_rejected_not_ignored():
    """A model lacking an observed epoch must not score well by omission."""
    ds = make_dataset()
    lk = TimingLikelihood(ds)
    model = {k: ds.transit_times_bjd[k].copy() for k in ds.planet_letters}
    model["d"][3] = np.nan
    assert lk.chi2(model) == INVALID_CHI2


def test_perturbation_vectors_are_consistent_with_chi2():
    """chi2(model) = | data_vector - model_vector |^2 when both are built against a control."""
    ds = make_dataset()
    lk = TimingLikelihood(ds)
    rng = np.random.default_rng(5)
    control = {k: ds.transit_times_bjd[k] + rng.normal(0, 0.02, len(ds.epochs[k])) for k in ds.planet_letters}
    model = {k: control[k] + rng.normal(0, 0.01, len(ds.epochs[k])) for k in ds.planet_letters}
    d = lk.data_vector(control)
    m = lk.model_vector(model, control)
    assert lk.chi2(model) == pytest.approx(float((d - m) @ (d - m)), rel=1e-9)


def test_weighted_ephemeris_fit_recovers_a_line_and_respects_weights():
    """The weighted fit must reproduce a noise-free line exactly, and a heavily down-weighted
    outlier must barely move it."""
    n = np.arange(12)
    t = 50.0 + 7.0 * n
    fit = fit_linear_ephemeris(t, epochs=n, sigma=np.full(12, 0.01))
    assert fit.t0 == pytest.approx(50.0, abs=1e-10) and fit.period == pytest.approx(7.0, abs=1e-12)

    t_bad = t.copy()
    t_bad[5] += 1.0
    sigma = np.full(12, 0.01)
    sigma[5] = 100.0
    weighted = fit_linear_ephemeris(t_bad, epochs=n, sigma=sigma)
    unweighted = fit_linear_ephemeris(t_bad, epochs=n)
    assert abs(weighted.period - 7.0) < 1e-4
    assert abs(unweighted.period - 7.0) > 1e-3
