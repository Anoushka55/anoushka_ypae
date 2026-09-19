"""
The real observing pattern and timing noise: data-integrity and provenance tests.

Each conversion from a raw published table to the pattern file is re-derived here
independently, so a mistake in the builder script cannot pass by being wrong in both places.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from invisible_planet.observations.pattern import load_pattern, select_at_epochs
from invisible_planet.observations.ttv import fit_linear_ephemeris

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


@pytest.fixture(scope="module")
def pattern():
    return load_pattern()


@pytest.fixture(scope="module")
def raw_json():
    return json.loads((ROOT / "data" / "kepler90_observed_transits.json").read_text(encoding="utf-8"))


def test_published_transit_counts(pattern):
    """The number of transits per planet must equal the published counts (Holczer: 16 for d,
    12 for e; Liang: 6 for g, 3 for h)."""
    assert {k: pattern[k].n_transits for k in ("d", "e", "g", "h")} == {"d": 16, "e": 12, "g": 6, "h": 3}


def test_holczer_times_are_rederived_independently(pattern):
    """measured time = BJD_offset + tn + O-C/1440, with the offset 2454900 (VizieR ReadMe)."""
    table = pd.read_csv(RAW / "holczer2016_table3_koi351.csv")
    for letter, koi in (("d", 351.03), ("e", 351.04)):
        rows = table[np.isclose(table["KOI"], koi)].sort_values("N")
        expected = 2454900.0 + rows["tn"].to_numpy() + rows["O-C"].to_numpy() / 1440.0
        assert np.allclose(pattern[letter].measured_bjd, expected, rtol=0, atol=1e-9)
        assert np.allclose(pattern[letter].sigma_days, rows["e_O-C"].to_numpy() / 1440.0, rtol=0, atol=1e-12)


def test_liang_times_are_rederived_independently(pattern):
    table = pd.read_csv(RAW / "liang2021_kepler90gh_transit_times.csv")
    for letter, name in (("g", "Kepler-90g"), ("h", "Kepler-90h")):
        rows = table[table["Planet"] == name].sort_values("HJD")
        assert np.allclose(pattern[letter].measured_bjd, 2454833.0 + rows["HJD"].to_numpy(), atol=1e-9)
        assert np.allclose(pattern[letter].sigma_days, rows["e_HJD"].to_numpy(), atol=1e-12)


def test_liang_time_convention_agrees_with_holczer_for_planet_g(pattern):
    """The Liang 'HJD' column is Kepler BJD - 2454833. Verified by agreement with Holczer's
    independent measurement of the same transits: within 3 combined sigma for the five transits
    that Holczer did not flag as outliers."""
    table = pd.read_csv(RAW / "holczer2016_table3_koi351.csv")
    rows = table[np.isclose(table["KOI"], 351.02) & (table["Out"] == 0)]
    holczer_t = 2454900.0 + rows["tn"].to_numpy() + rows["O-C"].to_numpy() / 1440.0
    holczer_s = rows["e_O-C"].to_numpy() / 1440.0
    liang_t, liang_s = pattern["g"].measured_bjd, pattern["g"].sigma_days
    assert len(rows) == 5
    for th, sh in zip(holczer_t, holczer_s):
        j = int(np.argmin(np.abs(liang_t - th)))
        assert abs(liang_t[j] - th) < 3.0 * math.hypot(sh, liang_s[j]), (th, liang_t[j])


def test_epoch_numbers_are_consistent_with_a_fitted_ephemeris(pattern):
    """Every transit must lie within 5% of a period of its integer epoch."""
    for letter in pattern.planets:
        p = pattern[letter]
        fit = fit_linear_ephemeris(p.measured_bjd, epochs=p.epochs, sigma=p.sigma_days)
        assert np.max(np.abs(fit.residuals)) / fit.period < 0.05
        assert np.all(np.diff(p.epochs) > 0)


def test_all_times_lie_inside_the_kepler_window(pattern):
    start, end = pattern.window_bjd
    for letter in pattern.planets:
        t = pattern[letter].measured_bjd
        assert t.min() > start - 20 and t.max() < end + 20


def test_all_uncertainties_are_positive_and_physically_sized(pattern):
    """Per-transit timing errors must be between 30 s and 1 hour."""
    for letter in pattern.planets:
        s = pattern[letter].sigma_days * 86400.0
        assert np.all(s > 30.0) and np.all(s < 3600.0), letter


def test_probe_rule_selects_d_and_e_only_and_for_the_stated_reasons(raw_json):
    assert raw_json["probe_planets"] == ["d", "e"]
    planets = raw_json["planets"]
    assert planets["d"]["linear_ephemeris_reduced_chi2"] <= raw_json["probe_rule"]["max_reduced_chi2"]
    assert planets["e"]["linear_ephemeris_reduced_chi2"] <= raw_json["probe_rule"]["max_reduced_chi2"]
    assert planets["g"]["linear_ephemeris_reduced_chi2"] > 100          # real, large TTV
    assert planets["h"]["n_transits"] < raw_json["probe_rule"]["min_transits"]


def test_pattern_probe_letters_match_the_rule(pattern):
    assert pattern.probe_letters == ["d", "e"]
    assert set(pattern.planets) == {"d", "e", "g", "h"}


def test_carter_photon_noise_formula_is_reproduced_independently(raw_json):
    """sigma_tc = (T/Q) sqrt(theta/2), theta = r/(1-b^2), T = T14/(1+theta), Q = SNR/sqrt(N)
    (Carter et al. 2008 eqs. 21-22), recomputed here from the raw KOI table."""
    koi = pd.read_csv(RAW / "kepler90_koi_cumulative.csv")
    row = koi[koi["kepler_name"] == "Kepler-90 d"].iloc[0]
    q = row["koi_model_snr"] / math.sqrt(row["koi_num_transits"])
    theta = row["koi_ror"] / (1 - row["koi_impact"] ** 2)
    t_fwhm = row["koi_duration"] * 3600.0 / (1 + theta)
    expected = (t_fwhm / q) * math.sqrt(theta / 2.0)
    assert raw_json["carter_photon_noise"]["d"]["sigma_seconds"] == pytest.approx(expected, rel=1e-9)


def test_published_errors_exceed_the_photon_noise_limit(raw_json):
    """The published per-transit uncertainty is a real fit and must be at or above the
    theoretical white-noise lower bound, by a factor of order 2-3."""
    for letter in ("d", "e", "g", "h"):
        ratio = (raw_json["planets"][letter]["median_sigma_seconds"]
                 / raw_json["carter_photon_noise"][letter]["sigma_seconds"])
        assert 1.5 < ratio < 4.0, (letter, ratio)


def test_implied_cadence_noise_is_the_same_for_all_well_measured_planets(raw_json):
    """Kepler noise on this star is ~white, so the per-cadence noise implied by each planet's
    depth, duration and SNR must agree: within 10% of the median for b, c, d, e, g, h."""
    values = {k: v for k, v in raw_json["implied_cadence_noise_ppm"].items() if k != "f"}
    median = float(np.median(list(values.values())))
    for k, v in values.items():
        assert abs(v / median - 1.0) < 0.1, (k, v, median)


def test_select_at_epochs_returns_nan_for_absent_epochs(pattern):
    """The simulation must not silently drop an observed epoch."""
    d = pattern["d"]
    sim = d.measured_bjd[:8]                       # only the first eight transits
    times, present = select_at_epochs(sim, d)
    assert present[:8].all() and not present[8:].any()
    assert np.all(np.isnan(times[8:]))
