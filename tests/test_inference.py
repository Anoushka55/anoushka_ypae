"""
Phase 9 & 10 gates - the observation model, ground-truth isolation, and the inverse problem.

The most important tests here are not about accuracy. They are about the INTEGRITY of the
experiment: if the inference can see the answer, recovering it proves nothing.

They use a test-local hidden planet (60 M_earth at 0.200 au, near the 2:1 resonance with d),
independent of the planet finally selected for the experiment.
"""

from __future__ import annotations

import ast
import inspect
import json

import numpy as np
import pytest

from invisible_planet.experiments.synthetic import simulate_observations
from invisible_planet.experiments.sweep import non_transiting_inclination_deg
from invisible_planet.inference import (
    domain,
    forward as forward_module,
    likelihood as likelihood_module,
    periodogram as periodogram_module,
    pipeline as pipeline_module,
    sampler as sampler_module,
    verify as verify_module,
)
from invisible_planet.inference.forward import ForwardModel
from invisible_planet.inference.likelihood import TimingLikelihood
from invisible_planet.inference.pipeline import recover
from invisible_planet.observations.dataset import (
    FORBIDDEN_TOKENS,
    ObserverDataset,
    assert_no_ground_truth_leakage,
    assert_parameters_absent,
)
from invisible_planet.observations.noise import TimingNoiseModel
from invisible_planet.observations.pattern import load_pattern
from invisible_planet.systems.kepler90 import load_reference
from invisible_planet.systems.planet_x import PlanetXParameters

TRUTH = PlanetXParameters(mass_earth=60.0, semi_major_axis_au=0.200, eccentricity=0.0, mean_anomaly=1.2345,
                          inclination_deg=non_transiting_inclination_deg(0.200, 0.005511))


@pytest.fixture(scope="module")
def reference():
    return load_reference()


@pytest.fixture(scope="module")
def dataset(reference):
    return simulate_observations(TRUTH, seed=1234, reference=reference)


# =============================================================================
# Phase 9 - the noise model
# =============================================================================

def test_noise_amplitudes_are_the_published_per_transit_uncertainties():
    """Each transit's sigma must equal the sigma PUBLISHED for that transit."""
    pattern = load_pattern()
    model = TimingNoiseModel(pattern, seed=1)
    for letter in pattern.probe_letters:
        assert np.array_equal(model.sigma_days[letter], pattern[letter].sigma_days)
    sig = np.concatenate([model.sigma_seconds(k) for k in pattern.probe_letters])
    assert sig.min() > 300.0          # per-transit errors of planets d and e are 5-23 minutes


def test_noise_is_reproducible_from_its_seed():
    pattern = load_pattern()
    times = pattern["d"].measured_bjd
    a = TimingNoiseModel(pattern, seed=42).apply("d", times)
    b = TimingNoiseModel(pattern, seed=42).apply("d", times)
    c = TimingNoiseModel(pattern, seed=43).apply("d", times)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_noise_has_the_requested_amplitude():
    """Standardised scatter must be N(0, 1) per transit: mean ~0, std ~1 over many draws."""
    pattern = load_pattern()
    times = pattern["e"].measured_bjd
    sigma = pattern["e"].sigma_days
    z = np.concatenate([(TimingNoiseModel(pattern, seed=s).apply("e", times) - times) / sigma
                        for s in range(400)])
    assert abs(z.mean()) < 0.05 and abs(z.std() - 1.0) < 0.03


def test_noise_model_rejects_a_series_of_the_wrong_length():
    pattern = load_pattern()
    with pytest.raises(ValueError, match="published uncertainties"):
        TimingNoiseModel(pattern, seed=1).apply("d", np.arange(5.0))


# =============================================================================
# Phase 9 - ground-truth isolation (the integrity of the whole experiment)
# =============================================================================

def test_observer_dataset_has_no_field_for_the_hidden_body(dataset):
    fields = set(dataset.to_dict().keys())
    for forbidden in ["planet_x", "n_bodies", "ground_truth", "true_parameters"]:
        assert forbidden not in fields


def test_observer_dataset_contains_no_forbidden_identifier(dataset):
    assert_no_ground_truth_leakage(dataset)


def test_observer_dataset_contains_no_parameter_value_of_the_hidden_body(dataset):
    assert_parameters_absent(dataset, TRUTH)


def test_leakage_detector_actually_catches_a_leak(dataset):
    """A detector that never fires is worthless: poison a copy and require both guards to fire."""
    poisoned = ObserverDataset(**{**dataset.__dict__})
    poisoned.provenance = dict(poisoned.provenance, planet_x="60 Earth masses")
    with pytest.raises(AssertionError, match="forbidden token"):
        assert_no_ground_truth_leakage(poisoned)

    by_value = ObserverDataset(**{**dataset.__dict__})
    by_value.provenance = dict(by_value.provenance, calibration_constant=TRUTH.semi_major_axis_au)
    with pytest.raises(AssertionError, match="leaked by value"):
        assert_parameters_absent(by_value, TRUTH)


def test_inference_modules_do_not_import_the_experiment_configuration():
    """The solver must not be able to reach the ground truth through an import. Parses the ACTUAL
    import statements of every inference module (the docstrings legitimately mention the
    experiment configuration in order to say it must not be imported)."""
    for module in (forward_module, likelihood_module, periodogram_module, pipeline_module,
                   sampler_module, verify_module, domain):
        tree = ast.parse(inspect.getsource(module))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
                imported.update(a.name for a in node.names)
        offending = [n for n in imported if "experiments" in n or "synthetic" in n or "CANONICAL" in n]
        assert not offending, f"{module.__name__} imports {offending}"


def test_dataset_survives_a_save_load_round_trip(dataset, tmp_path):
    path = tmp_path / "observations.json"
    dataset.save(path)
    reloaded = ObserverDataset.load(path)
    assert reloaded.planet_letters == dataset.planet_letters
    for k in dataset.planet_letters:
        assert np.allclose(reloaded.times_for(k), dataset.times_for(k), rtol=0, atol=1e-9)
        assert np.array_equal(reloaded.epochs[k], dataset.epochs[k])
    assert_no_ground_truth_leakage(reloaded)
    blob = json.dumps(json.loads(path.read_text(encoding="utf-8"))).lower()
    for token in FORBIDDEN_TOKENS:
        assert token not in blob


def test_dataset_epochs_and_uncertainties_are_the_real_ones(dataset):
    pattern = load_pattern()
    for k in dataset.planet_letters:
        assert np.array_equal(dataset.epochs[k], pattern[k].epochs)
        assert np.array_equal(dataset.timing_uncertainty_days[k], pattern[k].sigma_days)


# =============================================================================
# Phase 10 - the inverse problem
# =============================================================================

def test_the_true_parameters_fit_the_data_at_the_expected_level(dataset, reference):
    """Evaluated AT the truth (which only this test may know), the profile chi2 must be
    consistent with chi-squared statistics: within 4 sigma of its degrees of freedom."""
    forward = ForwardModel(dataset, reference=reference)
    likelihood = TimingLikelihood(dataset)
    chi2 = likelihood.chi2(forward.predict([TRUTH])[0])
    dof = likelihood.dof_data
    assert abs(chi2 - dof) < 4.0 * np.sqrt(2.0 * dof), f"chi2 = {chi2:.1f}, dof = {dof}"


def test_the_visible_only_model_is_strongly_disfavoured(dataset, reference):
    forward = ForwardModel(dataset, reference=reference)
    likelihood = TimingLikelihood(dataset)
    assert likelihood.chi2(forward.control()) > likelihood.chi2(forward.predict([TRUTH])[0]) + 50.0


def test_true_parameters_beat_a_clearly_wrong_candidate(dataset, reference):
    forward = ForwardModel(dataset, reference=reference)
    likelihood = TimingLikelihood(dataset)
    wrong = PlanetXParameters(mass_earth=200.0, semi_major_axis_au=0.16, mean_anomaly=3.0, inclination_deg=87.0)
    scores = [likelihood.chi2(p) for p in forward.predict([TRUTH, wrong])]
    assert scores[0] < scores[1]


@pytest.mark.slow
def test_end_to_end_recovery_of_a_detectable_planet(dataset, reference):
    """The whole pipeline, on a restricted search zone (so it runs in about two minutes): the
    detection must be significant and the period recovered to better than 0.5%."""
    result = recover(
        dataset, reference=reference, zones=[(0.185, 0.215)], n_peaks=2, n_null_trials=2000,
        sample=False, verbose=False, verify_settings={"n_phases": 8, "half_width_steps": 3},
    )
    assert result.detected, f"FAP = {result.false_alarm_probability:.2e}"
    best = result.best_parameters
    assert abs(best["semi_major_axis_au"] - TRUTH.semi_major_axis_au) / TRUTH.semi_major_axis_au < 0.005
    assert result.chi2_null - result.best_chi2 > 40.0


@pytest.mark.slow
def test_pure_noise_is_not_claimed_as_a_detection(reference):
    """A dataset generated WITHOUT a hidden planet must not produce a detection. Uses a zero-mass
    planet (which exerts no force) so the data are visible-planets-plus-noise only."""
    massless = PlanetXParameters(mass_earth=0.0, semi_major_axis_au=0.2, mean_anomaly=1.0, inclination_deg=87.0)
    empty = simulate_observations(massless, seed=99, reference=reference)
    result = recover(empty, reference=reference, zones=[(0.185, 0.215)], n_peaks=2, n_null_trials=2000,
                     sample=False, verbose=False, verify_settings={"n_phases": 8, "half_width_steps": 3})
    assert not result.detected, f"a planet was 'detected' in pure noise (FAP = {result.false_alarm_probability:.2e})"
