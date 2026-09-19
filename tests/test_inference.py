"""
Phase 9 & 10 gates — the observation model, ground-truth isolation, and the
inverse problem.

The most important tests here are not about accuracy. They are about the
INTEGRITY of the experiment: if the inference can see the answer, recovering it
proves nothing.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from invisible_planet.experiments.config import CANONICAL_PLANET_X, DEFAULT_CONFIG
from invisible_planet.inference import search as search_module
from invisible_planet.inference.forward import ForwardModel, residual_signature
from invisible_planet.inference.search import (
    build_data_residuals,
    chi_squared,
    coarse_to_fine_search,
    evaluate_candidates,
)
from invisible_planet.observations.dataset import (
    FORBIDDEN_TOKENS,
    ObserverDataset,
    assert_no_ground_truth_leakage,
    assert_parameters_absent,
)
from invisible_planet.observations.noise import (
    TimingNoiseModel,
    published_timing_uncertainties,
)
from invisible_planet.observations.transits import observe_transits
from invisible_planet.systems.kepler90 import build_kepler90, load_reference
from invisible_planet.systems.planet_x import PlanetXParameters, build_system_with_planet_x

PROBES = ["b", "c", "i", "d", "e"]
SHORT_DAYS = 500.0


@pytest.fixture(scope="module")
def reference():
    return load_reference()


@pytest.fixture(scope="module")
def dataset(reference, tmp_path_factory):
    """An observer dataset built from a system containing the hidden planet."""
    config = DEFAULT_CONFIG
    mystery = build_system_with_planet_x(CANONICAL_PLANET_X, reference=reference)
    names = [f"Kepler-90 {letter}" for letter in PROBES]
    series = observe_transits(mystery, SHORT_DAYS, config.timestep_days, tracked=names)

    sigmas = published_timing_uncertainties(reference, PROBES)
    noise = TimingNoiseModel(sigmas, seed=1234)
    return ObserverDataset.from_simulation(
        series=series,
        reference=reference,
        planet_letters=PROBES,
        noise_model=noise,
        observation_days=SHORT_DAYS,
        epoch_bjd=config.epoch_bjd,
        provenance={"generator": "tests"},
    )


# =============================================================================
# Phase 9 — the noise model
# =============================================================================

def test_noise_amplitudes_come_from_the_published_uncertainties(reference):
    """Noise levels must be data-derived, not invented.

    Each planet's timing uncertainty must equal the published transit-epoch
    uncertainty for that planet, so the synthetic data inherits the real
    system's precision structure.
    """
    sigmas = published_timing_uncertainties(reference, PROBES)
    for letter in PROBES:
        published = reference["planets"][letter]["transit_epoch_bjd"]["uncertainty"]
        assert sigmas[letter] == published

    # and they differ substantially between planets, as the real data do
    values = list(sigmas.values())
    assert max(values) / min(values) > 3.0


def test_noise_is_reproducible_from_its_seed(reference):
    """Identical seeds must produce identical noise; different seeds must not."""
    sigmas = published_timing_uncertainties(reference, PROBES)
    times = np.linspace(0.0, 100.0, 25)

    a = TimingNoiseModel(sigmas, seed=42).apply("b", times)
    b = TimingNoiseModel(sigmas, seed=42).apply("b", times)
    c = TimingNoiseModel(sigmas, seed=43).apply("b", times)

    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_noise_has_the_requested_amplitude(reference):
    """The scatter added must match the stated sigma, within sampling error."""
    sigmas = published_timing_uncertainties(reference, PROBES)
    model = TimingNoiseModel(sigmas, seed=7)
    times = np.zeros(20000)
    noisy = model.apply("d", times)
    measured = float(np.std(noisy))
    assert abs(measured - sigmas["d"]) / sigmas["d"] < 0.05


# =============================================================================
# Phase 9 — ground-truth isolation (the integrity of the whole experiment)
# =============================================================================

def test_observer_dataset_has_no_field_for_the_hidden_body(dataset):
    """Structural guarantee: the dataset CANNOT carry Planet X's parameters."""
    fields = set(dataset.to_dict().keys())
    for forbidden in ["planet_x", "n_bodies", "ground_truth", "true_parameters"]:
        assert forbidden not in fields


def test_observer_dataset_contains_no_forbidden_identifier(dataset):
    assert_no_ground_truth_leakage(dataset)


def test_observer_dataset_contains_no_parameter_value_of_the_hidden_body(dataset):
    """Value-level check: a renamed field cannot smuggle the answer through."""
    assert_parameters_absent(dataset, CANONICAL_PLANET_X)


def test_leakage_detector_actually_catches_a_leak(dataset):
    """The guard must fail when ground truth IS present.

    A detector that never fires is worthless. This deliberately poisons a copy
    of the dataset and requires both guards to catch it.
    """
    poisoned = ObserverDataset(**{**dataset.__dict__})
    poisoned.provenance = dict(poisoned.provenance)
    poisoned.provenance["planet_x"] = "40 Earth masses"
    with pytest.raises(AssertionError, match="forbidden token"):
        assert_no_ground_truth_leakage(poisoned)

    by_value = ObserverDataset(**{**dataset.__dict__})
    by_value.provenance = dict(by_value.provenance)
    by_value.provenance["calibration_constant"] = CANONICAL_PLANET_X.semi_major_axis_au
    with pytest.raises(AssertionError, match="leaked by value"):
        assert_parameters_absent(by_value, CANONICAL_PLANET_X)


def test_inference_module_does_not_import_the_experiment_config():
    """The solver must not be able to reach the ground truth through an import.

    experiments.config holds CANONICAL_PLANET_X. If inference imported it, the
    answer would be one attribute access away, and no amount of care elsewhere
    would make the recovery meaningful.
    """
    import ast
    import inspect

    from invisible_planet.inference import forward

    for module in (search_module, forward):
        # Parse the ACTUAL import statements rather than scanning the source
        # text: the module docstrings legitimately mention experiments.config in
        # order to explain that they must not import it, and a substring scan
        # would flag that explanation as the violation it warns about.
        tree = ast.parse(inspect.getsource(module))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
                imported.update(alias.name for alias in node.names)

        offending = [
            name for name in imported
            if "experiments" in name or "CANONICAL_PLANET_X" in name
        ]
        assert not offending, (
            f"{module.__name__} imports {offending}, which would expose the "
            "ground truth to the inference"
        )


def test_dataset_survives_a_save_load_round_trip(dataset, tmp_path):
    """The file on disk is the contract between experiment and inference."""
    path = tmp_path / "observations.json"
    dataset.save(path)
    reloaded = ObserverDataset.load(path)

    assert reloaded.planet_letters == dataset.planet_letters
    for letter in dataset.planet_letters:
        assert np.allclose(reloaded.times_for(letter), dataset.times_for(letter))
    assert_no_ground_truth_leakage(reloaded)

    raw = json.loads(path.read_text(encoding="utf-8"))
    blob = json.dumps(raw).lower()
    for token in FORBIDDEN_TOKENS:
        assert token not in blob


# =============================================================================
# Phase 10 — the inverse problem
# =============================================================================

def test_chi_squared_is_zero_for_a_perfect_match():
    residuals = {
        "b": {"epochs": np.arange(10), "residuals": np.linspace(-1e-4, 1e-4, 10)},
    }
    assert chi_squared(residuals, residuals, {"b": 1e-4}) == 0.0


def test_chi_squared_matches_on_epoch_not_index():
    """A missing transit must not shift the comparison.

    The model sequence here omits one epoch. Matching by array index would
    misalign every subsequent point and produce a large spurious chi-squared;
    matching by epoch gives zero.
    """
    epochs = np.arange(10)
    residuals = np.linspace(-1e-4, 1e-4, 10)
    data = {"b": {"epochs": epochs, "residuals": residuals}}
    keep = epochs != 4
    model = {"b": {"epochs": epochs[keep], "residuals": residuals[keep]}}

    assert chi_squared(model, data, {"b": 1e-4}) == 0.0


def test_forward_model_reproduces_the_truth_at_the_true_parameters(dataset, reference):
    """Evaluated AT the true parameters, the model must fit the data well.

    This is a sanity check on the forward model rather than on the search: if
    the truth itself did not fit, no search could succeed, and the failure would
    be in the physics or the pipeline rather than in the optimiser.

    The test is allowed to know the truth; the search is not.
    """
    forward = ForwardModel(
        planet_letters=dataset.planet_letters,
        observation_days=dataset.observation_days,
    )
    data_residuals = build_data_residuals(dataset)
    scores = evaluate_candidates([CANONICAL_PLANET_X], dataset, forward, data_residuals)

    n_points = sum(len(v["residuals"]) for v in data_residuals.values())
    reduced = scores[0] / max(n_points - 3, 1)
    assert reduced < 2.0, (
        f"the true parameters give reduced chi2 = {reduced:.2f}; the forward "
        "model does not reproduce the data it generated"
    )


def test_true_parameters_beat_a_clearly_wrong_candidate(dataset):
    """chi-squared must discriminate: truth should score better than a bad guess."""
    forward = ForwardModel(
        planet_letters=dataset.planet_letters,
        observation_days=dataset.observation_days,
    )
    data_residuals = build_data_residuals(dataset)
    wrong = PlanetXParameters(
        mass_earth=200.0, semi_major_axis_au=0.16, mean_anomaly=0.3, inclination_deg=88.0
    )
    scores = evaluate_candidates(
        [CANONICAL_PLANET_X, wrong], dataset, forward, data_residuals
    )
    assert scores[0] < scores[1], (
        f"a wrong candidate scored better than the truth: {scores}"
    )


def test_search_is_deterministic_given_its_seed(dataset):
    """Reproducibility: identical seeds must give identical answers."""
    kwargs = dict(
        mass_grid=(20.0, 60.0),
        axis_grid=(0.20, 0.24),
        n_phases=2,
        n_refine_stages=1,
        verbose=False,
    )
    a = coarse_to_fine_search(dataset, seed=5, **kwargs)
    b = coarse_to_fine_search(dataset, seed=5, **kwargs)

    assert a.best_parameters == b.best_parameters
    assert a.best_chi2 == b.best_chi2


def test_search_improves_on_the_null_hypothesis(dataset):
    """The search must find that an unseen body is needed.

    If the best fit did not improve on the no-extra-planet model, the honest
    conclusion would be that the data do not require a Planet X.
    """
    null = search_module.null_hypothesis_chi2(dataset)
    result = coarse_to_fine_search(
        dataset,
        mass_grid=(10.0, 40.0, 90.0),
        axis_grid=(0.19, 0.22, 0.25),
        n_phases=4,
        n_refine_stages=1,
        verbose=False,
    )
    assert result.best_chi2 < null, (
        f"search found nothing better than the null hypothesis "
        f"({result.best_chi2:.1f} vs {null:.1f})"
    )
