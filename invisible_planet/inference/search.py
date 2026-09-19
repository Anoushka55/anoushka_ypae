"""
The inverse problem: recovering Planet X from transit timing alone.

Phase 10 deliverable.

=============================================================================
WHAT THIS MODULE IS ALLOWED TO KNOW
=============================================================================
  * the observer dataset (noisy transit times + uncertainties)
  * the published parameters of the VISIBLE planets and the star
  * the laws of gravity

It must never import the true Planet X parameters. It does not import
experiments.config, and the search functions take only an ObserverDataset.
=============================================================================

METHOD: transparent coarse-to-fine search. No machine learning, no LLM, no
black box. Every evaluation is a full N-body integration scored against the
data by an explicit chi-squared.

    chi2 = sum over planets, transits of ((O-C_model - O-C_data) / sigma)^2

The comparison is made on O-C RESIDUALS rather than raw transit times, because
a real analysis never knows the true ephemeris a priori and always fits it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..constants import SECONDS_PER_DAY
from ..systems.planet_x import PlanetXParameters
from .forward import ForwardModel, residual_signature


@dataclass
class SearchResult:
    best_parameters: PlanetXParameters
    best_chi2: float
    best_reduced_chi2: float
    n_forward_simulations: int
    n_data_points: int
    stages: list = field(default_factory=list)
    samples: list = field(default_factory=list)  # (parameters, chi2) for every evaluation

    def to_dict(self) -> dict:
        return {
            "best_parameters": self.best_parameters.to_dict(),
            "best_chi2": self.best_chi2,
            "best_reduced_chi2": self.best_reduced_chi2,
            "n_forward_simulations": self.n_forward_simulations,
            "n_data_points": self.n_data_points,
            "stages": self.stages,
        }

    def parameter_spread(self, within_delta_chi2: float = 4.0) -> dict:
        """Range of parameters among samples close to the best fit.

        A crude but honest uncertainty estimate: the spread of candidates whose
        chi2 lies within `within_delta_chi2` of the minimum. It is NOT a formal
        confidence interval, and is labelled as such wherever it is reported.
        """
        good = [(p, c) for p, c in self.samples if c <= self.best_chi2 + within_delta_chi2]
        if not good:
            return {}
        return {
            "n_within": len(good),
            "delta_chi2": within_delta_chi2,
            "mass_earth": [
                min(p.mass_earth for p, _ in good),
                max(p.mass_earth for p, _ in good),
            ],
            "semi_major_axis_au": [
                min(p.semi_major_axis_au for p, _ in good),
                max(p.semi_major_axis_au for p, _ in good),
            ],
            "mean_anomaly": [
                min(p.mean_anomaly for p, _ in good),
                max(p.mean_anomaly for p, _ in good),
            ],
        }


def build_data_residuals(dataset, min_transits: int = 3) -> dict:
    """O-C residuals of the OBSERVED transit times, keyed by planet and epoch."""
    times = {letter: dataset.times_for(letter) for letter in dataset.planet_letters}
    return residual_signature(times, min_transits=min_transits)


def chi_squared(model_residuals: dict, data_residuals: dict, sigma_days: dict) -> float:
    """Chi-squared between modelled and observed O-C residuals.

    Matched on EPOCH NUMBER, never on array index: a transit present in one and
    absent in the other must not silently shift the comparison.
    """
    total = 0.0
    for letter, data in data_residuals.items():
        model = model_residuals.get(letter)
        if model is None:
            # The model produced too few transits to fit an ephemeris. Penalise
            # rather than silently ignore, so a candidate cannot score well by
            # failing to produce data.
            total += 1e6
            continue

        data_map = dict(zip(data["epochs"], data["residuals"]))
        model_map = dict(zip(model["epochs"], model["residuals"]))
        shared = set(data_map) & set(model_map)
        if not shared:
            total += 1e6
            continue

        sigma = sigma_days[letter]
        for epoch in shared:
            total += ((model_map[epoch] - data_map[epoch]) / sigma) ** 2
    return float(total)


def evaluate_candidates(
    candidates: list, dataset, forward: ForwardModel, data_residuals: dict
) -> list:
    """Chi-squared for each candidate."""
    predictions = forward.predict(candidates)
    scores = []
    for params, prediction in zip(candidates, predictions):
        model_residuals = residual_signature(prediction)
        scores.append(
            chi_squared(model_residuals, data_residuals, dataset.timing_uncertainty_days)
        )
    return scores


def coarse_to_fine_search(
    dataset,
    mass_grid: tuple = (5.0, 10.0, 20.0, 40.0, 80.0, 160.0),
    axis_grid: tuple = (0.14, 0.17, 0.20, 0.23, 0.26, 0.30),
    n_phases: int = 8,
    n_refine_stages: int = 3,
    n_keep: int = 4,
    seed: int = 7,
    verbose: bool = True,
) -> SearchResult:
    """Recover Planet X from the observer dataset alone.

    ==================================================================
    WARNING: THIS SEARCH IS INADEQUATE FOR PERIOD RECOVERY. DO NOT USE
    IT TO INFER A PLANET. Use `periodogram_search` followed by
    `verify_peaks_nonlinear` instead.
    ==================================================================

    Measured failure, kept on the record rather than deleted: on the canonical
    experiment this routine returned mass 8.8 M_earth and a = 0.162 au against a
    truth of 40 M_earth and 0.220 au, with chi2 = 551.9 — WORSE than the chi2 of
    551.9 > 516.8 achieved by simply evaluating the true parameters. The
    optimiser, not the physics, was at fault.

    The cause is that the chi2 minimum is only ~0.0005 au wide in semi-major
    axis while this routine's grid spacing is ~0.03 au, so the minimum is never
    sampled and the refinement stages polish a noise peak.

    It is retained because the tests that exercise it check determinism and
    improvement over the null hypothesis, which are properties it does have, and
    because its failure is instructive.

    Stage 1 samples a coarse grid in (mass, semi-major axis, phase). Later
    stages shrink the search box around the best candidates found so far. The
    procedure is deterministic given `seed` and fully reported.
    """
    rng = np.random.default_rng(seed)
    forward = ForwardModel(
        planet_letters=dataset.planet_letters,
        observation_days=dataset.observation_days,
    )
    data_residuals = build_data_residuals(dataset)
    n_data_points = sum(len(v["residuals"]) for v in data_residuals.values())

    all_samples: list = []
    stages: list = []

    # ---------------- stage 1: coarse grid ----------------
    phases = np.linspace(0.0, 2.0 * np.pi, n_phases, endpoint=False)
    candidates = [
        PlanetXParameters(
            mass_earth=float(m),
            semi_major_axis_au=float(a),
            eccentricity=0.0,
            mean_anomaly=float(phase),
            inclination_deg=88.0,
        )
        for m in mass_grid
        for a in axis_grid
        for phase in phases
    ]
    if verbose:
        print(f"  stage 1: coarse grid, {len(candidates)} candidates")
    scores = evaluate_candidates(candidates, dataset, forward, data_residuals)
    all_samples.extend(zip(candidates, scores))
    stages.append({"stage": 1, "kind": "coarse grid", "n_candidates": len(candidates),
                   "best_chi2": float(min(scores))})

    # ---------------- refinement stages ----------------
    for stage in range(2, n_refine_stages + 2):
        ranked = sorted(all_samples, key=lambda item: item[1])[:n_keep]
        shrink = 0.5 ** (stage - 1)

        candidates = []
        for params, _ in ranked:
            for _ in range(12):
                candidates.append(
                    PlanetXParameters(
                        mass_earth=float(
                            np.clip(
                                params.mass_earth
                                * (1.0 + rng.normal(0.0, 0.35 * shrink)),
                                0.5,
                                400.0,
                            )
                        ),
                        semi_major_axis_au=float(
                            np.clip(
                                params.semi_major_axis_au
                                + rng.normal(0.0, 0.03 * shrink),
                                0.13,
                                0.30,
                            )
                        ),
                        eccentricity=0.0,
                        mean_anomaly=float(
                            np.mod(
                                params.mean_anomaly + rng.normal(0.0, 0.8 * shrink),
                                2.0 * np.pi,
                            )
                        ),
                        inclination_deg=88.0,
                    )
                )
        if verbose:
            print(f"  stage {stage}: local refinement, {len(candidates)} candidates")
        scores = evaluate_candidates(candidates, dataset, forward, data_residuals)
        all_samples.extend(zip(candidates, scores))
        stages.append({"stage": stage, "kind": "local refinement",
                       "n_candidates": len(candidates), "best_chi2": float(min(scores))})

    best_params, best_chi2 = min(all_samples, key=lambda item: item[1])
    degrees_of_freedom = max(n_data_points - 3, 1)

    return SearchResult(
        best_parameters=best_params,
        best_chi2=float(best_chi2),
        best_reduced_chi2=float(best_chi2 / degrees_of_freedom),
        n_forward_simulations=forward.n_evaluations,
        n_data_points=n_data_points,
        stages=stages,
        samples=all_samples,
    )


# =============================================================================
# PERIODOGRAM SEARCH
#
# WHY THE COARSE-TO-FINE SEARCH ABOVE IS NOT ENOUGH — measured, not assumed.
#
# Scanning chi2 against the perturber's semi-major axis near the true value
# gives a minimum that is RAZOR SHARP: chi2 = 516.8 at a = 0.2200 au, rising to
# 564.7 just 0.0005 au away and past 600 by 0.0010 au. The half-width is
# ~0.25% in a.
#
# The reason is physical. The TTV signal is quasi-periodic, so an error in the
# perturber's period accumulates a phase error ~ 2*pi*(T/P)*(dP/P) across the
# observing baseline T. Over 1400 days with P ~ 36 d, the model decorrelates
# from the data once dP/P exceeds ~P/(2*pi*T) ~ 0.4%. A grid coarser than that
# does not sample the minimum at all — it samples noise.
#
# The original grid spacing of 0.03 au was ~60x too coarse, and the search
# converged to a spurious local minimum with a WORSE chi2 than the truth. That
# was a failure of the optimiser, not of the physics, and it is recorded here
# rather than quietly fixed.
#
# The remedy is the standard one for quasi-periodic signals: scan the period
# densely, and at each trial period solve for the remaining parameters
# ANALYTICALLY instead of searching them.
#
# To first order in the perturbing mass, the TTV perturbation is linear in that
# mass, and its dependence on the perturber's orbital phase is spanned by two
# basis solutions computed at phases 0 and pi/2. So at each trial period we run
# exactly TWO forward simulations and obtain the best mass and phase by linear
# least squares. That removes two dimensions from the search entirely.
# =============================================================================

def _weighted_perturbation_vectors(
    residuals: dict, control: dict, epochs_by_planet: dict, sigma_days: dict
) -> np.ndarray:
    """Flatten (residuals - control) into one noise-weighted vector.

    Working with the DIFFERENCE from the control is essential: the visible
    planets perturb each other regardless of Planet X, and that common signal
    must not be attributed to the unseen body. What scales with the perturber's
    mass is the difference.
    """
    pieces = []
    for letter, epochs in epochs_by_planet.items():
        model = residuals.get(letter)
        if model is None:
            return np.array([])
        model_map = dict(zip(model["epochs"], model["residuals"]))
        control_map = dict(zip(control[letter]["epochs"], control[letter]["residuals"]))
        sigma = sigma_days[letter]
        for epoch in epochs:
            if epoch not in model_map or epoch not in control_map:
                return np.array([])
            pieces.append((model_map[epoch] - control_map[epoch]) / sigma)
    return np.asarray(pieces)


def periodogram_search(
    dataset,
    axis_min: float = 0.15,
    axis_max: float = 0.28,
    axis_step: float = 0.0004,
    reference_mass_earth: float = 5.0,
    inclination_deg: float = 88.0,
    verbose: bool = True,
) -> dict:
    """Dense scan in semi-major axis, solving mass and phase analytically.

    Returns the full periodogram so the chi2 landscape can be inspected and
    reported, rather than only its minimum.

    THE REFERENCE MASS MUST BE SMALL — this is not a free tuning knob.

    The linearisation assumes the TTV perturbation is proportional to the
    perturbing mass. That holds only while the perturbation is small. If the
    basis simulations are run at a mass large enough to DESTABILISE the system
    at some trial period, the "basis" there is a chaotic trajectory rather than
    a linear response, and fitting a small coefficient to it can match noise
    beautifully while meaning nothing.

    This was not hypothetical. With a 40 M_earth reference mass the scan
    preferred a spurious peak at a = 0.2703 au (chi2 = 508.5) over the true
    value at 0.2200 (chi2 = 531.0) — and the full nonlinear model then scored
    that "solution" at chi2 = 1300, far worse than the null hypothesis. The
    reason is that a = 0.27 au lies only ~3.8 mutual Hill radii from planet d,
    so a 40 M_earth body there is violently unstable.

    Dropping the reference mass to 5 M_earth removes the spurious peak entirely
    (its chi2 rises to ~569, i.e. no improvement on the null) while leaving the
    true peak unchanged. The fitted mass at the true peak is also stable across
    reference masses of 2, 5, 10 and 40 M_earth (20.7-20.9), whereas at the
    spurious peak it swings between 0.12 and 2.83 — which is itself a usable
    diagnostic for an invalid linearisation.
    """
    forward = ForwardModel(
        planet_letters=dataset.planet_letters,
        observation_days=dataset.observation_days,
    )
    data_residuals = build_data_residuals(dataset)
    control_residuals = residual_signature(forward.control_prediction())

    # epochs present in BOTH the data and the control, per planet
    epochs_by_planet = {}
    for letter, data in data_residuals.items():
        control = control_residuals.get(letter)
        if control is None:
            continue
        shared = sorted(set(data["epochs"]) & set(control["epochs"]))
        if shared:
            epochs_by_planet[letter] = shared

    data_vector = _weighted_perturbation_vectors(
        data_residuals, control_residuals, epochs_by_planet, dataset.timing_uncertainty_days
    )
    null_chi2 = float(np.sum(data_vector**2))

    axes = np.arange(axis_min, axis_max + 0.5 * axis_step, axis_step)
    if verbose:
        print(f"  periodogram: {len(axes)} trial axes, 2 simulations each "
              f"({2 * len(axes)} total)")

    candidates = []
    for a in axes:
        for phase in (0.0, 0.5 * np.pi):
            candidates.append(
                PlanetXParameters(
                    mass_earth=reference_mass_earth,
                    semi_major_axis_au=float(a),
                    eccentricity=0.0,
                    mean_anomaly=phase,
                    inclination_deg=inclination_deg,
                )
            )

    predictions = forward.predict(candidates)

    results = []
    for index, a in enumerate(axes):
        basis = []
        for offset in (0, 1):
            residuals = residual_signature(predictions[2 * index + offset])
            vector = _weighted_perturbation_vectors(
                residuals, control_residuals, epochs_by_planet,
                dataset.timing_uncertainty_days,
            )
            basis.append(vector)

        if any(len(v) != len(data_vector) or len(v) == 0 for v in basis):
            results.append({"semi_major_axis_au": float(a), "chi2": null_chi2,
                            "delta_chi2_vs_null": 0.0,
                            "mass_earth": np.nan, "mean_anomaly": np.nan})
            continue

        design = np.vstack(basis).T  # (n_points, 2)
        coefficients, *_ = np.linalg.lstsq(design, data_vector, rcond=None)
        residual_vector = data_vector - design @ coefficients
        chi2 = float(np.sum(residual_vector**2))

        # Recover mass and phase from the two basis coefficients. To first order
        # the perturbation is m * (cos(phase) * basis_0 + sin(phase) * basis_1)
        # scaled relative to the reference mass.
        amplitude = float(np.hypot(coefficients[0], coefficients[1]))
        results.append({
            "semi_major_axis_au": float(a),
            "chi2": chi2,
            "delta_chi2_vs_null": null_chi2 - chi2,
            "mass_earth": reference_mass_earth * amplitude,
            "mean_anomaly": float(np.mod(np.arctan2(coefficients[1], coefficients[0]),
                                          2.0 * np.pi)),
            "coefficients": [float(c) for c in coefficients],
        })

    best = min(results, key=lambda r: r["chi2"])
    return {
        "null_chi2": null_chi2,
        "n_data_points": len(data_vector),
        "n_forward_simulations": forward.n_evaluations,
        "reference_mass_earth": reference_mass_earth,
        "periodogram": results,
        "best": best,
        "linearisation_note": (
            "Mass and phase are obtained by first-order linear least squares at "
            "each trial period. The result is a starting point; a nonlinear "
            "refinement is required for final parameters."
        ),
    }


def scan_axis_phase_grid(
    dataset,
    axes,
    phases,
    reference_mass_earth: float = 5.0,
    inclination_deg: float = 88.0,
    verbose: bool = True,
) -> dict:
    """Joint scan over (semi-major axis, phase), with mass solved analytically.

    The reference mass is small for the reason documented in
    `periodogram_search`: a large one invalidates the linearisation wherever it
    would destabilise the system, and manufactures spurious peaks.

    WHY A JOINT SCAN. Coordinate-wise refinement fails here, and measurably so:
    started from the periodogram's estimate it stalls at chi2 = 524.3 while the
    truth scores 516.8. Mass, phase and period are coupled, and the chi2 valley
    is both narrow and curved, so improving one parameter at a time cannot move
    along it.

    Phase is scanned explicitly rather than linearised, because near a
    mean-motion resonance the TTV response contains harmonics of the perturber's
    phase and is not a single sinusoid — which is exactly why the two-basis
    periodogram returns a biased mass.

    Mass is still obtained analytically. At fixed period and phase the
    perturbation is linear in the perturbing mass to first order, so for a model
    perturbation vector r and data vector d (both noise-weighted),

        best scale  alpha = (r . d) / (r . r)
        chi2        = |d|^2 - (r . d)^2 / (r . r)

    which is exact for the linear model and costs no extra simulations.
    """
    forward = ForwardModel(
        planet_letters=dataset.planet_letters,
        observation_days=dataset.observation_days,
    )
    data_residuals = build_data_residuals(dataset)
    control_residuals = residual_signature(forward.control_prediction())

    epochs_by_planet = {}
    for letter, data in data_residuals.items():
        control = control_residuals.get(letter)
        if control is None:
            continue
        shared = sorted(set(data["epochs"]) & set(control["epochs"]))
        if shared:
            epochs_by_planet[letter] = shared

    data_vector = _weighted_perturbation_vectors(
        data_residuals, control_residuals, epochs_by_planet,
        dataset.timing_uncertainty_days,
    )
    null_chi2 = float(np.sum(data_vector**2))

    candidates = [
        PlanetXParameters(
            mass_earth=reference_mass_earth,
            semi_major_axis_au=float(a),
            eccentricity=0.0,
            mean_anomaly=float(phase),
            inclination_deg=inclination_deg,
        )
        for a in axes
        for phase in phases
    ]
    if verbose:
        print(f"  joint (a, phase) scan: {len(candidates)} simulations")

    predictions = forward.predict(candidates)

    best = None
    grid = []
    for params, prediction in zip(candidates, predictions):
        model_vector = _weighted_perturbation_vectors(
            residual_signature(prediction), control_residuals, epochs_by_planet,
            dataset.timing_uncertainty_days,
        )
        if len(model_vector) != len(data_vector) or len(model_vector) == 0:
            continue
        denominator = float(np.dot(model_vector, model_vector))
        if denominator <= 0.0:
            continue
        projection = float(np.dot(model_vector, data_vector))
        alpha = projection / denominator
        chi2 = null_chi2 - projection * alpha

        entry = {
            "semi_major_axis_au": params.semi_major_axis_au,
            "mean_anomaly": params.mean_anomaly,
            "mass_earth": reference_mass_earth * alpha,
            "chi2": float(chi2),
            "delta_chi2_vs_null": null_chi2 - float(chi2),
        }
        grid.append(entry)
        if best is None or entry["chi2"] < best["chi2"]:
            best = entry

    return {
        "null_chi2": null_chi2,
        "n_data_points": len(data_vector),
        "n_forward_simulations": forward.n_evaluations,
        "grid": grid,
        "best": best,
    }


def select_peaks(periodogram: list, n_peaks: int = 5, min_separation_au: float = 0.004) -> list:
    """Pick the strongest, well-separated peaks from a periodogram.

    Neighbouring grid points around one peak are not independent detections, so
    a simple "top N by chi2" would return the same peak five times. Peaks are
    therefore required to be separated in semi-major axis.
    """
    ordered = sorted(
        (entry for entry in periodogram if np.isfinite(entry.get("chi2", np.nan))),
        key=lambda entry: entry["chi2"],
    )
    chosen: list = []
    for entry in ordered:
        if all(
            abs(entry["semi_major_axis_au"] - other["semi_major_axis_au"])
            >= min_separation_au
            for other in chosen
        ):
            chosen.append(entry)
        if len(chosen) >= n_peaks:
            break
    return chosen


def verify_peaks_nonlinear(
    dataset,
    peaks: list,
    axis_halfwidth: float = 0.0020,
    axis_step: float = 0.0004,
    n_phases: int = 12,
    mass_grid=None,
    verbose: bool = True,
) -> dict:
    """Re-score candidate peaks with the FULL nonlinear model.

    The periodogram's ranking is based on a linearised model and must never be
    trusted on its own: a peak can score well there and then be catastrophically
    wrong under the real physics. (One did: a linearised chi2 of 508 became 1300
    — worse than the null hypothesis — when evaluated properly.)

    Each peak is examined with a local joint (a, phase) scan and a nonlinear mass
    scan, and the winner is chosen by TRUE chi2 across all peaks.
    """
    if mass_grid is None:
        mass_grid = np.arange(2.5, 120.1, 2.5)

    forward = ForwardModel(
        planet_letters=dataset.planet_letters,
        observation_days=dataset.observation_days,
    )
    data_residuals = build_data_residuals(dataset)

    evaluated = []
    for index, peak in enumerate(peaks, start=1):
        centre = peak["semi_major_axis_au"]
        axes = np.arange(centre - axis_halfwidth, centre + axis_halfwidth + 1e-9, axis_step)
        phases = np.linspace(0.0, 2.0 * np.pi, n_phases, endpoint=False)

        joint = scan_axis_phase_grid(dataset, axes, phases, verbose=False)
        local = joint["best"]

        candidates = [
            PlanetXParameters(
                mass_earth=float(m),
                semi_major_axis_au=local["semi_major_axis_au"],
                mean_anomaly=local["mean_anomaly"],
                inclination_deg=88.0,
            )
            for m in mass_grid
        ]
        scores = np.array(evaluate_candidates(candidates, dataset, forward, data_residuals))
        best_index = int(np.argmin(scores))

        record = {
            "peak_index": index,
            "periodogram_axis_au": centre,
            "linearised_chi2": peak["chi2"],
            "parameters": candidates[best_index],
            "nonlinear_chi2": float(scores[best_index]),
            "mass_grid": [float(m) for m in mass_grid],
            "mass_chi2": [float(s) for s in scores],
        }
        evaluated.append(record)
        if verbose:
            print(
                f"    peak {index}: a = {centre:.4f} au  "
                f"linearised chi2 = {peak['chi2']:.1f}  ->  "
                f"nonlinear chi2 = {record['nonlinear_chi2']:.1f}  "
                f"(m = {candidates[best_index].mass_earth:.1f} Me)"
            )

    best = min(evaluated, key=lambda record: record["nonlinear_chi2"])
    return {
        "peaks": evaluated,
        "best": best,
        "n_forward_simulations": forward.n_evaluations,
    }


def refine_nonlinear(
    dataset,
    start: PlanetXParameters,
    n_rounds: int = 3,
    verbose: bool = True,
) -> dict:
    """Coordinate-wise nonlinear refinement around a starting estimate.

    The periodogram localises the PERIOD reliably but returns biased mass and
    phase, because its linear two-basis model assumes the TTV response is
    sinusoidal in the perturber's orbital phase. Near a mean-motion resonance
    that response contains harmonics, so the linearisation is a good detector
    and a poor estimator. This stage fixes that with the full nonlinear model.

    Coordinate descent is used rather than a gradient method because the chi2
    surface in `a` is extremely narrow (see the note above) and finite-difference
    gradients across it are meaningless. Scanning one parameter at a time is
    slower but transparent and robust, and every evaluation is a full N-body
    integration.
    """
    forward = ForwardModel(
        planet_letters=dataset.planet_letters,
        observation_days=dataset.observation_days,
    )
    data_residuals = build_data_residuals(dataset)

    current = start
    history = []

    def score(candidates: list) -> list:
        return evaluate_candidates(candidates, dataset, forward, data_residuals)

    best_chi2 = score([current])[0]
    history.append({"round": 0, "stage": "start", "chi2": best_chi2,
                    "parameters": current.to_dict()})

    for round_index in range(1, n_rounds + 1):
        width = 0.5 ** (round_index - 1)

        # --- phase ---
        phases = np.mod(
            current.mean_anomaly + np.linspace(-np.pi, np.pi, 17) * width, 2.0 * np.pi
        )
        candidates = [replace_params(current, mean_anomaly=float(p)) for p in phases]
        scores = score(candidates)
        current, best_chi2 = _take_best(candidates, scores, current, best_chi2)

        # --- mass ---
        masses = np.clip(
            current.mass_earth * np.linspace(0.3, 2.2, 13) ** width, 0.5, 400.0
        )
        candidates = [replace_params(current, mass_earth=float(m)) for m in masses]
        scores = score(candidates)
        current, best_chi2 = _take_best(candidates, scores, current, best_chi2)

        # --- semi-major axis (fine: the minimum is ~0.0005 au wide) ---
        axes = current.semi_major_axis_au + np.linspace(-0.0025, 0.0025, 26) * width
        candidates = [replace_params(current, semi_major_axis_au=float(a)) for a in axes]
        scores = score(candidates)
        current, best_chi2 = _take_best(candidates, scores, current, best_chi2)

        history.append({"round": round_index, "stage": "coordinate descent",
                        "chi2": float(best_chi2), "parameters": current.to_dict()})
        if verbose:
            print(f"    round {round_index}: chi2 = {best_chi2:.1f}  "
                  f"m = {current.mass_earth:.1f} Me  a = {current.semi_major_axis_au:.5f} au")

    return {
        "best_parameters": current,
        "best_chi2": float(best_chi2),
        "n_forward_simulations": forward.n_evaluations,
        "history": history,
    }


def replace_params(params: PlanetXParameters, **changes) -> PlanetXParameters:
    from dataclasses import replace

    return replace(params, **changes)


def _take_best(candidates, scores, current, best_chi2):
    index = int(np.argmin(scores))
    if scores[index] < best_chi2:
        return candidates[index], float(scores[index])
    return current, best_chi2


def null_hypothesis_chi2(dataset) -> float:
    """Chi-squared of the NO-EXTRA-PLANET model.

    The number that says whether an unseen body is needed at all. If the best
    fit does not improve substantially on this, the honest conclusion is that
    the data do not require a Planet X.
    """
    forward = ForwardModel(
        planet_letters=dataset.planet_letters,
        observation_days=dataset.observation_days,
    )
    data_residuals = build_data_residuals(dataset)
    control_residuals = residual_signature(forward.control_prediction())
    return chi_squared(control_residuals, data_residuals, dataset.timing_uncertainty_days)
