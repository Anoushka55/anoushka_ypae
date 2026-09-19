"""
Phases 9-11 - the flagship experiment: one hidden planet, one dataset, one blind recovery.

  1. read the configuration (data/experiment_config.json), which holds Planet X's ground truth;
  2. integrate the system WITH Planet X and sample its transits as Kepler sampled the real
     system (real epochs, real per-transit uncertainties, noise from a seeded generator);
  3. write the OBSERVER DATASET to data/observations.json - the only thing that crosses to the
     inference - after asserting that it holds nothing but measurements;
  4. run the inference on that dataset alone;
  5. compare the recovery with the truth and write data/experiment_result.json, which the
     autonomous demonstration displays. Nothing is displayed that this script did not compute.

Run: uv run --python 3.12 --extra dev python scripts/run_experiment.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from invisible_planet.constants import EARTH_MASS_IN_SOLAR, kepler_third_law_period
from invisible_planet.experiments.config import load_config
from invisible_planet.experiments.synthetic import simulate_observations
from invisible_planet.inference.forward import ForwardModel
from invisible_planet.inference.likelihood import TimingLikelihood
from invisible_planet.inference.pipeline import equivalent_sigma, recover
from invisible_planet.observations.dataset import assert_no_ground_truth_leakage, assert_parameters_absent
from invisible_planet.observations.ttv import fit_linear_ephemeris
from invisible_planet.systems.kepler90 import load_reference
from invisible_planet.systems.planet_x import PlanetXParameters

ROOT = Path(__file__).resolve().parents[1]


def oc_minutes(times_bjd: np.ndarray, epochs: np.ndarray, sigma: np.ndarray) -> list:
    """O-C in minutes about the weighted linear ephemeris fitted to `times_bjd` itself."""
    fit = fit_linear_ephemeris(np.asarray(times_bjd), epochs=np.asarray(epochs), sigma=np.asarray(sigma))
    return [float(x) for x in fit.residuals * 1440.0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/experiment_result.json")
    args = parser.parse_args()

    config = load_config()
    reference = load_reference()
    truth: PlanetXParameters = config.planet_x
    started = time.time()

    print("PLANET X IS SYNTHETIC - a controlled numerical experiment, not a claim about Kepler-90.\n")
    dataset = simulate_observations(truth, seed=config.noise_seed, noise_scale=config.noise_scale, reference=reference)
    assert_no_ground_truth_leakage(dataset)
    assert_parameters_absent(dataset, truth)
    dataset.save(ROOT / "data" / "observations.json")
    print(f"observer dataset written: {dataset.counts()} transits (measurements only)\n")

    inf = config.inference
    result = recover(
        dataset, reference=reference, basis_cache=ROOT / "data" / "cache" / "periodogram_basis.npz",
        fap_threshold=inf["fap_threshold"], n_peaks=inf["n_peaks"], n_null_trials=inf["n_null_trials"],
        sample=True, n_walkers=inf["n_walkers"], n_steps=inf["n_steps"], seed=inf["sampler_seed"], verbose=True,
    )
    r = result.to_dict()

    # ---- what the demonstration shows: observed / null-model / best-fit-model O-C -----------
    likelihood = TimingLikelihood(dataset)
    forward = ForwardModel(dataset, reference=reference)
    control = forward.control()
    best_model = None
    if r["best_parameters"]:
        best_model = forward.predict([PlanetXParameters.from_dict(r["best_parameters"])])[0]
    per_planet = {}
    chi2_null_by_planet = likelihood.chi2_per_planet(control)
    chi2_best_by_planet = likelihood.chi2_per_planet(best_model) if best_model else None
    for k in dataset.planet_letters:
        epochs, sigma = dataset.epochs[k], dataset.timing_uncertainty_days[k]
        per_planet[k] = {
            "epochs": [int(e) for e in epochs],
            "observed_bjd": [float(t) for t in dataset.transit_times_bjd[k]],
            "sigma_minutes": [float(s * 1440.0) for s in sigma],
            "observed_oc_minutes": oc_minutes(dataset.transit_times_bjd[k], epochs, sigma),
            "visible_only_model_oc_minutes": oc_minutes(control[k], epochs, sigma),
            "best_fit_model_oc_minutes": oc_minutes(best_model[k], epochs, sigma) if best_model else None,
            "chi2_visible_only": chi2_null_by_planet[k],
            "chi2_with_recovered_planet": chi2_best_by_planet[k] if chi2_best_by_planet else None,
        }

    # ---- comparison with the truth (this is the ONLY place truth and recovery meet) ---------
    m_star = reference["star"]["mass_solar"]["value"]
    p_true = truth.period_days(m_star + truth.mass_solar)
    comparison = None
    if r["best_parameters"]:
        best = r["best_parameters"]
        p_best = kepler_third_law_period(best["semi_major_axis_au"], m_star + best["mass_earth"] * EARTH_MASS_IN_SOLAR)
        comparison = {"period_true_days": p_true, "period_recovered_days": p_best,
                      "period_relative_error": (p_best - p_true) / p_true}
        summary = r["posterior"]["summary"] if r["posterior"] else None
        if summary:
            def inside(name, value, lo="2.5", hi="97.5"):
                return bool(summary[name][lo] <= value <= summary[name][hi])
            comparison["truth_inside_95pct_interval"] = {
                "mass_earth": inside("mass_earth", truth.mass_earth),
                "semi_major_axis_au": inside("semi_major_axis_au", truth.semi_major_axis_au),
                "period_days": inside("period_days", p_true),
                "eccentricity": inside("eccentricity", truth.eccentricity),
            }

    output = {
        "planet_x_is_synthetic": True,
        "warning": "Planet X is not real and is not a claim about Kepler-90.",
        "config": config.to_dict(include_ground_truth=True),
        "ground_truth": truth.to_dict(),
        "true_period_days": p_true,
        "inference": {k: v for k, v in r.items()},
        "observations": per_planet,
        "comparison": comparison,
        "wall_seconds": time.time() - started,
    }
    (ROOT / args.out).write_text(json.dumps(output, indent=1, default=float), encoding="utf-8")

    print("\n" + "=" * 72)
    print("RESULT (SYNTHETIC Planet X)")
    print("=" * 72)
    print(f"detected: {r['detected']}   false-alarm probability {r['false_alarm_probability']:.2e} "
          f"(~{equivalent_sigma(r['false_alarm_probability']):.1f} sigma)")
    print(f"chi2: visible-only {r['chi2_null']:.1f}  ->  with recovered planet {r['best_chi2']}   (n = {r['n_points']})")
    if r["best_parameters"]:
        b = r["best_parameters"]
        print(f"recovered (best fit): m = {b['mass_earth']:.1f} M_earth, a = {b['semi_major_axis_au']:.4f} au, "
              f"P = {comparison['period_recovered_days']:.3f} d")
        print(f"truth               : m = {truth.mass_earth:.1f} M_earth, a = {truth.semi_major_axis_au:.4f} au, "
              f"P = {p_true:.3f} d")
        print(f"period error {comparison['period_relative_error']:+.3%}")
    print(f"\nwritten {args.out} ({time.time() - started:.0f} s)")


if __name__ == "__main__":
    main()
