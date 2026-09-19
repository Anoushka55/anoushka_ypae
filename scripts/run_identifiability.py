"""
Phase 11 — is Planet X uniquely identifiable from these observations?

Asks the question the project rules insist on: not "what is the best fit?" but
"how well is this actually determined, and what is degenerate with what?"

Measures:
  * the chi2 profile in each parameter (mass, semi-major axis, phase)
  * the width of the chi2 minimum in period, and why it is so narrow
  * the mass-phase correlation
  * dependence on observation duration
  * dependence on measurement noise

IF THE PROBLEM IS NOT UNIQUELY SOLVABLE, THAT IS THE RESULT. Nothing here is
tuned to make the recovery look better than it is.

Run:
  uv run --python 3.12 --extra dev python scripts/run_identifiability.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from invisible_planet.experiments.config import CANONICAL_PLANET_X, DEFAULT_CONFIG
from invisible_planet.inference.forward import ForwardModel
from invisible_planet.inference.search import (
    build_data_residuals,
    evaluate_candidates,
    null_hypothesis_chi2,
)
from invisible_planet.observations.dataset import ObserverDataset
from invisible_planet.observations.noise import (
    TimingNoiseModel,
    published_timing_uncertainties,
)
from invisible_planet.observations.transits import observe_transits
from invisible_planet.systems.kepler90 import load_reference
from invisible_planet.systems.planet_x import (
    PlanetXParameters,
    build_system_with_planet_x,
)

ROOT = Path(__file__).resolve().parents[1]
PROBE_PLANETS = ["b", "c", "i", "d", "e", "f"]


def make_dataset(reference, days: float, noise_scale: float, seed: int):
    config = DEFAULT_CONFIG
    mystery = build_system_with_planet_x(CANONICAL_PLANET_X, reference=reference)
    names = [f"Kepler-90 {letter}" for letter in PROBE_PLANETS]
    series = observe_transits(mystery, days, config.timestep_days, tracked=names)
    sigmas = published_timing_uncertainties(reference, PROBE_PLANETS)
    noise = TimingNoiseModel(sigmas, seed=seed, scale=noise_scale)
    return ObserverDataset.from_simulation(
        series=series,
        reference=reference,
        planet_letters=PROBE_PLANETS,
        noise_model=noise,
        observation_days=days,
        epoch_bjd=config.epoch_bjd,
    )


def profile(dataset, vary: str, values, base: PlanetXParameters) -> list:
    """chi2 as one parameter is varied, the others held at the truth."""
    from dataclasses import replace

    forward = ForwardModel(
        planet_letters=dataset.planet_letters, observation_days=dataset.observation_days
    )
    residuals = build_data_residuals(dataset)
    candidates = [replace(base, **{vary: float(v)}) for v in values]
    scores = evaluate_candidates(candidates, dataset, forward, residuals)
    return [{"value": float(v), "chi2": float(s)} for v, s in zip(values, scores)]


def interval_from_profile(entries: list, delta: float = 1.0):
    """Range of a parameter whose chi2 is within `delta` of the profile minimum."""
    best = min(e["chi2"] for e in entries)
    good = [e["value"] for e in entries if e["chi2"] <= best + delta]
    return (min(good), max(good)) if good else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data/identifiability.json")
    args = parser.parse_args()

    reference = load_reference()
    truth = CANONICAL_PLANET_X
    report: dict = {
        "warning": "Planet X is synthetic. Not a claim about the real Kepler-90.",
        "truth": truth.to_dict(),
    }

    print("=" * 74)
    print("IDENTIFIABILITY ANALYSIS")
    print("=" * 74)

    dataset = make_dataset(reference, DEFAULT_CONFIG.observation_days, 1.0,
                           DEFAULT_CONFIG.noise_seed)
    null = null_hypothesis_chi2(dataset)
    print(f"\nnull-hypothesis chi2 (no unseen planet): {null:.1f}")

    # ------------------------------------------------------------------
    print("\n1. CHI2 PROFILE IN MASS  (a and phase held at the truth)")
    masses = np.arange(5.0, 121.0, 5.0)
    mass_profile = profile(dataset, "mass_earth", masses, truth)
    best = min(mass_profile, key=lambda e: e["chi2"])
    interval = interval_from_profile(mass_profile)
    print(f"   best-fit mass {best['value']:.1f} M_earth (true {truth.mass_earth:.1f})")
    print(f"   1-sigma interval (delta chi2 < 1): {interval[0]:.1f} - {interval[1]:.1f}")
    print(f"   relative width: {(interval[1] - interval[0]) / best['value']:.0%}")
    report["mass_profile"] = mass_profile
    report["mass_interval"] = interval

    # ------------------------------------------------------------------
    print("\n2. CHI2 PROFILE IN SEMI-MAJOR AXIS  (mass and phase held at the truth)")
    axes = np.arange(0.2160, 0.2241, 0.0004)
    axis_profile = profile(dataset, "semi_major_axis_au", axes, truth)
    best_axis = min(axis_profile, key=lambda e: e["chi2"])
    axis_interval = interval_from_profile(axis_profile)
    print(f"   best-fit a {best_axis['value']:.5f} au (true {truth.semi_major_axis_au:.5f})")
    if axis_interval:
        width = axis_interval[1] - axis_interval[0]
        print(f"   1-sigma interval: {axis_interval[0]:.5f} - {axis_interval[1]:.5f} au "
              f"(width {width:.5f} au, {width / best_axis['value']:.2%})")
    report["axis_profile"] = axis_profile
    report["axis_interval"] = axis_interval

    # ------------------------------------------------------------------
    print("\n3. CHI2 PROFILE IN ORBITAL PHASE")
    phases = np.linspace(0.0, 2.0 * np.pi, 25, endpoint=False)
    phase_profile = profile(dataset, "mean_anomaly", phases, truth)
    best_phase = min(phase_profile, key=lambda e: e["chi2"])
    print(f"   best-fit phase {best_phase['value']:.3f} rad (true {truth.mean_anomaly:.3f})")
    report["phase_profile"] = phase_profile

    # ------------------------------------------------------------------
    print("\n4. MASS-PHASE CORRELATION  (a held at the truth)")
    from dataclasses import replace

    forward = ForwardModel(
        planet_letters=dataset.planet_letters, observation_days=dataset.observation_days
    )
    residuals = build_data_residuals(dataset)
    mass_axis = np.arange(10.0, 101.0, 10.0)
    phase_axis = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    grid_candidates = [
        replace(truth, mass_earth=float(m), mean_anomaly=float(p))
        for m in mass_axis
        for p in phase_axis
    ]
    grid_scores = np.array(
        evaluate_candidates(grid_candidates, dataset, forward, residuals)
    ).reshape(len(mass_axis), len(phase_axis))
    best_per_mass = grid_scores.min(axis=1)
    print(f"   {'mass':>6} {'best chi2 over phase':>22}")
    for m, s in zip(mass_axis, best_per_mass):
        print(f"   {m:6.1f} {s:22.1f}")
    degenerate = mass_axis[best_per_mass <= best_per_mass.min() + 1.0]
    print(f"   masses within delta chi2 < 1 when phase is free: "
          f"{degenerate.min():.0f} - {degenerate.max():.0f} M_earth")
    report["mass_phase_grid"] = {
        "mass_axis": mass_axis.tolist(),
        "phase_axis": phase_axis.tolist(),
        "chi2": grid_scores.tolist(),
    }

    # ------------------------------------------------------------------
    print("\n5. DEPENDENCE ON OBSERVATION DURATION")
    print(f"   {'days':>7} {'delta chi2 vs null':>20} {'significance':>14}")
    duration_rows = []
    for days in [400.0, 700.0, 1000.0, 1400.0]:
        subset = make_dataset(reference, days, 1.0, DEFAULT_CONFIG.noise_seed)
        subset_null = null_hypothesis_chi2(subset)
        subset_forward = ForwardModel(
            planet_letters=subset.planet_letters, observation_days=subset.observation_days
        )
        subset_residuals = build_data_residuals(subset)
        score = evaluate_candidates([truth], subset, subset_forward, subset_residuals)[0]
        delta = subset_null - score
        significance = np.sqrt(max(delta, 0.0))
        duration_rows.append({"days": days, "delta_chi2": float(delta),
                              "sigma": float(significance)})
        print(f"   {days:7.0f} {delta:20.1f} {significance:13.1f}s")
    report["duration_scan"] = duration_rows

    # ------------------------------------------------------------------
    print("\n6. DEPENDENCE ON MEASUREMENT NOISE  (1400 d baseline)")
    print(f"   {'noise scale':>12} {'delta chi2 vs null':>20} {'significance':>14}")
    noise_rows = []
    for scale in [0.25, 0.5, 1.0, 2.0]:
        subset = make_dataset(reference, DEFAULT_CONFIG.observation_days, scale,
                              DEFAULT_CONFIG.noise_seed)
        subset_null = null_hypothesis_chi2(subset)
        subset_forward = ForwardModel(
            planet_letters=subset.planet_letters, observation_days=subset.observation_days
        )
        subset_residuals = build_data_residuals(subset)
        score = evaluate_candidates([truth], subset, subset_forward, subset_residuals)[0]
        delta = subset_null - score
        significance = np.sqrt(max(delta, 0.0))
        noise_rows.append({"noise_scale": scale, "delta_chi2": float(delta),
                           "sigma": float(significance)})
        print(f"   {scale:12.2f} {delta:20.1f} {significance:13.1f}s")
    report["noise_scan"] = noise_rows

    (ROOT / args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWROTE {args.out}")


if __name__ == "__main__":
    main()
