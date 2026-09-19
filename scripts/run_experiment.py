"""
The full experiment, end to end.

  1. Simulate the system WITH the synthetic Planet X (ground truth).
  2. Generate an observer dataset: noisy transit times only.
  3. Verify no ground truth leaked into that dataset.
  4. Hand the dataset to the inference, which has never seen Planet X.
  5. Compare the recovered parameters with the truth — only at the very end.

Run:
  uv run --python 3.12 --extra dev python scripts/run_experiment.py
  uv run --python 3.12 --extra dev python scripts/run_experiment.py --quick
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from invisible_planet.constants import SECONDS_PER_DAY
from invisible_planet.experiments.config import DEFAULT_CONFIG
from invisible_planet.inference.forward import ForwardModel
from invisible_planet.inference.search import (
    build_data_residuals,
    null_hypothesis_chi2,
    periodogram_search,
    select_peaks,
    verify_peaks_nonlinear,
)
from invisible_planet.systems.planet_x import PlanetXParameters
from invisible_planet.observations.dataset import (
    ObserverDataset,
    assert_no_ground_truth_leakage,
    assert_parameters_absent,
)
from invisible_planet.observations.noise import (
    TimingNoiseModel,
    published_timing_uncertainties,
)
from invisible_planet.observations.transits import observe_transits
from invisible_planet.observations.ttv import compare_ephemerides, fit_linear_ephemeris
from invisible_planet.systems.kepler90 import build_kepler90, load_reference
from invisible_planet.systems.planet_x import build_system_with_planet_x  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

#: Planets used as timing probes. g and h are excluded: over a Kepler-length
#: baseline they transit only 4 and 3 times, too few to support an ephemeris fit
#: plus meaningful residuals.
PROBE_PLANETS = ["b", "c", "i", "d", "e", "f"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--outdir", type=str, default="data")
    args = parser.parse_args()

    config = DEFAULT_CONFIG
    if args.seed is not None:
        config.noise_seed = args.seed
    reference = load_reference()
    outdir = ROOT / args.outdir

    print("=" * 74)
    print("THE INVISIBLE PLANET — controlled numerical experiment")
    print("=" * 74)
    print("Planet X is SYNTHETIC. This is not a claim about the real Kepler-90.")
    print()

    # ---------------------------------------------------------------- step 1
    truth = config.planet_x
    mystery = build_system_with_planet_x(truth, reference=reference)
    control = build_kepler90(reference=reference)

    print("STEP 1 — simulate the system containing the unseen body")
    print(f"  observation baseline : {config.observation_days:.0f} days")
    print(f"  timestep             : {config.timestep_days:.6f} days")
    print(f"  bodies               : {mystery.n_bodies} "
          f"(star + 8 known planets + 1 unseen)")

    probe_names = [f"Kepler-90 {letter}" for letter in PROBE_PLANETS]
    mystery_series = observe_transits(
        mystery, config.observation_days, config.timestep_days, tracked=probe_names
    )
    control_series = observe_transits(
        control, config.observation_days, config.timestep_days, tracked=probe_names
    )

    print("\n  the signature Planet X imprints on the visible planets:")
    print(f"  {'planet':>8} {'transits':>9} {'O-C ampl [min]':>15} {'O-C rms [s]':>12}")
    for letter, name in zip(PROBE_PLANETS, probe_names):
        result = compare_ephemerides(
            control_series.for_planet(name), mystery_series.for_planet(name)
        )
        print(
            f"  {letter:>8} {len(mystery_series.for_planet(name)):9d} "
            f"{result['detrended_peak_to_peak_minutes']:15.2f} "
            f"{result['detrended_rms_seconds']:12.1f}"
        )

    # ---------------------------------------------------------------- step 2
    print("\nSTEP 2 — generate the observer dataset (noisy transit times only)")
    sigmas = published_timing_uncertainties(reference, PROBE_PLANETS)
    noise = TimingNoiseModel(sigmas, seed=config.noise_seed, scale=config.noise_scale)
    print("  per-planet timing uncertainty, from the published values:")
    for letter in PROBE_PLANETS:
        print(f"    {letter}: {noise.sigma_seconds(letter):8.1f} s")

    dataset = ObserverDataset.from_simulation(
        series=mystery_series,
        reference=reference,
        planet_letters=PROBE_PLANETS,
        noise_model=noise,
        observation_days=config.observation_days,
        epoch_bjd=config.epoch_bjd,
        provenance={
            "generator": "scripts/run_experiment.py",
            "note": "Synthetic observations. The number of bodies is not disclosed.",
        },
    )
    dataset_path = outdir / "observations.json"
    dataset.save(dataset_path)
    print(f"  wrote {dataset_path.relative_to(ROOT)}  ({dataset.counts()})")

    # ---------------------------------------------------------------- step 3
    print("\nSTEP 3 — verify no ground truth leaked into the observer dataset")
    assert_no_ground_truth_leakage(dataset)
    assert_parameters_absent(dataset, truth)
    print("  PASS: no forbidden identifier and no parameter VALUE of the hidden")
    print("        body appears anywhere in the serialised dataset.")

    # ---------------------------------------------------------------- step 4
    print("\nSTEP 4 — inference (reads the dataset file only)")
    reloaded = ObserverDataset.load(dataset_path)

    null_chi2 = null_hypothesis_chi2(reloaded)
    n_points = sum(
        len(v) for v in reloaded.transit_times.values()
    )
    print(f"  null hypothesis (no unseen planet): chi2 = {null_chi2:.1f} "
          f"over ~{n_points} transits")

    # ------------------------------------------------------------------
    # Three stages. The design is driven by a measured property of the problem:
    # the chi2 minimum is ~0.0005 au wide in semi-major axis, because a period
    # error decorrelates the quasi-periodic TTV signal across the baseline. A
    # coarse grid search cannot find it (an earlier version converged to a
    # spurious minimum with a WORSE chi2 than the truth).
    # ------------------------------------------------------------------
    print("\n  stage 1 — periodogram: dense scan in period, mass and phase linearised")
    # The chi2 minimum is ~0.0005 au wide (measured; see inference/search.py).
    # The scan step must be FINER than that or the minimum is never sampled and
    # the periodogram locks onto a noise peak instead — which is exactly what
    # happened at 0.0016 au: it chose a = 0.1952 and returned a fit worse than
    # the null hypothesis.
    step = 0.0005 if args.quick else 0.00025
    periodogram = periodogram_search(
        reloaded, axis_min=0.16, axis_max=0.28, axis_step=step, verbose=True
    )
    localised = periodogram["best"]["semi_major_axis_au"]
    print(f"    strongest period signal at a = {localised:.4f} au "
          f"(delta chi2 = {periodogram['best']['delta_chi2_vs_null']:.1f})")

    print("\n  stage 2 — verify the strongest peaks with the FULL nonlinear model")
    peaks = select_peaks(periodogram["periodogram"], n_peaks=5)
    print(f"    {len(peaks)} well-separated peaks shortlisted. The linearised")
    print("    ranking is NOT trusted on its own: a peak can score well there")
    print("    and be catastrophically wrong under the real physics.")
    verification = verify_peaks_nonlinear(reloaded, peaks, verbose=True)

    winner = verification["best"]
    best = winner["parameters"]
    best_chi2 = winner["nonlinear_chi2"]

    # 1-sigma interval on mass from delta chi2 = 1 (one free parameter)
    mass_grid = np.asarray(winner["mass_grid"])
    mass_scores = np.asarray(winner["mass_chi2"])
    within = mass_grid[mass_scores <= best_chi2 + 1.0]
    mass_interval = (float(within.min()), float(within.max())) if len(within) else None

    data_residuals = build_data_residuals(reloaded)
    n_points = sum(len(v["residuals"]) for v in data_residuals.values())
    total_sims = (
        periodogram["n_forward_simulations"] + verification["n_forward_simulations"]
    )

    print(f"\n  forward simulations run : {total_sims}")
    print(f"  best chi2               : {best_chi2:.1f}")
    print(f"  reduced chi2            : {best_chi2 / max(n_points - 3, 1):.2f}")
    print(f"  improvement over null   : {null_chi2 - best_chi2:.1f} "
          f"(~{np.sqrt(max(null_chi2 - best_chi2, 0.0)):.1f} sigma)")

    # ---------------------------------------------------------------- step 5
    print("\nSTEP 5 — compare with the truth (revealed only now)")
    print(f"  {'parameter':>22} {'true':>12} {'recovered':>12} {'error':>12}")
    rows = [
        ("mass [M_earth]", truth.mass_earth, best.mass_earth),
        ("semi-major axis [au]", truth.semi_major_axis_au, best.semi_major_axis_au),
        ("mean anomaly [rad]", truth.mean_anomaly, best.mean_anomaly),
    ]
    for label, true_value, recovered in rows:
        print(f"  {label:>22} {true_value:12.4f} {recovered:12.4f} "
              f"{recovered - true_value:+12.4f}")

    m_star = reference["star"]["mass_solar"]["value"]
    true_period = truth.period_days(m_star + truth.mass_solar)
    recovered_period = best.period_days(m_star + best.mass_solar)
    print(f"  {'period [days]':>22} {true_period:12.4f} {recovered_period:12.4f} "
          f"{recovered_period - true_period:+12.4f}")

    if mass_interval is not None:
        print(f"\n  mass 1-sigma interval (delta chi2 < 1): "
              f"{mass_interval[0]:.1f} to {mass_interval[1]:.1f} M_earth")
        inside = mass_interval[0] <= truth.mass_earth <= mass_interval[1]
        print(f"  the true mass ({truth.mass_earth:.1f}) lies "
              f"{'INSIDE' if inside else 'OUTSIDE'} that interval")
        print("  note: mass is the weakly constrained parameter; the chi2 curve")
        print("        is shallow in mass and extremely sharp in period.")

    out = {
        "warning": "Planet X is synthetic. Not a claim about the real Kepler-90.",
        "config": config.to_dict(include_ground_truth=True),
        "probe_planets": PROBE_PLANETS,
        "null_hypothesis_chi2": null_chi2,
        "periodogram_best": periodogram["best"],
        "verified_peaks": [
            {k: v for k, v in p.items() if k != "parameters"}
            for p in verification["peaks"]
        ],
        "best_parameters": best.to_dict(),
        "best_chi2": best_chi2,
        "mass_one_sigma_interval": mass_interval,
        "n_forward_simulations": total_sims,
        "errors": {
            "mass_earth": best.mass_earth - truth.mass_earth,
            "semi_major_axis_au": best.semi_major_axis_au - truth.semi_major_axis_au,
            "period_days": recovered_period - true_period,
        },
    }
    path = outdir / "experiment_result.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWROTE {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
