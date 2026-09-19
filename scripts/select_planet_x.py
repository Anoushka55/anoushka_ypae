"""
Phase 8 (second stage) - recoverability screening and final selection of the hidden planet.

Takes the candidates that satisfied the pre-declared stability / plausibility / detectability /
subtlety criteria in scripts/sweep_planet_x.py and applies the last criterion:

  recoverable  in >= 80% of independent noise realisations, the detection is significant
               (false-alarm probability < 1%) AND the true orbital period is among the
               periodogram's shortlisted peaks - the necessary condition for the full pipeline
               (nonlinear verification of those peaks) to recover the planet.

This screening needs no extra N-body integrations beyond one synthetic dataset per noise
realisation, because the periodogram basis does not depend on the data. The full pipeline is
then run on the finalist by scripts/run_experiment.py, and on many random planets by
scripts/run_blind_recovery.py.

SELECTION OBJECTIVE (declared before the sweep, not "largest signal"):
  among candidates meeting every criterion, minimise |ln(detectability / (4 x T99))| - clearly
  detectable but not overwhelming - with the larger minimum Hill separation breaking near-ties.

Writes data/planet_x_selection.json and data/experiment_config.json.

Run: uv run --python 3.12 --extra dev python scripts/select_planet_x.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from invisible_planet.experiments.config import default_raw
from invisible_planet.experiments.synthetic import observations_from_times, simulate_transit_times
from invisible_planet.inference.domain import axes_for_zones, axis_step, stable_zones
from invisible_planet.inference.forward import ForwardModel
from invisible_planet.inference.likelihood import TimingLikelihood
from invisible_planet.inference.periodogram import (
    PeriodogramBasis,
    false_alarm_probability,
    null_distribution,
    scan,
    select_peaks,
)
from invisible_planet.systems.kepler90 import DEFAULT_DURATION_DAYS, load_reference
from invisible_planet.systems.planet_x import PlanetXParameters

ROOT = Path(__file__).resolve().parents[1]
N_SEEDS = 20
REQUIRED_FRACTION = 0.8
FAP_THRESHOLD = 0.01


def screen(params: PlanetXParameters, seeds: list, basis, null_max, reference, m_star) -> dict:
    """Detection and localisation over independent noise realisations."""
    widths = np.array([axis_step(a, DEFAULT_DURATION_DAYS, m_star) * 5.0 for a in basis.axes])
    outcomes = []
    sim_times = simulate_transit_times(params, reference=reference)          # one integration per planet
    for seed in seeds:
        dataset = observations_from_times(sim_times, seed=seed, reference=reference)
        likelihood = TimingLikelihood(dataset)
        forward = ForwardModel(dataset, reference=reference)
        control = forward.control()
        result = scan(basis, likelihood.data_vector(control))
        peaks = select_peaks(result, n_peaks=6, widths=widths)
        best = float(result["delta_chi2"][peaks[0]])
        fap = false_alarm_probability(best, null_max)
        nearest = min(abs(basis.axes[i] - params.semi_major_axis_au) for i in peaks)
        tolerance = 3.0 * axis_step(params.semi_major_axis_au, DEFAULT_DURATION_DAYS, m_star) * 5.0
        outcomes.append({"seed": seed, "linear_delta_chi2": best, "fap": fap,
                         "true_period_in_shortlist": bool(nearest <= tolerance),
                         "nearest_peak_offset_au": float(nearest)})
    detected = [o["fap"] < FAP_THRESHOLD for o in outcomes]
    localised = [o["true_period_in_shortlist"] for o in outcomes]
    success = [d and l for d, l in zip(detected, localised)]
    return {"fraction_detected": float(np.mean(detected)), "fraction_localised": float(np.mean(localised)),
            "fraction_recoverable": float(np.mean(success)), "outcomes": outcomes}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", default="data/planet_x_sweep.json")
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    reference = load_reference()
    m_star = reference["star"]["mass_solar"]["value"]
    sweep = json.loads((ROOT / args.sweep).read_text(encoding="utf-8"))
    qualified = sweep["qualified"][: args.top]
    print(f"{sweep['n_qualified']} candidates satisfied the sweep criteria; screening the top {len(qualified)}.")
    if not qualified:
        raise SystemExit("no candidate satisfied the criteria - the criteria or the design must be revisited")

    basis = PeriodogramBasis.load(ROOT / "data" / "cache" / "periodogram_basis.npz")
    from invisible_planet.experiments.sweep import template_dataset

    template = template_dataset()
    likelihood = TimingLikelihood(template)
    null_max = null_distribution(likelihood, basis, n_trials=20000, seed=0)
    seeds = list(range(1000, 1000 + N_SEEDS))

    started = time.time()
    rows = []
    for index, record in enumerate(qualified, start=1):
        p = record["parameters"]
        params = PlanetXParameters(
            mass_earth=p["mass_earth"], semi_major_axis_au=p["semi_major_axis_au"], eccentricity=p["eccentricity"],
            mean_anomaly=p["mean_anomaly"], inclination_deg=p["inclination_deg"])
        result = screen(params, seeds, basis, null_max, reference, m_star)
        rows.append({"candidate": record, "screening": result})
        print(f"  {index:2d}/{len(qualified)}  m = {params.mass_earth:5.0f}  a = {params.semi_major_axis_au:.4f}  "
              f"dchi2/T99 = {record['detectability_ratio']:5.1f}  detected {result['fraction_detected']:.0%}  "
              f"true period shortlisted {result['fraction_localised']:.0%}   ({time.time() - started:.0f} s)")

    finalists = [r for r in rows if r["screening"]["fraction_recoverable"] >= REQUIRED_FRACTION]
    print(f"\n{len(finalists)} of {len(rows)} screened candidates are recoverable in >= {REQUIRED_FRACTION:.0%} of "
          f"{N_SEEDS} noise realisations.")
    if not finalists:
        raise SystemExit("no recoverable candidate - report this; do not relax the criterion")
    finalists.sort(key=lambda r: (round(r["candidate"]["objective_distance"], 1), -r["candidate"]["hill_separation_min"]))
    chosen = finalists[0]
    c = chosen["candidate"]
    p = c["parameters"]

    print("\nSELECTED hidden planet (SYNTHETIC):")
    print(f"  mass {p['mass_earth']:.0f} M_earth, a = {p['semi_major_axis_au']:.4f} au, P = {c['period_days']:.2f} d, "
          f"phase {p['mean_anomaly']:.3f}, i = {p['inclination_deg']:.2f} deg (b = {c['impact_parameter']:.2f})")
    print(f"  {c['hill_separation_min']:.1f} mutual Hill radii from its nearest neighbour;"
          f" detectability {c['detectability_ratio']:.1f} x T99;"
          f" recoverable in {chosen['screening']['fraction_recoverable']:.0%} of noise realisations")
    print(f"  tension with the REAL d, e data (delta chi2 vs visible-only model): {c['tension_with_real_data']}")

    config = default_raw()
    config["planet_x_ground_truth"] = {
        "mass_earth": p["mass_earth"], "semi_major_axis_au": p["semi_major_axis_au"],
        "eccentricity": p["eccentricity"], "mean_anomaly": p["mean_anomaly"],
        "inclination_deg": p["inclination_deg"], "argument_of_periapsis": 0.0,
        "longitude_of_ascending_node_deg": 0.0, "radius_earth": 4.0,
        "status": "synthetic", "warning": "Planet X is not real and is not a claim about Kepler-90.",
        "selected_by": "scripts/select_planet_x.py from data/planet_x_sweep.json",
    }
    (ROOT / "data" / "experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    (ROOT / "data" / "planet_x_selection.json").write_text(json.dumps({
        "criteria": sweep["criteria"], "null_threshold": sweep["null_threshold"],
        "required_recoverable_fraction": REQUIRED_FRACTION, "n_noise_realisations": N_SEEDS,
        "screened": [{"parameters": r["candidate"]["parameters"], "period_days": r["candidate"]["period_days"],
                      "detectability_ratio": r["candidate"]["detectability_ratio"],
                      "hill_separation_min": r["candidate"]["hill_separation_min"],
                      "fraction_detected": r["screening"]["fraction_detected"],
                      "fraction_localised": r["screening"]["fraction_localised"],
                      "fraction_recoverable": r["screening"]["fraction_recoverable"]} for r in rows],
        "selected": chosen["candidate"],
        "warning": "Planet X is synthetic. Not a claim about Kepler-90.",
    }, indent=2, default=float), encoding="utf-8")
    print("\nWROTE data/experiment_config.json and data/planet_x_selection.json")


if __name__ == "__main__":
    main()
