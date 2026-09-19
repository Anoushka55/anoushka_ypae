"""
Phase 8 - controlled parameter sweep over the hidden planet.

Every candidate is integrated once (with the matched visible system) and scored on the
eight metric families of the phase prompt; see invisible_planet/experiments/sweep.py.

SELECTION CRITERIA - fixed here, before any candidate is evaluated
------------------------------------------------------------------
  stable       bound; no orbit crossing; eccentricity < 0.25; energy error < 1e-6
  plausible    >= 8 mutual Hill radii from BOTH neighbours (the real g-h pair is 5.2);
               non-transiting geometry (impact parameter 1.4)
  detectable   expected (noise-free) delta chi2 >= 2 x T99, where T99 is the 99th percentile of
               the EXACT noise-only distribution of the detection statistic
  subtle       no single transit deviates by more than 4 sigma from the profiled model - the
               signal is NOT obvious by eye
  recoverable  (second stage) recovered in >= 80% of independent noise realisations

The objective is NOT the largest signal. Among candidates meeting every criterion, the choice
is the one whose detectability is closest (in log) to 4 x T99 - clearly detectable but not
overwhelming - with the larger minimum Hill separation breaking near-ties. See
docs/PLANET_X_DESIGN.md.

Run: uv run --python 3.12 --extra dev python scripts/sweep_planet_x.py
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import product
from pathlib import Path

import numpy as np

from invisible_planet.constants import EARTH_MASS_IN_SOLAR
from invisible_planet.experiments.sweep import (
    DURATION_KEY_FULL,
    DURATIONS_DAYS,
    build_duration_likelihoods,
    integrate_with_diagnostics,
    non_transiting_inclination_deg,
    orbital_residuals,
    signal_metrics,
    stability_metrics,
    template_dataset,
)
from invisible_planet.inference.domain import axes_for_zones, hill_separations, impact_parameter, stable_zones
from invisible_planet.inference.forward import ForwardModel
from invisible_planet.inference.likelihood import TimingLikelihood
from invisible_planet.inference.periodogram import PeriodogramBasis, null_distribution
from invisible_planet.systems.kepler90 import (
    DEFAULT_DURATION_DAYS,
    DEFAULT_EPOCH_BJD,
    build_kepler90,
    load_reference,
    recommended_timestep,
)
from invisible_planet.systems.planet_x import PlanetXParameters, build_system_with_planet_x

ROOT = Path(__file__).resolve().parents[1]

# ---- criteria (fixed before evaluation) ---------------------------------------------------
MAX_ECCENTRICITY = 0.25
MAX_ENERGY_ERROR = 1e-6
MIN_HILL = 8.0
DETECTABLE_FACTOR = 2.0          # expected delta chi2 >= 2 x T99
TARGET_FACTOR = 4.0              # preferred detectability: 4 x T99
MAX_SINGLE_TRANSIT_SIGMA = 4.0
BATCH = 48


def candidate_grid(reference: dict, stellar_radius_au: float, coarse: bool) -> list:
    masses = [10.0, 25.0, 50.0, 100.0, 200.0] if not coarse else [25.0, 100.0]
    # step 0.0015 au (~0.75% at 0.2 au): near-resonant effects vary on ~1% scales
    zones = [(0.14, 0.30, 0.0015), (0.52, 0.70, 0.004)]
    if coarse:
        zones = [(0.14, 0.30, 0.02)]
    axes = [a for lo, hi, st in zones for a in np.arange(lo, hi + 1e-9, st)]
    phases = [0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0]
    out = []
    for m, a, ph in product(masses, axes, phases):
        out.append(PlanetXParameters(
            mass_earth=float(m), semi_major_axis_au=float(a), eccentricity=0.0, mean_anomaly=float(ph),
            inclination_deg=non_transiting_inclination_deg(float(a), stellar_radius_au)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse", action="store_true", help="small grid for a smoke test")
    parser.add_argument("--out", default="data/planet_x_sweep.json")
    args = parser.parse_args()

    reference = load_reference()
    m_star = reference["star"]["mass_solar"]["value"]
    dt = recommended_timestep(reference)
    control_system = build_kepler90(reference=reference)
    stellar_radius = float(control_system.radii[0])
    from invisible_planet.observations.pattern import load_pattern
    probes = [f"Kepler-90 {k}" for k in load_pattern().probe_letters]

    template = template_dataset()
    forward = ForwardModel(template, reference=reference)
    likelihoods, full_template = build_duration_likelihoods()
    lk_full = likelihoods[DURATION_KEY_FULL]

    # ---- detection threshold from the exact noise-only distribution -------------------------
    print("Calibrating the detection threshold (exact, noise-only)...")
    zones = stable_zones(reference)
    axes = axes_for_zones(zones, DEFAULT_DURATION_DAYS, m_star)
    cache = ROOT / "data" / "cache"
    cache.mkdir(exist_ok=True)
    started = time.time()
    cache_file = cache / "periodogram_basis.npz"
    basis = None
    if cache_file.exists():
        cached = PeriodogramBasis.load(cache_file)
        signature = PeriodogramBasis.make_signature(
            lk_full, forward.control(), axes, cached.reference_mass_earth, cached.inclination_deg)
        if cached.signature == signature:
            basis = cached
            print(f"  periodogram basis loaded from {cache_file.name} (signature matches)")
    if basis is None:
        basis = PeriodogramBasis.compute(forward, lk_full, axes, verbose=True)
        basis.save(cache_file)
    null_max = null_distribution(lk_full, basis, n_trials=20000, seed=0)
    t99 = float(np.percentile(null_max, 99))
    t999 = float(np.percentile(null_max, 99.9))
    print(f"  {len(axes)} trial axes; T50 = {np.median(null_max):.1f}, T99 = {t99:.1f}, "
          f"T99.9 = {t999:.1f}   ({time.time() - started:.0f} s)")
    print(f"  detectable: expected delta chi2 >= {DETECTABLE_FACTOR} x T99 = {DETECTABLE_FACTOR * t99:.1f}")

    # ---- the control integration ------------------------------------------------------------
    control_run = integrate_with_diagnostics([control_system], probes, DEFAULT_DURATION_DAYS, dt)
    control_times = forward._at_dataset_epochs(control_run["series"][0])
    real_chi2_control = lk_full.chi2_per_planet(control_times)

    grid = candidate_grid(reference, stellar_radius, args.coarse)
    print(f"\nEvaluating {len(grid)} candidates (batches of {BATCH})...")
    results = []
    started = time.time()
    for start in range(0, len(grid), BATCH):
        chunk = grid[start:start + BATCH]
        systems = [build_system_with_planet_x(p, reference=reference, epoch_bjd=DEFAULT_EPOCH_BJD) for p in chunk]
        run = integrate_with_diagnostics(systems, probes, DEFAULT_DURATION_DAYS, dt)
        stab = stability_metrics(run["rel_r"], run["rel_v"], run["masses"], m_star)
        for i, params in enumerate(chunk):
            predicted = forward._at_dataset_epochs(run["series"][i])
            finite = all(np.all(np.isfinite(predicted[k])) for k in predicted)
            record = {"parameters": params.to_dict(), "stability": stab[i]}
            record["stability"]["max_energy_error"] = float(np.max(run["energy_error"][:, i]))
            record["stability"]["min_pairwise_separation_au"] = float(run["min_separation"][i])
            d_in, d_out = hill_separations(params.semi_major_axis_au, params.mass_solar, reference)
            record["hill_separation_min"] = float(min(d_in, d_out))
            record["impact_parameter"] = impact_parameter(params, stellar_radius)
            record["period_days"] = params.period_days(m_star + params.mass_solar)
            record["orbital_residuals"] = orbital_residuals(
                run["rel_r"][:, i], run["rel_v"][:, i], control_run["rel_r"][:, 0], control_run["rel_v"][:, 0])
            if finite:
                record["signal"] = signal_metrics(predicted, control_times, likelihoods, full_template)
                real = lk_full.chi2_per_planet(predicted)
                record["tension_with_real_data"] = {
                    k: float(real[k] - real_chi2_control[k]) for k in lk_full.letters}
            else:
                record["signal"] = None
            results.append(record)
        print(f"  {min(start + BATCH, len(grid))}/{len(grid)}  ({time.time() - started:.0f} s)")

    # ---- apply the pre-declared criteria ---------------------------------------------------
    def qualifies(r: dict) -> bool:
        s, sig = r["stability"], r["signal"]
        return bool(
            sig is not None and s["bound"] and not s["orbit_crossing"]
            and s["max_eccentricity"] < MAX_ECCENTRICITY and s["max_energy_error"] < MAX_ENERGY_ERROR
            and r["hill_separation_min"] >= MIN_HILL
            and sig["delta_chi2_expected"] >= DETECTABLE_FACTOR * t99
            and sig["max_single_transit_sigma"] < MAX_SINGLE_TRANSIT_SIGMA
        )

    qualified = [r for r in results if qualifies(r)]
    for r in qualified:
        r["detectability_ratio"] = r["signal"]["delta_chi2_expected"] / t99
        r["objective_distance"] = abs(np.log(r["detectability_ratio"] / TARGET_FACTOR))
    qualified.sort(key=lambda r: (round(r["objective_distance"], 1), -r["hill_separation_min"]))

    n_unstable = sum(1 for r in results if not (r["stability"]["bound"] and not r["stability"]["orbit_crossing"]))
    n_close = sum(1 for r in results if r["hill_separation_min"] < MIN_HILL)
    n_weak = sum(1 for r in results if r["signal"] and r["signal"]["delta_chi2_expected"] < DETECTABLE_FACTOR * t99)
    n_obvious = sum(1 for r in results if r["signal"] and r["signal"]["max_single_transit_sigma"] >= MAX_SINGLE_TRANSIT_SIGMA)
    print(f"\n{len(qualified)} of {len(results)} candidates satisfy every criterion.")
    print(f"  rejected for instability: {n_unstable}; too close (< {MIN_HILL} R_H): {n_close}; "
          f"too weak: {n_weak}; obvious by eye: {n_obvious}")
    print(f"\n{'m':>6} {'a [au]':>7} {'P [d]':>7} {'ph':>5} {'i':>6} {'RH':>5} {'dchi2/T99':>10} {'max|dt|/s':>10} "
          f"{'d-tension':>10}  probes with signal")
    for r in qualified[:20]:
        p, s = r["parameters"], r["signal"]
        print(f"{p['mass_earth']:6.0f} {p['semi_major_axis_au']:7.3f} {r['period_days']:7.2f} "
              f"{p['mean_anomaly']:5.2f} {p['inclination_deg']:6.2f} {r['hill_separation_min']:5.1f} "
              f"{r['detectability_ratio']:10.1f} {s['max_single_transit_sigma']:10.2f} "
              f"{r['tension_with_real_data']['d']:10.1f}  "
              + " ".join(f"{k}:{v:.1f}" for k, v in s["per_planet_delta_chi2"].items()))

    payload = {
        "criteria": {
            "max_eccentricity": MAX_ECCENTRICITY, "max_energy_error": MAX_ENERGY_ERROR,
            "min_hill_separation": MIN_HILL, "detectable_factor": DETECTABLE_FACTOR,
            "target_factor": TARGET_FACTOR, "max_single_transit_sigma": MAX_SINGLE_TRANSIT_SIGMA,
            "impact_parameter_target": 1.4,
        },
        "null_threshold": {"T50": float(np.median(null_max)), "T99": t99, "T99.9": t999,
                           "n_trials": len(null_max), "n_trial_axes": int(len(axes))},
        "n_candidates": len(results),
        "n_qualified": len(qualified),
        "qualified": qualified,
        "all_results": results,
        "warning": "Planet X is synthetic. Not a claim about Kepler-90.",
    }
    (ROOT / args.out).write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(f"\nWROTE {args.out}")


if __name__ == "__main__":
    main()
