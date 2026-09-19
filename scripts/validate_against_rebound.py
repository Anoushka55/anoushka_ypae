"""
Independent accuracy check: this project's engine vs REBOUND IAS15.

Reports, for the full 8-planet Kepler-90 system (no Planet X):
  * transit-count agreement
  * raw transit-time differences (dominated by the constant O(dt^2) frequency offset)
  * O-C RESIDUAL differences after each code's own linear-ephemeris fit
    (the quantity the inference actually reads)
  * final-state agreement

Run: uv run --python 3.12 --extra dev python scripts/validate_against_rebound.py --days 700
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from invisible_planet.constants import SECONDS_PER_DAY
from invisible_planet.observations.transits import observe_transits
from invisible_planet.observations.ttv import fit_linear_ephemeris
from invisible_planet.physics.engine import NBodyEngine
from invisible_planet.systems.kepler90 import build_kepler90, load_reference, recommended_timestep
from invisible_planet.validation.independent import rebound_final_state, rebound_transit_times

ROOT = Path(__file__).resolve().parents[1]


def compare(days: float, steps_per_inner_orbit: int) -> dict:
    reference = load_reference()
    system = build_kepler90(reference=reference)
    dt = recommended_timestep(reference, steps_per_inner_orbit)
    names = system.names[1:]

    mine = observe_transits(system, days, dt, tracked=names)
    theirs = rebound_transit_times(system, days, names)

    rows = {}
    for name in names:
        a, b = mine.for_planet(name), theirs[name]
        row = {"n_engine": int(len(a)), "n_rebound": int(len(b))}
        if len(a) == len(b) and len(a) >= 3:
            raw = (a - b) * SECONDS_PER_DAY
            fa, fb = fit_linear_ephemeris(a), fit_linear_ephemeris(b)
            oc_diff = (fa.residuals - fb.residuals) * SECONDS_PER_DAY
            row.update(
                {
                    "raw_diff_max_s": float(np.max(np.abs(raw))),
                    "period_diff_rel": float((fa.period - fb.period) / fb.period),
                    "oc_diff_max_s": float(np.max(np.abs(oc_diff))),
                    "oc_diff_rms_s": float(np.sqrt(np.mean(oc_diff**2))),
                    "oc_signal_rms_s": float(fb.rms_seconds),
                }
            )
        rows[name] = row

    # final state
    engine = NBodyEngine(n_bodies=system.n_bodies)
    engine.set_state(system.masses, system.positions, system.velocities)
    engine.run(dt, int(round(days / dt)))
    p_engine = engine.get_positions()[0]
    p_ref, _ = rebound_final_state(system, engine.time)
    state_diff = np.max(np.linalg.norm(p_engine - p_ref, axis=1))

    return {"days": days, "dt": dt, "steps_per_inner_orbit": steps_per_inner_orbit,
            "planets": rows, "max_position_difference_au": float(state_diff)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=float, default=700.0)
    parser.add_argument("--steps", type=int, nargs="+", default=[1000])
    parser.add_argument("--out", type=str, default="data/rebound_validation.json")
    args = parser.parse_args()

    results = []
    for steps in args.steps:
        result = compare(args.days, steps)
        results.append(result)
        print(f"\n=== {args.days:.0f} d, {steps} steps per inner orbit (dt = {result['dt']:.6f} d) ===")
        print(f"{'planet':>14} {'n(eng)':>7} {'n(reb)':>7} {'raw max [s]':>12} {'dP/P':>10} "
              f"{'O-C diff max [s]':>17} {'O-C diff rms [s]':>17} {'signal rms [s]':>15}")
        for name, row in result["planets"].items():
            if "oc_diff_max_s" in row:
                print(f"{name:>14} {row['n_engine']:7d} {row['n_rebound']:7d} "
                      f"{row['raw_diff_max_s']:12.2f} {row['period_diff_rel']:10.2e} "
                      f"{row['oc_diff_max_s']:17.4f} {row['oc_diff_rms_s']:17.4f} "
                      f"{row['oc_signal_rms_s']:15.2f}")
            else:
                print(f"{name:>14} {row['n_engine']:7d} {row['n_rebound']:7d}   (count mismatch or <3 transits)")
        print(f"  max |position difference| at t = {args.days:.0f} d : "
              f"{result['max_position_difference_au']:.3e} au")

    (ROOT / args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWROTE {args.out}")


if __name__ == "__main__":
    main()
