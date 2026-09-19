"""
Fit the simulated visible Kepler-90 system to the real transit times.

Writes data/kepler90_matched_initial_conditions.json, which build_kepler90() uses by
default. See invisible_planet/calibration/matching.py for the method and its limits.

Run: uv run --python 3.12 --extra dev python scripts/match_visible_system.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from invisible_planet.calibration.matching import (
    NON_PROBES,
    PROBES,
    MatchProblem,
    levenberg_marquardt,
    save_matched,
)
from invisible_planet.observations.ttv import fit_linear_ephemeris
from invisible_planet.systems.kepler90 import PLANET_ORDER

ROOT = Path(__file__).resolve().parents[1]


def describe(problem: MatchProblem, detail: dict, title: str) -> None:
    print(f"\n{title}")
    print(f"  {'planet':>6} {'role':>9} {'chi2':>10} {'n':>3} {'note':>60}")
    for k in PLANET_ORDER:
        d = detail[k]
        role = "probe" if k in PROBES else "ephemeris"
        if k in PROBES:
            note = (f"rms {d['rms_minutes']:8.2f} min, chi2(published sigma) "
                    f"{d['chi2_published_sigma']:9.1f}, {d['missing']} missing")
        else:
            note = f"dP/P {d['period_rel_offset']:+.2e}, epoch offset {d['epoch_offset_seconds']:+8.1f} s"
        print(f"  {k:>6} {role:>9} {d['chi2']:10.1f} {d['n']:3d} {note:>60}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--out", default="data/kepler90_matched_initial_conditions.json")
    args = parser.parse_args()

    problem = MatchProblem.create(steps_per_inner_orbit=args.steps)
    print(f"epoch BJD {problem.epoch_bjd}, window {problem.duration_days} d, "
          f"dt = {problem.dt:.6f} d ({args.steps} steps per innermost orbit)")

    x0 = np.zeros(2 * len(PLANET_ORDER))
    r0, detail0 = problem.residuals(problem.simulate([x0])[0])
    describe(problem, detail0, "BEFORE: catalogue periods used as osculating periods")

    started = time.time()
    result = levenberg_marquardt(problem, x0, max_iterations=args.iterations)
    print(f"\nfit took {time.time() - started:.0f} s")
    describe(problem, result["residual_detail"], "AFTER: fitted initial conditions")

    # mean transit period of each planet over the simulated window, for the record
    final = problem.simulate([result["x"]])[0]
    mean_periods = {}
    periods, _ = problem.unpack(result["x"])
    print(f"\n  {'planet':>6} {'catalogue P [d]':>16} {'osculating P [d]':>17} {'mean transit P [d]':>19}")
    for k in PLANET_ORDER:
        t = final[k]
        epochs = np.round((t - t[0]) / problem.base_period[k]).astype(int)
        mean_periods[k] = float(fit_linear_ephemeris(t, epochs=epochs).period)
        print(f"  {k:>6} {problem.base_period[k]:16.6f} {periods[k]:17.6f} {mean_periods[k]:19.6f}")

    save_matched(problem, result, ROOT / args.out, mean_periods)
    print(f"\nWROTE {args.out}")


if __name__ == "__main__":
    main()
