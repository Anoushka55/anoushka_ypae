"""
Phase 5.2 — baseline stability of the real Kepler-90 system, with NO Planet X.

Integrates the reference system and reports:
  * semi-major-axis evolution
  * eccentricity evolution
  * measured orbital periods vs. catalogue periods
  * minimum pairwise separation
  * energy, momentum and angular-momentum errors

If instability appears, this script is for DIAGNOSING it, not patching it. Do not
adjust initial conditions to make the numbers look better.

Run:
  uv run --python 3.12 --extra dev python scripts/run_baseline_stability.py
  uv run --python 3.12 --extra dev python scripts/run_baseline_stability.py --days 5000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from invisible_planet.physics.diagnostics import hill_separations, osculating_elements
from invisible_planet.physics.engine import NBodyEngine
from invisible_planet.systems.kepler90 import (
    PLANET_ORDER,
    build_kepler90,
    load_reference,
    recommended_timestep,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=float, default=1500.0,
                        help="integration span in days (default: the Kepler baseline)")
    parser.add_argument("--steps-per-inner-orbit", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--eccentric", action="store_true",
                        help="use the measured eccentricities for g and h")
    parser.add_argument("--out", type=str, default="data/baseline_stability.json")
    args = parser.parse_args()

    reference = load_reference()
    system = build_kepler90(use_measured_eccentricities=args.eccentric, reference=reference)
    dt = recommended_timestep(reference, args.steps_per_inner_orbit)
    n_steps = int(args.days / dt)
    chunk = max(1, n_steps // args.samples)

    print(f"System        : {system.metadata['system']}")
    print(f"Bodies        : {system.n_bodies} (1 star + {system.n_bodies - 1} planets)")
    print(f"Epoch         : BJD {system.epoch}")
    print(f"Eccentricities: {system.metadata['eccentricity_mode']}")
    print(f"Timestep      : {dt:.8f} d  ({args.steps_per_inner_orbit} steps/inner orbit)")
    print(f"Span          : {args.days:.1f} d  ({n_steps} steps)")
    print(f"Synthetic body: {system.metadata['contains_synthetic_body']}")
    print()

    engine = NBodyEngine(n_bodies=system.n_bodies)
    engine.set_state(system.masses, system.positions, system.velocities)

    d0 = engine.diagnostics()
    e0 = float(d0["total_energy"][0])
    l0 = d0["angular_momentum"][0]
    p0 = d0["linear_momentum"][0]

    times: list[float] = []
    axes: list[list[float]] = []
    eccs: list[list[float]] = []
    periods: list[list[float]] = []
    energy_err: list[float] = []
    angmom_err: list[float] = []
    momentum_err: list[float] = []
    min_sep: list[float] = []

    for _ in range(args.samples):
        engine.run(dt, chunk)
        d = engine.diagnostics()
        pos = engine.get_positions()[0]
        vel = engine.get_velocities()[0]
        elements = osculating_elements(system.masses, pos, vel)

        times.append(engine.time)
        axes.append([el["semi_major_axis"] for el in elements])
        eccs.append([el["eccentricity"] for el in elements])
        periods.append([el["period"] for el in elements])
        energy_err.append(abs(float(d["total_energy"][0]) - e0) / abs(e0))
        angmom_err.append(
            float(np.linalg.norm(d["angular_momentum"][0] - l0) / np.linalg.norm(l0))
        )
        momentum_err.append(float(np.linalg.norm(d["linear_momentum"][0] - p0)))
        min_sep.append(float(d["min_separation"][0]))

    axes_arr = np.array(axes)
    eccs_arr = np.array(eccs)
    periods_arr = np.array(periods)

    catalogue = {
        letter: reference["planets"][letter]["orbital_period_days"]["value"]
        for letter in PLANET_ORDER
    }

    print("SEMI-MAJOR AXIS AND ECCENTRICITY EVOLUTION")
    print(
        f"{'planet':>7} {'a_0 [au]':>10} {'da/a':>11} {'a trend':>11} "
        f"{'e_max':>10} {'e trend':>11}"
    )
    worst_da = 0.0
    for idx, letter in enumerate(PLANET_ORDER):
        a_series = axes_arr[:, idx]
        e_series = eccs_arr[:, idx]
        a0 = a_series[0]
        frac_range = (a_series.max() - a_series.min()) / a0
        worst_da = max(worst_da, frac_range)
        a_trend = np.polyfit(np.arange(len(a_series)), a_series, 1)[0] * len(a_series) / a0
        e_trend = np.polyfit(np.arange(len(e_series)), e_series, 1)[0] * len(e_series)
        print(
            f"{letter:>7} {a0:10.6f} {frac_range:11.3e} {a_trend:+11.3e} "
            f"{e_series.max():10.3e} {e_trend:+11.3e}"
        )

    print()
    print("MEASURED vs CATALOGUE PERIODS (osculating, time-averaged)")
    print(f"{'planet':>7} {'catalogue [d]':>14} {'measured [d]':>14} {'diff [s]':>12} {'rel':>11}")
    for idx, letter in enumerate(PLANET_ORDER):
        measured = float(np.mean(periods_arr[:, idx]))
        cat = catalogue[letter]
        print(
            f"{letter:>7} {cat:14.6f} {measured:14.6f} "
            f"{(measured - cat) * 86400.0:12.2f} {(measured - cat) / cat:11.2e}"
        )

    print()
    print("STABILITY / COMPACTNESS")
    deltas = hill_separations(
        system.masses[1:], axes_arr[0], system.masses[0]
    )
    pairs = [f"{PLANET_ORDER[i]}-{PLANET_ORDER[i+1]}" for i in range(len(PLANET_ORDER) - 1)]
    print("  adjacent separations in mutual Hill radii:")
    for pair, delta in zip(pairs, deltas):
        print(f"    {pair:>6}: {delta:6.2f}")
    print(f"  minimum separation over the run: {min(min_sep):.5f} au")

    print()
    print("CONSERVATION")
    print(f"  energy   max |dE/E|   : {max(energy_err):.3e}")
    print(f"  angmom   max |dL/L|   : {max(angmom_err):.3e}")
    print(f"  momentum max |dp|     : {max(momentum_err):.3e}")

    # -----------------------------------------------------------------
    # STABILITY CRITERIA
    #
    # These are deliberately PHYSICAL criteria, not thresholds on the size of
    # osculating variations. A compact multi-planet system legitimately shows
    # ~1% oscillations in its osculating elements; that is perturbation theory
    # working, not instability. What actually signals instability is orbit
    # crossing, unbounded secular drift, or a close approach.
    # -----------------------------------------------------------------
    print()
    print("ORBIT-CROSSING MARGIN  (apoapsis of inner vs periapsis of outer)")
    print(f"{'pair':>7} {'Q_inner [au]':>13} {'q_outer [au]':>13} {'margin [au]':>13}")
    crossing_margins = []
    for i in range(len(PLANET_ORDER) - 1):
        q_in_max = float(np.max(axes_arr[:, i] * (1.0 + eccs_arr[:, i])))
        q_out_min = float(np.min(axes_arr[:, i + 1] * (1.0 - eccs_arr[:, i + 1])))
        margin = q_out_min - q_in_max
        crossing_margins.append(margin)
        print(
            f"{pairs[i]:>7} {q_in_max:13.6f} {q_out_min:13.6f} {margin:13.6f}"
            + ("   <-- CROSSING" if margin <= 0 else "")
        )

    # bounded, not secular: the linear trend must be small compared with the
    # oscillation range it sits inside
    worst_drift_ratio = 0.0
    for idx in range(len(PLANET_ORDER)):
        a_series = axes_arr[:, idx]
        a_range = a_series.max() - a_series.min()
        trend = abs(np.polyfit(np.arange(len(a_series)), a_series, 1)[0] * len(a_series))
        if a_range > 0:
            worst_drift_ratio = max(worst_drift_ratio, trend / a_range)

    checks = {
        "no orbit crossing": min(crossing_margins) > 0.0,
        "semi-major axes bounded (no runaway)": worst_drift_ratio < 1.0,
        "all orbits remain elliptical": bool(eccs_arr.max() < 1.0),
        "no close approach": min(min_sep) > 0.005,
        "energy conserved": max(energy_err) < 1e-6,
        "angular momentum conserved": max(angmom_err) < 1e-10,
    }
    print()
    print("STABILITY CHECKS")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"  (largest osculating a variation: {worst_da:.3e}; "
          f"largest trend/range ratio: {worst_drift_ratio:.2f})")

    verdict_stable = all(checks.values())
    print()
    print(f"  VERDICT: {'STABLE' if verdict_stable else 'NEEDS INVESTIGATION'}")

    out = {
        "span_days": float(engine.time),
        "timestep_days": dt,
        "steps_per_inner_orbit": args.steps_per_inner_orbit,
        "eccentricity_mode": system.metadata["eccentricity_mode"],
        "contains_synthetic_body": False,
        "planets": {
            letter: {
                "a_initial": float(axes_arr[0, idx]),
                "a_fractional_range": float(
                    (axes_arr[:, idx].max() - axes_arr[:, idx].min()) / axes_arr[0, idx]
                ),
                "e_max": float(eccs_arr[:, idx].max()),
                "period_measured_mean": float(np.mean(periods_arr[:, idx])),
                "period_catalogue": catalogue[letter],
            }
            for idx, letter in enumerate(PLANET_ORDER)
        },
        "hill_separations": {p: float(d) for p, d in zip(pairs, deltas)},
        "orbit_crossing_margins_au": {p: float(m) for p, m in zip(pairs, crossing_margins)},
        "stability_checks": {k: bool(v) for k, v in checks.items()},
        "min_separation_au": float(min(min_sep)),
        "max_energy_error": float(max(energy_err)),
        "max_angular_momentum_error": float(max(angmom_err)),
        "max_momentum_error": float(max(momentum_err)),
        "stable": bool(verdict_stable),
    }
    path = ROOT / args.out
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWROTE {args.out}")


if __name__ == "__main__":
    main()
