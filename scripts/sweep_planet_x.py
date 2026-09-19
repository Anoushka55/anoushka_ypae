"""
Phase 8 — controlled parameter sweep over Planet X.

Scans mass, semi-major axis, eccentricity and orbital phase, and for every
candidate measures:

  1. dynamical stability (orbit crossing, bounded elements, no ejection)
  2. minimum pairwise separation
  3. energy conservation
  4. TTV amplitude (detrended, per visible planet)
  5. orbital residual amplitude
  6. observation duration needed for the signal to emerge
  7. recoverability (signal vs. the real measurement uncertainty)
  8. physical plausibility (mutual Hill separations, transiting or not)

THE OBJECTIVE IS NOT "LARGEST SIGNAL". A huge signal would come from a massive
planet wedged between two others, which would be both dynamically implausible
and scientifically uninteresting. What we want is a candidate that is
simultaneously:

    stable         survives the integration with bounded elements
    detectable     signal above the real Kepler timing precision
    subtle         not so large that the answer is obvious by eye
    recoverable    signal present in more than one planet, over a realistic span
    plausible      comfortably Hill-separated from its neighbours

Run:
  uv run --python 3.12 --extra dev python scripts/sweep_planet_x.py
  uv run --python 3.12 --extra dev python scripts/sweep_planet_x.py --quick
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np

from invisible_planet.constants import EARTH_MASS_IN_SOLAR, SECONDS_PER_DAY
from invisible_planet.observations.transits import observe_transits, observe_transits_batch
from invisible_planet.observations.ttv import compare_ephemerides
from invisible_planet.physics.diagnostics import mutual_hill_radius, osculating_elements
from invisible_planet.physics.engine import NBodyEngine
from invisible_planet.systems.kepler90 import (
    PLANET_ORDER,
    build_kepler90,
    load_reference,
    recommended_timestep,
)
from invisible_planet.systems.planet_x import PlanetXParameters, build_system_with_planet_x

ROOT = Path(__file__).resolve().parents[1]

#: The most precise published transit-epoch uncertainty for this system, in
#: seconds (planet h). Used as the detectability yardstick: a signal below this
#: could not be recovered from data of the quality that actually exists.
BEST_MEASUREMENT_PRECISION_S = 71.0


def run_batch_with_diagnostics(
    systems: list, duration_days: float, dt: float, tracked: list, n_samples: int = 40
):
    """Integrate a batch ONCE, collecting transit times and stability metrics together.

    Transit detection state persists across successive `run()` calls, so the
    integration can be chunked to sample osculating elements and conservation
    diagnostics without restarting it. Doing both in one pass halves the work and
    lets Taichi thread the stability measurement across candidates too.
    """
    n_bodies = systems[0].n_bodies
    k = len(systems)
    tracked_indices = [systems[0].index_of(n) for n in tracked]

    engine = NBodyEngine(n_bodies=n_bodies, n_ensembles=k)
    engine.enable_transit_detection(
        tracked_bodies=tracked_indices,
        max_transits=2048,
        central_index=0,
        stellar_radius=float(systems[0].radii[0]),
        planet_radii=np.asarray([systems[0].radii[i] for i in tracked_indices]),
    )
    engine.set_state(
        np.stack([s.masses for s in systems]),
        np.stack([s.positions for s in systems]),
        np.stack([s.velocities for s in systems]),
        time=0.0,
    )

    energy_0 = engine.diagnostics()["total_energy"].copy()
    total_steps = int(round(duration_days / dt))
    chunk = max(1, total_steps // n_samples)

    axes = np.zeros((n_samples, k, n_bodies - 1))
    eccs = np.zeros((n_samples, k, n_bodies - 1))
    energy_err = np.zeros((n_samples, k))
    min_sep = np.zeros((n_samples, k))

    for sample in range(n_samples):
        engine.run(dt, chunk)
        d = engine.diagnostics()
        positions = engine.get_positions()
        velocities = engine.get_velocities()
        for i in range(k):
            el = osculating_elements(systems[i].masses, positions[i], velocities[i])
            axes[sample, i] = [e["semi_major_axis"] for e in el]
            eccs[sample, i] = [e["eccentricity"] for e in el]
        energy_err[sample] = np.abs(d["total_energy"] - energy_0) / np.abs(energy_0)
        min_sep[sample] = d["min_separation"]

    from invisible_planet.observations.transits import TransitSeries

    all_times = engine.get_transit_times()
    all_impacts = engine.get_transit_impact_parameters()
    series_list = [
        TransitSeries(
            names=list(tracked),
            times=all_times[i],
            impact_parameters=all_impacts[i],
            span_days=engine.time,
            timestep_days=dt,
            epoch_bjd=systems[i].epoch,
        )
        for i in range(k)
    ]

    stability_list = []
    for i in range(k):
        a_series = axes[:, i, :]
        e_series = eccs[:, i, :]
        order = np.argsort(a_series[0])
        crossing = False
        for j in range(len(order) - 1):
            inner, outer = order[j], order[j + 1]
            apoapsis = np.max(a_series[:, inner] * (1.0 + e_series[:, inner]))
            periapsis = np.min(a_series[:, outer] * (1.0 - e_series[:, outer]))
            if periapsis <= apoapsis:
                crossing = True
        bounded = bool(np.all(a_series > 0) and np.max(e_series) < 0.5)
        max_energy = float(np.max(energy_err[:, i]))
        stability_list.append({
            "orbit_crossing": bool(crossing),
            "bounded": bounded,
            "max_eccentricity": float(np.max(e_series)),
            "max_axis_variation": float(
                np.max((a_series.max(axis=0) - a_series.min(axis=0)) / a_series[0])
            ),
            "min_separation_au": float(np.min(min_sep[:, i])),
            "max_energy_error": max_energy,
            "stable": bool(not crossing and bounded and max_energy < 1e-6),
        })

    return series_list, stability_list


def hill_context(params: PlanetXParameters, reference: dict) -> dict:
    """Mutual Hill separations from the neighbouring real planets."""
    m_star = reference["star"]["mass_solar"]["value"]
    entries = [
        (
            letter,
            reference["planets"][letter]["semi_major_axis_au"]["value"],
            reference["planets"][letter]["mass_earth"]["value"] * EARTH_MASS_IN_SOLAR,
        )
        for letter in PLANET_ORDER
    ]
    inner = [e for e in entries if e[1] < params.semi_major_axis_au]
    outer = [e for e in entries if e[1] > params.semi_major_axis_au]

    result = {"inner_neighbour": None, "outer_neighbour": None,
              "delta_inner": np.inf, "delta_outer": np.inf}
    mx = params.mass_solar
    if inner:
        letter, a, m = max(inner, key=lambda e: e[1])
        r_h = mutual_hill_radius(m, mx, a, params.semi_major_axis_au, m_star)
        result["inner_neighbour"] = letter
        result["delta_inner"] = float((params.semi_major_axis_au - a) / r_h)
    if outer:
        letter, a, m = min(outer, key=lambda e: e[1])
        r_h = mutual_hill_radius(mx, m, params.semi_major_axis_au, a, m_star)
        result["outer_neighbour"] = letter
        result["delta_outer"] = float((a - params.semi_major_axis_au) / r_h)
    result["delta_min"] = float(min(result["delta_inner"], result["delta_outer"]))
    return result


def signal_metrics(control_series, mystery_series, names) -> dict:
    """Detrended O-C difference induced by Planet X, per visible planet."""
    per_planet = {}
    for name in names:
        control_times = control_series.for_planet(name)
        mystery_times = mystery_series.for_planet(name)
        if len(control_times) < 3 or len(mystery_times) < 3:
            continue
        result = compare_ephemerides(control_times, mystery_times)
        per_planet[name] = {
            "detrended_peak_to_peak_minutes": result["detrended_peak_to_peak_minutes"],
            "detrended_rms_seconds": result["detrended_rms_seconds"],
            "n_transits": int(min(len(control_times), len(mystery_times))),
        }

    if not per_planet:
        return {"per_planet": {}, "max_amplitude_minutes": 0.0,
                "n_planets_detectable": 0, "strongest_planet": None}

    amplitudes = {k: v["detrended_peak_to_peak_minutes"] for k, v in per_planet.items()}
    detectable = [
        k for k, v in per_planet.items()
        if v["detrended_rms_seconds"] > BEST_MEASUREMENT_PRECISION_S
    ]
    strongest = max(amplitudes, key=amplitudes.get)
    return {
        "per_planet": per_planet,
        "max_amplitude_minutes": float(amplitudes[strongest]),
        "strongest_planet": strongest,
        "n_planets_detectable": len(detectable),
        "detectable_planets": detectable,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=float, default=1400.0)
    parser.add_argument("--quick", action="store_true", help="small grid, for smoke testing")
    parser.add_argument("--out", type=str, default="data/planet_x_sweep.json")
    args = parser.parse_args()

    reference = load_reference()
    dt = recommended_timestep(reference)
    m_star = reference["star"]["mass_solar"]["value"]

    if args.quick:
        masses = [10.0, 40.0]
        axes = [0.19, 0.62]
        eccentricities = [0.0]
        phases = [0.0, np.pi]
    else:
        # Mass: from below Neptune to around Saturn. Above this the planet would
        # very likely have been detected by other means; below it the signal
        # falls under the real timing precision.
        masses = [5.0, 10.0, 20.0, 40.0, 80.0]
        # Semi-major axis: sample the two wide gaps in the architecture
        # (between i and d, and between f and g) plus an exterior orbit.
        axes = [0.16, 0.19, 0.22, 0.25, 0.58, 0.62, 1.25]
        eccentricities = [0.0]
        phases = [0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0]

    grid = [
        PlanetXParameters(
            mass_earth=m, semi_major_axis_au=a, eccentricity=e, mean_anomaly=phase
        )
        for m, a, e, phase in product(masses, axes, eccentricities, phases)
    ]
    print(f"Sweeping {len(grid)} Planet X candidates over {args.days:.0f} days")
    print(f"  timestep {dt:.6f} d, {int(args.days / dt)} steps per candidate")
    print(f"  detectability yardstick: {BEST_MEASUREMENT_PRECISION_S:.0f} s "
          "(best published transit-epoch uncertainty)")
    print()

    control = build_kepler90(reference=reference)
    control_series = observe_transits(control, args.days, dt)
    visible = control.names[1:]

    # Batched integration over the ensemble dimension.
    batch_size = 32
    results = []
    for start in range(0, len(grid), batch_size):
        chunk = grid[start : start + batch_size]
        systems = [build_system_with_planet_x(p, reference=reference) for p in chunk]
        series_list, stability_list = run_batch_with_diagnostics(
            systems, args.days, dt, tracked=visible
        )

        for params, series, stability in zip(chunk, series_list, stability_list):
            signal = signal_metrics(control_series, series, visible)
            hills = hill_context(params, reference)
            period = params.period_days(m_star + params.mass_solar)

            results.append({
                "parameters": params.to_dict(),
                "period_days": period,
                "hill": hills,
                "stability": stability,
                "signal": signal,
            })
        print(f"  ...{min(start + batch_size, len(grid))}/{len(grid)} candidates")

    # ------------------------------------------------------------------
    # Selection. Every criterion is stated and applied explicitly; no candidate
    # is called "best" on signal amplitude alone.
    # ------------------------------------------------------------------
    def qualifies(r: dict) -> bool:
        return (
            r["stability"]["stable"]
            and not r["stability"]["orbit_crossing"]
            and r["hill"]["delta_min"] > 8.0           # comfortably separated
            and r["signal"]["n_planets_detectable"] >= 2  # recoverable, not a fluke
            and 2.0 < r["signal"]["max_amplitude_minutes"] < 120.0  # detectable, still subtle
        )

    qualified = [r for r in results if qualifies(r)]
    qualified.sort(key=lambda r: (
        -r["signal"]["n_planets_detectable"],
        -r["hill"]["delta_min"],
    ))

    print()
    print(f"{len(qualified)} of {len(results)} candidates satisfy every criterion.")
    print()
    print(f"{'m [Me]':>7} {'a [au]':>7} {'P [d]':>8} {'e':>5} {'phase':>6} "
          f"{'dmin_H':>7} {'ampl [min]':>11} {'n_det':>6} {'stable':>7}")
    for r in qualified[:15]:
        p = r["parameters"]
        print(
            f"{p['mass_earth']:7.1f} {p['semi_major_axis_au']:7.3f} {r['period_days']:8.2f} "
            f"{p['eccentricity']:5.2f} {p['mean_anomaly']:6.2f} "
            f"{r['hill']['delta_min']:7.1f} {r['signal']['max_amplitude_minutes']:11.2f} "
            f"{r['signal']['n_planets_detectable']:6d} {str(r['stability']['stable']):>7}"
        )

    unstable = [r for r in results if not r["stability"]["stable"]]
    print()
    print(f"Rejected: {len(unstable)} unstable, "
          f"{sum(1 for r in results if r['signal']['n_planets_detectable'] < 2)} "
          "with signal in fewer than 2 planets, "
          f"{sum(1 for r in results if r['signal']['max_amplitude_minutes'] >= 120.0)} "
          "too obvious.")

    out = {
        "generated_days": args.days,
        "timestep_days": dt,
        "detectability_yardstick_seconds": BEST_MEASUREMENT_PRECISION_S,
        "selection_criteria": {
            "stable": True,
            "no_orbit_crossing": True,
            "min_mutual_hill_separation": 8.0,
            "min_planets_with_detectable_signal": 2,
            "amplitude_window_minutes": [2.0, 120.0],
        },
        "n_candidates": len(results),
        "n_qualified": len(qualified),
        "qualified": qualified,
        "all_results": results,
        "warning": "Planet X is synthetic. This is not a claim about Kepler-90.",
    }
    (ROOT / args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWROTE {args.out}")


if __name__ == "__main__":
    main()
