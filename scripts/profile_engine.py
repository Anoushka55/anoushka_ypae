"""
Phase 17 - profile BEFORE optimising.

Measures, on the actual machine:
  * engine throughput vs ensemble size K (does the ensemble dimension parallelise?)
  * the cost of transit detection inside the integration loop
  * the cost of diagnostics and state read-back
  * peak memory

Run: uv run --python 3.12 --extra dev python scripts/profile_engine.py
"""

from __future__ import annotations

import argparse
import json
import os
import time
import tracemalloc
from pathlib import Path

import numpy as np

from invisible_planet.physics.engine import NBodyEngine
from invisible_planet.systems.kepler90 import (
    DEFAULT_DURATION_DAYS,
    build_kepler90,
    load_reference,
    recommended_timestep,
)

ROOT = Path(__file__).resolve().parents[1]


def make_engine(system, k: int, detect: bool) -> NBodyEngine:
    engine = NBodyEngine(n_bodies=system.n_bodies, n_ensembles=k)
    if detect:
        engine.enable_transit_detection(
            tracked_bodies=list(range(1, system.n_bodies)),
            max_transits=1024,
            stellar_radius=float(system.radii[0]),
            planet_radii=system.radii[1:],
        )
    engine.set_state(
        np.broadcast_to(system.masses, (k, system.n_bodies)).copy(),
        np.broadcast_to(system.positions, (k, system.n_bodies, 3)).copy(),
        np.broadcast_to(system.velocities, (k, system.n_bodies, 3)).copy(),
    )
    return engine


def time_run(system, k: int, days: float, dt: float, detect: bool, repeat: int = 2) -> float:
    engine = make_engine(system, k, detect)
    engine.run(dt, 200)  # compile / warm up
    best = np.inf
    n = int(days / dt)
    for _ in range(repeat):
        engine = make_engine(system, k, detect)
        t0 = time.perf_counter()
        engine.run(dt, n)
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=float, default=DEFAULT_DURATION_DAYS)
    parser.add_argument("--out", default="data/profile_engine.json")
    args = parser.parse_args()

    reference = load_reference()
    system = build_kepler90(reference=reference)
    dt = recommended_timestep(reference)
    steps = int(args.days / dt)
    print(f"machine: {os.cpu_count()} logical cores; system: {system.n_bodies} bodies; "
          f"{steps} steps of {dt:.6f} d over {args.days:.1f} d\n")

    rows = []
    print(f"{'K':>4} {'detect':>7} {'wall [s]':>9} {'s / system':>11} {'systems / s':>12} {'speed-up vs K=1':>16}")
    baseline = None
    for detect in (True, False):
        for k in (1, 4, 12, 24, 48, 96):
            wall = time_run(system, k, args.days, dt, detect)
            per = wall / k
            if baseline is None and k == 1 and detect:
                baseline = per
            rows.append({"k": k, "detect": detect, "wall_s": wall, "s_per_system": per})
            speed = (baseline / per) if baseline else float("nan")
            print(f"{k:4d} {str(detect):>7} {wall:9.2f} {per:11.3f} {1.0 / per:12.2f} {speed:16.2f}")

    # cost of state read-back and diagnostics
    engine = make_engine(system, 1, True)
    engine.run(dt, 1000)
    t0 = time.perf_counter()
    for _ in range(50):
        engine.get_positions()
    readback = (time.perf_counter() - t0) / 50
    t0 = time.perf_counter()
    for _ in range(20):
        engine.diagnostics()
    diagnostics = (time.perf_counter() - t0) / 20
    print(f"\nget_positions(): {readback * 1e3:.3f} ms;  diagnostics(): {diagnostics * 1e3:.3f} ms")

    tracemalloc.start()
    make_engine(system, 96, True)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Python-side allocation while building K=96 engine: {peak / 1e6:.2f} MB (Taichi fields are off-heap)")

    (ROOT / args.out).write_text(
        json.dumps({"cores": os.cpu_count(), "steps": steps, "dt": dt, "rows": rows,
                    "readback_ms": readback * 1e3, "diagnostics_ms": diagnostics * 1e3}, indent=2),
        encoding="utf-8",
    )
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    main()
