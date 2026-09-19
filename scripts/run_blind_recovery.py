"""
Phase 10.2 - blind recovery over many random hidden planets.

For each case:
  1. draw a random VALID hidden planet from the stated priors (mass, semi-major axis in the
     dynamically plausible zones, eccentricity, mutual-inclination tilt, phase), rejecting any
     that is unstable, crowds a neighbour, or would transit;
  2. simulate the system containing it and generate independent noisy observations at the real
     epochs with the real per-transit uncertainties;
  3. hand ONLY the observer dataset to the inference (the truth stays in this script);
  4. compare the recovery with the truth.

EVERY case is reported, including failures and non-detections. The summary reports detection
rate against injected signal strength, period recovery, and - where a posterior was sampled -
the CALIBRATION of the credible intervals (the fraction of cases whose truth falls inside the
central 68% and 95% intervals, which should be ~68% and ~95% if the posterior is honest).

Results are appended to data/blind_recovery.jsonl after each case, so an interrupted run resumes.

Run: uv run --python 3.12 --extra dev python scripts/run_blind_recovery.py --cases 24 --posterior-cases 8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from invisible_planet.experiments.config import load_config
from invisible_planet.experiments.sweep import template_dataset
from invisible_planet.experiments.synthetic import simulate_observations
from invisible_planet.inference.domain import is_plausible, stable_zones
from invisible_planet.inference.likelihood import TimingLikelihood
from invisible_planet.inference.forward import ForwardModel
from invisible_planet.inference.pipeline import recover
from invisible_planet.inference.sampler import tilt_to_inclination_node
from invisible_planet.observations.dataset import assert_no_ground_truth_leakage, assert_parameters_absent
from invisible_planet.systems.kepler90 import build_kepler90, load_reference
from invisible_planet.systems.planet_x import PlanetXParameters

ROOT = Path(__file__).resolve().parents[1]


def draw_planet(rng: np.random.Generator, reference: dict, stellar_radius: float, zones: list) -> PlanetXParameters:
    """A random valid hidden planet from the priors (rejection sampling)."""
    weights = np.array([hi - lo for lo, hi in zones])
    for _ in range(10000):
        lo, hi = zones[int(rng.choice(len(zones), p=weights / weights.sum()))]
        a = rng.uniform(lo, hi)
        mass = float(np.exp(rng.uniform(np.log(10.0), np.log(200.0))))
        ecc = float(rng.uniform(0.0, 0.15))
        omega = float(rng.uniform(0.0, 2 * np.pi))
        tilt = min(float(rng.rayleigh(2.0)), 5.9)
        psi = float(rng.uniform(0.0, 2 * np.pi))
        inc, node = tilt_to_inclination_node(tilt * np.cos(psi), tilt * np.sin(psi))
        p = PlanetXParameters(mass_earth=mass, semi_major_axis_au=float(a), eccentricity=ecc,
                              mean_anomaly=float(rng.uniform(0, 2 * np.pi)), inclination_deg=inc,
                              argument_of_periapsis=omega, longitude_of_ascending_node_deg=node)
        if is_plausible(p, reference, stellar_radius):
            return p
    raise RuntimeError("could not draw a valid planet")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=24)
    parser.add_argument("--posterior-cases", type=int, default=8)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--out", default="data/blind_recovery.jsonl")
    args = parser.parse_args()

    config = load_config()
    reference = load_reference()
    stellar_radius = float(build_kepler90(reference=reference).radii[0])
    zones = stable_zones(reference)
    master = int(config.raw["blind_recovery"]["master_seed"])
    cache = ROOT / "data" / "cache" / "periodogram_basis.npz"

    template = template_dataset()
    template_lk = TimingLikelihood(template)
    template_fw = ForwardModel(template, reference=reference)
    template_control = template_fw.control()

    out_path = ROOT / args.out
    done = set()
    if out_path.exists():
        done = {json.loads(line)["case"] for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    print(f"blind recovery: {args.cases} cases ({len(done)} already done), posterior for the first {args.posterior_cases}")

    for case in range(args.cases):
        rng = np.random.default_rng([master, case])
        truth = draw_planet(rng, reference, stellar_radius, zones)     # drawn ALWAYS, so seeds stay aligned
        if case in done:
            continue
        started = time.time()
        noise_seed = int(rng.integers(0, 2**31 - 1))
        dataset = simulate_observations(truth, seed=noise_seed, reference=reference)
        assert_no_ground_truth_leakage(dataset)
        assert_parameters_absent(dataset, truth)

        # expected (noise-free) signal, for relating success to detectability
        pred = template_fw.predict([truth])[0]
        v = template_lk.model_vector(pred, template_control)
        expected_dchi2 = float(v @ v)

        print(f"\n--- case {case}: m = {truth.mass_earth:.0f} M_earth, a = {truth.semi_major_axis_au:.4f} au, "
              f"e = {truth.eccentricity:.2f}, expected dchi2 = {expected_dchi2:.1f}")
        result = recover(dataset, reference=reference, basis_cache=cache, sample=(case < args.posterior_cases),
                         n_steps=args.steps, n_walkers=24, seed=case + 1, verbose=False)
        r = result.to_dict()
        record = {
            "case": case, "noise_seed": noise_seed, "truth": truth.to_dict(), "expected_delta_chi2": expected_dchi2,
            "detected": r["detected"], "false_alarm_probability": r["false_alarm_probability"],
            "chi2_null": r["chi2_null"], "best_chi2": r["best_chi2"], "best_parameters": r["best_parameters"],
            "posterior_summary": (r["posterior"]["summary"] if r["posterior"] else None),
            "posterior_chain_over_tau": (r["posterior"]["chain_length_over_tau"] if r["posterior"] else None),
            "verified_peaks": [{k: v for k, v in pk.items() if k != "parameters"} for pk in r["verified_peaks"]],
            "n_forward_simulations": r["n_forward_simulations"], "seconds": time.time() - started,
        }
        if r["best_parameters"]:
            best = r["best_parameters"]
            record["period_relative_error"] = None
            from invisible_planet.constants import EARTH_MASS_IN_SOLAR, kepler_third_law_period
            m_star = reference["star"]["mass_solar"]["value"]
            p_true = truth.period_days(m_star + truth.mass_solar)
            p_best = kepler_third_law_period(best["semi_major_axis_au"], m_star + best["mass_earth"] * EARTH_MASS_IN_SOLAR)
            record["period_relative_error"] = float((p_best - p_true) / p_true)
        with open(out_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=float) + "\n")
        print(f"    detected: {r['detected']}   FAP = {r['false_alarm_probability']:.2e}   "
              f"period error = {record.get('period_relative_error')}   ({record['seconds']:.0f} s)")

    print(f"\nresults in {args.out}")


if __name__ == "__main__":
    main()
