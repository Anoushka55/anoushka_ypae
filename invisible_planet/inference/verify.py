"""
Nonlinear verification of periodogram peaks.

The linearised periodogram ranks peaks with an approximate model and must never be trusted on
its own: in an earlier version the best-ranked peak scored chi2 = 519.8 linearised and 695.6
under the full N-body model (worse than no planet), while the true peak was ranked second.

Each shortlisted peak is therefore re-scored with the FULL model, in three rounds that each
use the true chi2 (no linearisation) except the first:

  1. local (a, phase) grid at the small periodogram reference mass, with mass solved
     analytically - a cheap way to find the right valley;
  2. mass scan at the best (a, phase);
  3. local (a, phase) grid again at that mass, then a final mass scan.

The winner across peaks is chosen by TRUE chi2.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..systems.planet_x import PlanetXParameters

MASS_GRID_EARTH = np.geomspace(1.5, 400.0, 33)


def _local_grid(centre_a: float, half_width: float, step: float) -> np.ndarray:
    n = int(round(half_width / step))
    return centre_a + step * np.arange(-n, n + 1)


def refine_peak(
    forward,
    likelihood,
    control: dict,
    data_vector: np.ndarray,
    axis_au: float,
    axis_step_au: float,
    reference_mass_earth: float = 5.0,
    inclination_deg: float = 88.0,
    n_phases: int = 12,
    half_width_steps: int = 4,
) -> dict:
    """Locate the best full-model candidate in the valley around one periodogram peak."""
    phases = np.linspace(0.0, 2.0 * np.pi, n_phases, endpoint=False)
    axes = _local_grid(axis_au, half_width_steps * axis_step_au, axis_step_au)
    trace = []

    # ---- round 1: small reference mass, analytic mass ---------------------------------
    candidates = [
        PlanetXParameters(mass_earth=reference_mass_earth, semi_major_axis_au=float(a),
                          mean_anomaly=float(p), inclination_deg=inclination_deg)
        for a in axes for p in phases
    ]
    predictions = forward.predict(candidates)
    best = None
    for c, pred in zip(candidates, predictions):
        if not all(np.all(np.isfinite(pred[k])) for k in likelihood.letters):
            continue
        v = likelihood.model_vector(pred, control)
        denominator = float(v @ v)
        if denominator <= 0.0:
            continue
        alpha = float(v @ data_vector) / denominator
        chi2_lin = float(data_vector @ data_vector) - alpha * float(v @ data_vector)
        if alpha > 0 and (best is None or chi2_lin < best[0]):
            best = (chi2_lin, c.semi_major_axis_au, c.mean_anomaly, reference_mass_earth * alpha)
    if best is None:
        return {"chi2": np.inf, "parameters": None, "trace": trace}
    _, a_best, phase_best, mass_start = best
    mass_start = float(np.clip(mass_start, 1.5, 400.0))
    trace.append({"round": 1, "a": a_best, "phase": phase_best, "mass": mass_start})

    current = PlanetXParameters(mass_earth=mass_start, semi_major_axis_au=a_best,
                                mean_anomaly=phase_best, inclination_deg=inclination_deg)

    def mass_scan(base: PlanetXParameters):
        cands = [replace(base, mass_earth=float(m)) for m in MASS_GRID_EARTH]
        preds = forward.predict(cands)
        chi2 = np.array([likelihood.chi2(p) for p in preds])
        i = int(np.argmin(chi2))
        return cands[i], float(chi2[i]), chi2

    # ---- round 2: nonlinear mass scan -------------------------------------------------
    current, chi2_best, _ = mass_scan(current)
    trace.append({"round": 2, "a": current.semi_major_axis_au, "phase": current.mean_anomaly,
                  "mass": current.mass_earth, "chi2": chi2_best})

    # ---- round 3: (a, phase) grid at the fitted mass, true chi2, then mass again -------
    fine = _local_grid(current.semi_major_axis_au, 2 * axis_step_au, axis_step_au / 2.0)
    phase_span = np.mod(current.mean_anomaly + np.linspace(-np.pi / 3, np.pi / 3, 9), 2 * np.pi)
    candidates = [replace(current, semi_major_axis_au=float(a), mean_anomaly=float(p))
                  for a in fine for p in phase_span]
    predictions = forward.predict(candidates)
    scores = np.array([likelihood.chi2(p) for p in predictions])
    i = int(np.argmin(scores))
    if scores[i] < chi2_best:
        current, chi2_best = candidates[i], float(scores[i])
    current, chi2_best, mass_chi2 = mass_scan(current)
    trace.append({"round": 3, "a": current.semi_major_axis_au, "phase": current.mean_anomaly,
                  "mass": current.mass_earth, "chi2": chi2_best})

    return {"chi2": chi2_best, "parameters": current, "trace": trace,
            "mass_grid": MASS_GRID_EARTH.tolist(), "mass_chi2": mass_chi2.tolist()}


def verify_peaks(forward, likelihood, control, data_vector, peaks: list, verbose=True,
                 settings: dict | None = None) -> dict:
    """Refine every shortlisted peak with the full model and return them ranked by true chi2.

    `peaks` is a list of dicts with keys axis_au, axis_step_au, linear_delta_chi2.
    `settings` may override refine_peak's n_phases and half_width_steps (used by fast tests).
    """
    results = []
    for index, peak in enumerate(peaks, start=1):
        r = refine_peak(forward, likelihood, control, data_vector, peak["axis_au"], peak["axis_step_au"],
                        **(settings or {}))
        r["peak"] = index
        r["axis_au"] = peak["axis_au"]
        r["linear_delta_chi2"] = peak["linear_delta_chi2"]
        results.append(r)
        if verbose and r["parameters"] is not None:
            p = r["parameters"]
            print(f"    peak {index}: a0 = {peak['axis_au']:.4f} au  linear dchi2 = {peak['linear_delta_chi2']:6.1f}"
                  f"  ->  full-model chi2 = {r['chi2']:8.1f}  (a = {p.semi_major_axis_au:.4f}, m = {p.mass_earth:.1f})")
    ranked = sorted(results, key=lambda r: r["chi2"])
    return {"peaks": results, "best": ranked[0]}
