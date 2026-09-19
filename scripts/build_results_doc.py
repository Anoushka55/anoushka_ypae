"""
Generate docs/RESULTS.md (and the results block of README.md) from the recorded result files.

Every number in those documents is read from data/*.json(l); none is typed by hand. Hand-typed
numbers were the source of several errors caught during this project (docs/CORRECTIONS.md).

Inputs : data/experiment_result.json, data/planet_x_selection.json, data/blind_recovery.jsonl,
         data/rebound_validation.json, data/baseline_stability.json (each optional)
Outputs: docs/RESULTS.md ; the block between <!-- RESULTS:START --> and <!-- RESULTS:END --> in README.md

Run: uv run --python 3.12 --extra dev python scripts/build_results_doc.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from invisible_planet.constants import EARTH_MASS_IN_SOLAR, kepler_third_law_period
from invisible_planet.systems.kepler90 import load_reference

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def load(name: str):
    path = DATA / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def load_jsonl(name: str) -> list:
    path = DATA / name
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def interval(q: dict, fmt: str) -> str:
    return f"{fmt.format(q['50'])} (68 %: {fmt.format(q['16'])} – {fmt.format(q['84'])}; 95 %: {fmt.format(q['2.5'])} – {fmt.format(q['97.5'])})"


def fap_text(fap: float, n_trials: int) -> str:
    floor = 1.0 / (n_trials + 1.0)
    return f"< {floor:.1e} (no noise-only trial in {n_trials} reached the statistic)" if fap <= floor * 1.0001 else f"{fap:.2e}"


def flagship_section() -> tuple:
    r = load("experiment_result.json")
    if r is None:
        return "*(data/experiment_result.json not found; run scripts/run_experiment.py)*\n", ""
    inf, truth = r["inference"], r["ground_truth"]
    lines = ["## Flagship blind recovery (one hidden planet, one dataset)", "",
             "**Planet X is synthetic.** The observer dataset holds only measured transit times of planets "
             + ", ".join(r["observations"].keys()) + " with per-transit uncertainties.", ""]
    ntr = inf["periodogram"]["n_null_trials"] if inf.get("periodogram") else 0
    lines += [
        f"* null hypothesis (visible planets only): χ² = {inf['chi2_null']:.1f} for {inf['n_points']} points "
        f"({inf['dof_null']} degrees of freedom after the linear ephemerides)",
        f"* strongest linearised periodogram peak: Δχ² = {inf['linear_best_delta_chi2']:.1f}",
        f"* false-alarm probability (exact noise-only distribution of the same statistic, "
        f"look-elsewhere corrected): {fap_text(inf['false_alarm_probability'], ntr)}",
        f"* detected: **{inf['detected']}**",
    ]
    if inf["best_chi2"] is not None:
        lines.append(f"* full N-body fit: χ² = {inf['best_chi2']:.1f} "
                     f"(improvement {inf['chi2_null'] - inf['best_chi2']:.1f}; 7 fitted parameters)")
    lines.append(f"* forward N-body simulations used: {inf['n_forward_simulations']}")
    lines.append("")
    block = "\n".join(lines[3:])
    if inf["detected"] and inf["posterior"]:
        s, cmp_ = inf["posterior"]["summary"], r["comparison"]
        flags = cmp_.get("truth_inside_95pct_interval", {})
        rows = [
            ("mass [M⊕]", f"{truth['mass_earth']:.1f}", interval(s["mass_earth"], "{:.1f}"), flags.get("mass_earth")),
            ("semi-major axis [au]", f"{truth['semi_major_axis_au']:.4f}", interval(s["semi_major_axis_au"], "{:.4f}"),
             flags.get("semi_major_axis_au")),
            ("period [d]", f"{r['true_period_days']:.3f}", interval(s["period_days"], "{:.3f}"), flags.get("period_days")),
            ("eccentricity", f"{truth['eccentricity']:.3f}", interval(s["eccentricity"], "{:.3f}"), flags.get("eccentricity")),
        ]
        lines += ["| parameter | truth | posterior median and intervals | truth inside 95 %? |", "|---|---|---|---|"]
        lines += [f"| {a} | {b} | {c} | {'yes' if d else 'NO'} |" for a, b, c, d in rows]
        post = inf["posterior"]
        lines += ["", f"Sampler: {len(post['parameter_names'])} parameters, mean acceptance "
                      f"{np.mean(post['acceptance_fraction']):.2f}, chain length / autocorrelation time "
                      f"{post['chain_length_over_tau']:.0f}, effective samples {post['effective_samples']:.0f}.",
                  f"Period error of the best fit: {cmp_['period_relative_error']:+.3%}."]
        for note in inf["notes"]:
            lines.append(f"* NOTE: {note}")
        block += "\n\n" + "\n".join(lines[-len(rows) - 8:])
    elif inf["detected"]:
        lines.append("Detected, but no posterior was sampled.")
    else:
        lines.append("No detection was made in this dataset; no planet is claimed.")
    lines.append("")
    return "\n".join(lines), block


def selection_section() -> str:
    sel = load("planet_x_selection.json")
    if sel is None:
        return ""
    c = sel["selected"]
    p = c["parameters"]
    lines = ["## How the hidden planet was chosen (pre-declared criteria, not cherry-picked)", ""]
    crit = sel.get("criteria", {})
    lines.append("Criteria fixed before the sweep (see `docs/PLANET_X_DESIGN.md`): " + json.dumps(crit) + ".")
    lines.append("")
    lines.append(f"Screened {len(sel['screened'])} candidates that met every criterion for recoverability in "
                 f"{sel['n_noise_realisations']} independent noise realisations "
                 f"(required ≥ {sel['required_recoverable_fraction']:.0%} detected AND localised).")
    lines.append("")
    lines.append(f"Selected: m = {p['mass_earth']:.0f} M⊕, a = {p['semi_major_axis_au']:.4f} au, "
                 f"P = {c['period_days']:.2f} d, e = {p['eccentricity']:.2f}, "
                 f"{c['hill_separation_min']:.1f} mutual Hill radii from its nearest neighbour, "
                 f"detectability {c['detectability_ratio']:.1f} × the noise-only 99 % threshold.")
    lines.append("")
    lines.append("**Tension with the real data:** " + json.dumps(c.get("tension_with_real_data")) +
                 " (Δχ² of the real d, e timing against the visible-only model when this planet is added). "
                 "A large value means the REAL Kepler-90 timing does not favour such a planet. This one is "
                 "not a candidate for the real star; it is a synthetic test object.")
    lines.append("")
    return "\n".join(lines)


def blind_section() -> tuple:
    records = load_jsonl("blind_recovery.jsonl")
    if not records:
        return "", ""
    reference = load_reference()
    m_star = reference["star"]["mass_solar"]["value"]
    n = len(records)
    detected = [r for r in records if r["detected"]]
    lines = ["## Blind recovery over random hidden planets", "",
             f"{n} planets drawn from stated priors (mass 10–200 M⊕ log-uniform in the dynamically allowed zones; "
             "e ≤ 0.15; mutual tilt Rayleigh(2°); non-transiting). Each was injected, observed with the real "
             "epochs and real per-transit uncertainties, and handed to the inference as a bare dataset. "
             "EVERY case is listed in `data/blind_recovery.jsonl`, including failures.", ""]
    exp = np.array([r["expected_delta_chi2"] for r in records])
    det = np.array([r["detected"] for r in records])
    edges = [0, 10, 25, 50, 100, np.inf]
    lines += ["| expected noise-free Δχ² | cases | detected |", "|---|---|---|"]
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (exp >= lo) & (exp < hi)
        label = f"{lo:g} – {hi:g}" if np.isfinite(hi) else f"≥ {lo:g}"
        lines.append(f"| {label} | {int(sel.sum())} | {int(det[sel].sum())} |")
    lines.append("")
    errs = []
    for r in detected:
        if r.get("period_relative_error") is not None:
            errs.append(abs(r["period_relative_error"]))
    if errs:
        lines.append(f"Period recovery (best fit, detected cases): median |error| = {np.median(errs):.3%}, "
                     f"90th percentile = {np.percentile(errs, 90):.3%}, worst = {np.max(errs):.3%} "
                     f"({len(errs)} cases).")
        lines.append("")
    # calibration of credible intervals
    with_post = [r for r in records if r.get("posterior_summary")]
    if with_post:
        counts = {"mass_earth": [0, 0], "semi_major_axis_au": [0, 0], "period_days": [0, 0], "eccentricity": [0, 0]}
        for r in with_post:
            t = r["truth"]
            true_values = {
                "mass_earth": t["mass_earth"], "semi_major_axis_au": t["semi_major_axis_au"],
                "period_days": kepler_third_law_period(t["semi_major_axis_au"], m_star + t["mass_earth"] * EARTH_MASS_IN_SOLAR),
                "eccentricity": t["eccentricity"],
            }
            for key, value in true_values.items():
                q = r["posterior_summary"][key]
                counts[key][0] += int(q["16"] <= value <= q["84"])
                counts[key][1] += int(q["2.5"] <= value <= q["97.5"])
        m = len(with_post)
        lines += [f"Calibration of the posterior over {m} cases with sampled posteriors "
                  "(an honest posterior contains the truth in ≈ 68 % / ≈ 95 % of cases; with so few cases the "
                  "binomial uncertainty is large, ±" + f"{100 * np.sqrt(0.68 * 0.32 / m):.0f} percentage points at 68 %):", "",
                  "| parameter | truth inside 68 % interval | truth inside 95 % interval |", "|---|---|---|"]
        for key, (a, b) in counts.items():
            lines.append(f"| {key} | {a} / {m} | {b} / {m} |")
        lines.append("")
        taus = [r["posterior_chain_over_tau"] for r in with_post if r.get("posterior_chain_over_tau")]
        if taus:
            lines.append(f"Chain length / autocorrelation time: median {np.median(taus):.0f}, minimum {np.min(taus):.0f} "
                         "(< 50 means the intervals are provisional).")
            lines.append("")
    lines.append(f"Detected {len(detected)} of {n}. Missed cases had expected Δχ² of "
                 + (", ".join(f"{r['expected_delta_chi2']:.1f}" for r in records if not r["detected"]) or "— (none missed)") + ".")
    lines.append("")
    block = f"Blind recovery: {len(detected)} of {n} random hidden planets detected; see docs/RESULTS.md."
    return "\n".join(lines), block


def validation_section() -> str:
    lines = ["## Validation numbers", ""]
    rb = load("rebound_validation.json")
    if rb:
        lines.append("Independent IAS15 (REBOUND) comparison: see `data/rebound_validation.json` and "
                     "`docs/VALIDATION.md`.")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    flagship, flagship_block = flagship_section()
    blind, blind_block = blind_section()
    text = "\n".join([
        "# Results", "",
        "Generated by `scripts/build_results_doc.py` from the files in `data/`. Nothing here is typed by hand.",
        "**Planet X is synthetic; nothing here is a claim about the real Kepler-90 system.**", "",
        flagship, selection_section(), blind, validation_section(),
    ])
    (ROOT / "docs" / "RESULTS.md").write_text(text, encoding="utf-8")
    readme = ROOT / "README.md"
    if readme.exists():
        body = readme.read_text(encoding="utf-8")
        start, end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
        if start in body and end in body:
            new = (f"{start}\n" + flagship_block + ("\n\n" + blind_block if blind_block else "")
                   + "\n\nFull tables and caveats: [docs/RESULTS.md](docs/RESULTS.md).\n" + end)
            head, rest = body.split(start, 1)
            _, tail = rest.split(end, 1)
            readme.write_text(head + new + tail, encoding="utf-8")
    print("wrote docs/RESULTS.md")


if __name__ == "__main__":
    main()
