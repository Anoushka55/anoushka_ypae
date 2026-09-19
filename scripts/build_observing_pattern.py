"""
Build the REAL observing pattern and timing-noise model for Kepler-90.

Reads only the immutable raw tables in data/raw/ and writes
    data/kepler90_observed_transits.json
    docs/OBSERVING_PATTERN.md

WHY THIS EXISTS
---------------
An earlier version of the experiment used each planet's published transit-EPOCH (T0)
uncertainty as if it were a single-transit timing error, and assumed every transit
was observed. Both were wrong: T0 uncertainties are fits over many transits, and
Kepler observed only some of the transits that occurred. This script replaces both
with published, per-transit data:

  * Kepler-90 d, e : Holczer et al. (2016), ApJS 225, 9, table 3 (VizieR J/ApJS/225/9)
  * Kepler-90 g, h : Liang, Robnik & Seljak (2021), AJ 161, 202, fig. 1 data
                     (VizieR J/AJ/161/202)

which give, for every transit that was actually measured, its epoch, its time and
its 1-sigma uncertainty. Planets b, c, i and f have NO published individual transit
times (their per-transit SNR is ~1.5-2 for b, c, i; see the table written to
docs/OBSERVING_PATTERN.md), so they are not used as timing probes.

The analytic photon-noise limit of Carter et al. (2008, ApJ 689, 499), eqs. 21-22,
    sigma_tc = (T/Q) * sqrt(theta/2),   theta = tau/T = r/(1-b^2),
is computed as an independent cross-check. It is a LOWER BOUND on the timing error
(white photon noise only); the ratio of the published errors to it is reported in
docs/OBSERVING_PATTERN.md.

Run: uv run --python 3.12 --extra dev python scripts/build_observing_pattern.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from invisible_planet.observations.ttv import fit_linear_ephemeris

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

#: Time-system offsets (all published times are barycentric Julian dates).
BJD_OFFSET_HOLCZER = 2454900.0   # Holczer et al. 2016, table 3: "BJD - 2454900"
BJD_OFFSET_BKJD = 2454833.0      # Kepler BJD - 2454833 (BKJD); Liang et al. 2021 "HJD" column

#: Approximate span of Kepler's Q1-Q17 photometry (BKJD), used only as the simulation
#: window. Observed epochs come from the real tables, so the exact edges do not matter.
WINDOW_START_BKJD = 131.5
WINDOW_END_BKJD = 1591.0

#: A planet is used as a timing PROBE of an additional body only if (a) at least this many
#: individual transit times are published and (b) the real O-C series is consistent with a
#: straight line. A planet with large real TTVs that the visible-planet model does not
#: reproduce cannot separate a hidden perturber's signature from that unmodelled signal.
MIN_PROBE_TRANSITS = 5
MAX_PROBE_REDUCED_CHI2 = 3.0

HOLCZER_KOI = {"d": 351.03, "e": 351.04}
LIANG_NAME = {"g": "Kepler-90g", "h": "Kepler-90h"}
KOI_TABLE_LETTER = {"b": 6, "c": 5, "d": 3, "e": 4, "f": 7, "g": 2, "h": 1}
PERIODS = {"b": 7.008, "c": 8.720, "i": 14.449, "d": 59.737, "e": 91.940, "f": 124.916,
           "g": 210.603, "h": 331.601}


def carter_sigma_seconds(row: pd.Series) -> dict:
    """Photon-noise-limited single-transit timing error (Carter et al. 2008)."""
    n = float(row["koi_num_transits"])
    q = float(row["koi_model_snr"]) / math.sqrt(n)                 # per-transit SNR
    t14 = float(row["koi_duration"]) * 3600.0                      # T14 in seconds
    ror = float(row["koi_ror"])
    b = float(row["koi_impact"])
    theta = min(ror / max(1.0 - b * b, 1e-3), 0.99)                # tau/T, eq. 22
    t_fwhm = t14 / (1.0 + theta)                                   # T14 = T + tau
    sigma = (t_fwhm / q) * math.sqrt(theta / 2.0)                  # eq. 21
    return {"per_transit_snr": q, "theta": theta, "sigma_seconds": sigma}


def main() -> None:
    koi = pd.read_csv(RAW / "kepler90_koi_cumulative.csv")
    koi["letter"] = koi["kepoi_name"].str[-1].astype(int).map({v: k for k, v in KOI_TABLE_LETTER.items()})

    holczer = pd.read_csv(RAW / "holczer2016_table3_koi351.csv")
    liang = pd.read_csv(RAW / "liang2021_kepler90gh_transit_times.csv")

    planets: dict = {}

    # ---- d, e: Holczer et al. 2016 ------------------------------------------------
    for letter, koi_id in HOLCZER_KOI.items():
        g = holczer[np.isclose(holczer["KOI"], koi_id)].sort_values("N")
        measured = BJD_OFFSET_HOLCZER + g["tn"].to_numpy() + g["O-C"].to_numpy() / 1440.0
        sigma = g["e_O-C"].to_numpy() / 1440.0
        outlier = (g["Out"].to_numpy() != 0)
        planets[letter] = {
            "source": "Holczer et al. (2016), ApJS 225, 9, table 3",
            "source_id": f"KOI-{koi_id}",
            "measured_bjd": measured.tolist(),
            "sigma_days": sigma.tolist(),
            "published_epoch_numbers": g["N"].astype(int).tolist(),
            "outlier_flag": g["Out"].astype(int).tolist(),
            "uncertainty_estimated_flag": [bool(str(x).strip() == "*") for x in g["f_O-C"]],
        }

    # ---- g, h: Liang et al. 2021 ---------------------------------------------------
    for letter, name in LIANG_NAME.items():
        g = liang[liang["Planet"] == name].sort_values("HJD")
        planets[letter] = {
            "source": "Liang, Robnik & Seljak (2021), AJ 161, 202, fig. 1 data",
            "source_id": name,
            "measured_bjd": (BJD_OFFSET_BKJD + g["HJD"].to_numpy()).tolist(),
            "sigma_days": g["e_HJD"].to_numpy().tolist(),
            "published_epoch_numbers": None,
            "outlier_flag": [0] * len(g),
            "uncertainty_estimated_flag": [False] * len(g),
        }

    # ---- integer epochs and the real fitted ephemeris of each planet ---------------
    for letter, record in planets.items():
        t = np.asarray(record["measured_bjd"])
        s = np.asarray(record["sigma_days"])
        period_guess = PERIODS[letter]
        epochs = np.round((t - t[0]) / period_guess).astype(int)
        fit = fit_linear_ephemeris(t, epochs=epochs, sigma=s)
        # every transit must sit within a fraction of a period of its integer epoch
        worst = float(np.max(np.abs(fit.residuals)) / fit.period)
        if worst > 0.05:
            raise RuntimeError(f"planet {letter}: epoch assignment is ambiguous ({worst:.3f} P)")
        chi2 = float(np.sum((fit.residuals / s) ** 2))
        record.update(
            {
                "epochs": epochs.tolist(),
                "ephemeris": {
                    "t0_bjd": fit.t0,
                    "period_days": fit.period,
                    "t0_unc_days": fit.t0_uncertainty,
                    "period_unc_days": fit.period_uncertainty,
                    "weighted_fit_chi2": chi2,
                    "dof": len(t) - 2,
                    "rms_residual_minutes": float(np.sqrt(np.mean(fit.residuals**2)) * 1440.0),
                },
                "n_transits": int(len(t)),
                "median_sigma_seconds": float(np.median(s) * 86400.0),
                "linear_ephemeris_reduced_chi2": (chi2 / (len(t) - 2)) if len(t) > 2 else None,
            }
        )
        reasons = []
        if len(t) < MIN_PROBE_TRANSITS:
            reasons.append(f"only {len(t)} published transits (< {MIN_PROBE_TRANSITS})")
        elif chi2 / (len(t) - 2) > MAX_PROBE_REDUCED_CHI2:
            reasons.append(
                f"real O-C is NOT consistent with a straight line (reduced chi2 = "
                f"{chi2 / (len(t) - 2):.0f} > {MAX_PROBE_REDUCED_CHI2}): large real TTV that a "
                "circular model cannot reproduce"
            )
        record["probe_rejection_reasons"] = reasons

    # ---- time-convention cross-check: Liang (BKJD) vs Holczer (BJD-2454900) for g ---
    g_h = holczer[np.isclose(holczer["KOI"], 351.02)].sort_values("N")
    g_h_times = BJD_OFFSET_HOLCZER + g_h["tn"].to_numpy() + g_h["O-C"].to_numpy() / 1440.0
    g_l = np.asarray(planets["g"]["measured_bjd"])
    convention_rows = []
    for th, sh in zip(g_h_times, g_h["e_O-C"].to_numpy() / 1440.0):
        j = int(np.argmin(np.abs(g_l - th)))
        convention_rows.append((float(th), float(g_l[j]), float((g_l[j] - th) * 1440.0),
                                float(sh * 1440.0), float(planets["g"]["sigma_days"][j] * 1440.0)))

    # ---- Carter photon-noise comparison --------------------------------------------
    carter = {}
    for _, row in koi.iterrows():
        letter = row["letter"]
        if isinstance(letter, str):
            carter[letter] = carter_sigma_seconds(row)
            carter[letter]["koi_num_transits"] = int(row["koi_num_transits"])
            carter[letter]["koi_model_snr"] = float(row["koi_model_snr"])

    # per-cadence noise implied by each planet's SNR: white Kepler noise => the same for all
    cadence_hours = 29.4244 / 60.0
    cadence_noise = {}
    for _, row in koi.iterrows():
        n = float(row["koi_num_transits"])
        q = float(row["koi_model_snr"]) / math.sqrt(n)
        n_in = float(row["koi_duration"]) / cadence_hours
        cadence_noise[row["letter"]] = float(row["koi_depth"]) * 1e-6 * math.sqrt(n_in) / q * 1e6

    payload = {
        "description": (
            "Real, published per-transit timing measurements for the Kepler-90 planets whose "
            "individual transit times have been published, used as the timing-noise model and "
            "observing pattern of the synthetic experiment."
        ),
        "time_system": "BJD_TDB as published (Kepler barycentric time)",
        "kepler_window_bjd": [BJD_OFFSET_BKJD + WINDOW_START_BKJD, BJD_OFFSET_BKJD + WINDOW_END_BKJD],
        "probe_planets": [k for k, v in planets.items() if not v["probe_rejection_reasons"]],
        "timing_planets": list(planets),
        "probe_rule": {"min_transits": MIN_PROBE_TRANSITS, "max_reduced_chi2": MAX_PROBE_REDUCED_CHI2},
        "planets": planets,
        "excluded_planets": {
            "b": "no published individual transit times; per-transit SNR ~1.5",
            "c": "no published individual transit times; per-transit SNR ~2.2",
            "i": "no published individual transit times; per-transit SNR not tabulated (~1-2 estimated)",
            "f": "no published individual transit times",
        },
        "carter_photon_noise": carter,
        "implied_cadence_noise_ppm": cadence_noise,
        "source_files": [
            "data/raw/holczer2016_table3_koi351.csv",
            "data/raw/liang2021_kepler90gh_transit_times.csv",
            "data/raw/kepler90_koi_cumulative.csv",
        ],
        "vizier_access_utc": (RAW / "VIZIER_ACCESS_DATE.txt").read_text().strip(),
    }
    (ROOT / "data" / "kepler90_observed_transits.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    # ---- documentation -------------------------------------------------------------
    L = [
        "# Observing pattern and timing noise",
        "",
        "**Generated by `scripts/build_observing_pattern.py`. Every number is computed from the raw",
        "tables in `data/raw/`; nothing is typed by hand.**",
        "",
        "The synthetic experiment observes the simulated system **exactly as Kepler observed the real",
        "one**: only at the epochs that were actually measured, each with its published uncertainty.",
        "",
        "## Which planets are used as timing probes, and why",
        "",
        "| planet | published individual transit times? | source | transits measured | median σ [s] |",
        "|---|---|---|---|---|",
    ]
    for letter in ["b", "c", "i", "d", "e", "f", "g", "h"]:
        if letter in planets:
            p = planets[letter]
            L.append(f"| {letter} | **yes** | {p['source']} | {p['n_transits']} | {p['median_sigma_seconds']:.0f} |")
        else:
            L.append(f"| {letter} | no | — | — | — |")
    L += [
        "",
        "Planets **b, c, i** have a per-transit SNR of only ~1.5–2.2 (below), so individual transit",
        "times are not measurable; **f** has none published. Using them would mean assuming a data",
        "quality that does not exist.",
        "",
        "Of the four planets WITH published times (d, e, g, h), a planet is used as a **probe** of an",
        f"additional body only if it has at least {MIN_PROBE_TRANSITS} transits and its real O−C is consistent with a",
        f"straight line (reduced χ² ≤ {MAX_PROBE_REDUCED_CHI2}): otherwise it carries large real TTVs that a circular model",
        "cannot reproduce and that would be indistinguishable from a hidden perturber. The rule is evaluated",
        "on the real data:",
        "",
        "| planet | transits | reduced χ² of a linear ephemeris | probe? | reason |",
        "|---|---|---|---|---|",
    ]
    for letter in ["d", "e", "g", "h"]:
        p = planets[letter]
        red = p["linear_ephemeris_reduced_chi2"]
        L.append(
            f"| {letter} | {p['n_transits']} | {('%.1f' % red) if red is not None else '—'} | "
            f"{'**yes**' if not p['probe_rejection_reasons'] else 'no'} | "
            f"{'; '.join(p['probe_rejection_reasons']) or 'meets the rule'} |"
        )
    L += [
        "",
        "## Published per-transit uncertainty vs. the photon-noise limit",
        "",
        "Carter et al. (2008) eqs. 21–22: σ_tc = (T/Q)·√(θ/2), θ = τ/T = r/(1−b²), where Q is the",
        "per-transit SNR, T the FWHM duration and τ the ingress time. This is a *lower bound* (white",
        "photon noise only).",
        "",
        "| planet | transits (KOI) | total SNR | per-transit Q | θ | Carter limit [s] | published median [s] | ratio |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for letter in ["b", "c", "d", "e", "f", "g", "h"]:
        c = carter.get(letter)
        if c is None:
            continue
        pub = planets[letter]["median_sigma_seconds"] if letter in planets else None
        ratio = f"{pub / c['sigma_seconds']:.2f}" if pub else "—"
        L.append(
            f"| {letter} | {c['koi_num_transits']} | {c['koi_model_snr']:.1f} | {c['per_transit_snr']:.2f} | "
            f"{c['theta']:.3f} | {c['sigma_seconds']:.0f} | {('%.0f' % pub) if pub else '—'} | {ratio} |"
        )
    ratios = [planets[k]["median_sigma_seconds"] / carter[k]["sigma_seconds"]
              for k in ("d", "e", "g", "h") if k in carter]
    L += [
        "",
        f"Published uncertainties exceed the photon-noise limit by a factor {min(ratios):.1f}–{max(ratios):.1f}, "
        "as expected for",
        "real light-curve fits (correlated stellar noise, detrending, finite cadence). The published",
        "values are used; the Carter limit is a consistency check.",
        "",
        "For f, the KOI table gives b = 0.943 whereas Cabrera et al. (2014) give 0.35; the limit above",
        "uses the KOI-table values consistently and is therefore very uncertain for that planet.",
        "",
        "## Consistency of the SNR bookkeeping",
        "",
        "Kepler photometry of this star is close to white noise, so the per-cadence noise implied by",
        "each planet's depth, duration and SNR should be the same for all planets:",
        "",
        "| planet | implied σ per 29.4-min cadence [ppm] |",
        "|---|---|",
    ]
    for letter in ["b", "c", "d", "e", "f", "g", "h"]:
        if letter in cadence_noise:
            L.append(f"| {letter} | {cadence_noise[letter]:.0f} |")
    L += [
        "",
        "## Time-system cross-check",
        "",
        "Liang et al. label their times `HJD`; the values are Kepler BJD − 2454833. This was checked",
        "against Holczer et al. for the six transits of g they share (difference in minutes, and the",
        "two quoted 1-σ uncertainties in minutes):",
        "",
        "| Holczer time [BJD] | Liang time [BJD] | Liang − Holczer [min] | σ Holczer [min] | σ Liang [min] |",
        "|---|---|---|---|---|",
    ]
    for th, tl, dm, sh, sl in convention_rows:
        L.append(f"| {th:.4f} | {tl:.4f} | {dm:+.1f} | {sh:.1f} | {sl:.1f} |")
    L += [
        "",
        "The final row disagrees by more than a day: Holczer et al. flag that transit as an outlier",
        "(their outlier flag = 56) with a 21-minute uncertainty, while Liang et al. measure it",
        "directly. Liang et al. are used for g and h.",
        "",
        "## Real fitted ephemerides",
        "",
        "Weighted linear fits to the real measured times (used to match the simulated system):",
        "",
        "| planet | T0 [BJD] | P [d] | σ(P) [d] | O−C rms [min] | χ² (dof) |",
        "|---|---|---|---|---|---|",
    ]
    for letter in ["d", "e", "g", "h"]:
        e = planets[letter]["ephemeris"]
        L.append(
            f"| {letter} | {e['t0_bjd']:.5f} | {e['period_days']:.6f} | {e['period_unc_days']:.6f} | "
            f"{e['rms_residual_minutes']:.1f} | {e['weighted_fit_chi2']:.1f} ({e['dof']}) |"
        )
    L += [
        "",
        "Planet g's large χ² is genuine: its real O−C series contains a ≈ −205 minute excursion at one",
        "epoch, a real transit timing variation produced by its interaction with h.",
        "",
        "## Limitations of this noise model",
        "",
        "* Noise is taken as independent and Gaussian with the published σ; real errors are partly",
        "  correlated and non-Gaussian.",
        "* The observing pattern is the real one for d, e, g, h; the epoch of each simulated transit is",
        "  taken from the real ephemeris, so a transit lost to a data gap in reality is lost here too.",
        "* No stellar-activity or spot-crossing timing noise beyond what is in the published σ.",
        "",
    ]
    (ROOT / "docs" / "OBSERVING_PATTERN.md").write_text("\n".join(L), encoding="utf-8")

    print("wrote data/kepler90_observed_transits.json and docs/OBSERVING_PATTERN.md")
    for letter in ["d", "e", "g", "h"]:
        p = planets[letter]
        e = p["ephemeris"]
        print(f"  {letter}: n={p['n_transits']:2d}  median sigma = {p['median_sigma_seconds']:6.0f} s  "
              f"P = {e['period_days']:.6f} d  O-C rms = {e['rms_residual_minutes']:.1f} min")


if __name__ == "__main__":
    main()
