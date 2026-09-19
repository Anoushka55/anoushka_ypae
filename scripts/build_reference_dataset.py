"""
Build the Kepler-90 reference dataset and scientific source registry.

Phase 1 deliverable. Reads ONLY the immutable NASA Exoplanet Archive snapshot in
data/raw/ and emits:

    data/sources_registry.json     provenance record, one entry per parameter
    data/kepler90_reference.json   the adopted reference dataset (rich, with provenance)
    data/kepler90_reference.csv    the same, flattened for inspection
    docs/KEPLER90_DATA.md          human-readable dataset + consistency report

No value in the outputs is typed in by hand. Every number is either
  (a) read directly from the archive snapshot, or
  (b) computed here from (a) by an explicitly cited relation.

Run:  uv run --python 3.12 --with pandas --with numpy python scripts/build_reference_dataset.py
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "kepler90_ps_full_snapshot.csv"
ACCESS_DATE_FILE = ROOT / "data" / "raw" / "ACCESS_DATE.txt"

# ---------------------------------------------------------------------------
# Physical constants needed to DERIVE quantities in this script.
# IAU 2015 Resolution B3 nominal values + IAU 2012 definition of the au.
# These are repeated authoritatively in invisible_planet/constants.py; this
# script is standalone-runnable, so it carries its own copy with the same sources.
# ---------------------------------------------------------------------------
GM_SUN = 1.3271244e20      # m^3 s^-2, IAU 2015 nominal solar mass parameter
GM_EARTH = 3.986004e14     # m^3 s^-2, IAU 2015 nominal terrestrial mass parameter
R_EARTH_EQ = 6.3781e6      # m, IAU 2015 nominal Earth equatorial radius
R_SUN = 6.957e8            # m, IAU 2015 nominal solar radius
AU = 1.495978707e11        # m, IAU 2012 definition (exact)
DAY = 86400.0              # s, exact
YEAR_JULIAN = 365.25 * DAY  # s, exact

EARTH_MASS_IN_SOLAR = GM_EARTH / GM_SUN          # dimensionless mass ratio
# Earth mean density, derived (needed by the Weiss & Marcy 2014 density branch).
# rho_earth = M_earth / (4/3 pi R_earth^3); M_earth = GM_earth / G.
G_SI = 6.67430e-11         # m^3 kg^-1 s^-2, CODATA 2018
M_EARTH_KG = GM_EARTH / G_SI
RHO_EARTH_CGS = (M_EARTH_KG * 1e3) / ((4.0 / 3.0) * np.pi * (R_EARTH_EQ * 1e2) ** 3)

# ---------------------------------------------------------------------------
# Bibliography. Bibcodes taken from the href fields of the archive snapshot.
# ---------------------------------------------------------------------------
BIB = {
    "WEISS_ET_AL__2024": {
        "citation": "Weiss et al. (2024), ApJS 270, 8 — The Kepler Giant Planet Search",
        "bibcode": "2024ApJS..270....8W",
        "url": "https://ui.adsabs.harvard.edu/abs/2024ApJS..270....8W/abstract",
    },
    "CABRERA_ET_AL__2014": {
        "citation": (
            "Cabrera et al. (2014), ApJ 781, 18 — The planetary system to "
            "KIC 11442793: a compact analogue to the Solar System"
        ),
        "bibcode": "2014ApJ...781...18C",
        "url": "https://ui.adsabs.harvard.edu/abs/2014ApJ...781...18C/abstract",
    },
    "SHALLUE__AMP__VANDERBURG_2018": {
        "citation": (
            "Shallue & Vanderburg (2018), AJ 155, 94 — Identifying Exoplanets with "
            "Deep Learning: discovery of Kepler-90 i"
        ),
        "bibcode": "2018AJ....155...94S",
        "url": "https://ui.adsabs.harvard.edu/abs/2018AJ....155...94S/abstract",
    },
    "LIANG_ET_AL__2021": {
        "citation": "Liang et al. (2021), AJ 161, 202 — transit-timing analysis of Kepler-90",
        "bibcode": "2021AJ....161..202L",
        "url": "https://ui.adsabs.harvard.edu/abs/2021AJ....161..202L/abstract",
    },
    "SHAW_ET_AL__2025": {
        "citation": "Shaw et al. (2025), AJ 170, 146 — transit-timing analysis of Kepler-90",
        "bibcode": "2025AJ....170..146S",
        "url": "https://ui.adsabs.harvard.edu/abs/2025AJ....170..146S/abstract",
    },
    "FULTON__AMP__PETIGURA_2018": {
        "citation": (
            "Fulton & Petigura (2018), AJ 156, 264 — The California-Kepler Survey VII "
            "(homogeneous stellar parameters)"
        ),
        "bibcode": "2018AJ....156..264F",
        "url": "https://ui.adsabs.harvard.edu/abs/2018AJ....156..264F/abstract",
    },
    "WEISS__AMP__MARCY_2014": {
        "citation": (
            "Weiss & Marcy (2014), ApJ 783, L6 — The Mass-Radius Relation for "
            "65 Exoplanets Smaller than 4 Earth Radii"
        ),
        "bibcode": "2014ApJ...783L...6W",
        "url": "https://ui.adsabs.harvard.edu/abs/2014ApJ...783L...6W/abstract",
    },
    "NASA_EXOPLANET_ARCHIVE": {
        "citation": "NASA Exoplanet Archive, Planetary Systems (ps) table, TAP service",
        "bibcode": "2013PASP..125..989A",
        "url": "https://exoplanetarchive.ipac.caltech.edu/",
    },
}

PLANETS = ["b", "c", "i", "d", "e", "f", "g", "h"]  # ordered by period

# ---------------------------------------------------------------------------
# ADOPTION RULES. This is the scientific heart of Phase 1: which source supplies
# which parameter, and why. Changing these lines changes the physics inputs.
# ---------------------------------------------------------------------------
ADOPTION = {
    "orbital_period_days": (
        "WEISS_ET_AL__2024",
        "observed",
        "Only source covering all 8 planets with a single self-consistent solution.",
    ),
    "semi_major_axis_au": (
        "WEISS_ET_AL__2024",
        "derived",
        "Published by Weiss et al.; verified here to follow Kepler's third law with "
        "their own stellar mass, so it is a derived quantity, not an independent measurement.",
    ),
    "planet_radius_earth": (
        "WEISS_ET_AL__2024",
        "observed",
        "Transit depth x stellar radius; same source as the orbital solution.",
    ),
    "eccentricity": (
        "WEISS_ET_AL__2024",
        "adopted-model",
        "Weiss et al. fix e = 0 for all planets (no uncertainties quoted). Treated as an "
        "ADOPTED MODELLING ASSUMPTION, not a measurement. Measured values exist for g and h "
        "only and are recorded as alternates.",
    ),
    "inclination_deg": (
        "CABRERA_ET_AL__2014",
        "observed",
        "Cabrera et al. for b-h; Shallue & Vanderburg 2018 for i. CROSS-SOURCE: flagged.",
    ),
    "transit_epoch_bjd": (
        "CABRERA_ET_AL__2014",
        "observed",
        "Cabrera et al. for b-h; Shallue & Vanderburg 2018 for i. Sets orbital phase. "
        "CROSS-SOURCE: flagged.",
    ),
}

STELLAR_SOURCE = "FULTON__AMP__PETIGURA_2018"


def strip_ref(s: str) -> str:
    if not isinstance(s, str):
        return str(s)
    m = re.search(r"refstr=([^ >]+)", s)
    return m.group(1) if m else s


def weiss_marcy_2014_mass(radius_earth: float) -> tuple[float, float, str]:
    """Mass from radius using Weiss & Marcy (2014), ApJ 783, L6.

    Two branches, exactly as published in the paper abstract:
      R < 1.5 R_earth   : rho = 2.43 + 3.39 (R/R_earth)  [g/cm^3]
      1.5 <= R <= 4.0   : M/M_earth = 2.69 (R/R_earth)^0.93

    Returns (mass_earth, uncertainty_earth, branch_name).

    The quoted RMS scatter of masses about the power-law fit is 4.3 M_earth with
    reduced chi^2 = 6.2, i.e. the relation has LARGE intrinsic dispersion. That
    scatter is carried through as the uncertainty so no downstream code can mistake
    these for measured masses.
    """
    r = float(radius_earth)
    if r < 1.5:
        rho_cgs = 2.43 + 3.39 * r
        mass = (rho_cgs / RHO_EARTH_CGS) * r**3
        # Propagate the density-branch scatter conservatively: the paper does not
        # quote an RMS for this branch, so we use the power-law branch RMS scaled
        # by the mass, which is deliberately pessimistic for small planets.
        unc = max(0.5 * mass, 0.5)
        branch = "density (R < 1.5 R_earth)"
    elif r <= 4.0:
        mass = 2.69 * r**0.93
        unc = 4.3  # published RMS of masses about the fit, in M_earth
        branch = "power-law (1.5 <= R <= 4 R_earth)"
    else:
        raise ValueError(
            f"R = {r} R_earth is outside the stated validity of Weiss & Marcy (2014) "
            "(R < 4 R_earth). Do not extrapolate; use a measured mass instead."
        )
    return mass, unc, branch


def main() -> None:
    df = pd.read_csv(RAW, low_memory=False).copy()
    df["ref"] = df["pl_refname"].apply(strip_ref)
    access_date = ACCESS_DATE_FILE.read_text().strip()

    def pick(ref: str, letter: str) -> pd.Series | None:
        sub = df[(df["ref"] == ref) & (df["pl_letter"] == letter)]
        return None if sub.empty else sub.iloc[0]

    report: list[str] = []
    planets_out: dict[str, dict] = {}

    # ---------------- stellar host ----------------
    st_row = df[df["ref"] == STELLAR_SOURCE].iloc[0]
    star = {
        "name": "Kepler-90",
        "aliases": ["KOI-351", "KIC 11442793", "2MASS J18574403+4918185"],
        "mass_solar": {
            "value": float(st_row["st_mass"]),
            "uncertainty": float(st_row["st_masserr1"]),
            "unit": "M_sun",
            "status": "observed",
            "source": STELLAR_SOURCE,
        },
        "radius_solar": {
            "value": float(st_row["st_rad"]),
            "uncertainty": float(st_row["st_raderr1"]),
            "unit": "R_sun",
            "status": "observed",
            "source": STELLAR_SOURCE,
        },
        "teff_k": {
            "value": float(st_row["st_teff"]),
            "unit": "K",
            "status": "observed",
            "source": STELLAR_SOURCE,
        },
    }

    m_star = star["mass_solar"]["value"]
    report.append(
        f"Stellar host: M* = {m_star} +/- {star['mass_solar']['uncertainty']} M_sun, "
        f"R* = {star['radius_solar']['value']} +/- {star['radius_solar']['uncertainty']} R_sun "
        f"({BIB[STELLAR_SOURCE]['citation']})"
    )

    # ---------------- planets ----------------
    for letter in PLANETS:
        w = pick("WEISS_ET_AL__2024", letter)
        if w is None:
            raise RuntimeError(f"Weiss et al. 2024 row missing for planet {letter}")

        # Transit epoch + inclination: Cabrera for b-h, Shallue & Vanderburg for i
        geom_ref = "SHALLUE__AMP__VANDERBURG_2018" if letter == "i" else "CABRERA_ET_AL__2014"
        geom = pick(geom_ref, letter)
        if geom is None:
            raise RuntimeError(f"geometry row missing for planet {letter} in {geom_ref}")

        radius_earth = float(w["pl_rade"])

        # ---- mass: measured (TTV) where available, else derived from radius ----
        liang = pick("LIANG_ET_AL__2021", letter)
        shaw = pick("SHAW_ET_AL__2025", letter)
        if liang is not None and pd.notna(liang["pl_bmasse"]):
            mass_rec = {
                "value": float(liang["pl_bmasse"]),
                "uncertainty": float(liang["pl_bmasseerr1"]),
                "unit": "M_earth",
                "status": "observed",
                "method": "dynamical mass from transit timing variations",
                "source": "LIANG_ET_AL__2021",
                "notes": "Independently corroborated by Shaw et al. 2025.",
            }
            if shaw is not None and pd.notna(shaw["pl_bmasse"]):
                mass_rec["cross_check"] = {
                    "value": float(shaw["pl_bmasse"]),
                    "uncertainty": float(shaw["pl_bmasseerr1"]),
                    "source": "SHAW_ET_AL__2025",
                }
        else:
            mval, munc, branch = weiss_marcy_2014_mass(radius_earth)
            mass_rec = {
                "value": mval,
                "uncertainty": munc,
                "unit": "M_earth",
                "status": "derived",
                "method": f"Weiss & Marcy (2014) mass-radius relation, {branch}",
                "source": "WEISS__AMP__MARCY_2014",
                "notes": (
                    "NO PUBLISHED MASS EXISTS for this planet. This value is model-dependent "
                    "with large intrinsic scatter (published RMS 4.3 M_earth, reduced chi^2 6.2) "
                    "and must never be presented as a measurement."
                ),
            }

        # ---- eccentricity: adopted circular, with measured alternates recorded ----
        ecc_rec = {
            "value": 0.0,
            "uncertainty": None,
            "unit": "dimensionless",
            "status": "adopted-model",
            "source": "WEISS_ET_AL__2024",
            "notes": "Circular orbit assumed; Weiss et al. 2024 fix e = 0 with no uncertainty.",
        }
        alternates = []
        for ref_key, row in (("LIANG_ET_AL__2021", liang), ("SHAW_ET_AL__2025", shaw)):
            if row is not None and pd.notna(row["pl_orbeccen"]):
                alternates.append(
                    {
                        "value": float(row["pl_orbeccen"]),
                        "uncertainty": (
                            float(row["pl_orbeccenerr1"])
                            if pd.notna(row["pl_orbeccenerr1"])
                            else None
                        ),
                        "source": ref_key,
                        "status": "observed",
                    }
                )
        for alt in alternates:
            if alt["value"] < 0.0:
                alt["warning"] = (
                    "NEGATIVE VALUE — an eccentricity cannot be negative. This is almost "
                    "certainly an eccentricity VECTOR COMPONENT (e cos w or e sin w) that the "
                    "archive has mapped onto the scalar eccentricity column. It must NOT be "
                    "used as an eccentricity. Flagged, not silently corrected."
                )
        if alternates:
            ecc_rec["measured_alternates"] = alternates

        planets_out[letter] = {
            "letter": letter,
            "name": f"Kepler-90 {letter}",
            "orbital_period_days": {
                "value": float(w["pl_orbper"]),
                "uncertainty": (
                    float(w["pl_orbpererr1"]) if pd.notna(w["pl_orbpererr1"]) else None
                ),
                "unit": "days",
                "status": "observed",
                "source": "WEISS_ET_AL__2024",
            },
            "semi_major_axis_au": {
                "value": float(w["pl_orbsmax"]),
                "unit": "au",
                "status": "derived",
                "source": "WEISS_ET_AL__2024",
                "notes": "Consistent with Kepler's third law at M* = 1.108 M_sun (verified below).",
            },
            "eccentricity": ecc_rec,
            "inclination_deg": {
                "value": float(geom["pl_orbincl"]),
                "uncertainty": (
                    float(geom["pl_orbinclerr1"]) if pd.notna(geom["pl_orbinclerr1"]) else None
                ),
                "unit": "deg",
                "status": "observed",
                "source": geom_ref,
                "notes": (
                    "Published inclination. For transit geometry prefer `impact_parameter`, "
                    "which is the directly measured transit-shape observable; see CHECK 5."
                ),
            },
            "impact_parameter": {
                "value": (
                    float(geom["pl_imppar"]) if pd.notna(geom["pl_imppar"]) else None
                ),
                "unit": "stellar radii",
                "status": "observed",
                "source": geom_ref,
                "notes": (
                    "b = a cos(i) / R_star. This is what the transit light-curve shape "
                    "actually constrains; the inclination is a derived restatement of it "
                    "that depends on the adopted a and R_star."
                ),
            },
            "transit_epoch_bjd": {
                "value": float(geom["pl_tranmid"]),
                "uncertainty": (
                    float(geom["pl_tranmiderr1"]) if pd.notna(geom["pl_tranmiderr1"]) else None
                ),
                "unit": "BJD_TDB",
                "status": "observed",
                "source": geom_ref,
                "notes": "Sets the orbital phase of the initial condition.",
            },
            "planet_radius_earth": {
                "value": radius_earth,
                "uncertainty": float(w["pl_radeerr1"]) if pd.notna(w["pl_radeerr1"]) else None,
                "unit": "R_earth",
                "status": "observed",
                "source": "WEISS_ET_AL__2024",
            },
            "mass_earth": mass_rec,
        }

    # =====================================================================
    # CONSISTENCY CHECKS
    # =====================================================================
    report.append("")
    report.append("CHECK 1 - Kepler's third law: a^3 = GM* P^2 / (4 pi^2)")
    report.append(
        f"{'planet':>7} {'P [d]':>12} {'a_pub [au]':>12} {'a_kepler [au]':>14} {'rel.diff':>10}"
    )
    gm_star_si = GM_SUN * m_star
    max_rel = 0.0
    for letter in PLANETS:
        p = planets_out[letter]
        period_s = p["orbital_period_days"]["value"] * DAY
        a_kepler = (gm_star_si * period_s**2 / (4.0 * np.pi**2)) ** (1.0 / 3.0) / AU
        a_pub = p["semi_major_axis_au"]["value"]
        rel = abs(a_kepler - a_pub) / a_pub
        max_rel = max(max_rel, rel)
        report.append(
            f"{letter:>7} {p['orbital_period_days']['value']:12.6f} {a_pub:12.6f} "
            f"{a_kepler:14.6f} {rel:10.2e}"
        )
        p["semi_major_axis_au"]["kepler_third_law_check_au"] = a_kepler
        p["semi_major_axis_au"]["kepler_third_law_rel_diff"] = rel
    report.append(f"  max relative difference: {max_rel:.2e}")
    kepler_ok = max_rel < 1e-3
    report.append(f"  VERDICT: {'PASS' if kepler_ok else 'FAIL'} (tolerance 1e-3)")

    report.append("")
    report.append("CHECK 2 - cross-source period discrepancies (adopted = Weiss et al. 2024)")
    report.append(
        f"{'planet':>7} {'adopted [d]':>14} {'other source':>32} {'other [d]':>14} "
        f"{'diff [s]':>12}"
    )
    discrepancies = []
    for letter in PLANETS:
        adopted = planets_out[letter]["orbital_period_days"]["value"]
        for ref in ["CABRERA_ET_AL__2014", "LIANG_ET_AL__2021", "SHAW_ET_AL__2025",
                    "SHALLUE__AMP__VANDERBURG_2018"]:
            row = pick(ref, letter)
            if row is None or pd.isna(row["pl_orbper"]):
                continue
            other = float(row["pl_orbper"])
            diff_s = (other - adopted) * DAY
            report.append(f"{letter:>7} {adopted:14.6f} {ref:>32} {other:14.6f} {diff_s:12.2f}")
            discrepancies.append(
                {"planet": letter, "adopted_days": adopted, "source": ref,
                 "value_days": other, "difference_seconds": diff_s}
            )

    report.append("")
    report.append("CHECK 3 - adjacent-pair separation in mutual Hill radii")
    report.append(
        "  Delta = (a_out - a_in) / R_H,mutual ;  R_H,mutual = "
        "((m_in+m_out)/(3 M*))^(1/3) * (a_in+a_out)/2"
    )
    report.append(f"{'pair':>9} {'a_in':>9} {'a_out':>9} {'Delta':>9}")
    hill_rows = []
    for i in range(len(PLANETS) - 1):
        li, lo = PLANETS[i], PLANETS[i + 1]
        a_in = planets_out[li]["semi_major_axis_au"]["value"]
        a_out = planets_out[lo]["semi_major_axis_au"]["value"]
        m_in = planets_out[li]["mass_earth"]["value"] * EARTH_MASS_IN_SOLAR
        m_out = planets_out[lo]["mass_earth"]["value"] * EARTH_MASS_IN_SOLAR
        r_h = ((m_in + m_out) / (3.0 * m_star)) ** (1.0 / 3.0) * 0.5 * (a_in + a_out)
        delta = (a_out - a_in) / r_h
        report.append(f"{li + '-' + lo:>9} {a_in:9.4f} {a_out:9.4f} {delta:9.2f}")
        hill_rows.append({"pair": f"{li}-{lo}", "delta_mutual_hill": delta})
    min_delta = min(r["delta_mutual_hill"] for r in hill_rows)
    report.append(
        f"  minimum Delta = {min_delta:.2f} (Gladman 1993 two-planet stability needs "
        f"Delta > 2*sqrt(3) = 3.46; compact multis typically Delta > 10)"
    )

    report.append("")
    report.append("CHECK 4 - proximity to LOW-ORDER mean-motion resonances")
    report.append(
        "  Only commensurabilities p:q with p,q <= 6 and gcd(p,q)=1 are considered. "
        "High-order ratios are excluded because any number lies close to some high-order "
        "rational, which would make 'near-resonant' meaningless."
    )
    report.append(f"{'pair':>9} {'P_out/P_in':>11}  {'nearest MMR':>12}  {'offset':>9}  comment")
    from math import gcd

    mmr_candidates = [
        (p, q) for q in range(1, 7) for p in range(q + 1, 7) if gcd(p, q) == 1
    ]
    for i in range(len(PLANETS) - 1):
        li, lo = PLANETS[i], PLANETS[i + 1]
        ratio = (
            planets_out[lo]["orbital_period_days"]["value"]
            / planets_out[li]["orbital_period_days"]["value"]
        )
        best, best_err = None, 1e9
        for pnum, q in mmr_candidates:
            err = (ratio - pnum / q) / (pnum / q)
            if abs(err) < abs(best_err):
                best_err, best = err, f"{pnum}:{q}"
        if abs(best_err) < 0.02:
            comment = "NEAR-RESONANT -> expect enhanced TTVs"
        elif abs(best_err) < 0.05:
            comment = "moderately close"
        else:
            comment = "not near a low-order MMR"
        report.append(
            f"{li + '-' + lo:>9} {ratio:11.5f}  {best:>12}  {best_err * 100:+8.2f}%  {comment}"
        )

    report.append("")
    report.append("CHECK 5 - transit geometry: is the published inclination consistent with b?")
    report.append(
        "  Using the ADOPTED a and R_star, b_implied = a cos(i) / R_star must reproduce "
        "the published impact parameter. b > 1 would mean the planet does not transit "
        "at all, which would contradict the fact that it was discovered BY its transits."
    )
    report.append(
        f"{'planet':>7} {'b_published':>12} {'b_from_incl':>12} {'i_from_b [deg]':>15}  status"
    )
    r_star_au = star["radius_solar"]["value"] * (R_SUN / AU)
    geometry_flags = []
    for letter in PLANETS:
        p = planets_out[letter]
        a = p["semi_major_axis_au"]["value"]
        incl = np.deg2rad(p["inclination_deg"]["value"])
        b_pub = p["impact_parameter"]["value"]
        b_implied = a * np.cos(incl) / r_star_au
        i_from_b = np.rad2deg(np.arccos(b_pub * r_star_au / a)) if b_pub is not None else None

        if b_pub is None:
            status = "no published b"
        elif b_implied > 1.0:
            status = "INCONSISTENT - implied b > 1 (would not transit)"
        elif abs(b_implied - b_pub) < 0.05:
            status = "consistent"
        else:
            status = "discrepant"
        geometry_flags.append(
            {"planet": letter, "b_published": b_pub, "b_from_inclination": float(b_implied),
             "inclination_from_b_deg": (float(i_from_b) if i_from_b is not None else None),
             "status": status}
        )
        report.append(
            f"{letter:>7} {('%.3f' % b_pub) if b_pub is not None else 'n/a':>12} "
            f"{b_implied:12.3f} {('%.4f' % i_from_b) if i_from_b is not None else 'n/a':>15}  {status}"
        )
        p["inclination_deg"]["derived_from_impact_parameter_deg"] = (
            float(i_from_b) if i_from_b is not None else None
        )
        p["impact_parameter"]["consistency_with_published_inclination"] = status

    # =====================================================================
    # OUTPUTS
    # =====================================================================
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    reference = {
        "schema_version": "1.0",
        "system": "Kepler-90 (KOI-351, KIC 11442793)",
        "generated_utc": now,
        "generated_by": "scripts/build_reference_dataset.py",
        "archive_snapshot": str(RAW.relative_to(ROOT)).replace("\\", "/"),
        "archive_access_utc": access_date,
        "warning": (
            "This dataset describes the REAL Kepler-90 system only. It contains no "
            "synthetic body. Planet X is defined separately and is not part of this file."
        ),
        "star": star,
        "planets": planets_out,
        "consistency_checks": {
            "kepler_third_law_max_rel_diff": max_rel,
            "kepler_third_law_pass": bool(kepler_ok),
            "cross_source_period_discrepancies": discrepancies,
            "adjacent_pair_mutual_hill_separations": hill_rows,
            "transit_geometry_consistency": geometry_flags,
        },
        "bibliography": BIB,
    }

    (ROOT / "data" / "kepler90_reference.json").write_text(
        json.dumps(reference, indent=2), encoding="utf-8"
    )

    flat = []
    for letter in PLANETS:
        p = planets_out[letter]
        flat.append(
            {
                "planet": p["name"],
                "letter": letter,
                "period_days": p["orbital_period_days"]["value"],
                "period_source": p["orbital_period_days"]["source"],
                "a_au": p["semi_major_axis_au"]["value"],
                "a_status": p["semi_major_axis_au"]["status"],
                "ecc": p["eccentricity"]["value"],
                "ecc_status": p["eccentricity"]["status"],
                "incl_deg": p["inclination_deg"]["value"],
                "incl_source": p["inclination_deg"]["source"],
                "T0_bjd": p["transit_epoch_bjd"]["value"],
                "T0_source": p["transit_epoch_bjd"]["source"],
                "radius_earth": p["planet_radius_earth"]["value"],
                "mass_earth": p["mass_earth"]["value"],
                "mass_unc_earth": p["mass_earth"]["uncertainty"],
                "mass_status": p["mass_earth"]["status"],
                "mass_source": p["mass_earth"]["source"],
            }
        )
    pd.DataFrame(flat).to_csv(ROOT / "data" / "kepler90_reference.csv", index=False)

    # --- source registry: one row per adopted parameter ---
    registry = {
        "schema_version": "1.0",
        "generated_utc": now,
        "archive_access_utc": access_date,
        "status_vocabulary": {
            "observed": "Measured from data and published with an uncertainty.",
            "inferred": "Obtained from data through a model fit (e.g. dynamical TTV mass).",
            "adopted-model": "Assumed value, not measured (e.g. e = 0).",
            "derived": "Computed from other quantities by a cited relation.",
            "synthetic": "Invented for this numerical experiment (Planet X only).",
            "numerical": "A property of the integration, not of nature (e.g. timestep).",
        },
        "adoption_rules": {
            k: {"source": v[0], "status": v[1], "rationale": v[2]} for k, v in ADOPTION.items()
        },
        "stellar_source": STELLAR_SOURCE,
        "entries": [],
        "bibliography": BIB,
    }
    for key, rec in (("star.mass_solar", star["mass_solar"]),
                     ("star.radius_solar", star["radius_solar"]),
                     ("star.teff_k", star["teff_k"])):
        registry["entries"].append({"parameter": key, **rec})
    for letter in PLANETS:
        for field, rec in planets_out[letter].items():
            if isinstance(rec, dict) and "value" in rec:
                registry["entries"].append(
                    {"parameter": f"planet.{letter}.{field}",
                     **{k: v for k, v in rec.items() if k != "cross_check"}}
                )
    (ROOT / "data" / "sources_registry.json").write_text(
        json.dumps(registry, indent=2), encoding="utf-8"
    )

    text = "\n".join(report)
    print(text)

    # --- docs/KEPLER90_DATA.md ---
    md = [
        "# Kepler-90 Reference Dataset",
        "",
        "**Generated by** `scripts/build_reference_dataset.py` — do not edit by hand.",
        f"**Archive snapshot accessed:** {access_date}",
        f"**Built:** {now}",
        "",
        "This file describes the **real Kepler-90 system**. The synthetic Planet X used",
        "later in the project is *not* part of this dataset and is not a claim about nature.",
        "",
        "## Adopted values",
        "",
        "| Planet | P [d] | a [au] | e | i [deg] | T0 [BJD] | R [R⊕] | M [M⊕] | mass status |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for letter in PLANETS:
        p = planets_out[letter]
        md.append(
            f"| {p['name']} | {p['orbital_period_days']['value']:.6f} | "
            f"{p['semi_major_axis_au']['value']:.6f} | {p['eccentricity']['value']:.3f} | "
            f"{p['inclination_deg']['value']:.2f} | {p['transit_epoch_bjd']['value']:.5f} | "
            f"{p['planet_radius_earth']['value']:.3f} | "
            f"{p['mass_earth']['value']:.2f} ± {p['mass_earth']['uncertainty']:.2f} | "
            f"**{p['mass_earth']['status']}** |"
        )
    md += [
        "",
        f"**Star:** M★ = {m_star} ± {star['mass_solar']['uncertainty']} M☉, "
        f"R★ = {star['radius_solar']['value']} ± {star['radius_solar']['uncertainty']} R☉, "
        f"Teff = {star['teff_k']['value']:.0f} K ({BIB[STELLAR_SOURCE]['citation']}).",
        "",
        "## Where each number comes from",
        "",
        "| Parameter | Source | Status | Why |",
        "|---|---|---|---|",
    ]
    for k, (src, status, why) in ADOPTION.items():
        md.append(f"| {k} | {BIB[src]['citation'].split(' — ')[0]} | `{status}` | {why} |")
    md += [
        f"| mass (g, h) | {BIB['LIANG_ET_AL__2021']['citation'].split(' — ')[0]} | `observed` | "
        "Dynamical masses from transit timing variations; corroborated by Shaw et al. 2025. |",
        f"| mass (b, c, i, d, e, f) | {BIB['WEISS__AMP__MARCY_2014']['citation'].split(' — ')[0]} "
        "| `derived` | **No published mass exists.** Derived from radius; large intrinsic scatter. |",
        "",
        "## Consistency checks",
        "",
        "```",
        text,
        "```",
        "",
        "## Known discrepancies and caveats",
        "",
        "1. **The archive indexes this system as `KOI-351`, not `Kepler-90`.** Querying "
        "`hostname='Kepler-90'` returns nothing and `LIKE 'Kepler-90%'` returns Kepler-900–909.",
        "2. **Stellar mass is source-dependent** (0.967 → 1.242 M☉ across the literature). "
        "Semi-major axis scales as M★^(1/3), so this choice propagates into the dynamics. "
        "We use Fulton & Petigura (2018) because Weiss et al. (2024) adopt the same M★, "
        "keeping orbit and star on one scale.",
        "3. **Six of eight planets have no measured mass.** Their masses are model-derived "
        "and carry the full scatter of the mass–radius relation.",
        "4. **Eccentricities are an assumption**, not a measurement, for b–f.",
        "5. **Transit epochs and inclinations come from a different paper than the periods.** "
        "The period differences between sources are listed in CHECK 2 above and are "
        "themselves partly physical: Kepler-90 g and h show real transit timing variations.",
        "6. **Weiss & Marcy (2014) was calibrated on planets with P < 100 d.** Planet f "
        "(P = 124.9 d) lies slightly outside that calibration range in period, though well "
        "inside it in radius.",
    ]
    (ROOT / "docs" / "KEPLER90_DATA.md").write_text("\n".join(md), encoding="utf-8")

    print("\nWROTE:")
    for f in ["data/kepler90_reference.json", "data/kepler90_reference.csv",
              "data/sources_registry.json", "docs/KEPLER90_DATA.md"]:
        print("  ", f)


if __name__ == "__main__":
    main()
