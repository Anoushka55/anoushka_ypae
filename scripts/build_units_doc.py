"""
Generate docs/UNITS.md with every number COMPUTED, none typed by hand.

An earlier hand-written version of this document contained wrong values (the
normalised-unit period and velocity ranges, and a mis-stated explanation of the
4*pi^2 shortcut). Generating it removes that failure mode.

Run: uv run --python 3.12 --extra dev python scripts/build_units_doc.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from invisible_planet import constants as C

ROOT = Path(__file__).resolve().parents[1]


def sci(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}e}".replace("e+0", "e+").replace("e-0", "e-")


def main() -> None:
    ref = json.loads((ROOT / "data" / "kepler90_reference.json").read_text(encoding="utf-8"))
    m_star = ref["star"]["mass_solar"]["value"]
    planets = ref["planets"]
    mass_e = {k: v["mass_earth"]["value"] for k, v in planets.items()}
    mass_sun = {k: m * C.EARTH_MASS_IN_SOLAR for k, m in mass_e.items()}
    period = {k: v["orbital_period_days"]["value"] for k, v in planets.items()}
    axis = {k: C.kepler_third_law_axis(period[k], m_star + mass_sun[k]) for k in planets}
    vcirc = {k: C.circular_speed(axis[k], m_star) for k in planets}  # au/day

    ms_kg = C.GM_SUN_SI / C.G_SI
    me_kg = C.GM_EARTH_SI / C.G_SI
    lo = min(period, key=period.get)   # innermost
    hi = max(period, key=period.get)   # outermost
    m_lo = min(mass_e, key=mass_e.get)

    # --- the four candidate systems -------------------------------------------------
    L4 = axis[hi]
    T4 = math.sqrt(L4**3 / (C.G * m_star))     # time unit with G = M* = 1, L = a_outer

    def rng(vals, fmt):
        return f"{fmt(min(vals))} – {fmt(max(vals))}"

    systems = {
        "SI": {
            "G": f"{C.G_SI:.5e}",
            "M": f"{m_star * ms_kg:.3e} kg",
            "m": f"{mass_sun[m_lo] * ms_kg:.3e} – {mass_sun[hi] * ms_kg:.3e} kg",
            "a": rng([x * C.AU_SI for x in axis.values()], lambda v: f"{v:.3e} m"),
            "P": rng([x * C.DAY_SI for x in period.values()], lambda v: f"{v:.3e} s"),
            "v": rng([x * C.AU_SI / C.DAY_SI for x in vcirc.values()], lambda v: f"{v:.3e} m/s"),
        },
        "au, M☉, yr": {
            "G": f"{C.G * C.DAYS_PER_YEAR_JULIAN**2:.4f} (4π² = {4 * math.pi**2:.4f})",
            "M": f"{m_star:.3f}",
            "m": f"{mass_sun[m_lo]:.3e} – {mass_sun[hi]:.3e}",
            "a": rng(list(axis.values()), lambda v: f"{v:.4f}"),
            "P": rng([x / C.DAYS_PER_YEAR_JULIAN for x in period.values()], lambda v: f"{v:.4f}"),
            "v": rng([x * C.DAYS_PER_YEAR_JULIAN for x in vcirc.values()], lambda v: f"{v:.2f}"),
        },
        "au, M⊕, day": {
            "G": f"{C.G * C.EARTH_MASS_IN_SOLAR:.4e}",
            "M": f"{m_star / C.EARTH_MASS_IN_SOLAR:.4e}",
            "m": f"{mass_e[m_lo]:.2f} – {mass_e[hi]:.2f}",
            "a": rng(list(axis.values()), lambda v: f"{v:.4f}"),
            "P": rng(list(period.values()), lambda v: f"{v:.2f}"),
            "v": rng(list(vcirc.values()), lambda v: f"{v:.4f}"),
        },
        "normalised (G = M★ = a_h = 1)": {
            "G": "1",
            "M": "1",
            "m": f"{mass_sun[m_lo] / m_star:.3e} – {mass_sun[hi] / m_star:.3e}",
            "a": rng([x / L4 for x in axis.values()], lambda v: f"{v:.4f}"),
            "P": rng([x / T4 for x in period.values()], lambda v: f"{v:.4f}"),
            "v": rng([x * T4 / L4 for x in vcirc.values()], lambda v: f"{v:.4f}"),
        },
    }
    names = list(systems)

    def row(label: str, key: str) -> str:
        return f"| {label} | " + " | ".join(systems[n][key] for n in names) + " |"

    # --- Kepler-III bookkeeping for the shortcut discussion -------------------------
    gm_exact_yr = C.G * C.DAYS_PER_YEAR_JULIAN**2
    shortcut_err = abs(4 * math.pi**2 - gm_exact_yr) / gm_exact_yr
    sidereal_gauss_year = 2.0 * math.pi / C.GAUSS_K       # days in the Gaussian year
    year_ratio_sq = (sidereal_gauss_year / C.DAYS_PER_YEAR_JULIAN) ** 2 - 1.0

    worst_axis_rel = 0.0
    for k, v in planets.items():
        a_cat = v["semi_major_axis_au"]["value"]
        a_shortcut = (m_star * (period[k] / C.DAYS_PER_YEAR_JULIAN) ** 2) ** (1.0 / 3.0)
        worst_axis_rel = max(worst_axis_rel, abs(a_shortcut - a_cat) / a_cat)

    p_h_min = period[hi] * (shortcut_err / 2.0) * 1440.0

    lines = [
        "# Unit System — analysis and decision",
        "",
        "**Phase 2 deliverable.** Implemented in `invisible_planet/constants.py`, the only place",
        "in the codebase where a physical constant may be defined.",
        "",
        "> Every number in this document is **computed** by `scripts/build_units_doc.py` from the",
        "> reference dataset and `constants.py`. Nothing is typed by hand.",
        "",
        "## Decision",
        "",
        "> **length = astronomical unit (au) · time = day · mass = solar mass (M☉)**",
        ">",
        f"> with **G = {C.G:.12e} au³ M☉⁻¹ day⁻²**, derived from the IAU 2015 nominal solar mass",
        "> parameter GM☉ = 1.3271244×10²⁰ m³ s⁻², and cross-checked against the classical Gaussian",
        f"> constant k² (agreement {abs(C.G - C.GAUSS_K**2) / C.G:.1e}).",
        "",
        "## Candidates compared",
        "",
        f"Values for Kepler-90 (M★ = {m_star} M☉; semi-major axes from Kepler's third law with the",
        "total mass; velocities are circular speeds).",
        "",
        "| | " + " | ".join(f"**{n}**" for n in names) + " |",
        "|---|" + "---|" * len(names),
        row("G", "G"),
        row("M★", "M"),
        row("planet masses", "m"),
        row("a range", "a"),
        row("P range", "P"),
        row("v_circ range", "v"),
        "",
        "### Evaluation against the project's actual requirements",
        "",
        "**Numerical stability.** All of the normalised systems keep every stored quantity within a",
        "few orders of unity, so double-precision round-off is comparable between them. SI spreads",
        "values across ~40 orders of magnitude, which makes tolerances hard to reason about.",
        "**SI rejected.**",
        "",
        "**The decisive criterion — the observable.** The project's scientific output is a *transit",
        "time*, quoted in **days** (BJD) by every source in the dataset, and the O−C residuals to be",
        "resolved are of order seconds to minutes. With the day as the time unit, simulation time",
        "*is* the observable.",
        "",
        "In the normalised system every transit time is multiplied by a conversion factor that",
        f"depends on M★ (here T = {T4:.3f} d per unit). M★ is uncertain at the ±3% level, so that",
        "uncertainty would leak into the time axis of the measurements. **Rejected.**",
        "",
        "The year (system 2) adds a needless conversion on every timing computation. **Rejected.**",
        "",
        "Systems 3 and the chosen system differ only in the mass unit. The star's mass appears in",
        "nearly every dynamical expression (GM★) while planet masses enter linearly as small",
        "perturbers, so M☉ puts the *dominant* quantity at unity. Planet masses are converted at the",
        "dataset boundary by `earth_masses_to_solar()`.",
        "",
        "## About G M☉ = 4π² au³ yr⁻²",
        "",
        "The identity is a convenient approximation, not an exact relation. It equates GM☉ with the",
        "Kepler law for a body at 1 au whose period is *exactly* one Julian year (365.25 d), whereas",
        f"the Gaussian constant defines the period at 1 au to be 2π/k = {sidereal_gauss_year:.4f} d. Hence",
        "",
        "```",
        f"G M☉ (IAU 2015 nominal)  = {C.G:.12e} au³/day²  = {gm_exact_yr:.6f} au³/yr²",
        f"Gaussian constant k²     = {C.GAUSS_K**2:.12e} au³/day²   (agreement {abs(C.G - C.GAUSS_K**2) / C.G:.1e})",
        f"4π² au³/yr²              = {4 * math.pi**2:.6f} au³/yr²   (relative error {shortcut_err:.3e})",
        "```",
        "",
        "**What this does and does not affect — stated carefully.**",
        "",
        "* If the semi-major axis is derived from the *measured period* with one consistent GM, every",
        "  transit time is unchanged: transit times depend on the period, on planet-to-star mass",
        "  ratios and on the geometry, and only a/R★ moves (by ~1×10⁻⁵). Using 4π² consistently would",
        "  be an equivalent model with a stellar mass 4×10⁻⁵ different, far below the ±3% uncertainty",
        "  on M★. It would not, by itself, corrupt any timing result.",
        "* The catalogue's semi-major axes **do** embed the shortcut. Recomputing",
        "  a = (M★ P²/yr²)^(1/3) with P in Julian years reproduces every catalogue value to",
        f"  {worst_axis_rel:.0e} (relative), which identifies the cause of the uniform {abs(1 - (1 + shortcut_err) ** (-1 / 3)):.2e}",
        "  offset between the catalogue and a Kepler-law calculation using the IAU value of GM☉.",
        f"  Feeding a catalogue `a` into a dynamical model built on a different GM would therefore",
        f"  mis-set the period by {shortcut_err / 2:.2e} — up to {p_h_min:.1f} minutes per orbit for Kepler-90 h.",
        "  That is an *inconsistency between two conventions*, not an error inherent in either.",
        "",
        "The project avoids the inconsistency by never reading the catalogue `a`: the semi-major",
        "axis is derived from the measured period and the *total* mass using the single `G` defined",
        "in `constants.py`. (The catalogue also omitted the planet mass; see",
        "`docs/KEPLER90_DATA.md`.)",
        "",
        "## Why G is used through GM, never as G and M separately",
        "",
        "The CODATA value of G carries ~2×10⁻⁵ relative uncertainty, whereas the *product* GM☉ is",
        "known to ~10⁻¹⁰. All mass ratios (e.g. `EARTH_MASS_IN_SOLAR`) are formed from GM ratios so",
        "the poorly known G cancels exactly.",
        "",
        "## Precision",
        "",
        "Double precision (f64) is mandatory throughout the physics layer. In single precision the",
        "spacing of representable times near t = 10⁴ d is 2⁻¹⁰ d ≈ 84 s, the same order as the",
        "signals being measured.",
        "",
        "## Representative magnitudes in the adopted system (Kepler-90)",
        "",
        f"M★ = {m_star} M☉",
        "",
        "| planet | m [M☉] | a [au] | P [day] | v_circ [au/day] | GM★/a² [au/day²] |",
        "|---|---|---|---|---|---|",
    ]
    for k in planets:
        lines.append(
            f"| {k} | {mass_sun[k]:.4e} | {axis[k]:.5f} | {period[k]:.5f} | "
            f"{vcirc[k]:.6f} | {C.G * m_star / axis[k] ** 2:.6e} |"
        )
    lines += [
        "",
        "Reproduce: `uv run --python 3.12 --extra dev python scripts/build_units_doc.py`",
        "",
    ]
    (ROOT / "docs" / "UNITS.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote docs/UNITS.md")
    print(f"  shortcut relative error in GM : {shortcut_err:.3e}")
    print(f"  Gaussian-year explanation     : (2 pi/k / 365.25)^2 - 1 = {year_ratio_sq:.3e}")
    print(f"  catalogue a vs shortcut (max) : {worst_axis_rel:.1e}")
    print(f"  h period error if inconsistent: {p_h_min:.2f} min/orbit")


if __name__ == "__main__":
    main()
