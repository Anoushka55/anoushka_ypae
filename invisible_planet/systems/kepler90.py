"""
Build the Cartesian initial state of the real Kepler-90 system.

Phase 5 deliverable. Reads the validated reference dataset produced in Phase 1 and
converts it into barycentric positions and velocities.

This module describes the REAL system only. The synthetic Planet X lives in
systems/planet_x.py and is added on top; nothing here knows it exists.

ORBITAL PHASE — the one genuinely delicate conversion
------------------------------------------------------
An N-body initial condition needs each planet's position *along* its orbit at the
reference epoch, which the catalogue does not state directly. What the catalogue
gives is the transit epoch T0: the time at which the planet crossed in front of
the star.

We use that. At t = T0 the planet is at the transit point, whose true anomaly is
nu = pi/2 - omega in this project's coordinate convention (+Z towards the
observer). Converting to mean anomaly and stepping backwards to the chosen
reference epoch t_ref gives

    M(t_ref) = M_transit + 2*pi * (t_ref - T0) / P

This is exact for a Keplerian orbit and is the standard way to phase a transiting
system. It is not an assumption about the phase — it is a measurement of it,
inherited from the measured transit epoch, and it carries that epoch's uncertainty.

ASSUMPTIONS, stated because they are choices and not facts
-----------------------------------------------------------
1. Eccentricities are taken as zero (the adopted-model value from the dataset).
   With e = 0 the periapsis is undefined, so omega is set to 0 and the transit
   condition reduces to nu = pi/2. Measured eccentricities exist for g and h and
   can be enabled via `use_measured_eccentricities=True`.
2. The longitude of the ascending node is unconstrained by transit photometry for
   a single-star system, so it is set to 0 for every planet. This makes the
   orbits mutually coplanar apart from their differing inclinations. Node
   misalignment would matter for long-term nodal dynamics; over a Kepler-length
   baseline in a system this flat it does not.
3. Inclination is DERIVED FROM THE IMPACT PARAMETER, not taken from the
   published inclination column. See `inclination_from_impact_parameter` below
   for why; this is a deliberate scientific choice, not a convenience.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..constants import (
    EARTH_MASS_IN_SOLAR,
    earth_radii_to_au,
    kepler_third_law_axis,
    solar_radii_to_au,
)
from ..physics.kepler import transit_mean_anomaly
from .build import BodySpec, SystemState, build_system

REFERENCE_PATH = Path(__file__).resolve().parents[2] / "data" / "kepler90_reference.json"

#: Planet order by increasing period, matching the dataset.
PLANET_ORDER = ["b", "c", "i", "d", "e", "f", "g", "h"]

#: Reference epoch for all initial conditions [BJD_TDB]: the start of Kepler's Q1
#: science photometry (BKJD 131.5 = BJD 2454964.5), so the integration spans the real
#: observing window. The simulation window ends at BKJD 1591.0 (end of Q17).
DEFAULT_EPOCH_BJD = 2454964.5
DEFAULT_DURATION_DAYS = 1459.5

MATCHED_PATH = Path(__file__).resolve().parents[2] / "data" / "kepler90_matched_initial_conditions.json"


def load_reference(path: Path | None = None) -> dict:
    return json.loads((path or REFERENCE_PATH).read_text(encoding="utf-8"))


def inclination_from_impact_parameter(
    impact_parameter: float, semi_major_axis_au: float, stellar_radius_au: float
) -> float:
    """Inclination [rad] from the measured transit impact parameter.

        b = a cos(i) / R_star     =>     i = arccos(b R_star / a)

    WHY THE IMPACT PARAMETER IS PRIMARY
    ------------------------------------
    A transit light curve constrains the impact parameter b directly, through the
    transit's duration and its ingress/egress shape. The inclination is a derived
    restatement of b that depends on the adopted a and R_star. Taking a published
    inclination and pairing it with a DIFFERENT paper's a and R_star therefore
    produces an inconsistent geometry.

    That is not hypothetical here. For Kepler-90 h, the published inclination
    (89.60 deg) combined with our adopted a and R_star implies b = 1.23 — meaning
    the planet would miss the stellar disk entirely and never transit, which
    contradicts the fact that it was discovered BY its transits. The same paper
    quotes b = 0.36, which is self-consistent and implies i = 89.883 deg. The
    inclination uncertainty for h is +/-1.3 deg, so the two are compatible; the
    inclination is simply the poorly determined way to express this geometry.

    Deriving i from b guarantees the modelled system reproduces the transits that
    were actually observed. The discrepancies are catalogued in CHECK 5 of
    docs/KEPLER90_DATA.md.
    """
    cos_i = impact_parameter * stellar_radius_au / semi_major_axis_au
    if not -1.0 <= cos_i <= 1.0:
        raise ValueError(
            f"impact parameter {impact_parameter} with a = {semi_major_axis_au} au and "
            f"R_star = {stellar_radius_au} au gives cos(i) = {cos_i}, which is impossible"
        )
    return float(np.arccos(cos_i))


def load_matched(path: Path | None = None) -> dict:
    """Initial osculating period and mean anomaly of each planet, fitted to the real
    transit times by scripts/match_visible_system.py (see systems/matching.py)."""
    target = path or MATCHED_PATH
    if not target.exists():
        raise FileNotFoundError(
            f"{target} does not exist. Run scripts/match_visible_system.py, or build the "
            "system with initial_conditions='catalogue'."
        )
    return json.loads(target.read_text(encoding="utf-8"))


def build_kepler90(
    epoch_bjd: float = DEFAULT_EPOCH_BJD,
    use_measured_eccentricities: bool = False,
    planets: list[str] | None = None,
    reference: dict | None = None,
    inclination_source: str = "impact_parameter",
    initial_conditions: str = "matched",
    period_overrides: dict | None = None,
    mean_anomaly_overrides: dict | None = None,
) -> SystemState:
    """Construct the Kepler-90 system as barycentric Cartesian initial conditions.

    Parameters
    ----------
    epoch_bjd : reference epoch; all mean anomalies refer to this time.
    use_measured_eccentricities : use the measured e for g and h instead of the
        adopted circular baseline. Negative archive values (which are eccentricity
        vector components mis-mapped onto the scalar column) are refused.
    planets : subset of planet letters to include. Used by tests that need an
        isolated single-planet system.
    initial_conditions :
        "matched"   - each planet's initial osculating period and mean anomaly are those
                      fitted to the real transit times (systems/matching.py). This is the
                      default: catalogue periods are OBSERVED MEAN TRANSIT PERIODS, which
                      in a near-resonant system differ from osculating periods by up to
                      0.3% (planet g), so using them directly as osculating values would
                      misrepresent the observed system.
        "catalogue" - catalogue period as the osculating period, and the phase from the
                      catalogue transit epoch. The starting point of the matching, and
                      used by tests of the raw catalogue conversion.
    period_overrides, mean_anomaly_overrides : {letter: value}. Applied last; used by the
        matching fit to perturb one planet at a time.
    """
    if initial_conditions not in ("matched", "catalogue"):
        raise ValueError(f"unknown initial_conditions: {initial_conditions!r}")
    ref = reference if reference is not None else load_reference()
    star = ref["star"]
    chosen = planets if planets is not None else PLANET_ORDER
    stellar_radius_au = solar_radii_to_au(star["radius_solar"]["value"])
    if inclination_source not in ("impact_parameter", "published"):
        raise ValueError(f"unknown inclination_source: {inclination_source!r}")

    matched = load_matched() if initial_conditions == "matched" else None
    if matched is not None and abs(matched["epoch_bjd"] - epoch_bjd) > 1e-9:
        raise ValueError(
            f"matched initial conditions are for epoch {matched['epoch_bjd']}, not {epoch_bjd}"
        )
    period_overrides = period_overrides or {}
    mean_anomaly_overrides = mean_anomaly_overrides or {}

    bodies = [
        BodySpec(
            name="Kepler-90",
            mass=star["mass_solar"]["value"],
            radius=solar_radii_to_au(star["radius_solar"]["value"]),
            is_central=True,
            metadata={"kind": "star"},
        )
    ]

    for letter in chosen:
        p = ref["planets"][letter]
        period = p["orbital_period_days"]["value"]
        transit_epoch = p["transit_epoch_bjd"]["value"]

        eccentricity = p["eccentricity"]["value"]
        if use_measured_eccentricities:
            for alt in p["eccentricity"].get("measured_alternates", []):
                if alt["value"] < 0.0:
                    raise ValueError(
                        f"planet {letter}: refusing to use a negative eccentricity "
                        f"({alt['value']}) from {alt['source']}; it is an eccentricity "
                        "vector component, not a scalar eccentricity"
                    )
                eccentricity = alt["value"]

        # With e = 0 the periapsis - and therefore omega - is undefined; 0 is the
        # conventional choice and the transit condition reduces to nu = pi/2.
        argument_of_periapsis = 0.0

        planet_mass = p["mass_earth"]["value"] * EARTH_MASS_IN_SOLAR

        # Catalogue-based starting values: the catalogue period is used as the osculating
        # period, and the phase is propagated from the catalogue transit epoch with the
        # osculating mean motion 2 pi / P.
        mean_anomaly_at_transit = transit_mean_anomaly(eccentricity, argument_of_periapsis)
        osculating_period = period
        mean_anomaly = mean_anomaly_at_transit + 2.0 * np.pi * (epoch_bjd - transit_epoch) / period

        if matched is not None and letter in matched["planets"]:
            osculating_period = matched["planets"][letter]["osculating_period_days"]
            mean_anomaly = matched["planets"][letter]["mean_anomaly_rad"]
        if letter in period_overrides:
            osculating_period = period_overrides[letter]
        if letter in mean_anomaly_overrides:
            mean_anomaly = mean_anomaly_overrides[letter]

        # The semi-major axis is DERIVED from the osculating period and the TOTAL mass
        # (Kepler's third law), never read from the catalogue: the catalogue values equal
        # (M_star P_yr^2)^(1/3), i.e. the 4 pi^2 shortcut with the stellar mass alone
        # (docs/UNITS.md; CHECK 1b in docs/KEPLER90_DATA.md).
        semi_major_axis = kepler_third_law_axis(
            osculating_period, star["mass_solar"]["value"] + planet_mass
        )
        impact_parameter = p.get("impact_parameter", {}).get("value")
        if inclination_source == "impact_parameter" and impact_parameter is not None:
            inclination = inclination_from_impact_parameter(
                impact_parameter, semi_major_axis, stellar_radius_au
            )
            inclination_provenance = "derived from impact parameter"
        else:
            inclination = np.deg2rad(p["inclination_deg"]["value"])
            inclination_provenance = "published inclination"

        bodies.append(
            BodySpec(
                name=f"Kepler-90 {letter}",
                mass=planet_mass,
                semi_major_axis=semi_major_axis,
                eccentricity=eccentricity,
                inclination=inclination,
                longitude_of_ascending_node=0.0,
                argument_of_periapsis=argument_of_periapsis,
                mean_anomaly=float(np.mod(mean_anomaly, 2.0 * np.pi)),
                radius=earth_radii_to_au(p["planet_radius_earth"]["value"]),
                metadata={
                    "letter": letter,
                    "kind": "planet",
                    "period_days": period,
                    "osculating_period_days": osculating_period,
                    "transit_epoch_bjd": transit_epoch,
                    "mass_status": p["mass_earth"]["status"],
                    "impact_parameter": impact_parameter,
                    "inclination_provenance": inclination_provenance,
                    "is_synthetic": False,
                },
            )
        )

    system = build_system(bodies, epoch=epoch_bjd)
    system.metadata.update(
        {
            "system": ref["system"],
            "epoch_bjd": epoch_bjd,
            "eccentricity_mode": (
                "measured (g, h)" if use_measured_eccentricities else "adopted circular"
            ),
            "inclination_source": inclination_source,
            "initial_conditions": initial_conditions,
            "contains_synthetic_body": False,
            "reference_dataset": ref.get("generated_utc"),
        }
    )
    return system


def planet_periods(reference: dict | None = None) -> dict[str, float]:
    ref = reference if reference is not None else load_reference()
    return {
        letter: ref["planets"][letter]["orbital_period_days"]["value"]
        for letter in PLANET_ORDER
    }


def recommended_timestep(reference: dict | None = None, steps_per_inner_orbit: int = 1000) -> float:
    """Production timestep: the innermost period divided by `steps_per_inner_orbit`.

    Policy justified in docs/NUMERICAL_METHODS.md. The innermost planet sets the
    timestep because it is the fastest-varying body in the system.
    """
    return min(planet_periods(reference).values()) / steps_per_inner_orbit
