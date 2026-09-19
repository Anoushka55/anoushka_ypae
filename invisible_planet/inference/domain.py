"""
The search domain and the priors of the inverse problem.

Everything here uses only PUBLIC information: the visible planets' published masses and
orbits, and the physical requirement that a planet be dynamically plausible. Nothing depends
on the hidden planet.

WHAT THE OBSERVER KNOWS ABOUT WHERE A PLANET CAN BE
---------------------------------------------------
A planet placed too close to a neighbour is dynamically unstable. The standard measure is the
separation in mutual Hill radii,

    R_H = ((m1 + m2) / (3 M*))^(1/3) * (a1 + a2) / 2,        Delta = (a2 - a1) / R_H,

with Delta > 2 sqrt(3) ~ 3.46 required for two circular planets to be Hill stable (Gladman
1993) and larger values for many-planet systems. The real system's tightest pair (g-h) sits
at 5.2. The search domain is therefore the set of semi-major axes whose separation from both
neighbours exceeds `MIN_HILL_SEARCH` for a small trial mass, and the prior for a sampled
candidate requires `MIN_HILL_PRIOR` for ITS mass.

NON-TRANSITING PRIOR
--------------------
Kepler saw no extra transit signal, so a candidate must not transit. With impact parameter
b = a cos(i) (1-e^2) / (1 + e sin(omega)) / R*, that requires b > 1 + Rp/R*. The planet
radius is unknown, so a generous Rp = 4 R_earth is assumed. This is a genuine observational
constraint, not knowledge of the hidden planet: had it transited, Kepler would have detected it
directly (a 4 R_earth transit around this star is ~1e-3 deep, far above the 2e-4 per-cadence
noise).
"""

from __future__ import annotations

import numpy as np

from ..constants import EARTH_MASS_IN_SOLAR, R_EARTH_IN_AU, kepler_third_law_axis
from ..systems.planet_x import PlanetXParameters

MIN_HILL_SEARCH = 8.0    # separation required of the trial mass when defining the search zones
MIN_HILL_PRIOR = 5.0     # separation required of a sampled candidate (the real g-h pair is 5.2)
TRIAL_MASS_EARTH = 5.0
ASSUMED_RADIUS_EARTH = 4.0


def visible_planets(reference: dict) -> list:
    """(letter, a_au, mass_solar) of every visible planet, sorted by semi-major axis."""
    m_star = reference["star"]["mass_solar"]["value"]
    out = []
    for letter, p in reference["planets"].items():
        m = p["mass_earth"]["value"] * EARTH_MASS_IN_SOLAR
        a = kepler_third_law_axis(p["orbital_period_days"]["value"], m_star + m)
        out.append((letter, a, m))
    return sorted(out, key=lambda x: x[1])


def hill_separations(a: float, mass_solar: float, reference: dict) -> tuple:
    """Separation (in mutual Hill radii) from the nearest inner and outer visible planet."""
    m_star = reference["star"]["mass_solar"]["value"]
    inner = [p for p in visible_planets(reference) if p[1] < a]
    outer = [p for p in visible_planets(reference) if p[1] > a]

    def delta(a1, m1, a2, m2):
        r_h = ((m1 + m2) / (3.0 * m_star)) ** (1.0 / 3.0) * 0.5 * (a1 + a2)
        return (a2 - a1) / r_h

    d_in = delta(inner[-1][1], inner[-1][2], a, mass_solar) if inner else np.inf
    d_out = delta(a, mass_solar, outer[0][1], outer[0][2]) if outer else np.inf
    return d_in, d_out


def stable_zones(
    reference: dict,
    a_min: float = 0.13,
    a_max: float = 1.6,
    trial_mass_earth: float = TRIAL_MASS_EARTH,
    min_hill: float = MIN_HILL_SEARCH,
    resolution: float = 0.0005,
) -> list:
    """Intervals of semi-major axis [au] that are dynamically plausible for a trial mass."""
    grid = np.arange(a_min, a_max, resolution)
    mass = trial_mass_earth * EARTH_MASS_IN_SOLAR
    ok = np.array([min(hill_separations(a, mass, reference)) >= min_hill for a in grid])
    zones, start = [], None
    for a, flag in zip(grid, ok):
        if flag and start is None:
            start = a
        if not flag and start is not None:
            zones.append((float(start), float(a - resolution)))
            start = None
    if start is not None:
        zones.append((float(start), float(grid[-1])))
    return [z for z in zones if z[1] - z[0] > 2 * resolution]


def axis_step(a: float, observation_days: float, m_star: float, oversample: float = 5.0) -> float:
    """Sampling step in semi-major axis needed to resolve the chi2 valley.

    A period error dP accumulates a phase error 2 pi (T/P)(dP/P) over the baseline T, so the
    model decorrelates from the data when dP/P ~ P / (2 pi T). With a ~ P^(2/3) that is a
    width in a of (2/3) a P / (2 pi T). The valley is sampled `oversample` times per width.
    """
    period = 2.0 * np.pi * np.sqrt(a**3 / (4.0 * np.pi**2 / 365.25**2 * m_star))
    width = (2.0 / 3.0) * a * period / (2.0 * np.pi * observation_days)
    return float(width / oversample)


def axes_for_zones(zones: list, observation_days: float, m_star: float) -> np.ndarray:
    """Adaptive grid of semi-major axes covering the zones."""
    out = []
    for lo, hi in zones:
        a = lo
        while a <= hi:
            out.append(a)
            a += axis_step(a, observation_days, m_star)
    return np.asarray(out)


def impact_parameter(params: PlanetXParameters, stellar_radius_au: float) -> float:
    e, w = params.eccentricity, params.argument_of_periapsis
    return float(
        params.semi_major_axis_au
        * abs(np.cos(np.deg2rad(params.inclination_deg)))
        * (1.0 - e * e)
        / (1.0 + e * np.sin(w))
        / stellar_radius_au
    )


def is_plausible(params: PlanetXParameters, reference: dict, stellar_radius_au: float) -> bool:
    """The physical prior: bound orbit, Hill-plausible, and non-transiting."""
    if not (0.0 <= params.eccentricity < 0.35):
        return False
    if params.mass_earth <= 0.0:
        return False
    # periapsis / apoapsis must also clear the neighbours, not just the semi-major axis
    a_peri = params.semi_major_axis_au * (1.0 - params.eccentricity)
    a_apo = params.semi_major_axis_au * (1.0 + params.eccentricity)
    m = params.mass_solar
    d_in, _ = hill_separations(a_peri, m, reference)
    _, d_out = hill_separations(a_apo, m, reference)
    if min(d_in, d_out) < MIN_HILL_PRIOR:
        return False
    limit = 1.0 + ASSUMED_RADIUS_EARTH * R_EARTH_IN_AU / stellar_radius_au
    return impact_parameter(params, stellar_radius_au) > limit
