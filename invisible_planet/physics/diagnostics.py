"""
Orbit diagnostics: osculating elements and stability metrics.

Pure NumPy, read-only with respect to simulation state. Used to answer "is this
system behaving itself?" without any reference to how the state was produced.

OSCULATING ELEMENTS
-------------------
A perturbed planet has no fixed orbit, so at each instant we compute the
Keplerian orbit it *would* follow if all other perturbations vanished — its
osculating orbit. Watching the osculating a and e over time is the standard way
to distinguish a stable system (small bounded variations) from an unstable one
(secular growth, sudden jumps, or ejection).

The elements are computed ASTROCENTRICALLY (relative to the star) with
mu = G(M_star + m_planet), matching the convention used to build the initial
conditions in systems/build.py.
"""

from __future__ import annotations

import numpy as np

from ..constants import G
from .kepler import state_to_elements


def osculating_elements(
    masses: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    central_index: int = 0,
) -> list[dict]:
    """Osculating elements of every non-central body, relative to the central one."""
    out = []
    r_star = positions[central_index]
    v_star = velocities[central_index]
    m_star = masses[central_index]

    for i in range(len(masses)):
        if i == central_index:
            continue
        mu = G * (m_star + masses[i])
        elements = state_to_elements(mu, positions[i] - r_star, velocities[i] - v_star)
        elements["body_index"] = i
        out.append(elements)
    return out


def minimum_pairwise_separation(positions: np.ndarray) -> float:
    """Smallest distance between any two bodies."""
    n = len(positions)
    best = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            best = min(best, float(np.linalg.norm(positions[j] - positions[i])))
    return best


def mutual_hill_radius(
    m_inner: float, m_outer: float, a_inner: float, a_outer: float, m_star: float
) -> float:
    """Mutual Hill radius of an adjacent pair.

        R_H = ((m_in + m_out) / (3 M*))^(1/3) * (a_in + a_out) / 2
    """
    return ((m_inner + m_outer) / (3.0 * m_star)) ** (1.0 / 3.0) * 0.5 * (a_inner + a_outer)


def hill_separations(
    masses: np.ndarray, semi_major_axes: np.ndarray, m_star: float
) -> np.ndarray:
    """Separation of each adjacent pair in mutual Hill radii.

    The standard compactness measure for multi-planet systems. Gladman (1993)
    showed that an isolated two-planet system on circular orbits is Hill-stable
    for Delta > 2*sqrt(3) ~ 3.46; densely packed multi-planet systems typically
    require considerably more to survive long timescales.
    """
    order = np.argsort(semi_major_axes)
    a = np.asarray(semi_major_axes)[order]
    m = np.asarray(masses)[order]
    deltas = []
    for i in range(len(a) - 1):
        r_h = mutual_hill_radius(m[i], m[i + 1], a[i], a[i + 1], m_star)
        deltas.append((a[i + 1] - a[i]) / r_h)
    return np.array(deltas)


def summarise_element_evolution(history: dict[str, np.ndarray]) -> dict[str, float]:
    """Reduce a time series of one element to stability statistics."""
    out = {}
    for name, series in history.items():
        series = np.asarray(series, dtype=float)
        initial = float(series[0])
        out[f"{name}_initial"] = initial
        out[f"{name}_mean"] = float(np.mean(series))
        out[f"{name}_min"] = float(np.min(series))
        out[f"{name}_max"] = float(np.max(series))
        out[f"{name}_peak_to_peak"] = float(np.max(series) - np.min(series))
        if abs(initial) > 0:
            out[f"{name}_fractional_range"] = float(
                (np.max(series) - np.min(series)) / abs(initial)
            )
        # linear trend, to separate bounded oscillation from secular drift
        t = np.arange(len(series), dtype=float)
        slope = float(np.polyfit(t, series, 1)[0])
        out[f"{name}_trend_per_sample"] = slope
        out[f"{name}_trend_over_run"] = slope * len(series)
    return out


def semi_major_axis_and_eccentricity(mu, r, v):
    """Vectorised osculating semi-major axis and eccentricity from relative state vectors.

    `r`, `v` have shape (..., 3) and `mu` broadcasts against the leading dimensions. Uses
    the vis-viva energy for a and the eccentricity vector for e, exactly as
    `physics.kepler.state_to_elements` does for a single body. Returns (a, e); a is negative
    for an unbound orbit, which callers must treat as a failure.
    """
    r = np.asarray(r, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    rn = np.linalg.norm(r, axis=-1)
    v2 = np.sum(v * v, axis=-1)
    energy = 0.5 * v2 - mu / rn
    a = -mu / (2.0 * energy)
    h = np.cross(r, v)
    e_vec = np.cross(v, h) / np.asarray(mu)[..., None] - r / rn[..., None]
    return a, np.linalg.norm(e_vec, axis=-1)
