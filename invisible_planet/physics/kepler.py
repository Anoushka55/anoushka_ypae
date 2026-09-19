"""
Keplerian orbital elements <-> Cartesian state vectors.

Used to build initial conditions from published orbital elements (Phase 5) and
to analyse orbits during a run. Pure NumPy — no Taichi, no simulation state.

COORDINATE CONVENTION (fixed for the whole project)
----------------------------------------------------
    +Z  points from the system towards the OBSERVER (Earth).
    XY  is therefore the plane of the sky.

Consequences, which the transit model in observations/ depends on:

  * Inclination i is the angle between the orbital angular-momentum vector and
    the +Z axis. i = 90 deg is an edge-on orbit, which is the transiting case.
  * A planet is in front of the star (transiting, rather than being occulted)
    when its z coordinate exceeds the star's.
  * For a circular edge-on orbit the transit occurs at true anomaly
    nu = 90 deg - omega. This identity is asserted in the test suite.

Angles are in radians inside this module. Callers converting from published
tables (degrees) must convert at the boundary.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "solve_kepler_equation",
    "mean_to_true_anomaly",
    "true_to_mean_anomaly",
    "elements_to_state",
    "state_to_elements",
    "transit_mean_anomaly",
]


def solve_kepler_equation(
    mean_anomaly: float | np.ndarray,
    eccentricity: float,
    tolerance: float = 1e-14,
    max_iterations: int = 100,
) -> np.ndarray:
    """Solve Kepler's equation  M = E - e sin E  for the eccentric anomaly E.

    Newton-Raphson with a standard starting guess. Converges quadratically for
    the small eccentricities relevant here. Raises if it fails to converge, so a
    silent bad solution can never propagate into the physics.
    """
    m = np.atleast_1d(np.asarray(mean_anomaly, dtype=np.float64))
    e = float(eccentricity)
    if not 0.0 <= e < 1.0:
        raise ValueError(f"eccentricity {e} outside the elliptic range [0, 1)")

    # Split off whole revolutions before solving, then add them back. Newton's
    # method is only well conditioned on a single branch, but discarding the
    # winding number would make the returned E satisfy Kepler's equation only
    # modulo 2*pi. Restoring it makes  E - e sin E == M  hold exactly for any M.
    winding = np.floor((m + np.pi) / (2.0 * np.pi))
    m_wrapped = m - 2.0 * np.pi * winding
    ecc_anom = m_wrapped + e * np.sin(m_wrapped)  # good first guess for small e

    for _ in range(max_iterations):
        residual = ecc_anom - e * np.sin(ecc_anom) - m_wrapped
        derivative = 1.0 - e * np.cos(ecc_anom)
        delta = residual / derivative
        ecc_anom -= delta
        if np.max(np.abs(delta)) < tolerance:
            break
    else:
        raise RuntimeError(
            f"Kepler equation failed to converge for e={e} after {max_iterations} iterations"
        )
    return ecc_anom + 2.0 * np.pi * winding


def mean_to_true_anomaly(mean_anomaly, eccentricity: float) -> np.ndarray:
    """Mean anomaly -> true anomaly."""
    ecc_anom = solve_kepler_equation(mean_anomaly, eccentricity)
    e = float(eccentricity)
    return 2.0 * np.arctan2(
        np.sqrt(1.0 + e) * np.sin(0.5 * ecc_anom),
        np.sqrt(1.0 - e) * np.cos(0.5 * ecc_anom),
    )


def true_to_mean_anomaly(true_anomaly, eccentricity: float) -> np.ndarray:
    """True anomaly -> mean anomaly (inverse of `mean_to_true_anomaly`)."""
    nu = np.atleast_1d(np.asarray(true_anomaly, dtype=np.float64))
    e = float(eccentricity)
    ecc_anom = 2.0 * np.arctan2(
        np.sqrt(1.0 - e) * np.sin(0.5 * nu), np.sqrt(1.0 + e) * np.cos(0.5 * nu)
    )
    return ecc_anom - e * np.sin(ecc_anom)


def elements_to_state(
    mu: float,
    semi_major_axis: float,
    eccentricity: float,
    inclination: float,
    longitude_of_ascending_node: float,
    argument_of_periapsis: float,
    mean_anomaly: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert Keplerian elements to a relative position/velocity pair.

    Parameters
    ----------
    mu : G * (M_central + m_body), in au^3/day^2.
    semi_major_axis : au.
    eccentricity, inclination, longitude_of_ascending_node,
    argument_of_periapsis, mean_anomaly : radians (except a, dimensionless e).

    Returns
    -------
    (r, v) : position [au] and velocity [au/day] of the body RELATIVE to the
    central body, in the observer frame defined in the module docstring.
    """
    a = float(semi_major_axis)
    e = float(eccentricity)
    ecc_anom = float(solve_kepler_equation(mean_anomaly, e)[0])

    # --- perifocal frame: periapsis along +x, orbit in the xy plane ---
    cos_e, sin_e = np.cos(ecc_anom), np.sin(ecc_anom)
    sqrt_1me2 = np.sqrt(1.0 - e * e)

    r_pf = np.array([a * (cos_e - e), a * sqrt_1me2 * sin_e, 0.0])
    r_mag = a * (1.0 - e * cos_e)
    v_factor = np.sqrt(mu * a) / r_mag
    v_pf = np.array([-v_factor * sin_e, v_factor * sqrt_1me2 * cos_e, 0.0])

    # --- rotate into the observer frame:  R = Rz(Omega) Rx(i) Rz(omega) ---
    rot = _rotation_z(longitude_of_ascending_node) @ _rotation_x(inclination) @ _rotation_z(
        argument_of_periapsis
    )
    return rot @ r_pf, rot @ v_pf


def state_to_elements(mu: float, position: np.ndarray, velocity: np.ndarray) -> dict:
    """Convert a relative state vector back to Keplerian elements.

    Inverse of `elements_to_state`. Used to measure the osculating elements of a
    perturbed orbit during a run (Phase 5 stability diagnostics).
    """
    r_vec = np.asarray(position, dtype=np.float64)
    v_vec = np.asarray(velocity, dtype=np.float64)
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)

    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)

    # eccentricity vector points at periapsis
    e_vec = np.cross(v_vec, h_vec) / mu - r_vec / r
    e = np.linalg.norm(e_vec)

    # specific orbital energy -> semi-major axis (vis-viva rearranged)
    energy = 0.5 * v * v - mu / r
    a = -mu / (2.0 * energy)

    inclination = np.arccos(np.clip(h_vec[2] / h, -1.0, 1.0))

    # node vector: intersection of the orbital plane with the sky plane
    n_vec = np.array([-h_vec[1], h_vec[0], 0.0])
    n = np.linalg.norm(n_vec)

    if n < 1e-12:  # orbit lies in the sky plane; node is undefined
        raan = 0.0
        argp = np.arctan2(e_vec[1], e_vec[0]) if e > 1e-12 else 0.0
    else:
        raan = np.arctan2(n_vec[1], n_vec[0])
        if e > 1e-12:
            argp = np.arccos(np.clip(np.dot(n_vec, e_vec) / (n * e), -1.0, 1.0))
            if e_vec[2] < 0.0:
                argp = 2.0 * np.pi - argp
        else:
            argp = 0.0

    if e > 1e-12:
        true_anom = np.arccos(np.clip(np.dot(e_vec, r_vec) / (e * r), -1.0, 1.0))
        if np.dot(r_vec, v_vec) < 0.0:
            true_anom = 2.0 * np.pi - true_anom
    else:  # circular: measure from the node instead
        if n < 1e-12:
            true_anom = np.arctan2(r_vec[1], r_vec[0])
        else:
            true_anom = np.arccos(np.clip(np.dot(n_vec, r_vec) / (n * r), -1.0, 1.0))
            if r_vec[2] < 0.0:
                true_anom = 2.0 * np.pi - true_anom

    return {
        "semi_major_axis": a,
        "eccentricity": e,
        "inclination": inclination,
        "longitude_of_ascending_node": np.mod(raan, 2.0 * np.pi),
        "argument_of_periapsis": np.mod(argp, 2.0 * np.pi),
        "true_anomaly": np.mod(true_anom, 2.0 * np.pi),
        "mean_anomaly": float(np.mod(true_to_mean_anomaly(true_anom, e)[0], 2.0 * np.pi)),
        "period": 2.0 * np.pi * np.sqrt(a**3 / mu) if a > 0 else np.inf,
        "specific_angular_momentum": h,
        "specific_energy": energy,
    }


def transit_mean_anomaly(eccentricity: float, argument_of_periapsis: float) -> float:
    """Mean anomaly at which a transit occurs.

    A transit happens when the planet passes between the star and the observer,
    i.e. at true anomaly  nu = pi/2 - omega  in the convention of this module
    (+Z towards the observer).
    """
    nu_transit = 0.5 * np.pi - argument_of_periapsis
    return float(np.mod(true_to_mean_anomaly(nu_transit, eccentricity)[0], 2.0 * np.pi))


def _rotation_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rotation_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
