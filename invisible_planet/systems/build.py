"""
Assemble N-body initial conditions from Keplerian orbital elements.

Shared by the two-body validation (Phase 3), the Kepler-90 baseline (Phase 5),
and the Planet X experiments (Phase 7) — one code path, so the synthetic planet
is initialised by exactly the same machinery as every real planet.

ELEMENT CONVENTION
------------------
Published exoplanet elements are ASTROCENTRIC: each planet's orbit is quoted
relative to the star, with mu = G(M_star + m_planet). We build each planet's
state that way, then shift the whole system into the barycentric frame so that
total linear momentum is exactly zero.

That choice is documented rather than silent because it matters: astrocentric
and Jacobi elements differ at order (m_planet / M_star), which for Kepler-90 h
is ~5.5e-4. For a system this hierarchical the difference is far below the
observational uncertainties, but it is a real modelling choice.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..constants import G
from ..physics.kepler import elements_to_state


@dataclass
class BodySpec:
    """One gravitating body, defined by its mass and (for planets) its orbit."""

    name: str
    mass: float  # M_sun
    semi_major_axis: float = 0.0  # au
    eccentricity: float = 0.0
    inclination: float = 0.0  # radians
    longitude_of_ascending_node: float = 0.0  # radians
    argument_of_periapsis: float = 0.0  # radians
    mean_anomaly: float = 0.0  # radians at the reference epoch
    radius: float = 0.0  # au, for transit geometry only; never affects gravity
    is_central: bool = False
    metadata: dict = field(default_factory=dict)


@dataclass
class SystemState:
    """Cartesian initial conditions, ready to load into the N-body engine."""

    names: list[str]
    masses: np.ndarray  # (N,)   M_sun
    positions: np.ndarray  # (N,3) au
    velocities: np.ndarray  # (N,3) au/day
    radii: np.ndarray  # (N,)   au
    epoch: float = 0.0  # days
    metadata: dict = field(default_factory=dict)

    @property
    def n_bodies(self) -> int:
        return len(self.names)

    def index_of(self, name: str) -> int:
        return self.names.index(name)


def build_system(bodies: list[BodySpec], epoch: float = 0.0) -> SystemState:
    """Turn a list of BodySpec into barycentric Cartesian initial conditions.

    Exactly one body must be flagged `is_central`.
    """
    central = [b for b in bodies if b.is_central]
    if len(central) != 1:
        raise ValueError(f"expected exactly one central body, got {len(central)}")
    star = central[0]
    orbiters = [b for b in bodies if not b.is_central]

    names = [star.name] + [b.name for b in orbiters]
    masses = np.array([star.mass] + [b.mass for b in orbiters], dtype=np.float64)
    radii = np.array([star.radius] + [b.radius for b in orbiters], dtype=np.float64)

    positions = np.zeros((len(names), 3), dtype=np.float64)
    velocities = np.zeros((len(names), 3), dtype=np.float64)

    # each planet is placed relative to the star using astrocentric elements
    for idx, body in enumerate(orbiters, start=1):
        mu = G * (star.mass + body.mass)
        r_rel, v_rel = elements_to_state(
            mu,
            body.semi_major_axis,
            body.eccentricity,
            body.inclination,
            body.longitude_of_ascending_node,
            body.argument_of_periapsis,
            body.mean_anomaly,
        )
        positions[idx] = r_rel
        velocities[idx] = v_rel

    # shift into the barycentric frame: total momentum becomes exactly zero
    total_mass = masses.sum()
    com = (masses[:, None] * positions).sum(axis=0) / total_mass
    com_vel = (masses[:, None] * velocities).sum(axis=0) / total_mass
    positions -= com
    velocities -= com_vel

    return SystemState(
        names=names,
        masses=masses,
        positions=positions,
        velocities=velocities,
        radii=radii,
        epoch=epoch,
        metadata={"central_body": star.name},
    )


def two_body_system(
    central_mass: float,
    orbiter_mass: float,
    semi_major_axis: float,
    eccentricity: float = 0.0,
    inclination: float = 0.0,
    mean_anomaly: float = 0.0,
    argument_of_periapsis: float = 0.0,
) -> SystemState:
    """Convenience constructor for the two-body validation problems."""
    return build_system(
        [
            BodySpec(name="central", mass=central_mass, is_central=True),
            BodySpec(
                name="orbiter",
                mass=orbiter_mass,
                semi_major_axis=semi_major_axis,
                eccentricity=eccentricity,
                inclination=inclination,
                argument_of_periapsis=argument_of_periapsis,
                mean_anomaly=mean_anomaly,
            ),
        ]
    )
