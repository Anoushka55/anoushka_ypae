"""
Planet X — the SYNTHETIC unseen body.

=============================================================================
SCIENTIFIC BOUNDARY — read this before using anything in this module
=============================================================================
Planet X does not exist. It is a synthetic body invented for a controlled
numerical experiment. Nothing in this project constitutes a claim that the real
Kepler-90 system contains a ninth planet, and no output of this project should
ever be presented as a detection.

What the experiment DOES demonstrate is a general and real proposition: an
unseen gravitating body leaves measurable timing signatures in the bodies we can
see, and those signatures constrain its properties. That proposition is not
speculative — it is how Neptune was found, and it is how the masses of the real
Kepler-90 g and h were measured.
=============================================================================

Planet X uses the SAME gravitational solver as every other body. There is no
special "hidden planet force", no scripted trajectory, and no term in the
equations of motion that knows which body is synthetic. It is hidden from the
OBSERVER, not from the physics.

PARAMETERISATION
----------------
    m_X     mass                  [Earth masses]
    a_X     semi-major axis       [au]
    e_X     eccentricity
    M_X     mean anomaly at epoch [rad]   (orbital phase)
    i_X     inclination           [rad]   (default: coplanar with the system)
    w_X     argument of periapsis [rad]

Phase and periapsis are separate parameters even though they are degenerate at
e = 0; the inference stage must be able to hold e fixed at zero and still search
phase.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..constants import (
    EARTH_MASS_IN_SOLAR,
    G,
    earth_radii_to_au,
    kepler_third_law_axis,
)
from ..physics.kepler import elements_to_state
from .build import SystemState
from .kepler90 import build_kepler90, load_reference

PLANET_X_NAME = "Planet X (synthetic)"


@dataclass(frozen=True)
class PlanetXParameters:
    """The synthetic planet's parameters. Immutable so it cannot be edited in place."""

    mass_earth: float
    semi_major_axis_au: float
    eccentricity: float = 0.0
    mean_anomaly: float = 0.0
    inclination_deg: float = 89.8
    argument_of_periapsis: float = 0.0
    radius_earth: float = 3.0  # affects transit geometry only, never gravity

    @property
    def mass_solar(self) -> float:
        return self.mass_earth * EARTH_MASS_IN_SOLAR

    def period_days(self, total_mass_solar: float) -> float:
        from ..constants import kepler_third_law_period

        return kepler_third_law_period(self.semi_major_axis_au, total_mass_solar)

    def to_dict(self) -> dict:
        return {
            "mass_earth": self.mass_earth,
            "semi_major_axis_au": self.semi_major_axis_au,
            "eccentricity": self.eccentricity,
            "mean_anomaly": self.mean_anomaly,
            "inclination_deg": self.inclination_deg,
            "argument_of_periapsis": self.argument_of_periapsis,
            "radius_earth": self.radius_earth,
            "status": "synthetic",
            "warning": "Planet X is not real and is not a claim about Kepler-90.",
        }

    @classmethod
    def from_period(cls, mass_earth: float, period_days: float,
                    total_mass_solar: float, **kwargs) -> "PlanetXParameters":
        """Construct from a period instead of a semi-major axis."""
        a = kepler_third_law_axis(period_days, total_mass_solar)
        return cls(mass_earth=mass_earth, semi_major_axis_au=a, **kwargs)

    def with_mass(self, mass_earth: float) -> "PlanetXParameters":
        """Copy with a different mass — used by the m_X -> 0 causality test."""
        return replace(self, mass_earth=mass_earth)


def build_system_with_planet_x(
    parameters: PlanetXParameters,
    epoch_bjd: float | None = None,
    reference: dict | None = None,
    planets: list | None = None,
) -> SystemState:
    """The Kepler-90 system PLUS the synthetic planet.

    The visible planets are built by exactly the same code path as the control
    system, so the two differ only by the presence of Planet X.
    """
    ref = reference if reference is not None else load_reference()
    control = build_kepler90(
        **({} if epoch_bjd is None else {"epoch_bjd": epoch_bjd}),
        reference=ref,
        planets=planets,
    )

    # Planet X is APPENDED to the already-built control state rather than the
    # whole system being re-derived. That guarantees the visible planets'
    # star-relative initial conditions are bit-for-bit identical between the
    # control and mystery systems, so no timing difference can be blamed on the
    # initial conditions.
    star_mass = ref["star"]["mass_solar"]["value"]
    mu_total = star_mass + parameters.mass_solar

    r_rel, v_rel = elements_to_state(
        G * mu_total,
        parameters.semi_major_axis_au,
        parameters.eccentricity,
        np.deg2rad(parameters.inclination_deg),
        0.0,
        parameters.argument_of_periapsis,
        parameters.mean_anomaly,
    )

    names = list(control.names) + [PLANET_X_NAME]
    masses = np.append(control.masses, parameters.mass_solar)
    radii = np.append(control.radii, earth_radii_to_au(parameters.radius_earth))

    # Planet X is placed relative to the STAR, matching the astrocentric
    # convention used for every real planet in systems/build.py.
    star_index = 0
    positions = np.vstack([control.positions, control.positions[star_index] + r_rel])
    velocities = np.vstack([control.velocities, control.velocities[star_index] + v_rel])

    # Re-shift to the barycentric frame now that a new mass has been added.
    total_mass = masses.sum()
    com = (masses[:, None] * positions).sum(axis=0) / total_mass
    com_vel = (masses[:, None] * velocities).sum(axis=0) / total_mass
    positions = positions - com
    velocities = velocities - com_vel

    system = SystemState(
        names=names,
        masses=masses,
        positions=positions,
        velocities=velocities,
        radii=radii,
        epoch=control.epoch,
        metadata={
            **control.metadata,
            "contains_synthetic_body": True,
            "planet_x": parameters.to_dict(),
        },
    )
    return system


def control_and_mystery(
    parameters: PlanetXParameters,
    epoch_bjd: float | None = None,
    reference: dict | None = None,
    planets: list | None = None,
) -> tuple[SystemState, SystemState]:
    """Build the matched pair for the controlled experiment.

    CONTROL  : the Kepler-90 reference system alone.
    MYSTERY  : the same system plus Planet X.

    The visible planets' STAR-RELATIVE initial states are bit-for-bit identical
    in both systems. Their barycentric coordinates differ, because adding a mass
    moves the barycentre, but that is a uniform translation and velocity boost
    applied to every body equally — a Galilean shift, which cannot affect any
    relative configuration and therefore cannot affect a single transit time.
    Both properties are asserted in tests/test_planet_x.py, because if the
    visible planets really differed the experiment would be confounded: a timing
    difference could then be blamed on the initial conditions rather than on
    Planet X's gravity.
    """
    ref = reference if reference is not None else load_reference()
    kwargs = {} if epoch_bjd is None else {"epoch_bjd": epoch_bjd}
    control = build_kepler90(reference=ref, planets=planets, **kwargs)
    mystery = build_system_with_planet_x(
        parameters, epoch_bjd=epoch_bjd, reference=ref, planets=planets
    )
    return control, mystery
