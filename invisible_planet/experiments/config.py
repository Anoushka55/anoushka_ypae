"""
The experiment configuration — one place, fully serialisable, fully seeded.

Phase 18 deliverable, used from Phase 8 onwards. A fresh clone reproduces every
result in this project from the values in this file plus the archive snapshot in
data/raw/.

NOTHING HERE IS TUNED FOR APPEARANCE. The timestep comes from the convergence
study (docs/NUMERICAL_METHODS.md); the Planet X parameters come from the
Phase 8 sweep and its stated selection criteria (docs/PLANET_X_DESIGN.md).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from ..systems.kepler90 import DEFAULT_EPOCH_BJD, recommended_timestep
from ..systems.planet_x import PlanetXParameters

# =============================================================================
# THE CANONICAL PLANET X
# =============================================================================
# Selected by scripts/sweep_planet_x.py from 105 candidates, against criteria
# fixed BEFORE the sweep was run (see docs/PLANET_X_DESIGN.md):
#
#   stable       no orbit crossing, bounded elements, energy error < 1e-6
#   plausible    9.7 mutual Hill radii from its nearest neighbour
#   detectable   signal above the 71 s best published timing precision
#   recoverable  detectable in 4 visible planets, not just one
#   subtle       23 minutes peak-to-peak — far from obvious by eye
#   INVISIBLE    b = 1.39, so it does not transit and cannot be seen directly
#
# The inclination deserves comment. At the system's typical 89.8 deg the planet
# WOULD transit, which would defeat the premise entirely: a transiting planet is
# directly observable and needs no inference. 88.0 deg is the most conservative
# non-transiting choice — only 1.8 deg from the other planets, so the system
# stays very nearly coplanar and no result depends on exotic geometry.
#
# This mirrors how non-transiting planets are found in reality: Kepler-19c was
# detected purely through the transit timing variations it induced in Kepler-19b.
# =============================================================================
CANONICAL_PLANET_X = PlanetXParameters(
    mass_earth=40.0,
    semi_major_axis_au=0.22,
    eccentricity=0.0,
    mean_anomaly=2.09,
    inclination_deg=88.0,
    argument_of_periapsis=0.0,
    radius_earth=3.9,
)


@dataclass
class ExperimentConfig:
    """Everything needed to reproduce the experiment."""

    # --- numerics -----------------------------------------------------------
    steps_per_inner_orbit: int = 1000
    epoch_bjd: float = DEFAULT_EPOCH_BJD

    # --- observing campaign -------------------------------------------------
    #: 1400 days approximates the Kepler primary mission baseline, which is the
    #: span over which this system's transits were actually measured.
    observation_days: float = 1400.0

    # --- measurement model --------------------------------------------------
    #: Per-transit timing uncertainty. The published transit-epoch uncertainties
    #: for Kepler-90 span 71 s (h) to 1210 s (f); `noise_scale` multiplies the
    #: per-planet published value, so the synthetic data inherits the real
    #: system's precision structure rather than a made-up constant.
    noise_scale: float = 1.0
    noise_seed: int = 20260919

    # --- inference ----------------------------------------------------------
    search_seed: int = 7
    coarse_grid_mass: tuple = (5.0, 10.0, 20.0, 40.0, 80.0, 160.0)
    coarse_grid_axis: tuple = (0.14, 0.17, 0.20, 0.23, 0.26, 0.30)
    coarse_grid_phase_count: int = 8

    # --- ground truth (NEVER visible to the inference) ----------------------
    planet_x: PlanetXParameters = field(default_factory=lambda: CANONICAL_PLANET_X)

    @property
    def timestep_days(self) -> float:
        return recommended_timestep(steps_per_inner_orbit=self.steps_per_inner_orbit)

    def to_dict(self, include_ground_truth: bool = True) -> dict:
        out = {
            "steps_per_inner_orbit": self.steps_per_inner_orbit,
            "timestep_days": self.timestep_days,
            "epoch_bjd": self.epoch_bjd,
            "observation_days": self.observation_days,
            "noise_scale": self.noise_scale,
            "noise_seed": self.noise_seed,
            "search_seed": self.search_seed,
        }
        if include_ground_truth:
            out["planet_x_ground_truth"] = self.planet_x.to_dict()
        return out

    def save(self, path) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)


DEFAULT_CONFIG = ExperimentConfig()
