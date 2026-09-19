"""
The experiment configuration: ONE serialisable file that reproduces the whole experiment.

Phase 18 deliverable. `data/experiment_config.json` holds every seed, numerical setting,
observation setting, hidden-planet parameter and inference setting. A fresh clone plus that file
plus the archive snapshots in data/raw/ reproduces, in order:

    1. the baseline            scripts/run_baseline_stability.py
    2. the Planet X selection  scripts/sweep_planet_x.py, scripts/select_planet_x.py
    3. the observations        scripts/run_experiment.py
    4. the inference           scripts/run_experiment.py
    5. the validation          scripts/run_validation.py

The hidden planet's parameters live here, in experiments/, which the inference is forbidden to
import (verified by tests/test_inference.py). The observer dataset that crosses to the inference
is written to disk and contains none of them.

NOTHING HERE IS TUNED FOR APPEARANCE. The timestep comes from the convergence study and the
independent IAS15 comparison (docs/NUMERICAL_METHODS.md); the hidden planet is chosen by the
pre-declared criteria of scripts/sweep_planet_x.py (docs/PLANET_X_DESIGN.md).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..systems.kepler90 import DEFAULT_DURATION_DAYS, DEFAULT_EPOCH_BJD, recommended_timestep
from ..systems.planet_x import PlanetXParameters

CONFIG_PATH = Path(__file__).resolve().parents[2] / "data" / "experiment_config.json"


@dataclass
class ExperimentConfig:
    raw: dict = field(default_factory=dict)

    # ---- numerics ---------------------------------------------------------------------
    @property
    def steps_per_inner_orbit(self) -> int:
        return int(self.raw["numerics"]["steps_per_inner_orbit"])

    @property
    def epoch_bjd(self) -> float:
        return float(self.raw["numerics"]["epoch_bjd"])

    @property
    def duration_days(self) -> float:
        return float(self.raw["numerics"]["duration_days"])

    @property
    def timestep_days(self) -> float:
        return recommended_timestep(steps_per_inner_orbit=self.steps_per_inner_orbit)

    # ---- observations -----------------------------------------------------------------
    @property
    def noise_scale(self) -> float:
        return float(self.raw["observations"]["noise_scale"])

    @property
    def noise_seed(self) -> int:
        return int(self.raw["observations"]["noise_seed"])

    # ---- inference --------------------------------------------------------------------
    @property
    def inference(self) -> dict:
        return dict(self.raw["inference"])

    # ---- ground truth (NEVER visible to the inference) ---------------------------------
    @property
    def planet_x(self) -> PlanetXParameters:
        p = self.raw["planet_x_ground_truth"]
        return PlanetXParameters(
            mass_earth=p["mass_earth"],
            semi_major_axis_au=p["semi_major_axis_au"],
            eccentricity=p.get("eccentricity", 0.0),
            mean_anomaly=p["mean_anomaly"],
            inclination_deg=p["inclination_deg"],
            argument_of_periapsis=p.get("argument_of_periapsis", 0.0),
            longitude_of_ascending_node_deg=p.get("longitude_of_ascending_node_deg", 0.0),
            radius_earth=p.get("radius_earth", 4.0),
        )

    def to_dict(self, include_ground_truth: bool = True) -> dict:
        out = {k: v for k, v in self.raw.items() if k != "planet_x_ground_truth"}
        out["timestep_days"] = self.timestep_days
        if include_ground_truth:
            out["planet_x_ground_truth"] = self.raw["planet_x_ground_truth"]
        return out

    def save(self, path=None) -> None:
        Path(path or CONFIG_PATH).write_text(json.dumps(self.raw, indent=2), encoding="utf-8")


def default_raw() -> dict:
    """The configuration written by scripts/select_planet_x.py (planet_x_ground_truth is filled
    from the selection; every other value is fixed here)."""
    return {
        "description": (
            "Complete configuration of the Invisible Planet experiment. Planet X is SYNTHETIC; "
            "this is not a claim about the real Kepler-90 system."
        ),
        "version": 2,
        "numerics": {
            "steps_per_inner_orbit": 1000,
            "epoch_bjd": DEFAULT_EPOCH_BJD,
            "duration_days": DEFAULT_DURATION_DAYS,
            "integrator": "velocity Verlet (kick-drift-kick), fixed step, f64",
        },
        "observations": {
            "pattern_file": "data/kepler90_observed_transits.json",
            "noise_scale": 1.0,
            "noise_seed": 20260919,
        },
        "inference": {
            "fap_threshold": 0.01,
            "n_peaks": 6,
            "n_null_trials": 20000,
            "n_walkers": 32,
            "n_steps": 400,
            "sampler_seed": 7,
        },
        "blind_recovery": {"n_cases": 20, "master_seed": 20260919},
    }


def load_config(path=None) -> ExperimentConfig:
    target = Path(path or CONFIG_PATH)
    if not target.exists():
        raise FileNotFoundError(
            f"{target} does not exist. Run scripts/select_planet_x.py to create it."
        )
    return ExperimentConfig(raw=json.loads(target.read_text(encoding="utf-8")))
