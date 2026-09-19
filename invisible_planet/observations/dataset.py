"""
The observer dataset — the ONLY channel between the experiment and the solver.

Phase 9 deliverable.

=============================================================================
GROUND-TRUTH ISOLATION
=============================================================================
The inference must never see Planet X's true parameters. In a single Python
process that is easy to violate by accident: a shared config object, a cached
system, an innocent import.

The defence is structural rather than a matter of discipline:

  1. `ObserverDataset` physically CANNOT hold Planet X's parameters — it has no
     field for them, and `from_simulation` never receives them.
  2. The dataset is written to disk, and the inference reads that file. The
     serialised form is the contract, and it can be inspected by hand.
  3. `assert_no_ground_truth_leakage` walks the serialised dataset looking for
     any trace of the synthetic body, and is run as a test.

What the observer legitimately knows is exactly what a real astronomer would:
the star, the transiting planets and their measured transit times with
uncertainties. It does not know how many unseen bodies exist, or whether any do.
=============================================================================
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..constants import SECONDS_PER_DAY

#: Substrings that must never appear in a serialised observer dataset.
#:
#: These name the hidden body or its parameters. The word "synthetic" is
#: deliberately NOT forbidden: the dataset should say plainly that it contains
#: simulated rather than real observations, and suppressing that would be worse
#: than the leak it guards against. Likewise "mass_earth" — a generic key name —
#: would produce false positives. Leakage is instead caught by name and, in
#: `assert_parameters_absent`, by VALUE.
FORBIDDEN_TOKENS = (
    "planet_x",
    "planet x",
    "ground_truth",
    "ground truth",
    "true_mass",
    "hidden_body",
)


@dataclass
class ObserverDataset:
    """What the observer is allowed to know.

    Deliberately minimal. There is no field for the number of unseen bodies, for
    Planet X's parameters, or for anything derived from them.
    """

    star_mass_solar: float
    star_radius_solar: float
    planet_letters: list
    transit_times: dict  # letter -> noisy transit times [days since epoch]
    timing_uncertainty_days: dict  # letter -> 1-sigma
    epoch_bjd: float
    observation_days: float
    noise_model: dict
    provenance: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    @classmethod
    def from_simulation(
        cls,
        series,
        reference: dict,
        planet_letters: list,
        noise_model,
        observation_days: float,
        epoch_bjd: float,
        provenance: dict | None = None,
    ) -> "ObserverDataset":
        """Build the dataset from a simulated transit series.

        `series` carries transit times only. Planet X's parameters are not an
        argument of this function and cannot enter the result.
        """
        noisy, sigmas = {}, {}
        for letter in planet_letters:
            name = f"Kepler-90 {letter}"
            true_times = series.for_planet(name)
            noisy[letter] = noise_model.apply(letter, true_times).tolist()
            sigmas[letter] = noise_model.sigma_days[letter]

        return cls(
            star_mass_solar=reference["star"]["mass_solar"]["value"],
            star_radius_solar=reference["star"]["radius_solar"]["value"],
            planet_letters=list(planet_letters),
            transit_times=noisy,
            timing_uncertainty_days=sigmas,
            epoch_bjd=epoch_bjd,
            observation_days=observation_days,
            noise_model=noise_model.describe(),
            provenance=provenance or {},
        )

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "description": (
                "Synthetic transit-timing observations of a Kepler-90-inspired "
                "system. The observer does not know how many bodies produced them."
            ),
            "star_mass_solar": self.star_mass_solar,
            "star_radius_solar": self.star_radius_solar,
            "planet_letters": self.planet_letters,
            "epoch_bjd": self.epoch_bjd,
            "observation_days": self.observation_days,
            # A reloaded dataset holds numpy arrays; normalise to lists so the
            # serialised form round-trips, and so the leakage scanners (which
            # work on the serialised form) never fail for the wrong reason.
            "transit_times_days": {
                k: (v.tolist() if isinstance(v, np.ndarray) else list(v))
                for k, v in self.transit_times.items()
            },
            "timing_uncertainty_days": self.timing_uncertainty_days,
            "timing_uncertainty_seconds": {
                k: v * SECONDS_PER_DAY for k, v in self.timing_uncertainty_days.items()
            },
            "noise_model": self.noise_model,
            "provenance": self.provenance,
        }

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "ObserverDataset":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            star_mass_solar=raw["star_mass_solar"],
            star_radius_solar=raw["star_radius_solar"],
            planet_letters=raw["planet_letters"],
            transit_times={k: np.asarray(v) for k, v in raw["transit_times_days"].items()},
            timing_uncertainty_days=raw["timing_uncertainty_days"],
            epoch_bjd=raw["epoch_bjd"],
            observation_days=raw["observation_days"],
            noise_model=raw["noise_model"],
            provenance=raw.get("provenance", {}),
        )

    def times_for(self, letter: str) -> np.ndarray:
        return np.asarray(self.transit_times[letter], dtype=np.float64)

    def counts(self) -> dict:
        return {k: len(v) for k, v in self.transit_times.items()}


def assert_no_ground_truth_leakage(dataset: ObserverDataset) -> None:
    """Fail loudly if anything about the synthetic body reached the observer.

    Scans the SERIALISED form, because that is what the inference actually
    consumes. Checking the object's attributes would miss anything smuggled
    through `provenance`.
    """
    blob = json.dumps(dataset.to_dict()).lower()
    for token in FORBIDDEN_TOKENS:
        if token in blob:
            raise AssertionError(
                f"observer dataset contains the forbidden token {token!r} — "
                "ground truth has leaked into the inference input"
            )


def assert_parameters_absent(dataset: ObserverDataset, parameters) -> None:
    """Fail if any of the hidden body's PARAMETER VALUES appear in the dataset.

    A name-based scan can be defeated by a renamed field, so this checks the
    numbers themselves. Called from code that legitimately knows the truth — the
    experiment script and the tests — never from the inference.
    """
    suspicious = {
        "mass_earth": parameters.mass_earth,
        "semi_major_axis_au": parameters.semi_major_axis_au,
        "mean_anomaly": parameters.mean_anomaly,
        "inclination_deg": parameters.inclination_deg,
    }

    def walk(node, path="root"):
        """Compare parsed NUMBERS, not rendered substrings.

        A substring scan is useless here: the literal "40" appears inside
        perfectly innocent transit times such as 140.27 and 3.409. Exact
        numerical comparison of parsed values has no such false positives,
        because a measured transit time coinciding with a parameter value to
        within 1e-9 is not credible.
        """
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{path}.{key}")
        elif isinstance(node, (list, tuple)):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            for name, target in suspicious.items():
                if target != 0.0 and abs(node - target) <= 1e-9 * max(1.0, abs(target)):
                    raise AssertionError(
                        f"observer dataset field {path} holds {node}, which equals the "
                        f"hidden body's {name} — ground truth has leaked by value"
                    )

    walk(json.loads(json.dumps(dataset.to_dict())))
