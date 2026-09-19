"""
The observer dataset - the ONLY channel between the experiment and the solver.

=============================================================================
GROUND-TRUTH ISOLATION
=============================================================================
The inference must never see Planet X's true parameters. In a single Python process that is
easy to violate by accident: a shared config object, a cached system, an innocent import.

The defence is structural rather than a matter of discipline:

  1. `ObserverDataset` physically CANNOT hold Planet X's parameters - it has no field for
     them, and `from_simulation` never receives them.
  2. The dataset is written to disk, and the inference reads that file.
  3. `assert_no_ground_truth_leakage` and `assert_parameters_absent` walk the serialised
     dataset for any trace of the synthetic body, and are run as tests.

What the observer legitimately knows is what a real astronomer would: the star, the
measured transit times of the planets, their epochs and their uncertainties. It does not
know how many unseen bodies exist, or whether any do.
=============================================================================
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..constants import SECONDS_PER_DAY
from .pattern import ObservingPattern, select_at_epochs

#: Substrings that must never appear in a serialised observer dataset. The word
#: "synthetic" is deliberately NOT forbidden: the dataset should say plainly that it
#: contains simulated rather than real observations. Leakage is instead caught by name
#: and, in `assert_parameters_absent`, by VALUE.
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
    """What the observer is allowed to know."""

    star_mass_solar: float
    star_radius_solar: float
    planet_letters: list
    epochs: dict                    # letter -> integer transit numbers
    transit_times_bjd: dict         # letter -> noisy measured times [BJD]
    timing_uncertainty_days: dict   # letter -> per-transit 1-sigma [days]
    reference_ephemeris: dict       # letter -> {"t0_bjd", "period_days"} used to number epochs
    epoch_bjd: float                # start of the simulated observing window
    observation_days: float
    noise_model: dict
    provenance: dict = field(default_factory=dict)

    @classmethod
    def from_simulation(
        cls,
        sim_times_bjd: dict,
        pattern: ObservingPattern,
        reference: dict,
        noise_model,
        epoch_bjd: float,
        observation_days: float,
        provenance: dict | None = None,
    ) -> "ObserverDataset":
        """Sample a simulated transit series at the REAL observed epochs and add noise.

        `sim_times_bjd` maps planet letter -> ALL simulated transit times [BJD]. Planet X's
        parameters are not an argument of this function and cannot enter the result.
        """
        epochs, times, sigmas, ephemeris = {}, {}, {}, {}
        for letter in pattern.probe_letters:
            observed = pattern[letter]
            simulated, present = select_at_epochs(sim_times_bjd[letter], observed)
            if not present.all():
                raise ValueError(
                    f"planet {letter}: simulation lacks {int((~present).sum())} of the "
                    f"{observed.n_transits} observed epochs"
                )
            epochs[letter] = observed.epochs.copy()
            times[letter] = noise_model.apply(letter, simulated)
            sigmas[letter] = noise_model.sigma_days[letter].copy()
            ephemeris[letter] = {"t0_bjd": observed.t0_bjd, "period_days": observed.period_days}

        return cls(
            star_mass_solar=reference["star"]["mass_solar"]["value"],
            star_radius_solar=reference["star"]["radius_solar"]["value"],
            planet_letters=list(pattern.probe_letters),
            epochs=epochs,
            transit_times_bjd=times,
            timing_uncertainty_days=sigmas,
            reference_ephemeris=ephemeris,
            epoch_bjd=float(epoch_bjd),
            observation_days=float(observation_days),
            noise_model=noise_model.describe(),
            provenance=provenance or {},
        )

    def to_dict(self) -> dict:
        def listify(mapping):
            return {k: np.asarray(v).tolist() for k, v in mapping.items()}

        return {
            "description": (
                "Synthetic transit-timing observations of a Kepler-90-inspired system, sampled at "
                "the epochs Kepler actually measured. The observer does not know how many bodies "
                "produced them."
            ),
            "star_mass_solar": self.star_mass_solar,
            "star_radius_solar": self.star_radius_solar,
            "planet_letters": self.planet_letters,
            "epochs": listify(self.epochs),
            "transit_times_bjd": listify(self.transit_times_bjd),
            "timing_uncertainty_days": listify(self.timing_uncertainty_days),
            "timing_uncertainty_seconds": {
                k: (np.asarray(v) * SECONDS_PER_DAY).tolist()
                for k, v in self.timing_uncertainty_days.items()
            },
            "reference_ephemeris": self.reference_ephemeris,
            "epoch_bjd": self.epoch_bjd,
            "observation_days": self.observation_days,
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
            epochs={k: np.asarray(v, dtype=np.int64) for k, v in raw["epochs"].items()},
            transit_times_bjd={k: np.asarray(v) for k, v in raw["transit_times_bjd"].items()},
            timing_uncertainty_days={
                k: np.asarray(v) for k, v in raw["timing_uncertainty_days"].items()
            },
            reference_ephemeris=raw["reference_ephemeris"],
            epoch_bjd=raw["epoch_bjd"],
            observation_days=raw["observation_days"],
            noise_model=raw["noise_model"],
            provenance=raw.get("provenance", {}),
        )

    def window(self, max_days: float, min_transits: int = 4) -> "ObserverDataset":
        """The dataset an observer would have had if the campaign had stopped `max_days` after the
        epoch: only transits up to then, and only planets left with at least `min_transits`
        (the linear ephemeris removes two parameters per planet)."""
        letters, epochs, times, sigmas = [], {}, {}, {}
        for k in self.planet_letters:
            keep = (np.asarray(self.transit_times_bjd[k]) - self.epoch_bjd) <= max_days
            if int(keep.sum()) >= min_transits:
                letters.append(k)
                epochs[k] = np.asarray(self.epochs[k])[keep]
                times[k] = np.asarray(self.transit_times_bjd[k])[keep]
                sigmas[k] = np.asarray(self.timing_uncertainty_days[k])[keep]
        return ObserverDataset(
            star_mass_solar=self.star_mass_solar, star_radius_solar=self.star_radius_solar,
            planet_letters=letters, epochs=epochs, transit_times_bjd=times,
            timing_uncertainty_days=sigmas,
            reference_ephemeris={k: self.reference_ephemeris[k] for k in letters},
            epoch_bjd=self.epoch_bjd, observation_days=float(min(max_days, self.observation_days)),
            noise_model=self.noise_model, provenance=dict(self.provenance),
        )

    def times_for(self, letter: str) -> np.ndarray:
        return np.asarray(self.transit_times_bjd[letter], dtype=np.float64)

    def counts(self) -> dict:
        return {k: len(v) for k, v in self.transit_times_bjd.items()}


def assert_no_ground_truth_leakage(dataset: ObserverDataset) -> None:
    """Fail loudly if anything about the synthetic body reached the observer.

    Scans the SERIALISED form, because that is what the inference actually consumes.
    """
    blob = json.dumps(dataset.to_dict()).lower()
    for token in FORBIDDEN_TOKENS:
        if token in blob:
            raise AssertionError(
                f"observer dataset contains the forbidden token {token!r} - "
                "ground truth has leaked into the inference input"
            )


def assert_parameters_absent(dataset: ObserverDataset, parameters) -> None:
    """Fail if any of the hidden body's PARAMETER VALUES appear in the dataset.

    A name-based scan can be defeated by a renamed field, so this checks the numbers
    themselves. Called from code that legitimately knows the truth - the experiment scripts
    and the tests - never from the inference.
    """
    suspicious = {
        "mass_earth": parameters.mass_earth,
        "semi_major_axis_au": parameters.semi_major_axis_au,
        "mean_anomaly": parameters.mean_anomaly,
        "inclination_deg": parameters.inclination_deg,
        "eccentricity": parameters.eccentricity,
    }

    def walk(node, path="root"):
        """Compare parsed NUMBERS, not rendered substrings (a substring scan would match
        the literal '40' inside any transit time). A measured time coinciding with a
        parameter value to within 1e-9 is not credible."""
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
                        f"hidden body's {name} - ground truth has leaked by value"
                    )

    walk(json.loads(json.dumps(dataset.to_dict())))
