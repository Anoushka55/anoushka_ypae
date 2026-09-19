"""
Measurement noise model.

Phase 9 deliverable.

    observation = true_value + noise

THE NOISE IS NOT INVENTED. Each planet's per-transit timing uncertainty is taken
from the published transit-epoch uncertainty for that planet in the reference
dataset, so the synthetic data inherits the real system's precision structure:
planet h is measured to ~71 s, planet f only to ~1210 s. A single made-up
constant would misrepresent how well this system is actually known, and would
make the inference look either easier or harder than reality.

The noise is Gaussian and independent between transits. Real transit timing
errors are close to Gaussian when a transit is well sampled, but they are not
strictly independent (shared systematics, stellar activity, detrending choices).
That simplification is stated in docs/ASSUMPTIONS.md rather than buried here.
"""

from __future__ import annotations

import numpy as np

from ..constants import SECONDS_PER_DAY


def published_timing_uncertainties(reference: dict, planet_letters: list) -> dict:
    """Per-planet 1-sigma transit-timing uncertainty [days], from the dataset."""
    out = {}
    for letter in planet_letters:
        record = reference["planets"][letter]["transit_epoch_bjd"]
        uncertainty = record.get("uncertainty")
        if uncertainty is None:
            raise ValueError(
                f"planet {letter} has no published transit-epoch uncertainty; "
                "a noise level cannot be invented for it"
            )
        out[letter] = float(uncertainty)
    return out


class TimingNoiseModel:
    """Adds Gaussian timing noise with per-planet, data-derived amplitudes."""

    def __init__(self, sigma_days: dict, seed: int, scale: float = 1.0):
        self.sigma_days = {k: v * scale for k, v in sigma_days.items()}
        self.seed = int(seed)
        self.scale = float(scale)
        self._rng = np.random.default_rng(seed)

    def sigma_seconds(self, key: str) -> float:
        return self.sigma_days[key] * SECONDS_PER_DAY

    def apply(self, key: str, times: np.ndarray) -> np.ndarray:
        """Return noisy transit times for one planet."""
        sigma = self.sigma_days[key]
        return np.asarray(times) + self._rng.normal(0.0, sigma, size=len(times))

    def describe(self) -> dict:
        return {
            "model": "independent Gaussian per transit",
            "seed": self.seed,
            "scale": self.scale,
            "sigma_seconds": {k: v * SECONDS_PER_DAY for k, v in self.sigma_days.items()},
            "source": "published transit-epoch uncertainties from the reference dataset",
        }
