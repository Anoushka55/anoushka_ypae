"""
Measurement noise model.

    observation = true_value + noise

THE NOISE IS NOT INVENTED. Each transit that Kepler actually measured is given the 1-sigma
timing uncertainty that was PUBLISHED for that transit (Holczer et al. 2016 for planets d
and e; Liang et al. 2021 for g and h). See docs/OBSERVING_PATTERN.md, which also compares
these against the analytic photon-noise limit of Carter et al. (2008): the published errors
are 2.1-2.9x that limit.

An earlier version of this module used each planet's transit-EPOCH (T0) uncertainty as a
per-transit error. That was wrong: T0 comes from a fit over many transits and is far smaller
than a single-transit error, so the synthetic data were unrealistically precise.

The noise is drawn independently and Gaussian for each transit. Real timing errors are
close to Gaussian for well-sampled transits but are not strictly independent (shared stellar
noise, detrending choices); that simplification is stated in the documentation.
"""

from __future__ import annotations

import numpy as np

from ..constants import SECONDS_PER_DAY


class TimingNoiseModel:
    """Adds Gaussian timing noise with a per-transit sigma taken from the real data."""

    def __init__(self, pattern, seed: int, scale: float = 1.0):
        self.sigma_days = {k: pattern[k].sigma_days * scale for k in pattern.probe_letters}
        self.seed = int(seed)
        self.scale = float(scale)
        self._rng = np.random.default_rng(seed)
        self._source = {k: pattern[k].source for k in pattern.probe_letters}

    def sigma_seconds(self, letter: str) -> np.ndarray:
        return self.sigma_days[letter] * SECONDS_PER_DAY

    def apply(self, letter: str, times: np.ndarray) -> np.ndarray:
        """Return noisy transit times for one planet (one draw per transit)."""
        sigma = self.sigma_days[letter]
        times = np.asarray(times, dtype=np.float64)
        if len(times) != len(sigma):
            raise ValueError(
                f"planet {letter}: {len(times)} transit times but {len(sigma)} published uncertainties"
            )
        return times + self._rng.normal(0.0, 1.0, size=len(times)) * sigma

    def describe(self) -> dict:
        return {
            "model": "independent Gaussian per transit, per-transit sigma as published",
            "seed": self.seed,
            "scale": self.scale,
            "median_sigma_seconds": {
                k: float(np.median(v) * SECONDS_PER_DAY) for k, v in self.sigma_days.items()
            },
            "sources": self._source,
        }
