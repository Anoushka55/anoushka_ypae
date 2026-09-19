"""
The real observing pattern: which transits Kepler measured, and how precisely.

Loads data/kepler90_observed_transits.json (built by scripts/build_observing_pattern.py
from published per-transit measurements; see docs/OBSERVING_PATTERN.md).

Only the planets with PUBLISHED individual transit times are probes: d, e, g, h.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..constants import SECONDS_PER_DAY

PATTERN_PATH = Path(__file__).resolve().parents[2] / "data" / "kepler90_observed_transits.json"


@dataclass(frozen=True)
class ObservedPlanet:
    """The real measured transits of one planet."""

    letter: str
    source: str
    epochs: np.ndarray            # integer transit numbers (see `t0_bjd`, `period_days`)
    measured_bjd: np.ndarray      # the REAL measured mid-transit times
    sigma_days: np.ndarray        # published 1-sigma per transit
    t0_bjd: float                 # real weighted linear ephemeris, used only to number epochs
    period_days: float

    @property
    def n_transits(self) -> int:
        return len(self.epochs)

    def epoch_of(self, time_bjd: np.ndarray) -> np.ndarray:
        """Integer epoch of arbitrary transit times, by the real ephemeris."""
        return np.round((np.asarray(time_bjd) - self.t0_bjd) / self.period_days).astype(np.int64)


@dataclass(frozen=True)
class ObservingPattern:
    planets: dict          # every planet with published individual transit times
    window_bjd: tuple
    probes: tuple = ()     # the subset used to look for an additional body

    @property
    def probe_letters(self) -> list:
        return list(self.probes)

    def __getitem__(self, letter: str) -> ObservedPlanet:
        return self.planets[letter]

    @property
    def window_days(self) -> float:
        return float(self.window_bjd[1] - self.window_bjd[0])


def load_pattern(path: Path | None = None) -> ObservingPattern:
    raw = json.loads((path or PATTERN_PATH).read_text(encoding="utf-8"))
    planets = {}
    for letter in raw["timing_planets"]:
        r = raw["planets"][letter]
        planets[letter] = ObservedPlanet(
            letter=letter,
            source=r["source"],
            epochs=np.asarray(r["epochs"], dtype=np.int64),
            measured_bjd=np.asarray(r["measured_bjd"], dtype=np.float64),
            sigma_days=np.asarray(r["sigma_days"], dtype=np.float64),
            t0_bjd=float(r["ephemeris"]["t0_bjd"]),
            period_days=float(r["ephemeris"]["period_days"]),
        )
    return ObservingPattern(planets=planets, window_bjd=tuple(raw["kepler_window_bjd"]),
                            probes=tuple(raw["probe_planets"]))


def sigma_seconds(observed: ObservedPlanet) -> np.ndarray:
    return observed.sigma_days * SECONDS_PER_DAY


def select_at_epochs(sim_times_bjd: np.ndarray, observed: ObservedPlanet) -> tuple:
    """Simulated transit times at the REAL observed epochs.

    Returns (times_bjd, present) where `present[j]` is False if the simulation has no
    transit at observed epoch j (e.g. it falls outside the simulated window).
    """
    sim = np.asarray(sim_times_bjd, dtype=np.float64)
    sim_epochs = observed.epoch_of(sim)
    lookup = {int(n): t for n, t in zip(sim_epochs, sim)}
    out = np.full(observed.n_transits, np.nan)
    present = np.zeros(observed.n_transits, dtype=bool)
    for j, n in enumerate(observed.epochs):
        if int(n) in lookup:
            out[j] = lookup[int(n)]
            present[j] = True
    return out, present
