"""
Forward model for the inverse problem.

Maps candidate Planet X parameters to predicted transit times AT THE OBSERVED EPOCHS, using
the SAME physics as the experiment: the same integrator, the same force law, the same transit
detector, and the same visible-planet initial conditions (which are public: they are fitted
to the real published transit times, see systems/matching.py). There is no separate
"inference physics".

This module may import the system builders and the engine. It must NOT import the experiment
configuration's ground truth; see inference/search.py.
"""

from __future__ import annotations

import numpy as np

from ..observations.transits import observe_transits_batch
from ..systems.kepler90 import build_kepler90, load_reference, recommended_timestep
from ..systems.planet_x import PlanetXParameters, build_system_with_planet_x


class ForwardModel:
    """Predicts probe-planet transit times for candidate parameter sets.

    Candidates are evaluated in BATCHES through the ensemble dimension of the N-body engine.
    """

    def __init__(
        self,
        dataset,
        reference: dict | None = None,
        steps_per_inner_orbit: int = 1000,
        batch_size: int = 48,
    ):
        self.reference = reference if reference is not None else load_reference()
        self.dataset = dataset
        self.letters = list(dataset.planet_letters)
        self.names = [f"Kepler-90 {k}" for k in self.letters]
        self.epoch_bjd = float(dataset.epoch_bjd)
        self.duration_days = float(dataset.observation_days)
        self.timestep = recommended_timestep(self.reference, steps_per_inner_orbit)
        self.batch_size = int(batch_size)
        self.n_evaluations = 0
        self._control_cache = None
        self.control_system = build_kepler90(reference=self.reference, epoch_bjd=self.epoch_bjd)

    # ------------------------------------------------------------------
    def _at_dataset_epochs(self, series) -> dict:
        """Simulated transit times [BJD] at the dataset's epochs; NaN where absent."""
        out = {}
        for letter, name in zip(self.letters, self.names):
            t = self.epoch_bjd + series.for_planet(name)
            ephemeris = self.dataset.reference_ephemeris[letter]
            sim_epochs = np.round((t - ephemeris["t0_bjd"]) / ephemeris["period_days"]).astype(np.int64)
            lookup = {int(n): float(x) for n, x in zip(sim_epochs, t)}
            out[letter] = np.array([lookup.get(int(n), np.nan) for n in self.dataset.epochs[letter]])
        return out

    def control(self) -> dict:
        """Transit times with NO extra planet - the null hypothesis."""
        if self._control_cache is None:
            series = observe_transits_batch(
                [self.control_system], self.duration_days, self.timestep, tracked=self.names
            )[0]
            self._control_cache = self._at_dataset_epochs(series)
            self.n_evaluations += 1
        return self._control_cache

    def predict(self, candidates: list) -> list:
        """Predicted probe transit times for each candidate PlanetXParameters."""
        out = []
        for start in range(0, len(candidates), self.batch_size):
            chunk = candidates[start : start + self.batch_size]
            systems = [
                build_system_with_planet_x(p, reference=self.reference, epoch_bjd=self.epoch_bjd)
                for p in chunk
            ]
            series_list = observe_transits_batch(
                systems, self.duration_days, self.timestep, tracked=self.names, max_transits=512
            )
            out.extend(self._at_dataset_epochs(s) for s in series_list)
            self.n_evaluations += len(chunk)
        return out
