"""
Forward model for the inverse problem.

Phase 10. Maps candidate Planet X parameters to predicted transit times, using
the SAME physics as the experiment: the same integrator, the same force law, the
same transit detector. There is no separate "inference physics".

This module may import the system builders and the engine. It must NOT import
the experiment configuration's ground truth; see inference/search.py.
"""

from __future__ import annotations

import numpy as np

from ..observations.transits import observe_transits_batch
from ..observations.ttv import fit_linear_ephemeris
from ..systems.kepler90 import build_kepler90, load_reference, recommended_timestep
from ..systems.planet_x import PlanetXParameters, build_system_with_planet_x


class ForwardModel:
    """Predicts transit times for candidate parameter vectors.

    Candidates are evaluated in BATCHES through the ensemble dimension of the
    N-body engine, which is what makes a search over thousands of candidates
    practical on a laptop.
    """

    def __init__(
        self,
        planet_letters: list,
        observation_days: float,
        reference: dict | None = None,
        steps_per_inner_orbit: int = 1000,
        batch_size: int = 32,
    ):
        self.reference = reference if reference is not None else load_reference()
        self.planet_letters = list(planet_letters)
        self.planet_names = [f"Kepler-90 {letter}" for letter in self.planet_letters]
        self.observation_days = float(observation_days)
        self.timestep = recommended_timestep(self.reference, steps_per_inner_orbit)
        self.batch_size = int(batch_size)
        self.n_evaluations = 0

        self._control = build_kepler90(reference=self.reference)

    def control_prediction(self) -> dict:
        """Transit times with NO extra planet — the null hypothesis."""
        from ..observations.transits import observe_transits

        series = observe_transits(
            self._control, self.observation_days, self.timestep, tracked=self.planet_names
        )
        self.n_evaluations += 1
        return {
            letter: series.for_planet(name)
            for letter, name in zip(self.planet_letters, self.planet_names)
        }

    def predict(self, candidates: list) -> list:
        """Predicted transit times for each candidate PlanetXParameters."""
        predictions = []
        for start in range(0, len(candidates), self.batch_size):
            chunk = candidates[start : start + self.batch_size]
            systems = [
                build_system_with_planet_x(p, reference=self.reference) for p in chunk
            ]
            series_list = observe_transits_batch(
                systems, self.observation_days, self.timestep, tracked=self.planet_names
            )
            for series in series_list:
                predictions.append(
                    {
                        letter: series.for_planet(name)
                        for letter, name in zip(self.planet_letters, self.planet_names)
                    }
                )
            self.n_evaluations += len(chunk)
        return predictions


def residual_signature(times_by_planet: dict, min_transits: int = 3) -> dict:
    """Reduce transit times to O-C residuals keyed by epoch.

    The comparison between model and data happens in RESIDUAL space, not in
    absolute transit times. This is essential and not a convenience:

      * A real observer does not know the true period; it is always fitted.
      * Fitting the ephemeris removes the constant frequency offset of the
        symplectic integrator (docs/NUMERICAL_METHODS.md), which would otherwise
        dominate the comparison while carrying no information about Planet X.
      * The TTV signal IS the residual structure.
    """
    out = {}
    for letter, times in times_by_planet.items():
        times = np.asarray(times, dtype=np.float64)
        if len(times) < min_transits:
            continue
        fit = fit_linear_ephemeris(times)
        out[letter] = {
            "epochs": fit.epochs,
            "residuals": fit.residuals,
            "period": fit.period,
            "t0": fit.t0,
        }
    return out
