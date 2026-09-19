"""
The timing likelihood: chi-squared with each planet's linear ephemeris profiled out.

THE STATISTIC
-------------
The observer measures transit times t_j (epoch n_j, 1-sigma s_j) of each probe planet. A
candidate model predicts times m_j. Because the observer never knows a planet's true mean
period or phase, each planet is allowed its own straight line in epoch number:

    chi2 = sum over planets  min over (a, b)  sum_j [ (t_j - m_j - a - b n_j) / s_j ]^2

This is a linear least-squares problem, so it can be written with a projector. With
whitened residuals r~ = (t - m)/s and the whitened design A~ = [1, n - mean]/s,

    chi2 = | P r~ |^2,        P = I - A~ (A~^T A~)^-1 A~^T,

and P is fixed by the data (epochs and uncertainties), so it is computed once.

WHY THIS AND NOT "FIT EACH SEQUENCE SEPARATELY"
-----------------------------------------------
An earlier version fitted a linear ephemeris to the data and another to the model and
compared the two residual series. That is only equivalent when both use identical epochs
and weights. With real, gappy data with unequal per-transit uncertainties they differ, so
the profiled statistic above is the correct one. It also has 2 fewer degrees of freedom per
planet, which enters the significance calculation.
"""

from __future__ import annotations

import numpy as np

INVALID_CHI2 = 1e12   # returned when a model lacks one of the observed epochs


class TimingLikelihood:
    """Profile chi-squared of a model against an ObserverDataset."""

    def __init__(self, dataset):
        self.letters = list(dataset.planet_letters)
        self.epochs = {k: np.asarray(dataset.epochs[k], dtype=np.int64) for k in self.letters}
        self.data = {k: np.asarray(dataset.transit_times_bjd[k], dtype=np.float64) for k in self.letters}
        self.sigma = {k: np.asarray(dataset.timing_uncertainty_days[k], dtype=np.float64) for k in self.letters}

        self._projector = {}
        for k in self.letters:
            n = self.epochs[k].astype(np.float64)
            s = self.sigma[k]
            w = 1.0 / s**2
            nbar = float(np.sum(w * n) / np.sum(w))
            design = np.vstack([np.ones_like(n), n - nbar]).T / s[:, None]
            q, _ = np.linalg.qr(design)
            self._projector[k] = np.eye(len(n)) - q @ q.T

        self.n_points = int(sum(len(v) for v in self.data.values()))
        self.n_linear_terms = 2 * len(self.letters)
        self.dof_data = self.n_points - self.n_linear_terms

    # ------------------------------------------------------------------
    def _whitened(self, letter: str, times: np.ndarray) -> np.ndarray:
        return (self.data[letter] - times) / self.sigma[letter]

    def chi2(self, model_times: dict) -> float:
        """Profile chi2 of `model_times` (BJD at the dataset epochs, one array per planet)."""
        total = 0.0
        for k in self.letters:
            m = np.asarray(model_times[k], dtype=np.float64)
            if not np.all(np.isfinite(m)):
                return INVALID_CHI2
            r = self._projector[k] @ self._whitened(k, m)
            total += float(r @ r)
        return total

    def chi2_per_planet(self, model_times: dict) -> dict:
        out = {}
        for k in self.letters:
            m = np.asarray(model_times[k], dtype=np.float64)
            r = self._projector[k] @ self._whitened(k, m)
            out[k] = float(r @ r)
        return out

    # ------------------------------------------------------------------
    # vectors for the linearised (periodogram) analysis
    # ------------------------------------------------------------------
    def data_vector(self, control_times: dict) -> np.ndarray:
        """P (t_data - t_control) / sigma, concatenated over planets: the part of the data that
        the visible-planet model does NOT already explain."""
        return np.concatenate(
            [self._projector[k] @ self._whitened(k, np.asarray(control_times[k])) for k in self.letters]
        )

    def model_vector(self, model_times: dict, control_times: dict) -> np.ndarray:
        """P (t_model - t_control) / sigma: the timing perturbation a candidate adds."""
        return np.concatenate(
            [
                self._projector[k]
                @ ((np.asarray(model_times[k]) - np.asarray(control_times[k])) / self.sigma[k])
                for k in self.letters
            ]
        )
