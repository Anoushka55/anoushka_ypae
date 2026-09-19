"""
Transit timing analysis: linear ephemerides and O-C residuals.

Phase 6.2 deliverable. Pure NumPy. This module knows nothing about the
simulation that produced the transit times — it works identically on simulated
times and on real published ones, which is the point.

THE MEASUREMENT
---------------
For a strictly Keplerian, unperturbed orbit, transits are exactly periodic:

    T_n = T_0 + n P

Fitting that straight line to the observed transit times and subtracting it
leaves the "observed minus calculated" residual:

    O-C = T_observed - (T_0 + n P)

For an isolated planet the residuals are zero. Gravitational perturbations from
other bodies make the residuals non-zero and STRUCTURED — that structure is the
transit timing variation (TTV), and it is the only signal this project uses to
detect the unseen planet.

WHY THE FIT MATTERS NUMERICALLY
--------------------------------
Fitting P rather than assuming it also removes any constant frequency offset,
including the small O(dt^2) frequency shift inherent to a symplectic integrator
(see docs/NUMERICAL_METHODS.md). This is not a trick to hide numerical error: it
is exactly what is done with real data, where the true period is never known a
priori and is always fitted.

THREE ERROR SOURCES, WHICH MUST NOT BE CONFLATED
-------------------------------------------------
  numerical      : integration error in the simulated transit time
  ephemeris-fit  : uncertainty in the fitted T_0 and P from finite sampling
  physical TTV   : the real dynamical signal

`fit_linear_ephemeris` reports the formal uncertainties of T_0 and P so the
second can be quantified rather than assumed negligible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..constants import SECONDS_PER_DAY


@dataclass
class LinearEphemeris:
    """Best-fit linear ephemeris and its O-C residuals."""

    t0: float  # days
    period: float  # days
    t0_uncertainty: float  # days
    period_uncertainty: float  # days
    epochs: np.ndarray  # integer transit numbers
    times: np.ndarray  # observed transit times [days]
    residuals: np.ndarray  # O-C [days]

    @property
    def residuals_seconds(self) -> np.ndarray:
        return self.residuals * SECONDS_PER_DAY

    @property
    def residuals_minutes(self) -> np.ndarray:
        return self.residuals * 1440.0

    @property
    def rms_seconds(self) -> float:
        return float(np.sqrt(np.mean(self.residuals**2)) * SECONDS_PER_DAY)

    @property
    def peak_to_peak_minutes(self) -> float:
        return float((self.residuals.max() - self.residuals.min()) * 1440.0)

    @property
    def n_transits(self) -> int:
        return len(self.times)


def assign_epochs(times: np.ndarray, period_guess: float | None = None) -> np.ndarray:
    """Assign integer transit numbers, tolerating gaps in the sequence.

    Real transit sequences have missing entries (data gaps, instrument
    downtime), so epochs are NOT simply 0,1,2,... They are recovered by rounding
    (T - T_first)/P, which is robust as long as the period estimate is good
    enough that no transit is misassigned by a whole epoch.
    """
    times = np.asarray(times, dtype=np.float64)
    if len(times) < 2:
        return np.zeros(len(times), dtype=np.int64)

    if period_guess is None:
        # median consecutive spacing is robust against gaps of several epochs
        # only if gaps are the minority; the median difference approximates P
        # when most transits are consecutive.
        diffs = np.diff(times)
        period_guess = float(np.median(diffs))

    epochs = np.round((times - times[0]) / period_guess).astype(np.int64)
    return epochs


def fit_linear_ephemeris(
    times: np.ndarray, period_guess: float | None = None
) -> LinearEphemeris:
    """Least-squares fit of T_n = T_0 + n P, returning the O-C residuals.

    Requires at least three transits: two determine the line exactly and would
    leave identically zero residuals, which would be meaningless.
    """
    times = np.asarray(times, dtype=np.float64)
    if len(times) < 3:
        raise ValueError(
            f"need at least 3 transits to measure O-C residuals, got {len(times)}"
        )

    epochs = assign_epochs(times, period_guess)

    # design matrix for [T_0, P]
    design = np.vstack([np.ones_like(epochs, dtype=np.float64), epochs.astype(np.float64)]).T
    solution, _, _, _ = np.linalg.lstsq(design, times, rcond=None)
    t0, period = float(solution[0]), float(solution[1])

    model = t0 + period * epochs
    residuals = times - model

    # formal parameter uncertainties from the fit residuals
    dof = max(len(times) - 2, 1)
    variance = float(np.sum(residuals**2) / dof)
    covariance = variance * np.linalg.inv(design.T @ design)

    return LinearEphemeris(
        t0=t0,
        period=period,
        t0_uncertainty=float(np.sqrt(covariance[0, 0])),
        period_uncertainty=float(np.sqrt(covariance[1, 1])),
        epochs=epochs,
        times=times,
        residuals=residuals,
    )


def ttv_amplitude_minutes(times: np.ndarray, period_guess: float | None = None) -> float:
    """Peak-to-peak O-C amplitude in minutes — the headline TTV number."""
    return fit_linear_ephemeris(times, period_guess).peak_to_peak_minutes


def compare_ephemerides(
    control_times: np.ndarray, perturbed_times: np.ndarray
) -> dict:
    """Difference between two transit sequences, epoch by epoch.

    Used by the control-vs-mystery experiment: both systems have identical
    visible planets, so a non-zero difference is caused ONLY by the extra body.
    Matching is done on epoch number, not on array index, so a missing transit in
    one sequence cannot silently shift the comparison.
    """
    control = fit_linear_ephemeris(control_times)
    perturbed = fit_linear_ephemeris(perturbed_times)

    shared = np.intersect1d(control.epochs, perturbed.epochs)
    control_map = dict(zip(control.epochs, control.times))
    perturbed_map = dict(zip(perturbed.epochs, perturbed.times))
    delta = np.array([perturbed_map[e] - control_map[e] for e in shared])

    # remove the best-fit line: a constant offset or a slightly different mean
    # period is absorbed by any real ephemeris fit, so the STRUCTURE is the signal
    if len(shared) >= 3:
        design = np.vstack([np.ones(len(shared)), shared.astype(np.float64)]).T
        coefficients, _, _, _ = np.linalg.lstsq(design, delta, rcond=None)
        detrended = delta - design @ coefficients
    else:
        detrended = delta - np.mean(delta)

    return {
        "epochs": shared,
        "delta_days": delta,
        "delta_seconds": delta * SECONDS_PER_DAY,
        "detrended_days": detrended,
        "detrended_seconds": detrended * SECONDS_PER_DAY,
        "rms_seconds": float(np.sqrt(np.mean(delta**2)) * SECONDS_PER_DAY),
        "detrended_rms_seconds": float(
            np.sqrt(np.mean(detrended**2)) * SECONDS_PER_DAY
        ),
        "peak_to_peak_minutes": float((delta.max() - delta.min()) * 1440.0)
        if len(delta)
        else 0.0,
        "detrended_peak_to_peak_minutes": float(
            (detrended.max() - detrended.min()) * 1440.0
        )
        if len(detrended)
        else 0.0,
    }
