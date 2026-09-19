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
    times: np.ndarray,
    period_guess: float | None = None,
    epochs: np.ndarray | None = None,
    sigma: np.ndarray | None = None,
) -> LinearEphemeris:
    """Weighted least-squares fit of T_n = T_0 + n P, returning the O-C residuals.

    Parameters
    ----------
    times : transit times [days].
    epochs : integer transit numbers. If omitted they are inferred from the spacing,
        which fails when gaps dominate; real, gappy data should always pass them.
    sigma : per-transit 1-sigma uncertainties [days]. If omitted the fit is unweighted.
        With `sigma` the parameter uncertainties are the formal ones (they are NOT
        rescaled by the reduced chi-squared, which is reported separately by callers).

    Requires at least three transits: two determine the line exactly and would leave
    identically zero residuals, which would be meaningless.
    """
    times = np.asarray(times, dtype=np.float64)
    if len(times) < 3:
        raise ValueError(
            f"need at least 3 transits to measure O-C residuals, got {len(times)}"
        )

    epochs = (
        assign_epochs(times, period_guess)
        if epochs is None
        else np.asarray(epochs, dtype=np.int64)
    )
    if len(epochs) != len(times):
        raise ValueError("epochs and times must have the same length")

    weights = (
        np.ones(len(times))
        if sigma is None
        else 1.0 / np.asarray(sigma, dtype=np.float64) ** 2
    )

    # Fit about the weighted-mean epoch: the intercept and slope are then
    # uncorrelated and the normal equations are well conditioned even when the
    # epochs run to hundreds (planet b) or when the sequence has large gaps.
    mean_epoch = float(np.sum(weights * epochs) / np.sum(weights))
    x = epochs.astype(np.float64) - mean_epoch
    design = np.vstack([np.ones_like(x), x]).T
    normal = design.T @ (weights[:, None] * design)
    t_ref, period = np.linalg.solve(normal, design.T @ (weights * times))
    t0 = float(t_ref - period * mean_epoch)

    model = t0 + period * epochs
    residuals = times - model

    covariance_unit_weights = np.linalg.inv(normal)
    if sigma is None:
        dof = max(len(times) - 2, 1)
        variance = float(np.sum(residuals**2) / dof)
        covariance = variance * covariance_unit_weights
        cov_t0 = covariance[0, 0] + mean_epoch**2 * covariance[1, 1] - 2 * mean_epoch * covariance[0, 1]
        cov_p = covariance[1, 1]
    else:
        covariance = covariance_unit_weights
        cov_t0 = covariance[0, 0] + mean_epoch**2 * covariance[1, 1] - 2 * mean_epoch * covariance[0, 1]
        cov_p = covariance[1, 1]

    return LinearEphemeris(
        t0=t0,
        period=float(period),
        t0_uncertainty=float(np.sqrt(cov_t0)),
        period_uncertainty=float(np.sqrt(cov_p)),
        epochs=epochs,
        times=times,
        residuals=residuals,
    )


def profile_chi2(
    data_times: np.ndarray,
    model_times: np.ndarray,
    sigma: np.ndarray,
    epochs: np.ndarray,
) -> tuple[float, int]:
    """Chi-squared of a model against data with the linear ephemeris profiled out.

    The observer never knows a planet's true period and phase, so a candidate model
    is judged only up to a per-planet straight line in epoch number:

        chi2 = min over (a, b)  sum_n [ (data_n - model_n - a - b n) / sigma_n ]^2

    This is the correct statistic for O-C analysis with GAPS and UNEQUAL weights.
    Fitting the ephemeris separately to the model and to the data would be wrong in
    that case, because the two fits would use different weights or epochs. Returns
    (chi2, degrees_of_freedom_removed=2).
    """
    r = np.asarray(data_times, dtype=np.float64) - np.asarray(model_times, dtype=np.float64)
    n = np.asarray(epochs, dtype=np.float64)
    w = 1.0 / np.asarray(sigma, dtype=np.float64) ** 2
    mean_epoch = float(np.sum(w * n) / np.sum(w))
    design = np.vstack([np.ones_like(n), n - mean_epoch]).T
    normal = design.T @ (w[:, None] * design)
    coefficients = np.linalg.solve(normal, design.T @ (w * r))
    residual = r - design @ coefficients
    return float(np.sum(w * residual**2)), 2


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
