"""
The inference pipeline: from an observer dataset to a detection statement and a posterior.

INPUT: an ObserverDataset ONLY (noisy transit times, epochs, per-transit uncertainties). The
visible planets' published parameters are public knowledge and are read from the reference
dataset. This module must never import the experiment configuration or Planet X's parameters;
tests/test_inference.py verifies that by parsing its imports.

STAGES
------
 0. null hypothesis: chi2 of the visible planets alone.
 1. periodogram over the dynamically plausible search zones (linearised, 2 simulations per
    trial period) -> detection statistic.
 2. false-alarm probability of that statistic, from the EXACT noise-only distribution of the
    same statistic (look-elsewhere corrected).
 3. nonlinear verification of the strongest peaks with the full N-body model.
 4. if a detection is claimed: posterior sampling over seven parameters.

A detection is claimed only if the false-alarm probability is below `fap_threshold`. Otherwise
the result says so and reports no planet - the pipeline can and does return "nothing found".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..systems.kepler90 import build_kepler90, load_reference
from .domain import axes_for_zones, axis_step, stable_zones
from .forward import ForwardModel
from .likelihood import TimingLikelihood
from .periodogram import (
    PeriodogramBasis,
    false_alarm_probability,
    null_distribution,
    scan,
    select_peaks,
)
from .sampler import (
    Prior,
    PARAMETER_NAMES,
    initial_ball,
    integrated_autocorr_time,
    log_posterior_batch,
    parameters_to_vector,
    run_ensemble,
    vector_to_parameters,
)
from .verify import verify_peaks

FAP_THRESHOLD = 1e-2


@dataclass
class RecoveryResult:
    detected: bool
    chi2_null: float
    n_points: int
    dof_null: int
    linear_best_delta_chi2: float
    false_alarm_probability: float
    verified_peaks: list
    best_chi2: float | None
    best_parameters: dict | None
    posterior: dict | None
    n_forward_simulations: int
    notes: list = field(default_factory=list)
    periodogram: dict | None = None      # the linearised scan itself, kept for display

    def to_dict(self) -> dict:
        return {
            "detected": self.detected,
            "chi2_null": self.chi2_null,
            "n_points": self.n_points,
            "dof_null": self.dof_null,
            "linear_best_delta_chi2": self.linear_best_delta_chi2,
            "false_alarm_probability": self.false_alarm_probability,
            "verified_peaks": self.verified_peaks,
            "best_chi2": self.best_chi2,
            "best_parameters": self.best_parameters,
            "posterior": self.posterior,
            "n_forward_simulations": self.n_forward_simulations,
            "notes": self.notes,
            "periodogram": self.periodogram,
        }


def equivalent_sigma(p_value: float) -> float:
    """Gaussian-equivalent significance of a (two-sided) false-alarm probability."""
    from scipy.stats import norm

    return float(norm.isf(min(max(p_value, 1e-300), 0.5) / 2.0)) if p_value < 1.0 else 0.0


def summarise_samples(samples: np.ndarray, reference: dict) -> dict:
    """Marginal medians and credible intervals of the physical parameters."""
    from ..constants import EARTH_MASS_IN_SOLAR, kepler_third_law_period
    from .sampler import tilt_to_inclination_node

    m_star = reference["star"]["mass_solar"]["value"]
    mass = np.exp(samples[:, 0])
    axis = samples[:, 1]
    ecc = samples[:, 3] ** 2 + samples[:, 4] ** 2
    tilt = np.hypot(samples[:, 5], samples[:, 6])
    period = np.array([kepler_third_law_period(a, m_star + mm * EARTH_MASS_IN_SOLAR) for a, mm in zip(axis, mass)])
    phase = np.mod(samples[:, 2], 2 * np.pi)

    def q(v, qs=(2.5, 16, 50, 84, 97.5)):
        return {str(x): float(np.percentile(v, x)) for x in qs}

    return {
        "mass_earth": q(mass),
        "semi_major_axis_au": q(axis),
        "period_days": q(period),
        "eccentricity": q(ecc),
        "mutual_inclination_deg": q(tilt),
        "mean_anomaly_rad": q(phase),
        "n_samples": int(len(samples)),
    }


def recover(
    dataset,
    reference: dict | None = None,
    basis_cache: Path | None = None,
    fap_threshold: float = FAP_THRESHOLD,
    n_peaks: int = 6,
    n_null_trials: int = 4000,
    sample: bool = True,
    n_walkers: int = 32,
    n_steps: int = 250,
    seed: int = 1,
    verbose: bool = True,
    zones: list | None = None,
    verify_settings: dict | None = None,
) -> RecoveryResult:
    reference = reference if reference is not None else load_reference()
    likelihood = TimingLikelihood(dataset)
    forward = ForwardModel(dataset, reference=reference)
    m_star = reference["star"]["mass_solar"]["value"]
    stellar_radius_au = build_kepler90(reference=reference).radii[0]
    notes: list = []

    control = forward.control()
    chi2_null = likelihood.chi2(control)
    data_vector = likelihood.data_vector(control)
    if verbose:
        print(f"  null hypothesis: chi2 = {chi2_null:.1f} for {likelihood.n_points} points "
              f"({likelihood.dof_data} dof after the linear ephemerides)")

    # ---- stage 1: periodogram -----------------------------------------------------------
    zones = zones if zones is not None else stable_zones(reference)
    axes = axes_for_zones(zones, dataset.observation_days, m_star)
    if verbose:
        print("  search zones [au]: " + ", ".join(f"{lo:.3f}-{hi:.3f}" for lo, hi in zones))
    basis = None
    if basis_cache is not None and Path(basis_cache).exists():
        candidate = PeriodogramBasis.load(basis_cache)
        signature = PeriodogramBasis.make_signature(likelihood, control, axes,
                                                    candidate.reference_mass_earth, candidate.inclination_deg)
        if candidate.signature == signature:
            basis = candidate
            if verbose:
                print(f"  periodogram basis loaded from cache ({len(axes)} trial axes)")
    if basis is None:
        basis = PeriodogramBasis.compute(forward, likelihood, axes, verbose=verbose)
        if basis_cache is not None:
            basis.save(basis_cache)

    result = scan(basis, data_vector)
    widths = np.array([axis_step(a, dataset.observation_days, m_star) * 5.0 for a in basis.axes])
    peak_index = select_peaks(result, n_peaks=n_peaks, widths=widths)
    best_linear = float(result["delta_chi2"][peak_index[0]])

    # ---- stage 2: false-alarm probability ------------------------------------------------
    null_max = null_distribution(likelihood, basis, n_trials=n_null_trials, seed=seed)
    fap = false_alarm_probability(best_linear, null_max)
    periodogram = {
        "axes_au": [float(a) for a in basis.axes],
        "linear_delta_chi2": [float(x) for x in result["delta_chi2"]],
        "noise_only_threshold_1pct": float(np.percentile(null_max, 99.0)),
        "noise_only_threshold_5pct": float(np.percentile(null_max, 95.0)),
        "n_null_trials": int(n_null_trials),
    }
    if verbose:
        print(f"  strongest periodogram peak: a = {basis.axes[peak_index[0]]:.4f} au, "
              f"linear delta chi2 = {best_linear:.1f}")
        print(f"  false-alarm probability (exact, noise-only, {n_null_trials} trials): {fap:.2e}"
              f"  (~{equivalent_sigma(fap):.1f} sigma)")

    detected = fap < fap_threshold
    peaks = [
        {"axis_au": float(basis.axes[i]), "axis_step_au": float(widths[i] / 5.0),
         "linear_delta_chi2": float(result["delta_chi2"][i])}
        for i in peak_index
    ]

    # ---- stage 3: nonlinear verification --------------------------------------------------
    if verbose:
        print("  nonlinear verification of the shortlisted peaks (full N-body model):")
    verification = verify_peaks(forward, likelihood, control, data_vector, peaks, verbose=verbose,
                                settings=verify_settings)
    best = verification["best"]
    best_chi2 = float(best["chi2"]) if best["parameters"] is not None else None
    verified_table = [
        {"peak": r["peak"], "axis_au": r["axis_au"], "linear_delta_chi2": r["linear_delta_chi2"],
         "full_model_chi2": r["chi2"],
         "parameters": (r["parameters"].to_dict() if r["parameters"] is not None else None)}
        for r in verification["peaks"]
    ]

    if not detected:
        notes.append(
            f"No significant detection: false-alarm probability {fap:.2e} exceeds "
            f"the threshold {fap_threshold:g}. No planet is claimed."
        )
        return RecoveryResult(False, chi2_null, likelihood.n_points, likelihood.dof_data, best_linear,
                              fap, verified_table, best_chi2, None, None, forward.n_evaluations, notes,
                              periodogram)

    # ---- stage 4: posterior ---------------------------------------------------------------
    posterior = None
    if sample and best["parameters"] is not None:
        mode = parameters_to_vector(best["parameters"])
        a0 = mode[1]
        step = axis_step(a0, dataset.observation_days, m_star) * 5.0
        prior = Prior(axis_bounds=(a0 - 6 * step, a0 + 6 * step),
                      phase_bounds=(mode[2] - np.pi, mode[2] + np.pi),
                      reference=reference, stellar_radius_au=stellar_radius_au)
        start = initial_ball(mode, prior, n_walkers, seed,
                             scale={"ln_mass": 0.25, "axis": 0.4 * step, "phase": 0.15, "ecc": 0.05, "tilt": 0.8})
        if verbose:
            print(f"  posterior sampling: {n_walkers} walkers x {n_steps} steps "
                  f"({n_walkers * n_steps} forward simulations)")

        def log_prob(xs):
            return log_posterior_batch(xs, forward, likelihood, prior)

        run = run_ensemble(log_prob, start, n_steps=n_steps, seed=seed, verbose=verbose)
        chain = run["chain"]
        burn = n_steps // 3
        taus = integrated_autocorr_time(chain[burn:])
        flat = chain[burn:].reshape(-1, chain.shape[2])
        ess = float(len(flat) / np.nanmax(taus)) if np.all(np.isfinite(taus)) else float("nan")
        posterior = {
            "parameter_names": PARAMETER_NAMES,
            "acceptance_fraction": run["acceptance"],
            "burn_in_steps": int(burn),
            "autocorrelation_time_steps": [float(t) for t in taus],
            "effective_samples": ess,
            "chain_length_over_tau": float((n_steps - burn) / np.nanmax(taus)) if np.all(np.isfinite(taus)) else None,
            "summary": summarise_samples(flat, reference),
            "max_lnp": float(np.max(run["lnp"])),
            "samples_thinned": flat[:: max(1, len(flat) // 2000)].tolist(),
        }
        if posterior["chain_length_over_tau"] is not None and posterior["chain_length_over_tau"] < 50:
            notes.append(
                f"chain is only {posterior['chain_length_over_tau']:.0f} autocorrelation times long "
                "(< 50): the posterior intervals are PROVISIONAL"
            )

    return RecoveryResult(
        True, chi2_null, likelihood.n_points, likelihood.dof_data, best_linear, fap,
        verified_table, best_chi2, best["parameters"].to_dict() if best["parameters"] else None,
        posterior, forward.n_evaluations, notes, periodogram,
    )
