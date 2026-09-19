"""
Metrics for the Planet X design sweep (Phases 7.2 and 8).

For every candidate hidden planet this measures, from ONE integration of the mystery system
(sampled against a matched integration of the control system):

  1. dynamical stability        orbit crossing, boundedness, eccentricity, energy error
  2. minimum pairwise separation   and initial separation in mutual Hill radii
  3. energy conservation
  4. TTV amplitude              peak-to-peak of the ephemeris-profiled perturbation, per probe
  5. orbital residual amplitude Delta r_i(t), Delta v_i(t) of every visible planet, defined
                                relative to the star so the barycentric shift cancels
  6. observation duration needed  expected Delta chi2 as the baseline grows
  7. detectability              expected (noise-free) Delta chi2 against the null
  8. plausibility               Hill separation, and NON-TRANSITING geometry

The expected Delta chi2 of a candidate is |P dt / sigma|^2, the amount by which noise-free data
from that candidate disfavour the visible-planets-only model, with each probe planet's linear
ephemeris profiled out (inference/likelihood.py). It uses the REAL observing pattern and the
real published per-transit uncertainties.
"""

from __future__ import annotations

import types

import numpy as np

from ..constants import EARTH_MASS_IN_SOLAR, G
from ..inference.domain import hill_separations, impact_parameter
from ..inference.likelihood import TimingLikelihood
from ..observations.pattern import load_pattern
from ..observations.transits import TransitSeries
from ..physics.diagnostics import semi_major_axis_and_eccentricity
from ..physics.engine import NBodyEngine
from ..systems.kepler90 import DEFAULT_DURATION_DAYS, DEFAULT_EPOCH_BJD, build_kepler90, recommended_timestep
from ..systems.planet_x import PlanetXParameters, build_system_with_planet_x

NON_TRANSIT_IMPACT_TARGET = 1.4   # impact parameter (stellar radii) used to set the inclination
DURATIONS_DAYS = (400.0, 700.0, 1000.0, DEFAULT_DURATION_DAYS)


def non_transiting_inclination_deg(a_au: float, stellar_radius_au: float,
                                   b_target: float = NON_TRANSIT_IMPACT_TARGET) -> float:
    """Inclination that gives the requested impact parameter, capped at 89.7 degrees.

    b = a cos(i) / R*, so cos(i) = b R* / a. A candidate at a small semi-major axis needs a
    larger inclination offset to avoid transiting; a larger one needs almost none.
    """
    cos_i = min(b_target * stellar_radius_au / a_au, np.cos(np.deg2rad(86.0)))
    return float(min(np.degrees(np.arccos(cos_i)), 89.7))


def template_dataset(epoch_bjd: float = DEFAULT_EPOCH_BJD, duration_days: float = DEFAULT_DURATION_DAYS,
                     max_days: float | None = None):
    """A dataset-like object carrying the REAL epochs and uncertainties (data = real times)."""
    pattern = load_pattern()
    epochs, times, sigma, ephemeris = {}, {}, {}, {}
    for k in pattern.probe_letters:
        obs = pattern[k]
        keep = np.ones(obs.n_transits, dtype=bool)
        if max_days is not None:
            keep = (obs.measured_bjd - epoch_bjd) <= max_days
        epochs[k] = obs.epochs[keep]
        times[k] = obs.measured_bjd[keep]
        sigma[k] = obs.sigma_days[keep]
        ephemeris[k] = {"t0_bjd": obs.t0_bjd, "period_days": obs.period_days}
    return types.SimpleNamespace(
        planet_letters=[k for k in pattern.probe_letters if len(epochs[k]) >= 3],
        epochs=epochs, transit_times_bjd=times, timing_uncertainty_days=sigma,
        reference_ephemeris=ephemeris, epoch_bjd=epoch_bjd,
        observation_days=duration_days if max_days is None else max_days,
    )


def integrate_with_diagnostics(systems: list, tracked_names: list, duration_days: float, dt: float,
                               n_samples: int = 73) -> dict:
    """Integrate a batch once, recording transit times, star-relative states and energy error."""
    k = len(systems)
    n = systems[0].n_bodies
    engine = NBodyEngine(n_bodies=n, n_ensembles=k)
    indices = [systems[0].index_of(x) for x in tracked_names]
    engine.enable_transit_detection(
        tracked_bodies=indices, max_transits=512, stellar_radius=float(systems[0].radii[0]),
        planet_radii=np.asarray([systems[0].radii[i] for i in indices]),
    )
    engine.set_state(np.stack([s.masses for s in systems]), np.stack([s.positions for s in systems]),
                     np.stack([s.velocities for s in systems]), time=0.0)
    e0 = engine.diagnostics()["total_energy"].copy()

    total = int(round(duration_days / dt))
    chunk = max(1, total // n_samples)
    rel_r = np.zeros((n_samples, k, n, 3))
    rel_v = np.zeros((n_samples, k, n, 3))
    energy_err = np.zeros((n_samples, k))
    min_sep = np.full(k, np.inf)
    for s in range(n_samples):
        engine.run(dt, chunk)
        pos, vel = engine.get_positions(), engine.get_velocities()
        rel_r[s] = pos - pos[:, :1, :]
        rel_v[s] = vel - vel[:, :1, :]
        d = engine.diagnostics()
        energy_err[s] = np.abs(d["total_energy"] - e0) / np.abs(e0)
        min_sep = np.minimum(min_sep, d["min_separation"])

    times = engine.get_transit_times()
    series = [
        TransitSeries(names=list(tracked_names), times=times[i], impact_parameters=None,
                      span_days=engine.time, timestep_days=dt, epoch_bjd=systems[i].epoch)
        for i in range(k)
    ]
    return {"rel_r": rel_r, "rel_v": rel_v, "energy_error": energy_err, "min_separation": min_sep,
            "series": series, "masses": np.stack([s.masses for s in systems])}


def stability_metrics(rel_r, rel_v, masses, m_star: float) -> list:
    """Per-member stability from the sampled star-relative states."""
    n_samples, k, n, _ = rel_r.shape
    out = []
    for m in range(k):
        mu = G * (m_star + masses[m, 1:])
        a, e = semi_major_axis_and_eccentricity(mu[None, :], rel_r[:, m, 1:, :], rel_v[:, m, 1:, :])
        bound = bool(np.all(a > 0) and np.all(np.isfinite(a)))
        order = np.argsort(a[0])
        crossing = False
        if bound:
            for j in range(len(order) - 1):
                inner, outer = order[j], order[j + 1]
                if np.min(a[:, outer] * (1 - e[:, outer])) <= np.max(a[:, inner] * (1 + e[:, inner])):
                    crossing = True
        out.append({"bound": bound, "orbit_crossing": bool(crossing),
                    "max_eccentricity": float(np.nanmax(e)),
                    "max_relative_axis_change": float(np.nanmax(np.abs(a - a[0]) / a[0])) if bound else np.inf})
    return out


def signal_metrics(predicted: dict, control: dict, likelihood_by_duration: dict, template_full) -> dict:
    """Expected timing signal of a candidate: total and per-planet delta chi2, TTV amplitude,
    largest single-transit excursion in sigma, and the baseline needed."""
    lk = likelihood_by_duration[DURATION_KEY_FULL]
    v = lk.model_vector(predicted, control)
    total = float(v @ v)
    per_planet, amplitude_min, off = {}, {}, 0
    max_single = 0.0
    for k in lk.letters:
        n = len(lk.epochs[k])
        seg = v[off:off + n]
        per_planet[k] = float(seg @ seg)
        max_single = max(max_single, float(np.max(np.abs(seg))) if n else 0.0)
        # peak-to-peak of the profiled perturbation, in minutes
        seg_t = seg * lk.sigma[k] * 1440.0
        amplitude_min[k] = float(seg_t.max() - seg_t.min()) if n else 0.0
        off += n
    by_duration = {}
    for days, lkd in likelihood_by_duration.items():
        if days == DURATION_KEY_FULL:
            continue
        sub = {k: np.asarray(predicted[k])[_mask(template_full, lkd, k)] for k in lkd.letters}
        ctl = {k: np.asarray(control[k])[_mask(template_full, lkd, k)] for k in lkd.letters}
        w = lkd.model_vector(sub, ctl)
        by_duration[str(days)] = float(w @ w)
    by_duration[str(DURATIONS_DAYS[-1])] = total
    return {"delta_chi2_expected": total, "per_planet_delta_chi2": per_planet,
            "ttv_amplitude_minutes": amplitude_min, "max_single_transit_sigma": max_single,
            "delta_chi2_vs_baseline": by_duration}


DURATION_KEY_FULL = "full"


def build_duration_likelihoods(epoch_bjd: float = DEFAULT_EPOCH_BJD) -> tuple:
    """TimingLikelihood objects for the full pattern and for truncated baselines."""
    full = template_dataset(epoch_bjd)
    out = {DURATION_KEY_FULL: TimingLikelihood(full)}
    for days in DURATIONS_DAYS[:-1]:
        sub = template_dataset(epoch_bjd, max_days=days)
        if len(sub.planet_letters) > 0:
            out[days] = TimingLikelihood(sub)
    return out, full


def _mask(full, sub_likelihood, letter: str) -> np.ndarray:
    """Boolean mask selecting the epochs of `sub_likelihood` from the full pattern's arrays."""
    return np.isin(full.epochs[letter], sub_likelihood.epochs[letter])


def orbital_residuals(rel_r_cand, rel_v_cand, rel_r_ctrl, rel_v_ctrl) -> dict:
    """Delta r_i(t) and Delta v_i(t) of the visible planets (star-relative)."""
    dr = np.linalg.norm(rel_r_cand[:, 1:9, :] - rel_r_ctrl[:, 1:9, :], axis=-1)   # (samples, 8)
    dv = np.linalg.norm(rel_v_cand[:, 1:9, :] - rel_v_ctrl[:, 1:9, :], axis=-1)
    return {"max_dr_au": dr.max(axis=0).tolist(), "max_dv_au_per_day": dv.max(axis=0).tolist()}
