"""
Match the simulated visible system to the real transit times.

WHY THIS IS NECESSARY
---------------------
Catalogue periods are OBSERVED MEAN TRANSIT PERIODS. A simulation needs OSCULATING
elements at a reference epoch, and in a near-resonant multi-planet system the two differ
(measured with IAS15: 0.32% for planet g, 0.04% for e, 0.03% for h, ~3e-5 for b, c, i).
Feeding catalogue periods in as osculating periods therefore mis-represents the observed
system, and since near-resonant TTV amplitudes and super-periods depend on the offset from
resonance, that matters.

WHAT IS FITTED
--------------
Two parameters per planet: the initial osculating period and mean anomaly at the epoch
(16 parameters). Masses are held at their adopted values, orbits are circular and
coplanar-ish, as elsewhere in the project.

  * PROBE planets d, e, g, h: the simulated transit times at the REAL observed epochs are
    fitted to the REAL measured transit times, weighted by their published uncertainties.
    This is a small dynamical fit to real data with 2 free parameters per planet.
  * NON-PROBE planets b, c, i, f (no published individual transit times): the simulated
    linear ephemeris is matched to the catalogue ephemeris (Weiss period; Cabrera / Shallue
    & Vanderburg transit epoch) to a tolerance of 1e-5 in period and the published epoch
    uncertainty.

The model is NOT a full dynamical fit of the real system: eccentricities are zero, so the
large real TTVs of planet g (its O-C series has a -205 min excursion and a +1.1 d one) are
only partly reproduced. The remaining data-minus-model mismatch is reported, not hidden.

Integration uses 4000 steps per innermost orbit, so that the O(dt^2) frequency error of the
integrator (8.5e-7 for planet b, far smaller for the others; measured against IAS15) is
below the precision of the data being matched.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from ..constants import SECONDS_PER_DAY
from ..observations.pattern import ObservingPattern, load_pattern, select_at_epochs
from ..observations.transits import observe_transits_batch
from ..observations.ttv import fit_linear_ephemeris
from ..physics.kepler import transit_mean_anomaly
from ..systems.kepler90 import (
    DEFAULT_DURATION_DAYS,
    DEFAULT_EPOCH_BJD,
    PLANET_ORDER,
    build_kepler90,
    load_reference,
)

PROBES = ["d", "e", "g", "h"]
NON_PROBES = [k for k in PLANET_ORDER if k not in PROBES]

SCALE_PERIOD = 1e-4   # parameter unit: a relative period change of 1e-4
SCALE_PHASE = 1e-2    # parameter unit: 0.01 rad of mean anomaly
STEP = 0.02           # finite-difference step in those units
PERIOD_TOLERANCE = 1e-5
PENALTY = 1e3         # residual assigned to an observed epoch the simulation does not produce

#: Model-error (jitter) terms, added in quadrature to the published sigma IN THE MATCHING FIT
#: ONLY. Planet g's real O-C series has a ~1 day amplitude driven by its near-3:2 interaction
#: with h; a circular, coplanar model reproduces most of it but not to the 2-4 minute precision
#: of the measurements (rms misfit ~83 min after fitting). Without a jitter term that
#: structural mismatch (chi2 ~ 5900 for 6 points) would dominate the fit and distort every
#: other planet. 1 hour is the order of the residual misfit. This is NOT used in the
#: inference noise model, where synthetic data are generated from the model itself.
MODEL_JITTER_DAYS = {"g": 1.0 / 24.0}


@dataclass
class MatchProblem:
    reference: dict
    pattern: ObservingPattern
    epoch_bjd: float
    duration_days: float
    dt: float
    base_period: dict
    base_phase: dict

    @classmethod
    def create(cls, steps_per_inner_orbit: int = 4000, reference: dict | None = None):
        ref = reference if reference is not None else load_reference()
        pattern = load_pattern()
        base_period = {k: ref["planets"][k]["orbital_period_days"]["value"] for k in PLANET_ORDER}
        # starting phase: propagate a transit forward with the osculating mean motion
        base_phase = {}
        for k in PLANET_ORDER:
            if k in pattern.planets:
                t0, period = pattern[k].t0_bjd, pattern[k].period_days
            else:
                t0 = ref["planets"][k]["transit_epoch_bjd"]["value"]
                period = base_period[k]
            m_transit = transit_mean_anomaly(0.0, 0.0)
            base_phase[k] = float(np.mod(m_transit + 2.0 * np.pi * (DEFAULT_EPOCH_BJD - t0) / period, 2.0 * np.pi))
        dt = min(base_period.values()) / steps_per_inner_orbit
        return cls(ref, pattern, DEFAULT_EPOCH_BJD, DEFAULT_DURATION_DAYS, dt, base_period, base_phase)

    # ---- parameter vector <-> system -----------------------------------------------
    def unpack(self, x: np.ndarray) -> tuple:
        periods, phases = {}, {}
        for i, k in enumerate(PLANET_ORDER):
            periods[k] = self.base_period[k] * (1.0 + x[2 * i] * SCALE_PERIOD)
            phases[k] = self.base_phase[k] + x[2 * i + 1] * SCALE_PHASE
        return periods, phases

    def build(self, x: np.ndarray):
        periods, phases = self.unpack(x)
        return build_kepler90(
            epoch_bjd=self.epoch_bjd,
            reference=self.reference,
            initial_conditions="catalogue",
            period_overrides=periods,
            mean_anomaly_overrides=phases,
        )

    # ---- simulation and residuals --------------------------------------------------
    def simulate(self, xs: list) -> list:
        """Transit times [BJD] of every planet, for each parameter vector, in one batch."""
        systems = [self.build(x) for x in xs]
        names = systems[0].names[1:]
        series = observe_transits_batch(
            systems, self.duration_days, self.dt, tracked=names, max_transits=4096
        )
        out = []
        for s in series:
            out.append({k: self.epoch_bjd + s.for_planet(f"Kepler-90 {k}") for k in PLANET_ORDER})
        return out

    def residuals(self, times: dict) -> tuple:
        """Weighted residual vector and a per-planet breakdown."""
        pieces, detail = [], {}
        for k in PROBES:
            observed = self.pattern[k]
            sim, present = select_at_epochs(times[k], observed)
            sigma_eff = np.sqrt(observed.sigma_days**2 + MODEL_JITTER_DAYS.get(k, 0.0) ** 2)
            r = np.where(present, (observed.measured_bjd - sim) / sigma_eff, PENALTY)
            pieces.append(r)
            raw = np.where(present, (observed.measured_bjd - sim) / observed.sigma_days, PENALTY)
            detail[k] = {"chi2": float(np.sum(r**2)), "chi2_published_sigma": float(np.sum(raw**2)),
                         "n": int(len(r)),
                         "missing": int(np.sum(~present)),
                         "rms_minutes": float(np.sqrt(np.mean(((observed.measured_bjd - sim)[present]) ** 2)) * 1440.0)
                         if present.any() else float("nan")}
        for k in NON_PROBES:
            planet = self.reference["planets"][k]
            p_cat = planet["orbital_period_days"]["value"]
            t0_cat = planet["transit_epoch_bjd"]["value"]
            sigma_t0 = max(planet["transit_epoch_bjd"].get("uncertainty") or 0.0, 2e-3)
            sim = times[k]
            epochs = np.round((sim - t0_cat) / p_cat).astype(np.int64)
            fit = fit_linear_ephemeris(sim, epochs=epochs)
            r = np.array([(fit.t0 - t0_cat) / sigma_t0, (fit.period - p_cat) / (p_cat * PERIOD_TOLERANCE)])
            pieces.append(r)
            detail[k] = {"chi2": float(np.sum(r**2)), "n": 2,
                         "period_rel_offset": float((fit.period - p_cat) / p_cat),
                         "epoch_offset_seconds": float((fit.t0 - t0_cat) * SECONDS_PER_DAY)}
        return np.concatenate(pieces), detail


def levenberg_marquardt(problem: MatchProblem, x0: np.ndarray, max_iterations: int = 25,
                        verbose: bool = True) -> dict:
    """Minimise the weighted residual vector over the 16 scaled parameters."""
    n = len(x0)
    x = x0.copy()
    history = []

    r, detail = problem.residuals(problem.simulate([x])[0])
    cost = 0.5 * float(r @ r)
    lam = 1e-2
    history.append({"iteration": 0, "cost": cost})
    if verbose:
        print(f"  iteration  0: cost = {cost:12.2f}")

    for iteration in range(1, max_iterations + 1):
        perturbed = [x + STEP * np.eye(n)[i] for i in range(n)]
        sims = problem.simulate(perturbed)
        columns = []
        for s in sims:
            rj, _ = problem.residuals(s)
            columns.append((rj - r) / STEP)
        jac = np.array(columns).T                     # (n_residuals, n)
        gradient = jac.T @ r
        hessian = jac.T @ jac

        candidates, lams = [], [lam * 0.1, lam, lam * 10.0, lam * 100.0]
        for l in lams:
            step = np.linalg.solve(hessian + l * np.diag(np.diag(hessian) + 1e-12), -gradient)
            candidates.append(x + np.clip(step, -50.0, 50.0))
        trial_sims = problem.simulate(candidates)
        trial = []
        for c, s in zip(candidates, trial_sims):
            rc, dc = problem.residuals(s)
            trial.append((0.5 * float(rc @ rc), c, rc, dc, s))
        best = min(trial, key=lambda item: item[0])

        if best[0] < cost:
            improvement = (cost - best[0]) / max(cost, 1e-12)
            lam = lams[int(np.argmin([t[0] for t in trial]))] * 0.5
            x, cost, r, detail = best[1], best[0], best[2], best[3]
            history.append({"iteration": iteration, "cost": cost})
            if verbose:
                print(f"  iteration {iteration:2d}: cost = {cost:12.2f}  (lambda {lam:.1e})")
            if improvement < 1e-5:
                break
        else:
            lam *= 100.0
            if verbose:
                print(f"  iteration {iteration:2d}: no improvement, lambda -> {lam:.1e}")
            if lam > 1e8:
                break
    return {"x": x, "cost": cost, "residual_detail": detail, "history": history}


def save_matched(problem: MatchProblem, result: dict, path, mean_periods: dict) -> None:
    periods, phases = problem.unpack(result["x"])
    payload = {
        "description": (
            "Initial osculating period and mean anomaly of each Kepler-90 planet, fitted to the "
            "real transit times (scripts/match_visible_system.py, systems/matching.py)."
        ),
        "epoch_bjd": problem.epoch_bjd,
        "duration_days": problem.duration_days,
        "integration_steps_per_inner_orbit": int(round(min(problem.base_period.values()) / problem.dt)),
        "planets": {
            k: {
                "osculating_period_days": float(periods[k]),
                "mean_anomaly_rad": float(np.mod(phases[k], 2.0 * np.pi)),
                "catalogue_period_days": float(problem.base_period[k]),
                "osculating_minus_catalogue_period_rel": float(periods[k] / problem.base_period[k] - 1.0),
                "fitted_mean_transit_period_days": mean_periods.get(k),
            }
            for k in PLANET_ORDER
        },
        "fit": {
            "final_cost": result["cost"],
            "history": result["history"],
            "residual_detail": result["residual_detail"],
        },
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
