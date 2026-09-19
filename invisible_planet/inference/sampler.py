"""
Posterior sampling for the hidden planet's parameters.

WHY SAMPLE, NOT JUST OPTIMISE
-----------------------------
A best fit and a one-dimensional chi2 profile understate the uncertainty when parameters are
correlated. This project already met that: mass is degenerate with the orbital inclination
(the best-fit mass changed from 20 to 50 Earth masses when the assumed inclination moved from
85 to 88 degrees). Quoting a mass interval from a profile with the other parameters frozen would
be wrong. The posterior over all free parameters, marginalised, is the honest answer.

PARAMETRISATION (seven free parameters, none periodic)
------------------------------------------------------
    ln m        log-uniform mass, 1.5 - 400 Earth masses
    a           semi-major axis, uniform in a window around the verified mode
    lambda      mean anomaly at the epoch, uniform in a window of +-pi around the mode
    k, h        sqrt(e) cos(omega), sqrt(e) sin(omega): uniform in a disc, so e < 0.3 uniform
    tx, ty      tilt of the orbit relative to the visible planets' plane, in degrees, uniform
                in a disc of radius 6 degrees (Kepler multi-planet systems are nearly
                coplanar; the real g-h and inner pairs are within ~1 degree)

The tilt vector fixes both the inclination and the ascending node of the candidate. Sampling
i and Omega independently would allow orbits perpendicular to the visible planets (an edge-on
orbit with a different node is inclined to the others by the node difference), which no
plausible planetary system contains.

SAMPLER
-------
The affine-invariant ensemble sampler of Goodman & Weare (2010), with the stretch move
(a = 2). It is invariant under linear transformations of the parameters, which suits a
posterior that is extremely narrow in period and broad in mass. Each half of the ensemble is
updated in one batch through the N-body ensemble engine.

CONVERGENCE
-----------
The integrated autocorrelation time is estimated with the standard FFT method and windowing
(Sokal 1997; Goodman & Weare 2010). A chain shorter than ~50 autocorrelation times is flagged
and the reported intervals are marked provisional.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..systems.planet_x import PlanetXParameters
from .domain import is_plausible

LN_MASS_BOUNDS = (np.log(1.5), np.log(400.0))
MAX_ECCENTRICITY = 0.30
MAX_TILT_DEG = 6.0
VISIBLE_PLANE_INCLINATION_DEG = 89.8     # mean of the visible planets' inclinations (b-derived)
PARAMETER_NAMES = ["ln_mass", "axis_au", "mean_anomaly", "k=sqrt(e)cos(w)", "h=sqrt(e)sin(w)",
                   "tilt_x_deg", "tilt_y_deg"]


def tilt_to_inclination_node(tilt_x_deg: float, tilt_y_deg: float) -> tuple:
    """(inclination, node) in degrees of an orbit tilted by (tx, ty) from the visible plane.

    The visible planets share an ascending node of 0 and an inclination close to 90 degrees, so
    their orbit normal is n_v = (0, -sin i_v, cos i_v). A tilt of magnitude theta about the
    in-plane axis at angle psi from the line of nodes rotates n_v to the candidate's normal n;
    then cos i = n_z and Omega = atan2(n_x, -n_y).
    """
    theta = np.hypot(tilt_x_deg, tilt_y_deg)
    iv = np.deg2rad(VISIBLE_PLANE_INCLINATION_DEG)
    normal = np.array([0.0, -np.sin(iv), np.cos(iv)])
    if theta < 1e-12:
        n = normal
    else:
        psi = np.arctan2(tilt_y_deg, tilt_x_deg)
        axis_1 = np.array([1.0, 0.0, 0.0])                       # the line of nodes
        axis_2 = np.cross(normal, axis_1)                        # in-plane, perpendicular to it
        axis = np.cos(psi) * axis_1 + np.sin(psi) * axis_2
        t = np.deg2rad(theta)
        n = (normal * np.cos(t) + np.cross(axis, normal) * np.sin(t)
             + axis * np.dot(axis, normal) * (1.0 - np.cos(t)))    # Rodrigues rotation
    inclination = np.degrees(np.arccos(np.clip(n[2], -1.0, 1.0)))
    node = np.degrees(np.arctan2(n[0], -n[1]))
    return float(inclination), float(node)


def vector_to_parameters(x: np.ndarray, radius_earth: float = 4.0) -> PlanetXParameters:
    ln_m, a, lam, k, h, tx, ty = x
    e = float(k * k + h * h)
    omega = float(np.arctan2(h, k)) if e > 0 else 0.0
    inclination, node = tilt_to_inclination_node(tx, ty)
    return PlanetXParameters(
        mass_earth=float(np.exp(ln_m)),
        semi_major_axis_au=float(a),
        eccentricity=e,
        mean_anomaly=float(np.mod(lam, 2.0 * np.pi)),
        inclination_deg=inclination,
        argument_of_periapsis=omega,
        longitude_of_ascending_node_deg=node,
        radius_earth=radius_earth,
    )


def parameters_to_vector(p: PlanetXParameters) -> np.ndarray:
    """Inverse for the coplanar-tilt-free case (used to seed walkers from a fitted mode)."""
    k = np.sqrt(p.eccentricity) * np.cos(p.argument_of_periapsis)
    h = np.sqrt(p.eccentricity) * np.sin(p.argument_of_periapsis)
    return np.array([np.log(p.mass_earth), p.semi_major_axis_au, p.mean_anomaly, k, h, 0.0, 0.0])


@dataclass
class Prior:
    axis_bounds: tuple
    phase_bounds: tuple
    reference: dict
    stellar_radius_au: float

    def in_support(self, x: np.ndarray) -> bool:
        ln_m, a, lam, k, h, tx, ty = x
        if not (LN_MASS_BOUNDS[0] <= ln_m <= LN_MASS_BOUNDS[1]):
            return False
        if not (self.axis_bounds[0] <= a <= self.axis_bounds[1]):
            return False
        if not (self.phase_bounds[0] <= lam <= self.phase_bounds[1]):
            return False
        if k * k + h * h > MAX_ECCENTRICITY or np.hypot(tx, ty) > MAX_TILT_DEG:
            return False
        return is_plausible(vector_to_parameters(x), self.reference, self.stellar_radius_au)


def log_posterior_batch(xs: np.ndarray, forward, likelihood, prior: Prior) -> np.ndarray:
    """ln posterior (up to a constant) for a batch of parameter vectors."""
    out = np.full(len(xs), -np.inf)
    valid = [i for i, x in enumerate(xs) if prior.in_support(x)]
    if not valid:
        return out
    params = [vector_to_parameters(xs[i]) for i in valid]
    predictions = forward.predict(params)
    for i, pred in zip(valid, predictions):
        chi2 = likelihood.chi2(pred)
        out[i] = -0.5 * chi2 if chi2 < 1e11 else -np.inf
    return out


def integrated_autocorr_time(chain: np.ndarray, c: float = 5.0) -> np.ndarray:
    """Integrated autocorrelation time per parameter for a (steps, walkers, dims) chain."""
    n_steps, n_walkers, n_dim = chain.shape
    taus = np.zeros(n_dim)
    for d in range(n_dim):
        x = chain[:, :, d] - chain[:, :, d].mean(axis=0)
        f = np.fft.rfft(x, n=2 * n_steps, axis=0)
        acf = np.fft.irfft(f * np.conj(f), axis=0)[:n_steps].real.mean(axis=1)
        if acf[0] <= 0:
            taus[d] = np.nan
            continue
        acf /= acf[0]
        cumulative = 2.0 * np.cumsum(acf) - 1.0
        window = np.arange(len(cumulative)) < c * cumulative
        m = int(np.argmin(window)) if not window.all() else len(cumulative) - 1
        taus[d] = cumulative[m]
    return taus


def run_ensemble(
    log_prob,
    start: np.ndarray,
    n_steps: int,
    seed: int,
    stretch: float = 2.0,
    verbose: bool = True,
) -> dict:
    """Goodman-Weare stretch-move sampler. `start` is (walkers, dims); walkers must be even."""
    rng = np.random.default_rng(seed)
    n_walkers, n_dim = start.shape
    if n_walkers % 2:
        raise ValueError("number of walkers must be even")
    halves = [np.arange(0, n_walkers // 2), np.arange(n_walkers // 2, n_walkers)]

    coords = start.copy()
    logp = log_prob(coords)
    if not np.all(np.isfinite(logp)):
        raise ValueError(f"{int(np.sum(~np.isfinite(logp)))} starting walkers have zero posterior")

    chain = np.zeros((n_steps, n_walkers, n_dim))
    lnp = np.zeros((n_steps, n_walkers))
    accepted = np.zeros(n_walkers)

    for step in range(n_steps):
        for s in (0, 1):
            active, complement = halves[s], halves[1 - s]
            partners = rng.choice(complement, size=len(active))
            z = ((stretch - 1.0) * rng.random(len(active)) + 1.0) ** 2 / stretch
            proposal = coords[partners] + z[:, None] * (coords[active] - coords[partners])
            new_logp = log_prob(proposal)
            log_ratio = (n_dim - 1) * np.log(z) + new_logp - logp[active]
            accept = np.log(rng.random(len(active))) < log_ratio
            coords[active[accept]] = proposal[accept]
            logp[active[accept]] = new_logp[accept]
            accepted[active[accept]] += 1
        chain[step] = coords
        lnp[step] = logp
        if verbose and (step + 1) % max(1, n_steps // 10) == 0:
            print(f"    step {step + 1:4d}/{n_steps}   mean acceptance {accepted.mean() / (step + 1):.2f}"
                  f"   best ln p = {np.max(lnp[: step + 1]):.2f}")
    return {"chain": chain, "lnp": lnp, "acceptance": float(accepted.mean() / n_steps)}


def initial_ball(mode: np.ndarray, prior: Prior, n_walkers: int, seed: int, scale: dict) -> np.ndarray:
    """Walkers scattered around a fitted mode, inside the prior support."""
    rng = np.random.default_rng(seed)
    walkers = []
    attempts = 0
    while len(walkers) < n_walkers and attempts < 20000:
        attempts += 1
        x = mode + rng.standard_normal(len(mode)) * np.array(
            [scale["ln_mass"], scale["axis"], scale["phase"], scale["ecc"], scale["ecc"],
             scale["tilt"], scale["tilt"]]
        )
        if prior.in_support(x):
            walkers.append(x)
    if len(walkers) < n_walkers:
        raise RuntimeError("could not place walkers inside the prior support")
    return np.array(walkers)
