"""
The linearised periodogram - the DETECTION statistic.

WHAT IT IS
----------
To first order in the perturber's mass, the timing perturbation is linear in that mass. For
each trial semi-major axis the forward model is run at two orbital phases (0 and pi/2) with a
small reference mass, giving two basis vectors b_0, b_1 (projected, whitened). The data vector
d = P (t_data - t_control) / sigma is fitted by

    d ~ c_0 b_0 + c_1 b_1,          Delta chi2(a) = |d|^2 - |d - B c|^2,

so mass and phase are solved analytically at every trial period and only 2 simulations per
period are needed.

WHAT IT IS NOT
--------------
It is a DETECTOR, not an estimator. It assumes the response to the orbital phase is a single
sinusoid, which fails near a resonance where it contains harmonics, and it assumes a circular,
nearly coplanar orbit. It has produced spurious peaks that scored better than the truth and then
scored WORSE than no planet at all under the full model, so its candidates are always
re-scored nonlinearly (verify.py).

WHY THE REFERENCE MASS IS SMALL
-------------------------------
The linearisation is only valid while the perturbation is small. A reference mass large
enough to destabilise the system at some trial period makes the "basis" there a chaotic
trajectory, not a linear response. With a 40 M_earth reference this produced a spurious peak
that beat the true one; at 5 M_earth it vanished while the true peak was unchanged.

FALSE-ALARM CALIBRATION
-----------------------
The basis vectors depend only on the observing pattern and the visible-planet model, NOT on
the data. So the distribution of the maximum Delta chi2 under pure noise can be computed
exactly and cheaply: draw white noise, project, and take the maximum over trial periods. That
is the look-elsewhere-corrected significance of a detection - the honest replacement for
quoting sqrt(Delta chi2) as a number of sigma.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from ..systems.planet_x import PlanetXParameters

REFERENCE_MASS_EARTH = 5.0
PHASES = (0.0, 0.5 * np.pi)


@dataclass
class PeriodogramBasis:
    axes: np.ndarray            # (n,) trial semi-major axes [au]
    basis: np.ndarray           # (n, 2, n_points) projected, whitened perturbation vectors
    reference_mass_earth: float
    inclination_deg: float
    signature: str

    @staticmethod
    def make_signature(likelihood, control: dict, axes: np.ndarray, mass: float, inclination: float) -> str:
        h = hashlib.sha256()
        for k in likelihood.letters:
            h.update(np.asarray(likelihood.epochs[k]).tobytes())
            h.update(np.asarray(likelihood.sigma[k]).tobytes())
            h.update(np.asarray(control[k]).tobytes())
        h.update(np.asarray(axes).tobytes())
        h.update(f"{mass}:{inclination}".encode())
        return h.hexdigest()

    @classmethod
    def compute(cls, forward, likelihood, axes, reference_mass_earth=REFERENCE_MASS_EARTH,
                inclination_deg=88.0, verbose=True):
        axes = np.asarray(axes, dtype=np.float64)
        control = forward.control()
        candidates = [
            PlanetXParameters(
                mass_earth=reference_mass_earth,
                semi_major_axis_au=float(a),
                eccentricity=0.0,
                mean_anomaly=float(phase),
                inclination_deg=inclination_deg,
            )
            for a in axes
            for phase in PHASES
        ]
        if verbose:
            print(f"  periodogram basis: {len(axes)} trial axes x 2 phases = {len(candidates)} simulations")
        predictions = forward.predict(candidates)
        n_points = likelihood.n_points
        basis = np.zeros((len(axes), 2, n_points))
        for i in range(len(axes)):
            for j in range(2):
                p = predictions[2 * i + j]
                if all(np.all(np.isfinite(p[k])) for k in likelihood.letters):
                    basis[i, j] = likelihood.model_vector(p, control)
        sig = cls.make_signature(likelihood, control, axes, reference_mass_earth, inclination_deg)
        return cls(axes, basis, reference_mass_earth, inclination_deg, sig)

    def save(self, path) -> None:
        np.savez_compressed(path, axes=self.axes, basis=self.basis, m=self.reference_mass_earth,
                            inc=self.inclination_deg, signature=self.signature)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        return cls(z["axes"], z["basis"], float(z["m"]), float(z["inc"]), str(z["signature"]))


def _orthonormal(basis: np.ndarray) -> np.ndarray:
    """(n, 2, n_points) -> (n, 2, n_points) orthonormal bases of each 2-D span (rank-deficient
    spans are zeroed, which contributes no signal)."""
    out = np.zeros_like(basis)
    for i in range(basis.shape[0]):
        b = basis[i].T                                    # (n_points, 2)
        if np.linalg.norm(b) < 1e-12:
            continue
        q, r = np.linalg.qr(b)
        if abs(r[0, 0]) < 1e-10 or abs(r[1, 1]) < 1e-10 * abs(r[0, 0]):
            continue
        out[i] = q.T
    return out


def scan(basis: PeriodogramBasis, data_vector: np.ndarray) -> dict:
    """Delta chi2, mass and phase at every trial axis for one data vector."""
    n = len(basis.axes)
    delta = np.zeros(n)
    mass = np.full(n, np.nan)
    phase = np.full(n, np.nan)
    for i in range(n):
        b = basis.basis[i].T
        if np.linalg.norm(b) < 1e-12:
            continue
        c, *_ = np.linalg.lstsq(b, data_vector, rcond=None)
        residual = data_vector - b @ c
        delta[i] = float(data_vector @ data_vector - residual @ residual)
        mass[i] = basis.reference_mass_earth * float(np.hypot(c[0], c[1]))
        phase[i] = float(np.mod(np.arctan2(c[1], c[0]), 2 * np.pi))
    return {"axes": basis.axes, "delta_chi2": delta, "mass_earth": mass, "mean_anomaly": phase}


def select_peaks(result: dict, n_peaks: int = 6, min_separation_widths: float = 6.0,
                 widths: np.ndarray | None = None) -> list:
    """Strongest, well-separated peaks (neighbouring grid points are not independent)."""
    order = np.argsort(-result["delta_chi2"])
    chosen = []
    for i in order:
        a = result["axes"][i]
        sep = min_separation_widths * (widths[i] if widths is not None else 0.003)
        if all(abs(a - result["axes"][j]) >= sep for j in chosen):
            chosen.append(int(i))
        if len(chosen) >= n_peaks:
            break
    return chosen


def null_distribution(likelihood, basis: PeriodogramBasis, n_trials: int = 4000, seed: int = 0) -> np.ndarray:
    """Maximum Delta chi2 of the periodogram under PURE NOISE, for the same pattern and basis.

    Exact and cheap because the basis does not depend on the data: a noise-only dataset gives
    d = P xi with xi ~ N(0, I) (whitened noise), the control model cancelling out.
    """
    rng = np.random.default_rng(seed)
    n = likelihood.n_points
    projector = np.zeros((n, n))
    off = 0
    for k in likelihood.letters:
        m = len(likelihood.epochs[k])
        projector[off:off + m, off:off + m] = likelihood._projector[k]
        off += m
    q = _orthonormal(basis.basis)                         # (A, 2, n)
    xi = rng.standard_normal((n, n_trials))
    d = projector @ xi                                     # (n, T)
    proj = np.einsum("aqn,nt->aqt", q, d)                  # (A, 2, T)
    delta = np.sum(proj**2, axis=1)                        # (A, T)
    return np.max(delta, axis=0)


def false_alarm_probability(statistic: float, null_max: np.ndarray) -> float:
    """P(max Delta chi2 >= statistic | no planet), with the standard +1 correction."""
    return float((np.sum(null_max >= statistic) + 1.0) / (len(null_max) + 1.0))
