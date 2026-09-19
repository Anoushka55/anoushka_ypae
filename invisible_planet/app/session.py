"""
A live simulation session, independent of any window.

Wraps the N-body engine for the demonstration and investigation modes: it advances the physics,
collects transits, and exposes everything the UI displays - conservation errors, osculating
elements, the O-C of any planet, and the perturbation a hidden planet exerts on a visible one.

Keeping this separate from the drawing code means every number the UI shows can be tested
headlessly, and the UI can never influence the physics: it only READS from a Session.

The fixed timestep is the one derived from the convergence study and the IAS15 comparison;
`advance` takes a number of STEPS, never a step SIZE, so the speed control changes how many
fixed-size steps run per frame and cannot alter the numerics.
"""

from __future__ import annotations

import numpy as np

from ..constants import G, SECONDS_PER_DAY
from ..observations.ttv import fit_linear_ephemeris
from ..physics.diagnostics import semi_major_axis_and_eccentricity
from ..physics.engine import NBodyEngine
from ..systems.kepler90 import (
    DEFAULT_EPOCH_BJD,
    PLANET_ORDER,
    build_kepler90,
    load_reference,
    recommended_timestep,
)
from ..systems.planet_x import PlanetXParameters, build_system_with_planet_x


class Session:
    def __init__(self, hidden: PlanetXParameters | None = None, reference: dict | None = None,
                 steps_per_inner_orbit: int = 1000):
        self.reference = reference if reference is not None else load_reference()
        self.hidden = hidden
        self.system = (build_system_with_planet_x(hidden, reference=self.reference)
                       if hidden is not None else build_kepler90(reference=self.reference))
        self.dt = recommended_timestep(self.reference, steps_per_inner_orbit)
        self.letters = list(PLANET_ORDER)
        self.n_bodies = self.system.n_bodies
        self.hidden_index = self.n_bodies - 1 if hidden is not None else None

        self.engine = NBodyEngine(n_bodies=self.n_bodies)
        indices = [self.system.index_of(f"Kepler-90 {k}") for k in self.letters]
        self.engine.enable_transit_detection(
            tracked_bodies=indices, max_transits=4096, stellar_radius=float(self.system.radii[0]),
            planet_radii=np.asarray([self.system.radii[i] for i in indices]))
        self.engine.set_state(self.system.masses, self.system.positions, self.system.velocities, time=0.0)

        d0 = self.engine.diagnostics()
        self._e0 = float(d0["total_energy"][0])
        self._l0 = np.array(d0["angular_momentum"][0])
        self._p_scale = float(np.sum(self.system.masses * np.linalg.norm(self.system.velocities, axis=1)))
        self._diag_cache = None
        self._diag_time = -1.0
        self._seen = {k: 0 for k in self.letters}

    # ------------------------------------------------------------------
    @property
    def time_days(self) -> float:
        return float(self.engine.time)

    @property
    def epoch_bjd(self) -> float:
        return float(self.system.epoch)

    def advance(self, n_steps: int) -> None:
        self.engine.run(self.dt, int(n_steps))

    def positions(self) -> np.ndarray:
        return self.engine.get_positions()[0]

    def velocities(self) -> np.ndarray:
        return self.engine.get_velocities()[0]

    # ------------------------------------------------------------------
    def diagnostics(self, max_age_days: float = 5.0) -> dict:
        """Conservation errors, refreshed at most every `max_age_days` of simulated time
        (each evaluation costs ~35 ms, which would be wasteful every frame)."""
        if self._diag_cache is None or abs(self.time_days - self._diag_time) >= max_age_days:
            d = self.engine.diagnostics()
            lvec = np.array(d["angular_momentum"][0])
            self._diag_cache = {
                "time_days": self.time_days,
                "energy_error": abs(float(d["total_energy"][0]) - self._e0) / abs(self._e0),
                "angular_momentum_error": float(np.linalg.norm(lvec - self._l0) / np.linalg.norm(self._l0)),
                "linear_momentum_error": float(np.linalg.norm(d["linear_momentum"][0]) / self._p_scale),
                "min_separation_au": float(d["min_separation"][0]),
            }
            self._diag_time = self.time_days
        return self._diag_cache

    def osculating(self, letter: str) -> dict:
        """Astrocentric osculating a, e and the corresponding period of a visible planet."""
        i = self.system.index_of(f"Kepler-90 {letter}")
        pos, vel = self.positions(), self.velocities()
        mu = G * (self.system.masses[0] + self.system.masses[i])
        a, e = semi_major_axis_and_eccentricity(np.array([mu]), (pos[i] - pos[0])[None], (vel[i] - vel[0])[None])
        a, e = float(a[0]), float(e[0])
        return {"a_au": a, "e": e, "period_days": float(2.0 * np.pi * np.sqrt(a**3 / mu))}

    def transit_bjd(self, letter: str) -> np.ndarray:
        idx = self.letters.index(letter)
        return self.epoch_bjd + np.asarray(self.engine.get_transit_times()[0][idx])

    def o_minus_c(self, letter: str) -> dict | None:
        """O-C of the transits recorded so far (needs at least three)."""
        t = self.transit_bjd(letter)
        if len(t) < 3:
            return None
        period_guess = self.reference["planets"][letter]["orbital_period_days"]["value"]
        epochs = np.round((t - t[0]) / period_guess).astype(np.int64)
        fit = fit_linear_ephemeris(t, epochs=epochs)
        return {"epochs": epochs, "residual_minutes": fit.residuals * 1440.0, "period_days": fit.period,
                "n_transits": fit.n_transits, "rms_seconds": fit.rms_seconds,
                "peak_to_peak_minutes": fit.peak_to_peak_minutes}

    # ------------------------------------------------------------------
    def perturbation_from_hidden(self, letter: str) -> dict | None:
        """Acceleration of a visible planet due to the hidden planet ALONE, and its size relative
        to the acceleration the star exerts on it. None if the system has no hidden planet."""
        if self.hidden_index is None:
            return None
        i = self.system.index_of(f"Kepler-90 {letter}")
        a_hidden = self.engine.acceleration_due_to(self.hidden_index)[0, i]
        a_star = self.engine.acceleration_due_to(0)[0, i]
        return {"vector_au_per_day2": a_hidden, "magnitude": float(np.linalg.norm(a_hidden)),
                "fraction_of_stellar_pull": float(np.linalg.norm(a_hidden) / np.linalg.norm(a_star))}

    def transit_count(self, letter: str) -> int:
        return int(len(self.engine.get_transit_times()[0][self.letters.index(letter)]))

    def new_transits(self) -> list:
        """Letters of planets that have transited since the last call (for transit markers)."""
        fresh = []
        counts = self.engine.transit_count.to_numpy()[0]
        for index, letter in enumerate(self.letters):
            n = int(counts[index])
            if n > self._seen[letter]:
                fresh.append(letter)
                self._seen[letter] = n
        return fresh

    def days_to_bjd(self, days: float) -> float:
        return DEFAULT_EPOCH_BJD + days
