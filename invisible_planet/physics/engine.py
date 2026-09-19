"""
Newtonian N-body engine.

Phase 4 production engine, validated in Phase 3 on the two-body problem.

EQUATION OF MOTION — the only force in this project
---------------------------------------------------
For body i, with no softening, no drag, no fudge terms:

    a_i = G * sum_{j != i}  m_j * (r_j - r_i) / |r_j - r_i|^3

Every body obeys this, including the synthetic Planet X. There is no special
"hidden planet force" and no scripted trajectory anywhere in the codebase.

WHY NO SOFTENING
----------------
The galaxy-scale prototype in prototypes/ uses Plummer softening, which is
correct there (its particles represent huge stellar populations, not points).
Here, softening would be a *lie*: it would alter the two-body force law that the
entire inference rests on. A planetary system that is dynamically stable never
brings two bodies close enough to need it. If a close approach does occur, that
is physically meaningful information about instability and must be REPORTED, not
smoothed away — see `min_separation()`.

INTEGRATOR — velocity Verlet in kick-drift-kick form
-----------------------------------------------------
    v(t + dt/2) = v(t)        + a(t)      * dt/2      [kick]
    r(t + dt)   = r(t)        + v(t+dt/2) * dt        [drift]
    a(t + dt)   = a(r(t+dt))                          [one force evaluation]
    v(t + dt)   = v(t+dt/2)   + a(t+dt)   * dt/2      [kick]

This is symplectic and time-reversible, so the energy error stays bounded and
oscillatory instead of growing secularly as it does with forward Euler. It is
symplectic ONLY at fixed dt: dt must never be varied mid-run, and must never be
tied to a frame rate.

ARRAY LAYOUT
------------
All state carries a leading ENSEMBLE index k:  shape (K, N).
K = 1 for a single simulation; K = many when the inference stage integrates a
population of candidate Planet X parameters simultaneously. This is where Taichi
actually earns its place, since N ~ 10 is far too small for per-body parallelism
to matter.
"""

# NOTE: `from __future__ import annotations` must NOT be added to this file.
# It converts annotations to strings (PEP 563), which Taichi cannot resolve when
# parsing @ti.kernel signatures, producing
# "Invalid type annotation (argument 1) of Taichi kernel: ti.f64".
from typing import Optional

import numpy as np
import taichi as ti

from ..constants import G
from .backend import init_taichi


@ti.data_oriented
class NBodyEngine:
    """Fixed-step velocity-Verlet N-body integrator over an ensemble of systems."""

    def __init__(self, n_bodies: int, n_ensembles: int = 1, arch: str = "cpu"):
        init_taichi(arch)
        self.n = int(n_bodies)
        self.k = int(n_ensembles)

        shape = (self.k, self.n)
        self.pos = ti.Vector.field(3, ti.f64, shape=shape)
        self.vel = ti.Vector.field(3, ti.f64, shape=shape)
        self.acc = ti.Vector.field(3, ti.f64, shape=shape)
        self.partial_acc = ti.Vector.field(3, ti.f64, shape=shape)
        self.mass = ti.field(ti.f64, shape=shape)

        # scalar diagnostics, one per ensemble member
        self._ke = ti.field(ti.f64, shape=self.k)
        self._pe = ti.field(ti.f64, shape=self.k)
        self._min_sep = ti.field(ti.f64, shape=self.k)
        self._mom = ti.Vector.field(3, ti.f64, shape=self.k)
        self._angmom = ti.Vector.field(3, ti.f64, shape=self.k)
        self._com = ti.Vector.field(3, ti.f64, shape=self.k)
        self._com_vel = ti.Vector.field(3, ti.f64, shape=self.k)

        self.time = 0.0

        # ------------------------------------------------------------------
        # Transit detection support.
        #
        # WHY THIS LIVES IN THE ENGINE: detecting a transit centre to the
        # required sub-second precision requires the state at the two steps that
        # BRACKET the event, together with velocity and acceleration there. Doing
        # it outside the integration loop would mean either reading the full
        # state back every step (which made an earlier version ~25x slower) or
        # quantising transit times to the timestep (~10 minutes), which is far
        # larger than the signals being measured.
        #
        # Only the GEOMETRY of the observation is computed here. The force law is
        # untouched, and all interpretation — ephemeris fitting, O-C residuals,
        # measurement noise — lives in observations/, which never imports this.
        # ------------------------------------------------------------------
        self.transit_detection_enabled = False
        self.tracked_bodies: list[int] = []
        self.max_transits = 0
        self._n_tracked = 0

    # ------------------------------------------------------------------
    # state I/O
    # ------------------------------------------------------------------
    def set_state(
        self,
        masses: np.ndarray,
        positions: np.ndarray,
        velocities: np.ndarray,
        time: float = 0.0,
    ) -> None:
        """Load initial conditions. Accepts (N,) / (N,3) and broadcasts across the
        ensemble, or full (K,N) / (K,N,3) arrays."""
        m = np.asarray(masses, dtype=np.float64)
        p = np.asarray(positions, dtype=np.float64)
        v = np.asarray(velocities, dtype=np.float64)

        if m.ndim == 1:
            m = np.broadcast_to(m, (self.k, self.n)).copy()
        if p.ndim == 2:
            p = np.broadcast_to(p, (self.k, self.n, 3)).copy()
        if v.ndim == 2:
            v = np.broadcast_to(v, (self.k, self.n, 3)).copy()

        if m.shape != (self.k, self.n):
            raise ValueError(f"masses shape {m.shape} != {(self.k, self.n)}")
        if p.shape != (self.k, self.n, 3) or v.shape != (self.k, self.n, 3):
            raise ValueError("position/velocity shape mismatch")

        self.mass.from_numpy(m)
        self.pos.from_numpy(p)
        self.vel.from_numpy(v)
        self.time = float(time)
        self._compute_accelerations()  # velocity Verlet needs a valid a(t) on entry

    def get_positions(self) -> np.ndarray:
        return self.pos.to_numpy()

    def get_velocities(self) -> np.ndarray:
        return self.vel.to_numpy()

    def get_masses(self) -> np.ndarray:
        return self.mass.to_numpy()

    # ------------------------------------------------------------------
    # transit detection setup
    # ------------------------------------------------------------------
    def enable_transit_detection(
        self,
        tracked_bodies: list,
        max_transits: int = 512,
        central_index: int = 0,
        stellar_radius: float = 0.0,
        planet_radii=None,
    ) -> None:
        """Record transit centre times for the given bodies during integration.

        A transit centre is the instant of MINIMUM sky-projected star-planet
        separation while the planet is in front of the star (z_planet > z_star,
        with +Z towards the observer). This is the standard geometric definition
        and coincides with the midpoint of a symmetric transit light curve.
        """
        self.tracked_bodies = list(tracked_bodies)
        self._n_tracked = len(self.tracked_bodies)
        self.max_transits = int(max_transits)
        self.central_index = int(central_index)
        self.stellar_radius = float(stellar_radius)

        shape = (self.k, self._n_tracked)
        self._tracked_index = ti.field(ti.i32, shape=self._n_tracked)
        self._tracked_index.from_numpy(np.asarray(self.tracked_bodies, dtype=np.int32))

        self._planet_radius = ti.field(ti.f64, shape=self._n_tracked)
        radii = (
            np.zeros(self._n_tracked)
            if planet_radii is None
            else np.asarray(planet_radii, dtype=np.float64)
        )
        self._planet_radius.from_numpy(radii)

        self.transit_times = ti.field(ti.f64, shape=(self.k, self._n_tracked, self.max_transits))
        self.transit_impact = ti.field(ti.f64, shape=(self.k, self._n_tracked, self.max_transits))
        self.transit_count = ti.field(ti.i32, shape=shape)
        self._overflow = ti.field(ti.i32, shape=shape)

        # bracketing state for the root find
        self._prev_sdot = ti.field(ti.f64, shape=shape)
        self._prev_sddot = ti.field(ti.f64, shape=shape)
        self._prev_valid = ti.field(ti.i32, shape=shape)

        self.transit_detection_enabled = True
        self._reset_transit_state()

    @ti.kernel
    def _reset_transit_state(self):
        for k, p in ti.ndrange(self.k, self._n_tracked):
            self.transit_count[k, p] = 0
            self._prev_valid[k, p] = 0
            self._prev_sdot[k, p] = 0.0
            self._prev_sddot[k, p] = 0.0
            self._overflow[k, p] = 0

    def reset_transits(self) -> None:
        if self.transit_detection_enabled:
            self._reset_transit_state()

    def get_transit_times(self) -> list:
        """Recorded transit times as a list (per ensemble member) of arrays per body."""
        if not self.transit_detection_enabled:
            raise RuntimeError("transit detection is not enabled")
        times = self.transit_times.to_numpy()
        counts = self.transit_count.to_numpy()
        overflow = self._overflow.to_numpy()
        if overflow.any():
            raise RuntimeError(
                "transit buffer overflowed; increase max_transits "
                f"(currently {self.max_transits})"
            )
        return [
            [times[k, p, : counts[k, p]].copy() for p in range(self._n_tracked)]
            for k in range(self.k)
        ]

    def get_transit_impact_parameters(self) -> list:
        times = self.transit_impact.to_numpy()
        counts = self.transit_count.to_numpy()
        return [
            [times[k, p, : counts[k, p]].copy() for p in range(self._n_tracked)]
            for k in range(self.k)
        ]

    # ------------------------------------------------------------------
    # gravity
    # ------------------------------------------------------------------
    @ti.func
    def _evaluate_accelerations(self, k):
        """Newtonian accelerations of every body of ensemble member k.

            a_i = G * sum_{j != i} m_j (r_j - r_i) / |r_j - r_i|^3

        PERFORMANCE NOTE (measured in scripts/profile_engine.py, docs/PERFORMANCE.md): this
        plain double loop is the fastest of the variants tried. Evaluating each pair once and
        accumulating into the global acceleration field was 2.6x SLOWER (Taichi compiles the
        += on a global field to an atomic add); unrolling the loops with ti.static was 1.2x
        slower; a symmetric-pair loop with a local accumulator matrix was within 7% of this
        one and not worth the added complexity. The cost is dominated by the square root and
        division in each of the ~90 pair evaluations per step.
        """
        for i in range(self.n):
            a = ti.Vector([0.0, 0.0, 0.0], dt=ti.f64)
            for j in range(self.n):
                if j != i:
                    d = self.pos[k, j] - self.pos[k, i]
                    r2 = d.dot(d)
                    inv_r3 = 1.0 / (r2 * ti.sqrt(r2))
                    a += d * (G * self.mass[k, j] * inv_r3)
            self.acc[k, i] = a

    @ti.kernel
    def _partial_accelerations(self, source: ti.i32):
        """Acceleration of every body due to body `source` ALONE (the single term j = source
        of the force sum), stored in `partial_acc`."""
        for k, i in ti.ndrange(self.k, self.n):
            a = ti.Vector([0.0, 0.0, 0.0], dt=ti.f64)
            if i != source:
                d = self.pos[k, source] - self.pos[k, i]
                r2 = d.dot(d)
                a = d * (G * self.mass[k, source] / (r2 * ti.sqrt(r2)))
            self.partial_acc[k, i] = a

    def acceleration_due_to(self, source: int) -> np.ndarray:
        """(K, N, 3) acceleration of each body caused by body `source` alone.

        This is one term of the same Newtonian sum that drives the integration, so it is exactly
        the part of each body's acceleration attributable to that source - what the 'perturbation
        vector' display shows for the hidden planet. It adds no force; it only reads one.
        """
        self._partial_accelerations(int(source))
        return self.partial_acc.to_numpy()

    @ti.kernel
    def _compute_accelerations(self):
        for k in range(self.k):
            self._evaluate_accelerations(k)

    @ti.kernel
    def _integrate(self, dt: ti.f64, n_steps: ti.i32, start_time: ti.f64):
        """Fused kick-drift-kick loop.

        PARALLELISATION: the outermost loop is over ENSEMBLE MEMBERS, so Taichi
        threads across candidate systems while each member's time evolution stays
        strictly sequential (as it must — time cannot be parallelised). With N ~ 10
        bodies, per-body parallelism would be worthless, and launching separate
        kernels per step made launch overhead dominate the run time by orders of
        magnitude.
        """
        half = 0.5 * dt
        for k in range(self.k):
            for _step in range(n_steps):
                # --- kick: v(t) -> v(t + dt/2), using the cached a(t) ---
                for i in range(self.n):
                    self.vel[k, i] += self.acc[k, i] * half

                # --- drift: r(t) -> r(t + dt) ---
                for i in range(self.n):
                    self.pos[k, i] += self.vel[k, i] * dt

                # --- single force evaluation at the new positions ---
                self._evaluate_accelerations(k)

                # --- kick: v(t + dt/2) -> v(t + dt) ---
                for i in range(self.n):
                    self.vel[k, i] += self.acc[k, i] * half

                # --- transit detection at the completed step ---
                if ti.static(self.transit_detection_enabled):
                    t_now = start_time + (_step + 1) * dt
                    c = self.central_index
                    for p in range(self._n_tracked):
                        i = self._tracked_index[p]
                        d = self.pos[k, i] - self.pos[k, c]
                        dv = self.vel[k, i] - self.vel[k, c]
                        da = self.acc[k, i] - self.acc[k, c]

                        # s(t) = sky-projected squared separation; a transit centre
                        # is a minimum of s, i.e. a zero of s' with s'' > 0.
                        sdot = 2.0 * (d[0] * dv[0] + d[1] * dv[1])
                        sddot = 2.0 * (
                            dv[0] * dv[0] + dv[1] * dv[1] + d[0] * da[0] + d[1] * da[1]
                        )

                        if self._prev_valid[k, p] == 1:
                            f0 = self._prev_sdot[k, p]
                            f1 = sdot
                            # sign change from negative to positive = minimum of s
                            if f0 < 0.0 and f1 >= 0.0 and d[2] > 0.0:
                                m0 = self._prev_sddot[k, p] * dt
                                m1 = sddot * dt

                                # Cubic Hermite interpolation of s'(t) across the
                                # bracketing step, solved by Newton iteration.
                                # Linear interpolation would quantise the root at
                                # O(dt^2) ~ seconds; the cubic uses the exactly
                                # known derivative s'' at both ends and reaches
                                # sub-millisecond accuracy.
                                u = f0 / (f0 - f1)  # linear first guess
                                for _iter in range(6):
                                    u2 = u * u
                                    u3 = u2 * u
                                    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
                                    h10 = u3 - 2.0 * u2 + u
                                    h01 = -2.0 * u3 + 3.0 * u2
                                    h11 = u3 - u2
                                    value = h00 * f0 + h10 * m0 + h01 * f1 + h11 * m1
                                    dh00 = 6.0 * u2 - 6.0 * u
                                    dh10 = 3.0 * u2 - 4.0 * u + 1.0
                                    dh01 = -6.0 * u2 + 6.0 * u
                                    dh11 = 3.0 * u2 - 2.0 * u
                                    slope = dh00 * f0 + dh10 * m0 + dh01 * f1 + dh11 * m1
                                    if slope != 0.0:
                                        u -= value / slope
                                    u = ti.min(ti.max(u, 0.0), 1.0)

                                t_transit = t_now - dt + u * dt

                                # sky separation at the refined centre, by
                                # Hermite interpolation of the position
                                d_prev = d - dv * dt
                                sep_x = d_prev[0] + dv[0] * (u * dt)
                                sep_y = d_prev[1] + dv[1] * (u * dt)
                                impact = ti.sqrt(sep_x * sep_x + sep_y * sep_y)

                                # only count it if the planet actually crosses the disk
                                limit = self.stellar_radius + self._planet_radius[p]
                                if impact < limit or limit == 0.0:
                                    idx = self.transit_count[k, p]
                                    if idx < self.max_transits:
                                        self.transit_times[k, p, idx] = t_transit
                                        self.transit_impact[k, p, idx] = impact
                                        self.transit_count[k, p] = idx + 1
                                    else:
                                        self._overflow[k, p] = 1

                        self._prev_sdot[k, p] = sdot
                        self._prev_sddot[k, p] = sddot
                        self._prev_valid[k, p] = 1

    def step(self, dt: float) -> None:
        """Advance every ensemble member by one fixed velocity-Verlet step."""
        self._integrate(dt, 1, self.time)
        self.time += dt

    def run(self, dt: float, n_steps: int) -> None:
        """Advance by n_steps of fixed size dt.

        dt must be identical for every step of a run: velocity Verlet is
        symplectic only at constant step size.
        """
        n = int(n_steps)
        if n <= 0:
            return
        self._integrate(dt, n, self.time)
        self.time += dt * n

    # ------------------------------------------------------------------
    # conservation diagnostics
    # ------------------------------------------------------------------
    @ti.kernel
    def _compute_diagnostics(self):
        for k in range(self.k):
            self._ke[k] = 0.0
            self._pe[k] = 0.0
            self._min_sep[k] = 1.0e30
            self._mom[k] = ti.Vector([0.0, 0.0, 0.0], dt=ti.f64)
            self._angmom[k] = ti.Vector([0.0, 0.0, 0.0], dt=ti.f64)
            self._com[k] = ti.Vector([0.0, 0.0, 0.0], dt=ti.f64)
            self._com_vel[k] = ti.Vector([0.0, 0.0, 0.0], dt=ti.f64)

        for k, i in ti.ndrange(self.k, self.n):
            m = self.mass[k, i]
            v = self.vel[k, i]
            r = self.pos[k, i]
            self._ke[k] += 0.5 * m * v.dot(v)
            self._mom[k] += m * v
            self._angmom[k] += m * r.cross(v)
            self._com[k] += m * r
            self._com_vel[k] += m * v

        # potential energy: strictly j > i so no pair is counted twice
        for k, i in ti.ndrange(self.k, self.n):
            for j in range(i + 1, self.n):
                d = self.pos[k, j] - self.pos[k, i]
                r = ti.sqrt(d.dot(d))
                self._pe[k] -= G * self.mass[k, i] * self.mass[k, j] / r
                ti.atomic_min(self._min_sep[k], r)

        for k in range(self.k):
            total_mass = 0.0
            for i in range(self.n):
                total_mass += self.mass[k, i]
            self._com[k] /= total_mass
            self._com_vel[k] /= total_mass

    def diagnostics(self) -> dict[str, np.ndarray]:
        """Conserved quantities for every ensemble member.

        Returns arrays of shape (K,) or (K,3). Nothing here is smoothed or
        hidden: drift in these numbers is the honest error of the integration.
        """
        self._compute_diagnostics()
        ke = self._ke.to_numpy()
        pe = self._pe.to_numpy()
        return {
            "kinetic_energy": ke,
            "potential_energy": pe,
            "total_energy": ke + pe,
            "linear_momentum": self._mom.to_numpy(),
            "angular_momentum": self._angmom.to_numpy(),
            "center_of_mass": self._com.to_numpy(),
            "center_of_mass_velocity": self._com_vel.to_numpy(),
            "min_separation": self._min_sep.to_numpy(),
            "time": self.time,
        }

    def total_energy(self) -> np.ndarray:
        return self.diagnostics()["total_energy"]

    def min_separation(self) -> np.ndarray:
        return self.diagnostics()["min_separation"]
