"""
Independent reference integration with REBOUND (IAS15).

Purpose: an ACCURACY check that does not share code with this project's engine.
Everything else in the test-suite validates the engine against analytic two-body
solutions or against itself (convergence); this validates it against an unrelated,
widely used N-body code on the full multi-planet system.

IAS15 (Rein & Spiegel 2015) is a 15th-order adaptive integrator whose error is at
the level of machine precision for this kind of problem, so over the durations used
here it may be treated as the exact solution of the same initial-value problem.

Only the INTEGRATION is independent. The initial state is handed over unchanged, so
this checks the dynamics and the transit-timing extraction, not the initial
conditions. Transit centres are located here by a separate method (secant/Brent
root-finding on the time derivative of the projected separation, integrating a copy
of the simulation to each trial time) rather than by the Hermite interpolation used
inside the engine.

REBOUND is a development-only dependency (`--extra dev`).
"""

from __future__ import annotations

import numpy as np

from ..constants import G
from ..systems.build import SystemState


def _projected_derivative(sim, index: int) -> tuple[float, float]:
    """Return (d/dt of projected separation squared, z_planet - z_star)."""
    p = sim.particles
    dx = p[index].x - p[0].x
    dy = p[index].y - p[0].y
    dz = p[index].z - p[0].z
    dvx = p[index].vx - p[0].vx
    dvy = p[index].vy - p[0].vy
    return 2.0 * (dx * dvx + dy * dvy), dz


def make_simulation(system: SystemState, epsilon: float = 1e-9):
    import rebound

    sim = rebound.Simulation()
    sim.G = G  # au^3 / (Msun day^2): the same G as the engine
    sim.integrator = "ias15"
    sim.integrator.epsilon = epsilon  # REBOUND >= 5 API (was sim.ri_ias15.epsilon)
    for m, r, v in zip(system.masses, system.positions, system.velocities):
        sim.add(m=float(m), x=r[0], y=r[1], z=r[2], vx=v[0], vy=v[1], vz=v[2])
    sim.t = 0.0
    return sim


def rebound_transit_times(
    system: SystemState,
    duration_days: float,
    tracked_names: list,
    stride_days: float = 0.02,
    epsilon: float = 1e-9,
) -> dict:
    """Transit-centre times [days since epoch] for each tracked body, from IAS15."""
    from scipy.optimize import brentq

    indices = {name: system.index_of(name) for name in tracked_names}
    sim = make_simulation(system, epsilon)
    times = {name: [] for name in tracked_names}

    previous = {name: _projected_derivative(sim, idx) for name, idx in indices.items()}
    snapshot = sim.copy()
    t_previous = sim.t

    n_steps = int(np.ceil(duration_days / stride_days))
    for step in range(1, n_steps + 1):
        t_now = min(step * stride_days, duration_days)
        sim.integrate(t_now, exact_finish_time=1)

        for name, idx in indices.items():
            f_now, z_now = _projected_derivative(sim, idx)
            f_prev, _ = previous[name]
            # minimum of the projected separation while in front of the star
            if f_prev < 0.0 <= f_now and z_now > 0.0:

                def root_function(t, _idx=idx):
                    trial = snapshot.copy()
                    trial.integrate(t, exact_finish_time=1)
                    return _projected_derivative(trial, _idx)[0]

                t_transit = brentq(root_function, t_previous, t_now, xtol=1e-12, rtol=1e-14)
                times[name].append(t_transit)
            previous[name] = (f_now, z_now)

        snapshot = sim.copy()
        t_previous = sim.t

    return {name: np.asarray(v) for name, v in times.items()}


def rebound_final_state(system: SystemState, duration_days: float, epsilon: float = 1e-9):
    """Positions and velocities after `duration_days`, from IAS15."""
    sim = make_simulation(system, epsilon)
    sim.integrate(duration_days, exact_finish_time=1)
    positions = np.array([[p.x, p.y, p.z] for p in sim.particles])
    velocities = np.array([[p.vx, p.vy, p.vz] for p in sim.particles])
    return positions, velocities
