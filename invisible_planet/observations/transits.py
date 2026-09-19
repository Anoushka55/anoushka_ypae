"""
Run a system and collect its transit times.

Phase 6.1 deliverable: the bridge between the physics engine and the timing
analysis. Consumes simulation state; never influences it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..physics.engine import NBodyEngine
from ..systems.build import SystemState


@dataclass
class TransitSeries:
    """Transit times for every tracked planet of one system."""

    names: list
    times: list  # list of arrays, days since the integration start
    impact_parameters: list
    span_days: float
    timestep_days: float
    epoch_bjd: float

    def for_planet(self, name: str) -> np.ndarray:
        return self.times[self.names.index(name)]

    def counts(self) -> dict:
        return {n: len(t) for n, t in zip(self.names, self.times)}


def observe_transits_batch(
    systems: list,
    duration_days: float,
    timestep_days: float,
    tracked: list | None = None,
    max_transits: int = 2048,
) -> list:
    """Integrate many systems SIMULTANEOUSLY and return each one's transit times.

    This is what the ensemble-leading array layout exists for. Every system must
    have the same number of bodies in the same order; only their masses and
    initial states differ. Taichi threads across the ensemble dimension, so a
    parameter sweep of K candidates costs far less than K separate runs.

    Used by the Phase 8 parameter sweep and by the Phase 10 inference, where
    thousands of candidate systems are evaluated.
    """
    if not systems:
        return []
    n_bodies = systems[0].n_bodies
    if any(s.n_bodies != n_bodies for s in systems):
        raise ValueError("all systems in a batch must have the same number of bodies")

    tracked_names = tracked if tracked is not None else systems[0].names[1:]
    tracked_indices = [systems[0].index_of(n) for n in tracked_names]

    k = len(systems)
    masses = np.stack([s.masses for s in systems])
    positions = np.stack([s.positions for s in systems])
    velocities = np.stack([s.velocities for s in systems])

    engine = NBodyEngine(n_bodies=n_bodies, n_ensembles=k)
    engine.enable_transit_detection(
        tracked_bodies=tracked_indices,
        max_transits=max_transits,
        central_index=0,
        stellar_radius=float(systems[0].radii[0]),
        planet_radii=np.asarray([systems[0].radii[i] for i in tracked_indices]),
    )
    engine.set_state(masses, positions, velocities, time=0.0)
    engine.run(timestep_days, int(round(duration_days / timestep_days)))

    all_times = engine.get_transit_times()
    all_impacts = engine.get_transit_impact_parameters()

    return [
        TransitSeries(
            names=list(tracked_names),
            times=all_times[i],
            impact_parameters=all_impacts[i],
            span_days=engine.time,
            timestep_days=timestep_days,
            epoch_bjd=systems[i].epoch,
        )
        for i in range(k)
    ]


def observe_transits(
    system: SystemState,
    duration_days: float,
    timestep_days: float,
    tracked: list | None = None,
    max_transits: int = 4096,
) -> TransitSeries:
    """Integrate `system` and return the transit times of its planets.

    Times are returned in DAYS SINCE THE EPOCH of the system, matching the
    convention used throughout the project. Add `system.epoch` to convert to BJD.
    """
    tracked_names = tracked if tracked is not None else system.names[1:]
    tracked_indices = [system.index_of(n) for n in tracked_names]

    engine = NBodyEngine(n_bodies=system.n_bodies)
    engine.enable_transit_detection(
        tracked_bodies=tracked_indices,
        max_transits=max_transits,
        central_index=0,
        stellar_radius=float(system.radii[0]),
        planet_radii=np.asarray([system.radii[i] for i in tracked_indices]),
    )
    engine.set_state(system.masses, system.positions, system.velocities, time=0.0)

    n_steps = int(round(duration_days / timestep_days))
    engine.run(timestep_days, n_steps)

    times = engine.get_transit_times()[0]
    impacts = engine.get_transit_impact_parameters()[0]

    return TransitSeries(
        names=list(tracked_names),
        times=times,
        impact_parameters=impacts,
        span_days=engine.time,
        timestep_days=timestep_days,
        epoch_bjd=system.epoch,
    )
