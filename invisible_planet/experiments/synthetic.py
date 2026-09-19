"""
Generate synthetic observations of a system containing a hidden planet.

This is the only place where a hidden planet's parameters meet the observation model, and it
lives in experiments/, which the inference is forbidden to import. The result is an
ObserverDataset: noisy transit times at the REAL observed epochs, with the REAL published
per-transit uncertainties, and no field able to carry the hidden parameters.
"""

from __future__ import annotations

from ..observations.dataset import ObserverDataset
from ..observations.noise import TimingNoiseModel
from ..observations.pattern import load_pattern
from ..observations.transits import observe_transits
from ..systems.kepler90 import (
    DEFAULT_DURATION_DAYS,
    DEFAULT_EPOCH_BJD,
    load_reference,
    recommended_timestep,
)
from ..systems.planet_x import PlanetXParameters, build_system_with_planet_x


def simulate_transit_times(
    parameters: PlanetXParameters,
    reference: dict | None = None,
    steps_per_inner_orbit: int = 1000,
) -> dict:
    """Integrate the system WITH the hidden planet once; return noiseless transit times (BJD)
    of every probe planet. Independent noise realisations of the same planet share this
    integration - only the measurement noise differs between them."""
    ref = reference if reference is not None else load_reference()
    pattern = load_pattern()
    system = build_system_with_planet_x(parameters, reference=ref, epoch_bjd=DEFAULT_EPOCH_BJD)
    names = [f"Kepler-90 {k}" for k in pattern.probe_letters]
    series = observe_transits(
        system, DEFAULT_DURATION_DAYS, recommended_timestep(ref, steps_per_inner_orbit), tracked=names
    )
    return {k: DEFAULT_EPOCH_BJD + series.for_planet(f"Kepler-90 {k}") for k in pattern.probe_letters}


def observations_from_times(
    sim_times: dict,
    seed: int,
    noise_scale: float = 1.0,
    reference: dict | None = None,
) -> ObserverDataset:
    """Sample noiseless transit times as Kepler sampled the real system (real observing pattern,
    real per-transit uncertainties, noise drawn from them)."""
    ref = reference if reference is not None else load_reference()
    pattern = load_pattern()
    noise = TimingNoiseModel(pattern, seed=seed, scale=noise_scale)
    return ObserverDataset.from_simulation(
        sim_times, pattern, ref, noise, DEFAULT_EPOCH_BJD, DEFAULT_DURATION_DAYS,
        provenance={"generator": "invisible_planet.experiments.synthetic",
                    "note": "Simulated observations. The number of bodies is not disclosed."},
    )


def simulate_observations(
    parameters: PlanetXParameters,
    seed: int,
    noise_scale: float = 1.0,
    reference: dict | None = None,
    steps_per_inner_orbit: int = 1000,
) -> ObserverDataset:
    """Integrate the system WITH the hidden planet and sample it as Kepler sampled the real one."""
    sim_times = simulate_transit_times(parameters, reference, steps_per_inner_orbit)
    return observations_from_times(sim_times, seed, noise_scale, reference)
