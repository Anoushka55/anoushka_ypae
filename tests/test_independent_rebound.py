"""
Independent accuracy validation against REBOUND (IAS15).

Unlike the rest of the suite, this checks the engine against a completely separate
N-body code on the FULL 8-planet system, using a separately written transit finder.
IAS15 is 15th order with an adaptive step tuned to machine precision, so over these
durations it is treated as the exact solution of the same initial-value problem.

The tests skip (rather than silently pass) if REBOUND is not installed; it is a
development dependency: `uv sync --extra dev`.
"""

from __future__ import annotations

import numpy as np
import pytest

rebound = pytest.importorskip("rebound")
pytest.importorskip("scipy")

from invisible_planet.constants import SECONDS_PER_DAY  # noqa: E402
from invisible_planet.observations.transits import observe_transits  # noqa: E402
from invisible_planet.observations.ttv import fit_linear_ephemeris  # noqa: E402
from invisible_planet.systems.kepler90 import (  # noqa: E402
    build_kepler90,
    load_reference,
    recommended_timestep,
)
from invisible_planet.validation.independent import rebound_transit_times  # noqa: E402

DAYS = 400.0
PROBES = ["Kepler-90 b", "Kepler-90 c", "Kepler-90 i", "Kepler-90 d", "Kepler-90 e", "Kepler-90 f"]


@pytest.fixture(scope="module")
def setup():
    reference = load_reference()
    system = build_kepler90(reference=reference)
    exact = rebound_transit_times(system, DAYS, PROBES)
    return reference, system, exact


def _oc_error_seconds(system, exact, dt):
    engine_series = observe_transits(system, DAYS, dt, tracked=PROBES)
    out = {}
    for name in PROBES:
        a, b = engine_series.for_planet(name), exact[name]
        assert len(a) == len(b), f"{name}: {len(a)} transits vs {len(b)} from IAS15"
        fa, fb = fit_linear_ephemeris(a), fit_linear_ephemeris(b)
        out[name] = (
            float(np.max(np.abs(fa.residuals - fb.residuals)) * SECONDS_PER_DAY),
            float(fb.rms_seconds),
        )
    return out


def test_engine_agrees_with_ias15_on_the_ttv_signal(setup):
    """The O-C signal from the engine must match IAS15 to within 5% of its own RMS.

    The 5% bound is the requirement, not the observed value: the measured error at
    the production timestep is ~2.5% of the signal for planet b (the worst case,
    because b sets the timestep) and 0.003% for planet d.
    """
    reference, system, exact = setup
    errors = _oc_error_seconds(system, exact, recommended_timestep(reference))
    for name, (error, signal) in errors.items():
        assert error < 0.05 * signal, f"{name}: O-C error {error:.3f} s vs signal RMS {signal:.1f} s"


def test_outer_probe_planets_are_numerically_negligible(setup):
    """For planets d, e, f the numerical O-C error must be below 0.05 s."""
    reference, system, exact = setup
    errors = _oc_error_seconds(system, exact, recommended_timestep(reference))
    for name in ("Kepler-90 d", "Kepler-90 e", "Kepler-90 f"):
        assert errors[name][0] < 0.05, f"{name}: {errors[name][0]:.4f} s"


def test_error_against_the_exact_solution_converges_at_second_order(setup):
    """Halving dt must reduce the error against IAS15 by ~4x.

    This is the strongest convergence statement in the suite: it is measured against
    an independent, effectively exact solution rather than against the engine itself.
    """
    reference, system, exact = setup
    base = recommended_timestep(reference)
    coarse = _oc_error_seconds(system, exact, base)["Kepler-90 b"][0]
    fine = _oc_error_seconds(system, exact, base / 2.0)["Kepler-90 b"][0]
    ratio = coarse / fine
    assert 3.0 < ratio < 5.5, f"error ratio per halving is {ratio:.2f}, expected ~4"
