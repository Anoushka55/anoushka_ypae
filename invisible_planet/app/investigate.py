"""
Investigation mode.

Phase 15 deliverable. The user tests their own hypothesis about the unseen body
and sees, live, how well it explains the observed transit times.

EVERY CONTROL IS A REAL MODEL PARAMETER
----------------------------------------
There are no arbitrary game mechanics. The sliders change the mass, semi-major
axis and orbital phase of a candidate planet in a full N-body integration, and
the chi-squared shown is the same statistic the automated inference minimises.
Two systems are integrated side by side with identical visible planets:

    OBSERVED  : the system containing the hidden body (the "data")
    CANDIDATE : the user's hypothesis

If the candidate is right, its O-C curve tracks the observed one and chi2 falls.
"""

from __future__ import annotations

import numpy as np
import taichi as ti

from ..constants import SECONDS_PER_DAY
from ..experiments.config import DEFAULT_CONFIG
from ..observations.ttv import fit_linear_ephemeris
from ..physics.engine import NBodyEngine
from ..render.scene import SystemRenderer
from ..systems.kepler90 import load_reference
from ..systems.planet_x import PlanetXParameters, build_system_with_planet_x

PROBE_LETTER = "d"
SEGMENT_DAYS = 1400.0


def _oc(times: np.ndarray):
    if len(times) < 3:
        return None
    return fit_linear_ephemeris(times)


def _chi2(candidate_fit, observed_fit, sigma_days: float) -> float:
    """Chi-squared between two O-C series, matched on epoch."""
    if candidate_fit is None or observed_fit is None:
        return float("nan")
    candidate = dict(zip(candidate_fit.epochs, candidate_fit.residuals))
    observed = dict(zip(observed_fit.epochs, observed_fit.residuals))
    shared = set(candidate) & set(observed)
    if not shared:
        return float("nan")
    return float(
        sum(((candidate[e] - observed[e]) / sigma_days) ** 2 for e in shared)
    )


def _sparkline(values: np.ndarray, width: int = 44) -> str:
    if values is None or len(values) == 0:
        return ""
    glyphs = " .:-=+*#%@"
    lo, hi = float(np.min(values)), float(np.max(values))
    if hi - lo < 1e-12:
        return "flat"
    sampled = values[np.linspace(0, len(values) - 1, min(width, len(values))).astype(int)]
    scaled = (sampled - lo) / (hi - lo)
    return "".join(glyphs[int(v * (len(glyphs) - 1))] for v in scaled)


def _make_engine(system, probe_index):
    engine = NBodyEngine(n_bodies=system.n_bodies)
    engine.enable_transit_detection(
        tracked_bodies=[probe_index],
        max_transits=512,
        stellar_radius=float(system.radii[0]),
        planet_radii=np.asarray([system.radii[probe_index]]),
    )
    engine.set_state(system.masses, system.positions, system.velocities, time=0.0)
    return engine


def run() -> None:
    config = DEFAULT_CONFIG
    reference = load_reference()
    probe_name = f"Kepler-90 {PROBE_LETTER}"
    sigma_days = reference["planets"][PROBE_LETTER]["transit_epoch_bjd"]["uncertainty"]

    # The "observations": the system that really contains the hidden body.
    observed_system = build_system_with_planet_x(config.planet_x, reference=reference)
    probe_index = observed_system.index_of(probe_name)
    observed_engine = _make_engine(observed_system, probe_index)

    # The user's hypothesis. Starts deliberately wrong.
    candidate = PlanetXParameters(
        mass_earth=20.0, semi_major_axis_au=0.19, mean_anomaly=0.0, inclination_deg=88.0
    )
    candidate_system = build_system_with_planet_x(candidate, reference=reference)
    candidate_engine = _make_engine(candidate_system, probe_index)

    renderer = SystemRenderer(n_bodies=candidate_system.n_bodies)
    paused = False
    steps_per_frame = 60

    print("=" * 70)
    print("THE INVISIBLE PLANET - investigation mode")
    print("=" * 70)
    print("Planet X is SYNTHETIC. Not a claim about the real Kepler-90.")
    print()
    print("  Adjust your hypothesis with the sliders, then press R to restart")
    print("  the integration and see whether it explains the observed timings.")
    print()
    print("  SPACE pause/resume   R restart with current hypothesis   ESC quit")
    print("  LMB orbit camera     W/S zoom")
    print()

    def restart():
        nonlocal candidate_system, candidate_engine, observed_engine
        candidate_system = build_system_with_planet_x(candidate, reference=reference)
        candidate_engine = _make_engine(candidate_system, probe_index)
        observed_engine = _make_engine(observed_system, probe_index)

    mass = candidate.mass_earth
    axis = candidate.semi_major_axis_au
    phase = candidate.mean_anomaly

    while renderer.running():
        for event in renderer.window.get_events(ti.ui.PRESS):
            if event.key == ti.ui.ESCAPE:
                renderer.window.running = False
            elif event.key == ti.ui.SPACE:
                paused = not paused
            elif event.key == "r":
                candidate = PlanetXParameters(
                    mass_earth=float(mass),
                    semi_major_axis_au=float(axis),
                    mean_anomaly=float(phase),
                    inclination_deg=88.0,
                )
                restart()

        if not paused and candidate_engine.time < SEGMENT_DAYS:
            candidate_engine.run(config.timestep_days, steps_per_frame)
            observed_engine.run(config.timestep_days, steps_per_frame)

        positions = candidate_engine.get_positions()[0]
        renderer.trails.push(positions)
        renderer.draw(
            positions, candidate_system.radii, float(candidate_system.radii[0]),
            reveal_planet_x=True,
        )

        candidate_fit = _oc(candidate_engine.get_transit_times()[0][0])
        observed_fit = _oc(observed_engine.get_transit_times()[0][0])
        chi2 = _chi2(candidate_fit, observed_fit, sigma_days)

        gui = renderer.window.get_gui()
        with gui.sub_window("Your hypothesis", 0.01, 0.01, 0.34, 0.24) as panel:
            panel.text("Every control is a real model parameter.")
            mass = panel.slider_float("mass [M_earth]", mass, 1.0, 150.0)
            axis = panel.slider_float("semi-major axis [au]", axis, 0.14, 0.30)
            phase = panel.slider_float("phase [rad]", phase, 0.0, 6.283)
            panel.text("")
            panel.text("Press R to run this hypothesis.")

        with gui.sub_window("Observed vs your model", 0.01, 0.27, 0.42, 0.26) as panel:
            if observed_fit is not None:
                panel.text(f"OBSERVED  planet {PROBE_LETTER}: "
                           f"{observed_fit.n_transits} transits")
                panel.text(f"  O-C p-p {observed_fit.peak_to_peak_minutes:7.2f} min")
                panel.text("  " + _sparkline(observed_fit.residuals))
            else:
                panel.text("OBSERVED: collecting transits...")
            panel.text("")
            if candidate_fit is not None:
                panel.text(f"YOUR MODEL: {candidate_fit.n_transits} transits")
                panel.text(f"  O-C p-p {candidate_fit.peak_to_peak_minutes:7.2f} min")
                panel.text("  " + _sparkline(candidate_fit.residuals))
            else:
                panel.text("YOUR MODEL: collecting transits...")

        with gui.sub_window("Goodness of fit", 0.01, 0.55, 0.42, 0.16) as panel:
            if np.isfinite(chi2):
                panel.text(f"chi2 = {chi2:10.1f}")
                panel.text("Lower is better. The automated inference minimises")
                panel.text("exactly this statistic.")
            else:
                panel.text("chi2: not enough transits yet")
            panel.text(f"t = {candidate_engine.time:8.1f} d of {SEGMENT_DAYS:.0f} d")
            panel.text("PAUSED" if paused else "running")

        renderer.show()
