"""
Autonomous demonstration mode.

Phase 14 deliverable. A judge launches one command and watches the scientific
argument unfold with no input required.

THE NARRATIVE FOLLOWS THE PHYSICS, NEVER THE REVERSE
-----------------------------------------------------
Each stage displays quantities the simulation and the real inference produced.
The O-C diagram is measured from the running integration. The "recovered"
parameters are read from data/experiment_result.json, written by
scripts/run_experiment.py — they are NOT re-derived here, and never invented.

Planet X is hidden from the RENDERER during the mystery stages. It is present in
the physics throughout, because that is the whole point: its gravity is what
bends the transit times.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import taichi as ti

from ..constants import SECONDS_PER_DAY
from ..experiments.config import DEFAULT_CONFIG
from ..observations.ttv import fit_linear_ephemeris
from ..physics.engine import NBodyEngine
from ..render.scene import SystemRenderer
from ..systems.kepler90 import build_kepler90, load_reference
from ..systems.planet_x import build_system_with_planet_x

ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = ROOT / "data" / "experiment_result.json"

#: Which planet's transit timing is shown in the O-C panel. Planet d carries the
#: detection: it is the only planet whose induced signal exceeds its own timing
#: uncertainty (see docs/IDENTIFIABILITY.md).
PROBE_LETTER = "d"


@dataclass
class Stage:
    key: str
    title: str
    lines: list
    days: float
    reveal_planet_x: bool


def build_stages(result: dict | None) -> list:
    """The narrative. Numbers come from the recorded experiment where available."""
    if result is not None:
        truth = result["config"]["planet_x_ground_truth"]
        best = result.get("best_parameters", {})
        delta = result["null_hypothesis_chi2"] - result.get("best_chi2", float("nan"))
        recovered = [
            f"recovered mass    : {best.get('mass_earth', float('nan')):.1f} M_earth"
            f"   (true {truth['mass_earth']:.1f})",
            f"recovered a       : {best.get('semi_major_axis_au', float('nan')):.4f} au"
            f"   (true {truth['semi_major_axis_au']:.4f})",
            f"detection         : delta chi2 = {delta:.1f}  (~{np.sqrt(max(delta,0)):.1f} sigma)",
        ]
    else:
        recovered = [
            "No recorded inference result found.",
            "Run: python scripts/run_experiment.py",
        ]

    return [
        Stage(
            key="system",
            title="1. A planetary system",
            lines=[
                "Kepler-90 (KOI-351): eight known planets around a Sun-like star.",
                "Every orbit you see is produced by Newtonian gravity, integrated",
                "with a symplectic velocity-Verlet scheme. Nothing is scripted.",
            ],
            days=120.0,
            reveal_planet_x=False,
        ),
        Stage(
            key="transits",
            title="2. We measure transits",
            lines=[
                "When a planet crosses the star we record the time of minimum",
                "separation. For an isolated planet those times are exactly",
                "periodic: T_n = T_0 + n P.",
                "Residuals for an isolated planet stay below 1 second.",
            ],
            days=400.0,
            reveal_planet_x=False,
        ),
        Stage(
            key="anomaly",
            title="3. Something is wrong",
            lines=[
                f"Planet {PROBE_LETTER}'s transits are NOT periodic.",
                "The observed-minus-calculated residuals drift by tens of minutes,",
                "far beyond the measurement uncertainty of a few hundred seconds.",
                "Gravity from something we have not accounted for is at work.",
            ],
            days=700.0,
            reveal_planet_x=False,
        ),
        Stage(
            key="inference",
            title="4. Inferring the unseen body",
            lines=[
                "The inference sees only noisy transit times. It never sees the",
                "hidden planet. It scans candidate periods, running a full N-body",
                "simulation for each, and scores them by chi-squared.",
                "The chi2 minimum in period is only ~0.25% wide, so a dense",
                "periodogram is required; a coarse search finds a false answer.",
            ],
            days=1000.0,
            reveal_planet_x=False,
        ),
        Stage(
            key="reveal",
            title="5. The invisible planet, revealed",
            lines=recovered
            + [
                "",
                "PLANET X IS SYNTHETIC. This is a controlled numerical experiment,",
                "not a claim that Kepler-90 has a ninth planet.",
            ],
            days=1400.0,
            reveal_planet_x=True,
        ),
    ]


def oc_panel_text(times: np.ndarray, letter: str) -> list:
    """O-C summary computed live from the running simulation."""
    if len(times) < 3:
        return [f"planet {letter}: {len(times)} transits so far (need 3 for O-C)"]
    fit = fit_linear_ephemeris(times)
    residuals = fit.residuals * SECONDS_PER_DAY
    return [
        f"planet {letter}: {fit.n_transits} transits",
        f"fitted period   : {fit.period:.5f} d",
        f"O-C rms         : {fit.rms_seconds:8.1f} s",
        f"O-C peak-to-peak: {fit.peak_to_peak_minutes:8.2f} min",
        _sparkline(residuals),
    ]


def _sparkline(values: np.ndarray, width: int = 46) -> str:
    """A tiny text O-C plot, so the signal is visible without a plotting window."""
    if len(values) == 0:
        return ""
    glyphs = " .:-=+*#%@"
    lo, hi = float(np.min(values)), float(np.max(values))
    if hi - lo < 1e-12:
        return "O-C: flat"
    sampled = values[np.linspace(0, len(values) - 1, min(width, len(values))).astype(int)]
    scaled = (sampled - lo) / (hi - lo)
    return "O-C: " + "".join(glyphs[int(v * (len(glyphs) - 1))] for v in scaled)


def run(speed: float = 1.0) -> None:
    """Run the autonomous demonstration."""
    config = DEFAULT_CONFIG
    reference = load_reference()
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8")) if RESULT_PATH.exists() else None

    system = build_system_with_planet_x(config.planet_x, reference=reference)
    control = build_kepler90(reference=reference)
    probe_name = f"Kepler-90 {PROBE_LETTER}"
    probe_index = system.index_of(probe_name)

    engine = NBodyEngine(n_bodies=system.n_bodies)
    engine.enable_transit_detection(
        tracked_bodies=[probe_index],
        max_transits=512,
        stellar_radius=float(system.radii[0]),
        planet_radii=np.asarray([system.radii[probe_index]]),
    )
    engine.set_state(system.masses, system.positions, system.velocities, time=0.0)

    renderer = SystemRenderer(n_bodies=system.n_bodies)
    stages = build_stages(result)
    stage_index = 0

    # Physics steps per frame. The TIMESTEP is fixed by the convergence study;
    # `speed` changes how many steps are taken per frame, never their size.
    steps_per_frame = max(1, int(40 * speed))
    paused = False

    print("=" * 70)
    print("THE INVISIBLE PLANET - autonomous demonstration")
    print("=" * 70)
    print("Planet X is SYNTHETIC. Not a claim about the real Kepler-90.")
    print()
    print("  SPACE  pause / resume      N  next stage")
    print("  LMB    orbit camera        W/S  zoom")
    print("  ESC    quit")
    print()

    while renderer.running():
        for event in renderer.window.get_events(ti.ui.PRESS):
            if event.key == ti.ui.ESCAPE:
                renderer.window.running = False
            elif event.key == ti.ui.SPACE:
                paused = not paused
            elif event.key == "n":
                stage_index = min(stage_index + 1, len(stages) - 1)

        stage = stages[stage_index]
        if not paused and engine.time < stage.days:
            engine.run(config.timestep_days, steps_per_frame)
        elif not paused and stage_index < len(stages) - 1:
            stage_index += 1

        positions = engine.get_positions()[0]
        renderer.trails.push(positions)
        renderer.draw(
            positions,
            system.radii,
            float(system.radii[0]),
            reveal_planet_x=stage.reveal_planet_x,
        )

        transit_times = engine.get_transit_times()[0][0]

        gui = renderer.window.get_gui()
        with gui.sub_window("The Invisible Planet", 0.01, 0.01, 0.42, 0.34) as panel:
            panel.text(stage.title)
            panel.text("")
            for line in stage.lines:
                panel.text(line)

        with gui.sub_window("Transit timing (measured live)", 0.01, 0.37, 0.42, 0.20) as panel:
            for line in oc_panel_text(transit_times, PROBE_LETTER):
                panel.text(line)

        with gui.sub_window("Simulation", 0.01, 0.59, 0.42, 0.16) as panel:
            panel.text(f"t = {engine.time:8.1f} d  of {stage.days:.0f} d")
            panel.text(f"timestep = {config.timestep_days:.6f} d (fixed, symplectic)")
            panel.text(f"bodies = {system.n_bodies}")
            panel.text("PAUSED" if paused else "running")
            diagnostics = engine.diagnostics()
            panel.text(f"min separation = {float(diagnostics['min_separation'][0]):.4f} au")

        renderer.show()
