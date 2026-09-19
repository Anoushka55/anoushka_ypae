"""
The observatory-style interface shared by the autonomous demonstration and Investigation Mode.

Layout (normalised window coordinates)
    centre          the planetary system: star, planets, trails, transit flashes
    left column     narrative / controls, ANALYSIS (selected planet), VALIDATION (conservation
                    errors, timestep, time), INFERENCE (status and fit quality)
    right column    two framed plots (O-C, periodogram / model comparison)

EVERY NUMBER SHOWN IS READ FROM A `Session` (the running integration) OR FROM A RECORDED RESULT
FILE PRODUCED BY THE REAL INFERENCE. Nothing here computes physics, and nothing here can change
it: the UI reads; only `Session.advance` advances, and only in whole steps of the fixed timestep.
"""

# NOTE: no `from __future__ import annotations` - Taichi fields are built here.

import numpy as np
import taichi as ti

from ..render.plots import (
    Box, PlotPool, draw_frame, draw_grid, draw_points, draw_polyline, draw_segments, nice_limits,
)
from ..render.scene import SystemRenderer
from .session import Session

LEFT_X, LEFT_W = 0.01, 0.30
TOP_PLOT = Box(0.665, 0.39, 0.985, 0.63)        # canvas coords, origin bottom-left
BOTTOM_PLOT = Box(0.665, 0.06, 0.985, 0.30)

OBSERVED = (1.00, 1.00, 1.00)
VISIBLE_ONLY = (0.45, 0.60, 1.00)
CANDIDATE = (1.00, 0.80, 0.20)
LIVE = (0.40, 0.95, 0.55)
THRESHOLD = (1.00, 0.35, 0.40)


class Plot:
    """A framed plot in one rectangle of the canvas, drawn in layers."""

    def __init__(self, box: Box, n_layers: int = 6):
        self.box = box
        self.pools = [PlotPool() for _ in range(n_layers)]
        self.grid_pool, self.frame_pool = PlotPool(), PlotPool()
        self._layer = 0

    def begin(self, canvas) -> None:
        self._layer = 0
        draw_grid(canvas, self.grid_pool, self.box)
        draw_frame(canvas, self.frame_pool, self.box)

    def _next(self) -> PlotPool:
        pool = self.pools[min(self._layer, len(self.pools) - 1)]
        self._layer += 1
        return pool

    def line(self, canvas, x, y, xlim, ylim, colour, width=0.002) -> None:
        if len(x) >= 2:
            draw_polyline(canvas, self._next(), self.box.to_box(x, y, xlim, ylim), colour, width, self.box)

    def points(self, canvas, x, y, xlim, ylim, colour, radius=0.0045) -> None:
        if len(x):
            draw_points(canvas, self._next(), self.box.to_box(x, y, xlim, ylim), colour, radius, self.box)

    def error_bars(self, canvas, x, y, err, xlim, ylim, colour) -> None:
        if len(x):
            lo = self.box.to_box(x, np.asarray(y) - np.asarray(err), xlim, ylim)
            hi = self.box.to_box(x, np.asarray(y) + np.asarray(err), xlim, ylim)
            draw_segments(canvas, self._next(), np.stack([lo, hi], axis=1), colour, 0.0012, self.box)

    def hline(self, canvas, y, xlim, ylim, colour, width=0.0015) -> None:
        pair = self.box.to_box(np.array(xlim), np.array([y, y]), xlim, ylim)
        draw_segments(canvas, self._next(), pair[None], colour, width, self.box)

    def title(self, gui, title: str, subtitle: str = "") -> None:
        """Titles are GUI text: GGUI cannot draw text on the canvas."""
        with gui.sub_window(title, self.box.x0, 1.0 - self.box.y1 - 0.055, self.box.x1 - self.box.x0, 0.05) as g:
            g.text(subtitle if subtitle else " ")


class Observatory:
    def __init__(self, session: Session, title: str, window_size=(1500, 900), steps_per_frame: int = 60):
        self.session = session
        self.renderer = SystemRenderer(session.n_bodies, window_size=window_size, title=title)
        self.top_plot = Plot(TOP_PLOT)
        self.bottom_plot = Plot(BOTTOM_PLOT)
        self.selected = "d"
        self.paused = False
        self.steps_per_frame = int(steps_per_frame)
        self.show_velocity = False
        self.show_trails = True
        self.flash: dict = {}
        self.frame = 0

    # ---- window ------------------------------------------------------------------------
    @property
    def gui(self):
        return self.renderer.window.get_gui()

    @property
    def canvas(self):
        return self.renderer.canvas

    def running(self) -> bool:
        return self.renderer.running()

    def poll_keys(self) -> list:
        keys = [e.key for e in self.renderer.window.get_events(ti.ui.PRESS)]
        if ti.ui.ESCAPE in keys:
            self.renderer.window.running = False
        if ti.ui.SPACE in keys:
            self.paused = not self.paused
        for i, letter in enumerate(self.session.letters, start=1):
            if str(i) in keys:
                self.selected = letter
        return keys

    # ---- simulation ----------------------------------------------------------------------
    def advance(self, companions=()) -> None:
        """Advance the physics by whole fixed-size steps. `companions` are further sessions
        (e.g. the candidate model) advanced in lock-step so all share one clock."""
        if self.paused:
            return
        self.session.advance(self.steps_per_frame)
        for other in companions:
            other.advance(self.steps_per_frame)
        self.renderer.trails.push(self.session.positions())
        for letter in self.session.new_transits():
            self.flash[letter] = 12
        self.frame += 1

    def draw_world(self, reveal_hidden: bool, ghost=None, show_perturbation: bool = False) -> None:
        s = self.session
        flash = []
        for letter in list(self.flash):
            flash.append(s.system.index_of(f"Kepler-90 {letter}"))
            self.flash[letter] -= 1
            if self.flash[letter] <= 0:
                del self.flash[letter]
        pos, vel = s.positions(), s.velocities()
        perturbation = None
        if show_perturbation and s.hidden_index is not None and reveal_hidden:
            perturbation = s.engine.acceleration_due_to(s.hidden_index)[0]
        selected = s.system.index_of(f"Kepler-90 {self.selected}")
        self.renderer.draw(
            pos, float(s.system.radii[0]), velocities=vel, hidden_index=s.hidden_index,
            reveal_hidden=reveal_hidden, selected=selected, show_trails=self.show_trails,
            show_velocity=self.show_velocity, perturbation=perturbation, flash=flash, ghost=ghost,
        )

    # ---- panels ----------------------------------------------------------------------------
    def analysis_panel(self, reveal_perturbation: bool) -> None:
        s = self.session
        letter = self.selected
        ref = s.reference["planets"][letter]
        with self.gui.sub_window("ANALYSIS - selected planet", LEFT_X, 0.39, LEFT_W, 0.21) as g:
            osc = s.osculating(letter)
            g.text(f"Kepler-90 {letter}   (keys 1-8 select)")
            g.text(f"osculating a  {osc['a_au']:.4f} au   e  {osc['e']:.3f}")
            g.text(f"osculating P  {osc['period_days']:.3f} d")
            g.text(f"catalogue  P  {ref['orbital_period_days']['value']:.3f} d")
            g.text(f"transits recorded: {s.transit_count(letter)}")
            oc = s.o_minus_c(letter)
            if oc is None:
                g.text("O-C: need 3 transits")
            else:
                g.text(f"O-C rms {oc['rms_seconds']:.1f} s   p-p {oc['peak_to_peak_minutes']:.2f} min")
            if reveal_perturbation and s.hidden_index is not None:
                p = s.perturbation_from_hidden(letter)
                g.text(f"pull of hidden planet / pull of star: {p['fraction_of_stellar_pull']:.2e}")
            else:
                g.text("pull of an unmodelled body: not yet identified")

    def validation_panel(self) -> None:
        s = self.session
        d = s.diagnostics()
        with self.gui.sub_window("VALIDATION - conservation", LEFT_X, 0.61, LEFT_W, 0.17) as g:
            g.text(f"|dE/E|   {d['energy_error']:.2e}")
            g.text(f"|dL/L|   {d['angular_momentum_error']:.2e}")
            g.text(f"|dP|/P   {d['linear_momentum_error']:.2e}")
            g.text(f"timestep {s.dt:.6f} d (fixed)   t = {s.time_days:7.1f} d")
            g.text("integrator: velocity Verlet, symplectic")

    def controls_hint(self) -> str:
        return "SPACE pause   1-8 planet   LMB orbit   W/S zoom   ESC quit"


def oc_layers(session: Session, letter: str):
    """(days since epoch, O-C in minutes) of the transits the running integration has produced."""
    oc = session.o_minus_c(letter)
    if oc is None:
        return None
    t = session.transit_bjd(letter) - session.epoch_bjd
    return np.asarray(t), np.asarray(oc["residual_minutes"])


def limits_from(*arrays, pad: float = 0.12, minimum_span: float = 1.0):
    values = np.concatenate([np.asarray(a, dtype=np.float64).ravel() for a in arrays if len(a)])
    return nice_limits(values, pad, minimum_span)
