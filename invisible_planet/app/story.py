"""
Autonomous demonstration (Phase 14).

A judge launches one command and the ten-step argument runs by itself; keyboard input can pause
or inspect but is never required.

THE NARRATIVE FOLLOWS THE PHYSICS, NEVER THE REVERSE
    * The planetary system moving on screen is the live N-body integration (`Session`), advanced
      in whole steps of the fixed timestep. Planet X is inside that integration throughout - its
      gravity is what bends the transit times - but it is not DRAWN until step 9.
    * The observed transit times, the periodogram, the false-alarm probability, the recovered
      parameters and their credible intervals are read from data/experiment_result.json, written by
      scripts/run_experiment.py running the real inference on the observer dataset. Nothing is
      invented, interpolated or scripted here, and if that file is absent the demonstration says so.
    * The candidate planet drawn in step 7 is a second live integration of the visible system plus
      a planet with the RECOVERED parameters - the same solver, the same force law.

Planet X is synthetic. The demonstration says so on screen.
"""

# NOTE: no `from __future__ import annotations` - Taichi fields are created downstream.

import json
import time
from pathlib import Path

import numpy as np
import taichi as ti

from ..render.scene import HIDDEN_COLOUR  # noqa: F401  (documented colour of the revealed planet)
from ..systems.planet_x import PlanetXParameters
from .observatory import (
    CANDIDATE, LIVE, OBSERVED, THRESHOLD, VISIBLE_ONLY, LEFT_W, LEFT_X, Observatory, limits_from,
)
from .session import Session

ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = ROOT / "data" / "experiment_result.json"

STEP_NAMES = [
    "1  A normal planetary system", "2  The system evolves", "3  A transit anomaly appears",
    "4  The O-C signal", "5  A structured perturbation", "6  The inference begins",
    "7  A candidate planet is reconstructed", "8  The model against the observations",
    "9  Planet X is revealed", "10 True versus recovered",
]
#: (simulated-time target [d] that must be reached, minimum dwell [s]) per step
TIMING = [(90.0, 9), (400.0, 10), (1459.5, 12), (None, 12), (None, 14), (None, 16), (None, 14), (None, 16),
          (None, 12), (None, 20)]


def load_result(path: Path = RESULT_PATH):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def probe_letters(result: dict) -> list:
    return list(result["observations"].keys())


def pick_probe(result: dict) -> str:
    """The probe planet carrying the most of the signal (largest chi2 improvement, else largest chi2)."""
    best, best_score = None, -np.inf
    for k, o in result["observations"].items():
        gain = o["chi2_visible_only"] - (o["chi2_with_recovered_planet"] or 0.0)
        if gain > best_score:
            best, best_score = k, gain
    return best


def format_parameter(summary: dict, key: str, fmt: str = "{:.3g}") -> str:
    q = summary[key]
    return f"{fmt.format(q['50'])}  [{fmt.format(q['16'])}, {fmt.format(q['84'])}]"


class Story:
    def __init__(self, result: dict, speed: float = 1.0):
        self.result = result
        self.inference = result["inference"]
        truth = PlanetXParameters.from_dict(result["ground_truth"])
        self.world = Session(hidden=truth)
        self.candidate = None
        if self.inference["best_parameters"]:
            self.candidate = Session(hidden=PlanetXParameters.from_dict(self.inference["best_parameters"]))
        self.obs = Observatory(self.world, "The Invisible Planet - autonomous demonstration",
                               steps_per_frame=max(1, int(60 * speed)))
        self.duration = float(result["config"]["numerics"]["duration_days"])
        self.probe = pick_probe(result)
        others = [k for k in probe_letters(result) if k != self.probe]
        self.other = others[0] if others else None
        self.step = 0
        self.step_started = time.time()
        self.step_paused_for = 0.0

    # ---- progression -------------------------------------------------------------------------
    def goto(self, step: int) -> None:
        self.step = int(np.clip(step, 0, len(STEP_NAMES) - 1))
        self.step_started = time.time()

    def maybe_advance(self) -> None:
        if self.obs.paused:
            self.step_started += 1.0 / 60.0            # the dwell clock stops while paused
            return
        target, dwell = TIMING[self.step]
        reached = target is None or self.world.time_days >= target
        if reached and time.time() - self.step_started >= dwell and self.step < len(STEP_NAMES) - 1:
            self.goto(self.step + 1)

    def fraction_elapsed(self) -> float:
        return float(np.clip((time.time() - self.step_started) / TIMING[self.step][1], 0.0, 1.0))

    # ---- text ---------------------------------------------------------------------------------
    def narrative(self) -> list:
        r, inf, s = self.result, self.inference, self.step
        o = r["observations"][self.probe]
        sigma = float(np.median(o["sigma_minutes"]))
        oc = np.asarray(o["observed_oc_minutes"])
        n_pts, dof = inf["n_points"], inf["dof_null"]
        if s == 0:
            return ["Kepler-90: a Sun-like star with eight known planets.",
                    "Every orbit on screen is produced by Newtonian gravity,",
                    "integrated with a fixed-step symplectic scheme.",
                    "Nothing is scripted. Transit flashes mark the moment a",
                    "planet crosses the star as seen from Earth."]
        if s == 1:
            return ["Planets pull on each other. The pulls are tiny compared",
                    "with the star's, so orbits look periodic - but the",
                    "transit times of interacting planets are not exactly",
                    "periodic. That deviation is the TTV.",
                    "Simulated Kepler observations: real epochs, real",
                    "per-transit uncertainties, seeded noise."]
        if s == 2:
            return [f"Planet {self.probe}: observed minus calculated (O-C) transit",
                    "times, drawn as transits accumulate (plot, lower right).",
                    f"Scatter of O-C: {oc.std():.1f} min (1 sigma)",
                    f"Typical timing uncertainty: {sigma:.1f} min",
                    "Only the scatter beyond the noise is informative."]
        if s == 3:
            return [f"Planet {self.probe}, all {len(oc)} transits.",
                    "The O-C curve is measured against the best linear",
                    "ephemeris; a lone planet would leave only noise.",
                    "Visible-only N-body model (blue) vs data (white):",
                    f"chi2 = {inf['chi2_null']:.1f} for {n_pts} points ({dof} dof)"]
        if s == 4:
            from scipy.stats import chi2 as chi2_dist
            p = float(chi2_dist.sf(inf["chi2_null"], dof))
            return ["Is the excess consistent with noise? Under the",
                    "visible-only model, chi2 this large would occur with",
                    f"probability {p:.1e}.",
                    "Note: this alone does NOT establish a new planet -",
                    "the fit is shown to be structured, not white noise.",
                    "What structure? Next: a search."]
        if s == 5:
            n_zones = len(inf["periodogram"]["axes_au"]) if inf.get("periodogram") else 0
            return ["The inference sees ONLY noisy transit times.",
                    "It scans candidate orbits of an extra planet",
                    f"({n_zones} trial semi-major axes in dynamically stable",
                    "zones); each is a full N-body forward model, scored by",
                    "chi2 improvement. The red line is the 1% noise-only",
                    "threshold for the best peak (look-elsewhere corrected)."]
        if s == 6:
            if not inf["detected"]:
                return ["NO SIGNIFICANT DETECTION.",
                        f"False-alarm probability {inf['false_alarm_probability']:.2e}.",
                        "No planet is claimed."]
            b = inf["best_parameters"]
            return ["Best candidate after nonlinear refit with the full",
                    "N-body model (amber marker = candidate planet):",
                    f"  mass  {b['mass_earth']:.1f} M_earth",
                    f"  a     {b['semi_major_axis_au']:.4f} au",
                    f"  e     {b['eccentricity']:.3f}",
                    f"false-alarm probability {inf['false_alarm_probability']:.2e}"]
        if s == 7:
            gain = inf["chi2_null"] - (inf["best_chi2"] or np.nan)
            return [f"Planet {self.probe}: observed (white, with error bars)",
                    "vs visible-only model (blue) vs model with the",
                    "candidate planet (amber). Each curve detrended by",
                    "its own linear ephemeris.",
                    f"chi2: {inf['chi2_null']:.1f} -> {inf['best_chi2']:.1f}",
                    f"improvement {gain:.1f} for 7 extra parameters"]
        if s == 8:
            return ["REVEAL. The red planet was in the integration all",
                    "along. Orange arrows are its pull on each planet -",
                    "one term of the Newtonian force sum.",
                    "",
                    "PLANET X IS SYNTHETIC: a controlled numerical",
                    "experiment, not a claim about the real Kepler-90."]
        return ["TRUE vs RECOVERED (posterior median [16%, 84%])."]

    def inference_lines(self) -> list:
        inf, s = self.inference, self.step
        if s < 4:
            return ["status: waiting for a structured anomaly"]
        if s == 4:
            return ["status: anomaly under examination", f"chi2 (visible only) {inf['chi2_null']:.1f}"]
        if s == 5:
            return ["status: searching", f"forward N-body runs so far: {int(self.fraction_elapsed() * inf['n_forward_simulations'])}"]
        if not inf["detected"]:
            return ["status: NO DETECTION", f"FAP {inf['false_alarm_probability']:.2e}"]
        b = inf["best_parameters"]
        lines = [f"status: {'candidate planet' if self.step < 8 else 'revealed'}",
                 f"fit chi2 {inf['best_chi2']:.1f} (was {inf['chi2_null']:.1f})",
                 f"candidate: m {b['mass_earth']:.1f} M_E, a {b['semi_major_axis_au']:.4f} au"]
        if inf["posterior"]:
            lines.append(f"chain length / tau: {inf['posterior']['chain_length_over_tau']:.0f}")
        return lines

    # ---- drawing ------------------------------------------------------------------------------
    def draw_panels(self) -> None:
        gui = self.obs.gui
        with gui.sub_window(f"THE INVISIBLE PLANET   {STEP_NAMES[self.step]}", LEFT_X, 0.01, LEFT_W, 0.36) as g:
            for line in self.narrative():
                g.text(line)
            if self.step == 9:
                self.true_vs_recovered(g)
            g.text("")
            g.text("PLANET X IS SYNTHETIC - not a claim about Kepler-90")
            g.text(self.obs.controls_hint() + "   N next   B back")
        self.obs.analysis_panel(reveal_perturbation=self.step >= 8)
        self.obs.validation_panel()
        with gui.sub_window("INFERENCE", LEFT_X, 0.79, LEFT_W, 0.20) as g:
            for line in self.inference_lines():
                g.text(line)

    def true_vs_recovered(self, g) -> None:
        inf, truth, cmp_ = self.inference, self.result["ground_truth"], self.result["comparison"]
        if not inf["detected"] or not inf["posterior"]:
            g.text("(no posterior: the inference made no detection)")
            return
        s = inf["posterior"]["summary"]
        flags = cmp_.get("truth_inside_95pct_interval", {})

        def row(label, true_value, key, fmt):
            g.text(f"{label:<6} true {fmt.format(true_value)}   got {format_parameter(s, key, fmt)}"
                   f"   {'in 95%' if flags.get(key) else 'OUTSIDE 95%'}")

        row("mass", truth["mass_earth"], "mass_earth", "{:.1f}")
        row("a", truth["semi_major_axis_au"], "semi_major_axis_au", "{:.4f}")
        row("P [d]", self.result["true_period_days"], "period_days", "{:.2f}")
        row("e", truth["eccentricity"], "eccentricity", "{:.3f}")
        g.text(f"period error {cmp_['period_relative_error']:+.2%}   (eccentricity is weakly constrained)")

    def draw_plots(self) -> None:
        obs, canvas, gui = self.obs, self.obs.canvas, self.obs.gui
        s, sim_t = self.step, self.world.time_days
        # ---- lower right: O-C of the probe planet ----------------------------------------------
        o = self.result["observations"][self.probe]
        t = np.asarray(o["observed_bjd"]) - self.world.epoch_bjd
        y, e = np.asarray(o["observed_oc_minutes"]), np.asarray(o["sigma_minutes"])
        arrays = [y - e, y + e, np.asarray(o["visible_only_model_oc_minutes"])]
        if o["best_fit_model_oc_minutes"]:
            arrays.append(np.asarray(o["best_fit_model_oc_minutes"]))
        ylim, xlim = limits_from(*arrays), (-20.0, self.duration + 20.0)
        shown = t <= sim_t if s < 3 else np.ones_like(t, dtype=bool)
        plot = obs.bottom_plot
        plot.begin(canvas)
        if s >= 3:
            plot.line(canvas, t, o["visible_only_model_oc_minutes"], xlim, ylim, VISIBLE_ONLY)
        if s >= 7 and o["best_fit_model_oc_minutes"]:
            plot.line(canvas, t, o["best_fit_model_oc_minutes"], xlim, ylim, CANDIDATE)
        plot.error_bars(canvas, t[shown], y[shown], e[shown], xlim, ylim, (0.6, 0.6, 0.65))
        plot.points(canvas, t[shown], y[shown], xlim, ylim, OBSERVED)
        legend = "white: observed   blue: visible-only model" + ("   amber: with candidate" if s >= 7 else "")
        plot.title(gui, f"O-C of Kepler-90 {self.probe}  (min vs days)  y: {ylim[0]:.0f}..{ylim[1]:.0f}", legend)

        # ---- upper right: periodogram (from step 6) or the other probe's O-C ---------------------
        plot = obs.top_plot
        plot.begin(canvas)
        pg = self.inference.get("periodogram")
        if s >= 5 and pg:
            a = np.asarray(pg["axes_au"])
            d = np.asarray(pg["linear_delta_chi2"])
            frac = self.fraction_elapsed() if s == 5 else 1.0
            keep = a <= a[0] + frac * (a[-1] - a[0])
            top = max(float(d.max()), pg["noise_only_threshold_1pct"]) * 1.15
            xlim2, ylim2 = (float(a[0]), float(a[-1])), (0.0, top)
            plot.hline(canvas, pg["noise_only_threshold_1pct"], xlim2, ylim2, THRESHOLD)
            plot.points(canvas, a[keep], d[keep], xlim2, ylim2, LIVE, radius=0.0025)
            if s >= 6 and self.inference["best_parameters"]:
                ab = self.inference["best_parameters"]["semi_major_axis_au"]
                plot.points(canvas, [ab], [0.03 * top], xlim2, ylim2, CANDIDATE, radius=0.007)
            plot.title(gui, "Periodogram: chi2 gain vs a [au]", "green: linearised scan   red: 1% noise-only threshold")
        elif self.other is not None:
            oo = self.result["observations"][self.other]
            t2 = np.asarray(oo["observed_bjd"]) - self.world.epoch_bjd
            y2, e2 = np.asarray(oo["observed_oc_minutes"]), np.asarray(oo["sigma_minutes"])
            yl2 = limits_from(y2 - e2, y2 + e2)
            keep = t2 <= sim_t
            plot.error_bars(canvas, t2[keep], y2[keep], e2[keep], xlim, yl2, (0.6, 0.6, 0.65))
            plot.points(canvas, t2[keep], y2[keep], xlim, yl2, OBSERVED)
            plot.title(gui, f"O-C of Kepler-90 {self.other}  (min vs days)  y: {yl2[0]:.0f}..{yl2[1]:.0f}", "second probe planet")
        else:
            plot.title(gui, "-", "")

    # ---- main loop -----------------------------------------------------------------------------
    def run(self) -> None:
        obs = self.obs
        print("THE INVISIBLE PLANET - autonomous demonstration. Planet X is SYNTHETIC.")
        print("SPACE pause | N next | B back | 1-8 select planet | LMB orbit | W/S zoom | ESC quit")
        while obs.running():
            keys = obs.poll_keys()
            if "n" in keys:
                self.goto(self.step + 1)
            if "b" in keys:
                self.goto(self.step - 1)
            self.maybe_advance()
            obs.advance(companions=[self.candidate] if self.candidate else [])
            if self.candidate is not None:
                obs.renderer.ghost_trails.push(self.candidate.positions()[self.candidate.hidden_index][None])
            if not obs.renderer.window.is_pressed(ti.ui.LMB):      # slow camera drift; a view choice only
                obs.renderer.azimuth += 0.0012
            ghost = (self.candidate.positions()[self.candidate.hidden_index]
                     if self.candidate is not None and 6 <= self.step < 8 else None)
            obs.draw_world(reveal_hidden=self.step >= 8, ghost=ghost, show_perturbation=self.step >= 8)
            self.draw_panels()
            self.draw_plots()
            obs.renderer.show()


def run(speed: float = 1.0) -> None:
    result = load_result()
    if result is None:
        raise SystemExit(
            "data/experiment_result.json not found. It is produced by the real inference:\n"
            "  uv run --python 3.12 --extra dev python scripts/run_experiment.py"
        )
    Story(result, speed).run()
