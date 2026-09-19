"""
Investigation Mode (Phase 15): the optional, interactive counterpart of the demonstration.

EVERY CONTROL IS A REAL MODEL PARAMETER OR A REAL OBSERVING CHOICE - there are no game mechanics:

    planet selector          which planet's elements, transits and O-C are inspected
    pause / speed            whole fixed-size steps per frame; the timestep itself is never changed
    observation window       how long the campaign ran (only transits up to then are used)
    timing-noise scale       multiplies the real per-transit uncertainties (the noise drawn AND the
                             uncertainty quoted to the fit)
    candidate parameters     mass, semi-major axis, eccentricity, mean anomaly and mutual
                             inclination of a hypothesised extra planet
    Run hypothesis           integrates the visible system plus that planet with the SAME solver and
                             compares the predicted transit times with the observations by chi2
    Load recovered           sets the candidate to what the real inference found (from
                             data/experiment_result.json)
    Reveal Planet X          draws the hidden planet and the force it exerts

The data are simulated observations of the synthetic world: the same physics that generated them is
what the hypothesis is tested with. The chi2 shown is a goodness of fit for THAT candidate; it is not
corrected for the look-elsewhere effect a search incurs (the periodogram false-alarm probability in
the demonstration is).
"""

# NOTE: no `from __future__ import annotations` - Taichi fields are created downstream.

import numpy as np

from ..experiments.synthetic import observations_from_times, simulate_transit_times
from ..inference.domain import stable_zones
from ..inference.forward import ForwardModel
from ..inference.likelihood import TimingLikelihood
from ..inference.sampler import tilt_to_inclination_node
from ..observations.ttv import fit_linear_ephemeris
from ..systems.planet_x import PlanetXParameters
from .observatory import CANDIDATE, LEFT_W, LEFT_X, OBSERVED, VISIBLE_ONLY, Observatory, limits_from
from .session import Session
from .story import load_result

SAMPLE_INTERVAL_DAYS = 2.0


def detrended(times_bjd, epochs, sigma):
    fit = fit_linear_ephemeris(np.asarray(times_bjd), epochs=np.asarray(epochs), sigma=np.asarray(sigma))
    return fit.residuals * 1440.0


class Investigation:
    def __init__(self, result: dict):
        self.result = result
        truth = PlanetXParameters.from_dict(result["ground_truth"])
        self.world = Session(hidden=truth)
        self.obs = Observatory(self.world, "The Invisible Planet - investigation mode", steps_per_frame=40)
        self.duration = float(result["config"]["numerics"]["duration_days"])
        self.noise_seed = int(result["config"]["observations"]["noise_seed"])
        self.sim_times = simulate_transit_times(truth, reference=self.world.reference)   # noiseless, once
        zones = stable_zones(self.world.reference)
        self.a_range = (float(min(lo for lo, _ in zones)), float(max(hi for _, hi in zones)))

        # observing choices
        self.window_days = self.duration
        self.noise_scale = 1.0
        self.reveal = False
        self.show_candidate = False
        # candidate Planet X
        self.mass, self.a, self.ecc, self.phase, self.tilt = 20.0, 0.5 * sum(self.a_range), 0.0, 0.0, 0.0
        self.candidate: Session | None = None
        self.hypothesis: dict | None = None
        self.pending = False
        self._exact = None
        self.message = "Set a candidate, then press Run hypothesis."
        # orbital-residual history (osculating elements recorded from the live integration)
        self.history = {k: {"t": [], "a": []} for k in self.world.letters}
        self._last_sample = -1e9

    # ---- candidate & hypothesis ---------------------------------------------------------------
    def _sliders(self) -> tuple:
        return (self.mass, self.a, self.ecc, self.phase, self.tilt)

    def candidate_parameters(self) -> PlanetXParameters:
        if self._exact is not None and self._exact[0] == self._sliders():
            return self._exact[1]                 # loaded from the inference and not edited since
        inc, node = tilt_to_inclination_node(self.tilt, 0.0)
        return PlanetXParameters(mass_earth=self.mass, semi_major_axis_au=self.a, eccentricity=self.ecc,
                                 mean_anomaly=self.phase, inclination_deg=inc,
                                 longitude_of_ascending_node_deg=node)

    def load_recovered(self) -> None:
        best = self.result["inference"]["best_parameters"]
        if not best:
            self.message = "The recorded inference made no detection: nothing to load."
            return
        self.mass, self.a, self.ecc = best["mass_earth"], best["semi_major_axis_au"], best["eccentricity"]
        self.phase = best["mean_anomaly"] % (2 * np.pi)
        self.tilt = 0.0
        # the inference also fitted inclination / node / argument of periapsis; keep them exactly
        # until a slider is moved (the tilt slider alone cannot represent them)
        self._exact = (self._sliders(), PlanetXParameters.from_dict({**best, "mean_anomaly": best["mean_anomaly"]}))
        self.message = "Loaded the recovered best fit (all its fitted parameters)."

    def run_hypothesis(self) -> None:
        dataset = observations_from_times(self.sim_times, seed=self.noise_seed, noise_scale=self.noise_scale,
                                          reference=self.world.reference).window(self.window_days)
        if len(dataset.planet_letters) == 0:
            self.hypothesis = None
            self.message = "Window too short: no planet has enough transits."
            return
        forward = ForwardModel(dataset, reference=self.world.reference)
        likelihood = TimingLikelihood(dataset)
        control = forward.control()
        params = self.candidate_parameters()
        model = forward.predict([params])[0]
        chi2_null, chi2_model = likelihood.chi2(control), likelihood.chi2(model)
        self.hypothesis = {
            "dataset": dataset, "control": control, "model": model, "chi2_null": chi2_null, "chi2_model": chi2_model,
            "per_null": likelihood.chi2_per_planet(control), "per_model": likelihood.chi2_per_planet(model),
            "n_points": likelihood.n_points, "dof": likelihood.dof_data, "params": params,
        }
        self.message = "Hypothesis evaluated."
        # a live candidate, brought up to the present moment by the same fixed-size steps
        self.obs.renderer.ghost_trails.clear()
        self.candidate = Session(hidden=params, reference=self.world.reference)
        self.candidate.advance(int(round(self.world.time_days / self.candidate.dt)))

    # ---- panels ----------------------------------------------------------------------------------
    def controls(self) -> None:
        gui = self.obs.gui
        o = self.obs
        with gui.sub_window("INVESTIGATION - observation & candidate", LEFT_X, 0.01, LEFT_W, 0.37) as g:
            g.text("OBSERVATION")
            index = g.slider_int("planet", self.world.letters.index(o.selected) + 1, 1, len(self.world.letters))
            o.selected = self.world.letters[index - 1]
            o.steps_per_frame = g.slider_int("steps/frame", o.steps_per_frame, 1, 400)
            self.window_days = g.slider_float("window [d]", self.window_days, 300.0, self.duration)
            self.noise_scale = g.slider_float("noise x", self.noise_scale, 0.2, 3.0)
            g.text("CANDIDATE PLANET X")
            self.mass = g.slider_float("mass [M_E]", self.mass, 1.0, 300.0)
            self.a = g.slider_float("a [au]", self.a, self.a_range[0], self.a_range[1])
            self.ecc = g.slider_float("e", self.ecc, 0.0, 0.3)
            self.phase = g.slider_float("mean anomaly", self.phase, 0.0, 6.283)
            self.tilt = g.slider_float("tilt [deg]", self.tilt, 0.0, 6.0)
            if g.button("Run hypothesis"):
                self.pending = True
                self.message = "integrating the candidate model..."
            if g.button("Load recovered parameters"):
                self.load_recovered()
            self.show_candidate = g.checkbox("draw candidate", self.show_candidate)
            self.reveal = g.checkbox("reveal Planet X", self.reveal)
            o.show_velocity = g.checkbox("velocity vectors", o.show_velocity)
            o.show_trails = g.checkbox("trails", o.show_trails)
            g.text(self.message)

    def result_panel(self) -> None:
        with self.obs.gui.sub_window("HYPOTHESIS - candidate vs observations", LEFT_X, 0.79, LEFT_W, 0.20) as g:
            h = self.hypothesis
            if h is None:
                g.text("no hypothesis evaluated yet")
                return
            g.text(f"visible only : chi2 {h['chi2_null']:.1f}   ({h['n_points']} pts, {h['dof']} dof)")
            g.text(f"+ candidate  : chi2 {h['chi2_model']:.1f}")
            g.text(f"improvement  : {h['chi2_null'] - h['chi2_model']:.1f}   (7 extra parameters)")
            per = "  ".join(f"{k}:{h['per_null'][k]:.0f}->{h['per_model'][k]:.0f}" for k in h["dataset"].planet_letters)
            g.text("per planet " + per)
            g.text("not corrected for the search (look-elsewhere)")

    # ---- plots -------------------------------------------------------------------------------------
    def plots(self) -> None:
        o, canvas, gui = self.obs, self.obs.canvas, self.obs.gui
        letter = o.selected
        # top: orbital residual = osculating a(t) minus its mean, in the live integration
        plot = o.top_plot
        plot.begin(canvas)
        h = self.history[letter]
        if len(h["t"]) >= 2:
            t, a = np.asarray(h["t"]), np.asarray(h["a"])
            resid = (a - a.mean()) * 1.0e4
            xlim, ylim = (0.0, max(float(t[-1]), 50.0)), limits_from(resid, minimum_span=1e-3)
            plot.line(canvas, t, resid, xlim, ylim, (0.4, 0.95, 0.55))
            plot.title(gui, f"Orbital residual of {letter}: osculating a - mean [1e-4 au]  y: {ylim[0]:.2f}..{ylim[1]:.2f}",
                       "recorded from the live integration (x: days)")
        else:
            plot.title(gui, f"Orbital residual of {letter}", "collecting...")

        # bottom: O-C of the selected planet - observed vs null vs candidate
        plot = o.bottom_plot
        plot.begin(canvas)
        hyp = self.hypothesis
        if hyp is not None and letter in hyp["dataset"].planet_letters:
            d = hyp["dataset"]
            t = np.asarray(d.transit_times_bjd[letter]) - d.epoch_bjd
            sig = np.asarray(d.timing_uncertainty_days[letter])
            ep = d.epochs[letter]
            y = detrended(d.transit_times_bjd[letter], ep, sig)
            e = sig * 1440.0
            null = detrended(hyp["control"][letter], ep, sig)
            cand = detrended(hyp["model"][letter], ep, sig)
            ylim, xlim = limits_from(y - e, y + e, null, cand), (-20.0, self.window_days + 20.0)
            plot.line(canvas, t, null, xlim, ylim, VISIBLE_ONLY)
            plot.line(canvas, t, cand, xlim, ylim, CANDIDATE)
            plot.error_bars(canvas, t, y, e, xlim, ylim, (0.6, 0.6, 0.65))
            plot.points(canvas, t, y, xlim, ylim, OBSERVED)
            plot.title(gui, f"O-C of {letter} (min)  y: {ylim[0]:.0f}..{ylim[1]:.0f}",
                       "white observed   blue visible-only   amber candidate")
        else:
            live = self.world.o_minus_c(letter)
            if live is None:
                plot.title(gui, f"O-C of {letter}", "run the hypothesis (only d and e have observations)")
            else:
                t = self.world.transit_bjd(letter) - self.world.epoch_bjd
                y = np.asarray(live["residual_minutes"])
                ylim, xlim = limits_from(y), (0.0, max(float(t[-1]), 50.0))
                plot.points(canvas, t, y, xlim, ylim, (0.4, 0.95, 0.55))
                plot.title(gui, f"live O-C of {letter} (min)  y: {ylim[0]:.1f}..{ylim[1]:.1f}",
                           "noise-free, from the running integration; no observations for this planet")

    def sample_elements(self) -> None:
        if self.world.time_days - self._last_sample < SAMPLE_INTERVAL_DAYS:
            return
        self._last_sample = self.world.time_days
        for letter in self.world.letters:
            self.history[letter]["t"].append(self.world.time_days)
            self.history[letter]["a"].append(self.world.osculating(letter)["a_au"])

    # ---- main loop ----------------------------------------------------------------------------------
    def run(self) -> None:
        o = self.obs
        print("INVESTIGATION MODE. Planet X is SYNTHETIC. SPACE pause | 1-8 planet | LMB orbit | W/S zoom | ESC quit")
        while o.running():
            o.poll_keys()
            if self.pending:
                self.run_hypothesis()
                self.pending = False
            o.advance(companions=[self.candidate] if (self.candidate is not None) else [])
            if self.candidate is not None:
                o.renderer.ghost_trails.push(self.candidate.positions()[self.candidate.hidden_index][None])
            self.sample_elements()
            ghost = (self.candidate.positions()[self.candidate.hidden_index]
                     if (self.show_candidate and self.candidate is not None) else None)
            o.draw_world(reveal_hidden=self.reveal, ghost=ghost, show_perturbation=self.reveal)
            self.controls()
            o.analysis_panel(reveal_perturbation=self.reveal)
            o.validation_panel()
            self.result_panel()
            self.plots()
            o.renderer.show()


def run() -> None:
    result = load_result()
    if result is None:
        raise SystemExit(
            "data/experiment_result.json not found. It is produced by the real inference:\n"
            "  uv run --python 3.12 --extra dev python scripts/run_experiment.py"
        )
    Investigation(result).run()
