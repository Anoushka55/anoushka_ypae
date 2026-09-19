# Hostile physics review

**Phase 20 deliverable.** Written in the voice of a skeptical astrophysics judge
looking for reasons to reject the work. No praise. Issues are rated
**critical / major / minor**, with the fix (or the admission) recorded.

Findings marked **[MEASURED]** were established by running an experiment, not by
argument.

---

## CRITICAL

### C1. The inference is handed the true inclination, and the recovered mass depends on it **[MEASURED]**

**Problem.** `periodogram_search` and `scan_axis_phase_grid` hardcode
`inclination_deg = 88.0`, which is exactly the true value of the synthetic planet.
The solver is therefore told one of the answers.

**Why it matters.** If the recovered mass is sensitive to that assumption, the
headline mass recovery is partly a consequence of being given the answer.

**Test.** Fix a and phase at the truth, vary the inclination the *solver* assumes,
and refit the mass:

| assumed i | best-fit mass | χ² |
|---|---|---|
| 85.0° | 20.0 M⊕ | 547.1 |
| 87.0° | 35.0 M⊕ | **513.2** |
| **88.0° (true)** | 50.0 M⊕ | 514.8 |
| 89.0° | 50.0 M⊕ | 528.5 |
| 89.5° | 50.0 M⊕ | 532.4 |

**Verdict: the objection is VALID.** The inferred mass varies by a factor of 2.5
across a 4.5° range of assumed inclination, and the *best* χ² occurs at 87°, not at
the true 88°. There is a genuine **mass–mutual-inclination degeneracy**, which is a
well-known feature of TTV analysis: a more mutually inclined perturber produces a
weaker in-plane perturbation, which a larger mass can mimic.

**But the period survives.** Repeating the scan in semi-major axis at each assumed
inclination:

| assumed i | recovered a | (true 0.2200) |
|---|---|---|
| 86.0° | **0.2200** | ✓ |
| 87.0° | **0.2200** | ✓ |
| 88.0° | **0.2200** | ✓ |
| 89.0° | **0.2200** | ✓ |

**Resolution.** The claim is weakened rather than the code patched:

- **Period/semi-major axis is robustly recovered** and does not depend on the
  inclination assumption.
- **Mass is constrained only to within a factor of ~2** once inclination is treated
  as unknown, not to the ~20% suggested by a fixed-inclination χ² profile.
- README and `docs/IDENTIFIABILITY.md` now state this explicitly.

This is a limitation of the experiment, not a bug, and it is exactly the sort of
degeneracy the identifiability phase exists to expose.

---

## MAJOR

### M1. The noise model probably understates real single-transit timing errors

**Problem.** `TimingNoiseModel` uses the published **transit-epoch (T₀)** uncertainty
as the per-transit timing uncertainty. These are not the same quantity. T₀ comes from
a global fit over N transits, so it is smaller than a single transit's timing error by
roughly √N. For planet b (200 transits) that is a factor of ~14.

**Why it matters.** The reported 7.1σ detection would be substantially weaker with
realistic per-transit errors.

**Partial defence [MEASURED].** The noise-sensitivity scan is already in
`docs/IDENTIFIABILITY.md`:

| noise scale | significance |
|---|---|
| 1.0× | 7.3σ |
| 2.0× | 4.2σ |

**Verdict: the objection is VALID and is not fully fixed.** It is recorded as a stated
limitation: *the adopted noise level is optimistic, and at several times the adopted
uncertainties this signal would not be recoverable.* The experiment remains internally
consistent — the same noise model generates the data and defines the χ² weights — but
it should not be read as a claim about what is achievable with real Kepler data.

### M2. Short-baseline stability does not establish stability

**Problem.** The Planet X sweep found **0 of 105** candidates unstable over 1200 days
and used that as a stability criterion. Instabilities in compact systems commonly take
10⁴–10⁷ orbits to appear.

**Verdict: VALID.** `docs/PLANET_X_DESIGN.md` states plainly that the short-baseline
check is *necessary but not sufficient*, and that the Hill-separation criterion
(> 8 mutual Hill radii, versus 5.16 for the real g–h pair) is doing the real work.

### M3. A known-inadequate search routine is still in the codebase

**Problem.** `coarse_to_fine_search` was demonstrated to fail on this problem —
it converged to a solution with a *worse* χ² than the truth. It remains importable,
and two tests still exercise it.

**Verdict: VALID.** Its docstring now carries an explicit warning that it must not be
used for period recovery, and states the measured failure. It is retained only because
the tests that use it verify determinism and null-hypothesis improvement, not accuracy.

---

## MINOR

### m1. Planet X's eccentricity is never searched

Both the truth and the solver fix e = 0. The experiment therefore says nothing about
whether an eccentric unseen planet would be recoverable. Stated as a limitation.

### m2. Only one noise realisation is reported

The blind recovery is a single draw. A proper study would repeat over many noise seeds
to characterise the distribution of recovered parameters. The identifiability analysis
partially compensates by profiling χ², but this remains a real gap. `--seed` is exposed
on `scripts/run_experiment.py` for anyone wanting to repeat it.

### m3. The detection rests on one planet

Only planet d's induced signal (415 s RMS) exceeds its own timing uncertainty (363 s).
A systematic error in planet d's transit times would propagate directly into the
inferred planet. Stated in README §13.

### m4. Unrelated prototype in the repository root

`sim.py` (a galaxy-collision visualiser) is unrelated to this project. Moved to
`prototypes/` so it cannot be mistaken for part of the experiment.

### m5. `refine_nonlinear` is superseded

Retained and documented, since it records the measured failure of coordinate descent
on this χ² surface (it stalls at 524.3 where the truth scores 516.8). That failure is
informative, so the function is kept with the explanation attached.

---

## Checked and found NOT to be problems

| Objection | Finding |
|---|---|
| "Planet X gets a special force" | **[MEASURED]** Engine accelerations match an independent Newtonian sum over all bodies to 10⁻¹³, Planet X included |
| "The trajectories are scripted" | Integrated trajectory matches the analytic Kepler solution; no closed-form path is used anywhere |
| "Euler integration" | **[MEASURED]** Convergence order 2.000; an explicit test fails if order < 1.5 |
| "Energy drift is hidden" | Reported as max/RMS/final in the conservation report; bounded, non-trending |
| "The inference can see the answer" | Enforced by AST inspection of imports, by two leakage scanners, and by a test that verifies the scanners fire |
| "Softening was used to avoid singularities" | **[MEASURED]** Force law matches G·m/r² to 10⁻¹⁴; no softening term exists |
| "The units are hand-waved" | G derived from IAU 2015 nominal GM☉, cross-checked against Gaussian k² to 3×10⁻¹⁰ |
| "The timestep was chosen so it looks smooth" | Derived from a convergence study and confirmed end-to-end by post-fit O−C residuals < 1 s |
| "The catalogue values were blended carelessly" | Per-parameter provenance with status flags; two internal inconsistencies in the source data found and corrected |
| "It claims a real discovery" | Every artefact — code, docs, UI, dataset — states Planet X is synthetic |

---

## Summary

The physics, numerics and experimental controls hold up. The **weakest link is the
inference's treatment of mass**: it is degenerate with inclination at the factor-of-2
level, and the adopted noise level is optimistic. The period determination is solid and
robust to the assumptions tested.

The claims in the README have been adjusted to match what the measurements actually
support.
