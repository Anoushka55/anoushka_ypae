# Numerical Methods

**Phase 3.2 deliverable.** Integrator, convergence study, and the timestep policy.
Every number in this document is produced by `tests/test_convergence.py`; run it
directly to regenerate the table.

## Integrator

**Velocity Verlet (kick–drift–kick leapfrog), fixed step.**

```
v(t + dt/2) = v(t)      + a(t)      * dt/2      kick
r(t + dt)   = r(t)      + v(t+dt/2) * dt        drift
a(t + dt)   = a(r(t+dt))                        one force evaluation per step
v(t + dt)   = v(t+dt/2) + a(t+dt)   * dt/2      kick
```

Chosen because it is **symplectic** and time-reversible: the energy error stays
bounded instead of growing without limit, which is what makes multi-thousand-orbit
integrations trustworthy. Forward Euler is explicitly prohibited by the project rules
and is guarded against by `test_integrator_is_not_first_order`.

Velocity Verlet is symplectic **only at constant dt**. The step size is therefore
never varied during a run and is never tied to a frame rate. Rendering samples the
state; it never drives it.

**No softening.** The force law is exactly `a_i = G Σ m_j (r_j − r_i)/|r_j − r_i|³`.
Softening would alter the very force law the inference rests on. Close approaches are
instead *detected and reported* (`min_separation`), because in a planetary system a
close approach is physically meaningful information about instability, not a
numerical nuisance to be smoothed away.

## Convergence study

Two-body test problem: M★ = 1.108 M☉, m = 17 M⊕, a = 0.4 au, e = 0.2, 5 orbits,
compared against the **exact analytic Kepler solution** (not against another
integration).

```
 steps/orbit       dt [d]      pos err      vel err    E oscill.    E secular     ang.mom.
         250     0.351130    1.384e-02    1.536e-02    1.648e-04    1.397e-08    3.696e-16
         500     0.175565    3.460e-03    3.840e-03    4.118e-05    2.176e-10    4.990e-15
        1000     0.087783    8.649e-04    9.599e-04    1.029e-05    3.405e-12    5.729e-15
        2000     0.043891    2.162e-04    2.400e-04    2.573e-06    5.977e-14    5.914e-15
        4000     0.021946    5.405e-05    5.999e-05    6.433e-07    2.625e-14    4.805e-15

measured order, position_error : 2.000
measured order, velocity_error : 2.000
measured order, energy (oscill): 2.000
```

**Measured order of accuracy = 2.000** in position, velocity and energy — exactly the
theoretical order of velocity Verlet. This, rather than any single error threshold, is
the primary evidence that the integrator is implemented correctly.

### Three error modes, which must not be conflated

Conflating these is an easy way to draw a false conclusion, so they are measured
separately.

| Mode | How it is measured | Behaviour | Does it threaten the science? |
|---|---|---|---|
| **Energy oscillation** | energy sampled *within* the orbit | bounded, O(dt²), ~10⁻⁵ at 1000 steps/orbit | **No** — bounded and non-accumulating |
| **Secular energy drift** | energy sampled *stroboscopically*, once per orbit at fixed phase | ~10⁻¹² and falling as dt² | **No** — this is the symplectic signature |
| **Along-track phase error** | position vs. analytic solution | grows *linearly* in time, O(dt²) rate | **No, but only because it is linear** — see below |

An earlier version of this study sampled energy only at orbit boundaries and
measured an apparent convergence order of ~5. That was a measurement artefact: the
oscillating term is phase-locked, so once-per-orbit sampling cancels it. Both
samplings are now reported, and they measure genuinely different physics.

### The along-track phase error, and why it is harmless

A symplectic integrator does not reproduce the exact Kepler frequency — it exactly
integrates a nearby *shadow* Hamiltonian whose orbital frequency differs by O(dt²).
At 1000 steps/orbit the resulting along-track error reaches ~10⁻³ of the semi-major
axis after only five orbits, which converts to **tens of minutes** of apparent
timing offset. For a project measuring transit times to the second, that looks fatal.

It is not, **provided the offset is a constant frequency shift**, because every
transit-timing analysis — this one and every real one — fits a linear ephemeris

```
T_n = T_0 + n·P
```

to the transit times and analyses the residuals. A constant period offset is
absorbed exactly by the fitted `P` and cancels out of the O−C residuals. What would
*not* be absorbed is an accelerating or erratic phase error.

`test_phase_error_grows_linearly_in_time` therefore tests the property we actually
depend on: fitting `φ(t) = c₁t + c₂t²` to the phase error, the quadratic term must be
below 10⁻³ of the linear term. The end-to-end confirmation — that an isolated planet
produces O−C residuals far below 1 s after ephemeris fitting — is established in
Phase 6.

### Angular momentum

Angular momentum drift sits at **round-off (~10⁻¹⁵) at every timestep, with no dt
dependence at all**. This is expected and is a stronger statement than the energy
result: each sub-step of the kick–drift–kick map is a shear that preserves r × v
exactly for a central force, so angular momentum is conserved identically rather than
approximately.

## Timestep policy

**Policy: `dt = P_innermost / 1000`.**

For Kepler-90 b (P = 7.0083 d) this gives **dt ≈ 0.00701 d ≈ 10.1 minutes**, and the
~1500-day Kepler baseline costs ~214,000 steps.

Justification:

1. **Second-order convergence is confirmed**, so the error budget is predictable and
   can be tightened by refinement if a later phase demands it.
2. **Secular energy drift at 1000 steps/orbit is ~3×10⁻¹²**, which by
   δP/P ≈ 1.5 δE/E corresponds to ~10⁻¹¹ relative period error — around 10⁻⁶ s over
   the baseline. Utterly negligible.
3. **Angular momentum is conserved to round-off** independent of dt.
4. **The along-track phase error is a constant frequency shift**, absorbed by the
   linear ephemeris fit (see above).
5. The innermost orbit sets the stability limit; every other planet is then sampled
   far more finely than 1000 steps per orbit (Kepler-90 h gets ~47,000).

The timestep was **not** chosen because the animation looked smooth, and it is not
tied to the frame rate.

### Phase 6 verdict on the timestep — CLOSED

The open item above has been settled end-to-end, through the real analysis pipeline
rather than by proxy argument. Measured in `tests/test_ttv.py`:

| Measurement | Result | Requirement | Verdict |
|---|---|---|---|
| O−C RMS, isolated planet (b, i, e, h) | **< 1 s** | ≪ minutes-scale TTVs | **PASS** |
| O−C reproducibility under dt → dt/2, isolated | **< 1 s** | — | **PASS** |
| O−C reproducibility under dt → dt/2, full 8-planet system | **≈ 1.6 s** | < 20% of best measurement | **PASS** |
| Physical TTV signal, full system (planet b) | **≈ 106 s RMS** | ≫ numerical floor | **PASS** |
| Fitted-period offset from the input period | **3.1 × 10⁻⁶**, scaling as dt² | absorbed by ephemeris fit | **PASS** |

**The timestep `dt = P_inner/1000` is confirmed for production.**

Two results deserve emphasis because they look alarming in isolation:

1. **Raw transit times shift with dt, growing linearly with epoch** — up to 170 s over
   200 days for planet b when dt is halved. This is the constant frequency offset of
   the shadow Hamiltonian. The *residuals* after ephemeris fitting agree to < 1 s, and
   the residuals are what the inference reads.
2. **The fitted period sits 3.1 × 10⁻⁶ above the input period** (~4 s per orbit for
   planet i). Verified to scale as dt² under refinement, confirming it is the known
   symplectic frequency shift and not a coding error. Real transit-timing analyses
   always fit the period rather than assuming it, so this is absorbed exactly.

**The decisive comparison is against the data, not against an arbitrary threshold.**
The published transit-epoch uncertainties for Kepler-90 range from **71 s** (planet h)
to **1210 s** (planet f). The numerical reproducibility of ~1.6 s is more than **40×
below the most precise measurement that exists for this system**, so it cannot
influence any inference drawn from timing data of that quality.

## Precision

All physics runs in **f64**. In f32 the representable time resolution at t ~ 10⁴ days
is ~86 s — the same order as the signals being measured — so f32 would fabricate
"TTVs" indistinguishable from physical ones. Physics runs on the CPU backend because
Taichi's Vulkan backend (the GPU available on the development machine) does not
support f64. With N ~ 10 bodies this costs nothing: the parallelism that matters is
across *ensemble members* during parameter search, which the CPU backend threads
perfectly well.
