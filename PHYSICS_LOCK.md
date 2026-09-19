# PHYSICS_LOCK.md

**Phase 21 deliverable. This file is the authoritative scientific contract.**

From this point on:

- **No visual change may modify physics.**
- **No performance change may alter numerical behaviour without rerunning validation.**
- **No narrative change may introduce a scientifically false claim.**

---

## 1. Equation of motion

```
a_i = G · Σ_{j≠i}  m_j (r_j − r_i) / |r_j − r_i|³
```

Newtonian point masses. **No softening. No drag. No relativistic terms. No tides.
No oblateness. No special term for any body.** Planet X obeys this identically to
every real planet.

Implemented once, in `invisible_planet/physics/engine.py::NBodyEngine._integrate`.

## 2. Constants and units

| Quantity | Value | Source |
|---|---|---|
| Length unit | 1 au = 1.495978707×10¹¹ m | IAU 2012 (exact) |
| Time unit | 1 day = 86400 s | exact |
| Mass unit | 1 M☉ | — |
| **G** | **2.959122081921×10⁻⁴ au³ M☉⁻¹ day⁻²** | derived from IAU 2015 nominal GM☉ = 1.3271244×10²⁰ m³s⁻² |
| GM⊕ | 3.986004×10¹⁴ m³s⁻² | IAU 2015 nominal |
| R☉ | 6.957×10⁸ m | IAU 2015 nominal |
| R⊕ | 6.3781×10⁶ m | IAU 2015 nominal |

G is cross-checked against the Gaussian constant k² to 3×10⁻¹⁰.
**G M☉ = 4π² au³/yr² is explicitly rejected** (wrong by 3.8×10⁻⁵).

Defined in exactly one place: `invisible_planet/constants.py`.

## 3. Integrator

Velocity Verlet (kick–drift–kick), **fixed timestep**, symplectic.

- `dt = P_innermost / 1000 = 0.00700828 d`
- Measured convergence order: **2.000**
- dt is never varied mid-run and is never tied to frame rate.

## 4. Precision

**f64 throughout the physics layer.** CPU backend (Taichi's Vulkan backend lacks
f64). Rendering may use f32.

## 5. Data sources and status

Adopted from the NASA Exoplanet Archive snapshot at
`data/raw/kepler90_ps_full_snapshot.csv`, accessed **2026-09-19T06:03:26Z**.

| Parameter | Source | Status |
|---|---|---|
| Stellar mass, radius | Fulton & Petigura (2018) | observed |
| Periods, radii (all 8) | Weiss et al. (2024) | observed |
| Semi-major axes | derived here from P and total mass | **derived** |
| Transit epochs, impact parameters (b–h) | Cabrera et al. (2014) | observed |
| Transit epoch, impact parameter (i) | Shallue & Vanderburg (2018) | observed |
| Inclinations | derived here from impact parameter | **derived** |
| Masses of g, h | Liang et al. (2021) | observed (TTV dynamical) |
| Masses of b, c, i, d, e, f | Weiss & Marcy (2014) M–R relation | **derived** |
| Eccentricities | assumed zero | **adopted-model** |
| Longitude of ascending node | assumed zero | **adopted-model** |

## 6. Initial conditions

Astrocentric Keplerian elements → barycentric Cartesian state
(`systems/build.py::build_system`). Orbital phase is inherited from each planet's
**measured transit epoch**. Reference epoch BJD 2454967.0.

## 7. Planet X (SYNTHETIC)

```
mass                 40.0 M⊕
semi-major axis      0.22 au
period               35.80 d
eccentricity         0.0
mean anomaly         2.09 rad
inclination          88.0°  (impact parameter 1.39 — does not transit)
```

Selected by `scripts/sweep_planet_x.py` under criteria fixed before the sweep
(`docs/PLANET_X_DESIGN.md`). **Not real. Not a claim about Kepler-90.**

## 8. Observation model

- Transit centre = minimum sky-projected separation with the planet in front,
  located by cubic Hermite root-finding on d(separation²)/dt.
- Probe planets: b, c, i, d, e, f. (g and h transit too few times.)
- Baseline: 1400 days.

## 9. Noise model

Independent Gaussian per transit, with **per-planet σ taken from the published
transit-epoch uncertainties** (147–1210 s). Seed 20260919.

**Known limitation (locked in as a stated caveat):** epoch uncertainties
underestimate single-transit timing errors by roughly √N.

## 10. Inference

χ² on O−C residuals, matched by epoch number:

```
chi2 = Σ_planets Σ_transits ((O−C_model − O−C_data) / σ)²
```

Three stages: dense periodogram (reference mass **5 M⊕**) → shortlist of
well-separated peaks → **full nonlinear re-scoring** → nonlinear mass scan.

**No machine learning. No ground-truth access.** Enforced by AST import checks and
two leakage scanners.

## 11. Validation tolerances

| Check | Tolerance |
|---|---|
| Force law vs G·m/r² | 1×10⁻¹⁴ |
| Newton's third law | 1×10⁻¹³ |
| Convergence order | 1.7 – 2.3 |
| Angular momentum drift | < 1×10⁻¹² |
| Linear momentum | < 1×10⁻¹⁶ |
| Energy (N-body, bounded) | < 1×10⁻⁶ |
| Secular energy drift (two-body) | < 1×10⁻⁹ |
| O−C, isolated planet | < 1 s RMS |
| Massless Planet X shift | < 1×10⁻³ s |

## 12. Locked limitations

1. Mass is degenerate with inclination — constrained only to a factor of ~2.
2. The adopted noise level is optimistic (see §9).
3. Detection rests mainly on planet d.
4. Planet X eccentricity is never searched.
5. Short-baseline stability ≠ long-term stability.
6. Single noise realisation reported for the headline recovery.

## 13. The claim

An unseen gravitating body leaves measurable timing signatures in observable
bodies, and those signatures constrain **its period tightly and its mass loosely**.

**Planet X is synthetic. Kepler-90 is not claimed to have a ninth planet.**
