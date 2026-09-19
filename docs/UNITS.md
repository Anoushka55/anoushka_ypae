# Unit System — analysis and decision

**Phase 2 deliverable.** Implemented in `invisible_planet/constants.py`, the only place
in the codebase where a physical constant may be defined.

> Every number in this document is **computed** by `scripts/build_units_doc.py` from the
> reference dataset and `constants.py`. Nothing is typed by hand.

## Decision

> **length = astronomical unit (au) · time = day · mass = solar mass (M☉)**
>
> with **G = 2.959122081921e-04 au³ M☉⁻¹ day⁻²**, derived from the IAU 2015 nominal solar mass
> parameter GM☉ = 1.3271244×10²⁰ m³ s⁻², and cross-checked against the classical Gaussian
> constant k² (agreement 3.2e-10).

## Candidates compared

Values for Kepler-90 (M★ = 1.108 M☉; semi-major axes from Kepler's third law with the
total mass; velocities are circular speeds).

| | **SI** | **au, M☉, yr** | **au, M⊕, day** | **normalised (G = M★ = a_h = 1)** |
|---|---|---|---|---|
| G | 6.67430e-11 | 39.4769 (4π² = 39.4784) | 8.8877e-10 | 1 |
| M★ | 2.203e+30 kg | 1.108 | 3.6890e+05 | 1 |
| planet masses | 1.633e+25 – 1.212e+27 kg | 8.210e-06 – 6.097e-04 | 2.73 – 203.00 | 7.410e-06 – 5.503e-04 |
| a range | 1.109e+10 m – 1.452e+11 m | 0.0742 – 0.9704 | 0.0742 – 0.9704 | 0.0764 – 1.0000 |
| P range | 6.055e+05 s – 2.865e+07 s | 0.0192 – 0.9079 | 7.01 – 331.60 | 0.1328 – 6.2815 |
| v_circ range | 3.183e+04 m/s – 1.151e+05 m/s | 6.71 – 24.29 | 0.0184 – 0.0665 | 1.0000 – 3.6172 |

### Evaluation against the project's actual requirements

**Numerical stability.** All of the normalised systems keep every stored quantity within a
few orders of unity, so double-precision round-off is comparable between them. SI spreads
values across ~40 orders of magnitude, which makes tolerances hard to reason about.
**SI rejected.**

**The decisive criterion — the observable.** The project's scientific output is a *transit
time*, quoted in **days** (BJD) by every source in the dataset, and the O−C residuals to be
resolved are of order seconds to minutes. With the day as the time unit, simulation time
*is* the observable.

In the normalised system every transit time is multiplied by a conversion factor that
depends on M★ (here T = 52.790 d per unit). M★ is uncertain at the ±3% level, so that
uncertainty would leak into the time axis of the measurements. **Rejected.**

The year (system 2) adds a needless conversion on every timing computation. **Rejected.**

Systems 3 and the chosen system differ only in the mass unit. The star's mass appears in
nearly every dynamical expression (GM★) while planet masses enter linearly as small
perturbers, so M☉ puts the *dominant* quantity at unity. Planet masses are converted at the
dataset boundary by `earth_masses_to_solar()`.

## About G M☉ = 4π² au³ yr⁻²

The identity is a convenient approximation, not an exact relation. It equates GM☉ with the
Kepler law for a body at 1 au whose period is *exactly* one Julian year (365.25 d), whereas
the Gaussian constant defines the period at 1 au to be 2π/k = 365.2569 d. Hence

```
G M☉ (IAU 2015 nominal)  = 2.959122081921e-04 au³/day²  = 39.476926 au³/yr²
Gaussian constant k²     = 2.959122082856e-04 au³/day²   (agreement 3.2e-10)
4π² au³/yr²              = 39.478418 au³/yr²   (relative error 3.777e-05)
```

**What this does and does not affect — stated carefully.**

* If the semi-major axis is derived from the *measured period* with one consistent GM, every
  transit time is unchanged: transit times depend on the period, on planet-to-star mass
  ratios and on the geometry, and only a/R★ moves (by ~1×10⁻⁵). Using 4π² consistently would
  be an equivalent model with a stellar mass 4×10⁻⁵ different, far below the ±3% uncertainty
  on M★. It would not, by itself, corrupt any timing result.
* The catalogue's semi-major axes **do** embed the shortcut. Recomputing
  a = (M★ P²/yr²)^(1/3) with P in Julian years reproduces every catalogue value to
  7e-10 (relative), which identifies the cause of the uniform 1.26e-05
  offset between the catalogue and a Kepler-law calculation using the IAU value of GM☉.
  Feeding a catalogue `a` into a dynamical model built on a different GM would therefore
  mis-set the period by 1.89e-05 — up to 9.0 minutes per orbit for Kepler-90 h.
  That is an *inconsistency between two conventions*, not an error inherent in either.

The project avoids the inconsistency by never reading the catalogue `a`: the semi-major
axis is derived from the measured period and the *total* mass using the single `G` defined
in `constants.py`. (The catalogue also omitted the planet mass; see
`docs/KEPLER90_DATA.md`.)

## Why G is used through GM, never as G and M separately

The CODATA value of G carries ~2×10⁻⁵ relative uncertainty, whereas the *product* GM☉ is
known to ~10⁻¹⁰. All mass ratios (e.g. `EARTH_MASS_IN_SOLAR`) are formed from GM ratios so
the poorly known G cancels exactly.

## Precision

Double precision (f64) is mandatory throughout the physics layer. In single precision the
spacing of representable times near t = 10⁴ d is 2⁻¹⁰ d ≈ 84 s, the same order as the
signals being measured.

## Representative magnitudes in the adopted system (Kepler-90)

M★ = 1.108 M☉

| planet | m [M☉] | a [au] | P [day] | v_circ [au/day] | GM★/a² [au/day²] |
|---|---|---|---|---|---|
| b | 8.2102e-06 | 0.07416 | 7.00828 | 0.066490 | 5.961056e-02 |
| c | 1.2086e-05 | 0.08579 | 8.71983 | 0.061819 | 4.454445e-02 |
| i | 8.6802e-06 | 0.12014 | 14.44912 | 0.052241 | 2.271694e-02 |
| d | 2.1510e-05 | 0.30947 | 59.73703 | 0.032550 | 3.423552e-03 |
| e | 1.9732e-05 | 0.41253 | 91.94046 | 0.028192 | 1.926608e-03 |
| f | 2.0839e-05 | 0.50605 | 124.91634 | 0.025454 | 1.280294e-03 |
| g | 4.5052e-05 | 0.71685 | 210.60311 | 0.021386 | 6.380321e-04 |
| h | 6.0971e-04 | 0.97037 | 331.60111 | 0.018382 | 3.481984e-04 |

Reproduce: `uv run --python 3.12 --extra dev python scripts/build_units_doc.py`
