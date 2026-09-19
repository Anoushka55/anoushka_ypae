# Unit System — analysis and decision

**Phase 2 deliverable.** Implemented in `invisible_planet/constants.py`, which is the
only place in the codebase where a physical constant may be defined.

## Decision

> **length = astronomical unit (au) · time = day · mass = solar mass (M☉)**
>
> with **G = 2.959122081921 × 10⁻⁴ au³ M☉⁻¹ day⁻²**, derived from the IAU 2015 nominal
> solar mass parameter — *not* from the textbook approximation G M☉ = 4π² au³ yr⁻².

## Candidates compared

| | **(1) SI** | **(2) au, M☉, yr** | **(3) au, M⊕, day** | **(4) Normalised (a_h = M★ = G = 1)** |
|---|---|---|---|---|
| G | 6.674×10⁻¹¹ | 39.478 (≈4π²) | 8.888×10⁻¹⁰ | 1 (exact) |
| M★ | 2.20×10³⁰ kg | 1.108 | 3.689×10⁵ | 1 |
| m_b (smallest planet) | 1.63×10²⁵ kg | 8.21×10⁻⁶ | 2.73 | 7.41×10⁻⁶ |
| a (range) | 1.11×10¹⁰ – 1.45×10¹¹ m | 0.074 – 0.970 | 0.074 – 0.970 | 0.076 – 1.000 |
| P (range) | 6.06×10⁵ – 2.87×10⁷ s | 0.0192 – 0.908 | 7.01 – 331.6 | 0.48 – 22.8 |
| v (range) | 2.9×10⁴ – 1.1×10⁵ m/s | 6.7 – 24.3 | 0.018 – 0.066 | 0.21 – 0.76 |
| Dynamic range of stored values | ~10⁴⁰ | ~10⁷ | ~10⁶ | ~10⁶ |

### Evaluation against the project's actual requirements

**Numerical stability.** All of (2), (3), (4) keep every stored quantity within a few
orders of unity, which is what matters for f64 — there is no meaningful round-off
difference between them. (1) SI spreads values across ~40 orders of magnitude; while f64
technically copes, it destroys the ability to reason about tolerances and makes test
thresholds meaningless. **SI rejected.**

**The decisive criterion — the observable.** This project's entire scientific output is a
*transit time*, and transit times are quoted in **days** (BJD) by every source in our
dataset. The O−C residuals we must resolve are of order **seconds to minutes**.

- In system (3) or the chosen system, simulation time *is* the observable. A transit at
  t = 1523.4172 d is compared directly against an ephemeris in days.
- In system (2) or (4), every transit time is multiplied by a conversion factor before it
  can be compared with data. In (4) that factor depends on **M★**, so the stellar-mass
  uncertainty (±3.2%) would leak into the *time axis* of our measurements — contaminating
  the very residuals we are trying to interpret.

**(4) rejected**: normalising by M★ entangles an observational uncertainty with the
timing observable. Elegant for pure dynamics, wrong for an inference project built on
timing.

**(2) rejected**: the year is a needless conversion layer on every timing computation, and
it invites the 4π² trap discussed below.

**(3) vs the chosen system** — these differ only in the mass unit. M⊕ makes planet masses
read naturally (2.73, 203) but makes the star an awkward 3.689 × 10⁵. Since the star's
mass is what appears in nearly every dynamical expression (GM★), and since planet masses
enter only linearly as small perturbers, **M☉ is the better choice**: it puts the
*dominant* quantity at unity. Planet masses are converted at the dataset boundary by
`earth_masses_to_solar()`.

## The 4π² trap (why G is derived, not assumed)

It is conventional to write G M☉ = 4π² au³ yr⁻². That identity holds only for a test
particle orbiting at exactly 1 au with a period of exactly one Julian year — which is not
quite true of the real Earth.

```
G M☉ from IAU 2015 nominal value : 2.959122081921e-04 au³/day²
Gaussian constant k² (classical) : 2.959122082856e-04 au³/day²   agreement 3.2e-10
4π² au³/yr² converted to au³/day²: 2.959233859352e-04 au³/day²   error    3.8e-05
```

A **3.8 × 10⁻⁵** relative error in GM is not cosmetic for this project. Orbital period
scales as P ∝ (GM)^(−1/2), so every modelled period would be wrong by ~1.9 × 10⁻⁵ —
about **0.006 d (8 minutes) for Kepler-90 h**, accumulating coherently across the
observing baseline. The transit-timing signals we intend to detect are of exactly this
order. Using 4π² would therefore bury a systematic error inside the signal.

Two independent derivations (IAU nominal GM☉, and the classical Gaussian constant k²)
agree to 3 × 10⁻¹⁰, confirming the adopted value. This cross-check is asserted in the
test suite.

## Related precision policy

`G` is never formed as a product of the separately measured `G` and `M☉`. The CODATA
value of G carries ~2 × 10⁻⁵ relative uncertainty, whereas the *product* GM☉ is known to
~10⁻¹⁰. All mass ratios (e.g. `EARTH_MASS_IN_SOLAR`) are likewise formed from GM ratios,
so the poorly known G cancels exactly.

Double precision (f64) is mandatory throughout the physics layer — see
`docs/ARCHITECTURE.md` §3.1.

## Representative magnitudes in the adopted system (Kepler-90)

```
M* = 1.108 M_sun
 planet    m [M_sun]    a [au]     P [day] v_circ [au/d]  accel [au/d^2]
      b   8.2102e-06   0.07416     7.00828      0.066490    5.960936e-02
      c   1.2086e-05   0.08579     8.71983      0.061819    4.454365e-02
      i   8.6802e-06   0.12014    14.44912      0.052241    2.271649e-02
      d   2.1510e-05   0.30947    59.73703      0.032549    3.423510e-03
      e   1.9732e-05   0.41253    91.94046      0.028192    1.926582e-03
      f   2.0839e-05   0.50606   124.91634      0.025454    1.280278e-03
      g   4.5052e-05   0.71685   210.60311      0.021386    6.380334e-04
      h   6.0971e-04   0.97021   331.60111      0.018383    3.483174e-04
```

Reproduce with: `uv run --python 3.12 --with numpy python -m invisible_planet.constants`
