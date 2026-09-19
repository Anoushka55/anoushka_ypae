# Planet X — parameter-search design and selection

**Phase 7.1 and Phase 8 deliverable.**

> **Planet X is synthetic.** It is a body invented for a controlled numerical
> experiment. Nothing in this project is a claim that the real Kepler-90 system
> contains a ninth planet.

## 1. Parameterisation

| Symbol | Parameter | Units | Role in the inference |
|---|---|---|---|
| m_X | mass | M⊕ | amplitude of the TTV signal |
| a_X | semi-major axis | au | sets the period; dominates the signal's *structure* |
| e_X | eccentricity | — | fixed at 0 in the baseline experiment |
| M_X | mean anomaly at epoch | rad | orbital phase |
| i_X | inclination | deg | decides whether the planet transits |
| ω_X | argument of periapsis | rad | degenerate with M_X at e = 0 |

## 2. Physically plausible search ranges, set before any sweep was run

The ranges are derived from the system's architecture, not chosen for effect.

**Semi-major axis.** The known planets sit at 0.074, 0.086, 0.120, 0.309, 0.413,
0.506, 0.717 and 0.970 au. The widest gap in the interior of the system is between
planets **i (0.120 au) and d (0.309 au)** — a factor of 2.6 in radius, and a period
ratio of 4.13. A body there is dynamically plausible and perturbs planets on both
sides. The search range was set to **0.14 – 0.30 au**, with the wide f–g gap and an
exterior orbit also sampled in the sweep.

**Mass.** Bounded below by detectability against the *real* published timing
precision of this system (71 – 1210 s), and above by plausibility: a Saturn-mass or
larger body in this region would likely have been found by other means. Range
**5 – 150 M⊕**.

**Hill stability.** Gladman (1993) gives Δ > 2√3 ≈ 3.46 mutual Hill radii for
two-planet Hill stability; densely packed multi-planet systems typically need
considerably more. The real Kepler-90 system's tightest pair (g–h) sits at
**5.16**. Candidates were required to exceed **8** mutual Hill radii from both
neighbours — comfortably more conservative than the real system itself.

## 3. Selection criteria — fixed in advance

The objective is explicitly **not** "largest signal". A huge signal comes from a
massive planet wedged between two others, which is both implausible and
scientifically uninteresting. Every candidate had to satisfy all of:

| Criterion | Threshold | Why |
|---|---|---|
| dynamically stable | no orbit crossing, bounded elements, ΔE/E < 10⁻⁶ | an unstable system is not a model of anything |
| plausibly spaced | > 8 mutual Hill radii | more conservative than the real g–h pair |
| detectable | O−C RMS above the 71 s best published precision, in ≥ 2 planets | recoverable, not a fluke of one noisy planet |
| subtle | peak-to-peak amplitude between 2 and 120 minutes | obvious-by-eye is not an inference problem |
| **invisible** | impact parameter b > 1 | **a transiting planet needs no inference at all** |

## 4. Sweep results

`scripts/sweep_planet_x.py` evaluated **105 candidates** over a 1200-day baseline
(5 masses × 7 semi-major axes × 3 phases), measuring stability, minimum separation,
energy conservation, per-planet TTV amplitude and Hill separations for each.

- **16 candidates** satisfied every criterion.
- **0** were dynamically unstable over the baseline.
- **22** produced a detectable signal in fewer than two planets.
- **54** were rejected as too obvious (> 120 minutes peak-to-peak).

A caveat worth stating: *no candidate was unstable over 1200 days*. Dynamical
instability in compact systems often takes far longer to manifest, so a short-baseline
stability check is **necessary but not sufficient**. The Hill-separation criterion is
doing the real work of enforcing plausibility.

## 5. The invisibility requirement — a design flaw caught late

The sweep's best candidates were initially generated at the system's typical
inclination (89.8°). At a = 0.22 au that gives an impact parameter of **b = 0.14**,
meaning **Planet X would transit** — and a transiting planet is directly observable,
which defeats the entire premise.

Inclination was therefore added as a constraint rather than left at a default:

| i [deg] | b | transits in 1200 d | max O−C amplitude | planets with detectable signal |
|---|---|---|---|---|
| 89.8 | 0.139 | **33** | 20.3 min | 4 |
| **88.0** | **1.393** | **0** | **23.2 min** | **4** |
| 87.0 | 2.089 | 0 | 32.5 min | 4 |
| 86.0 | 2.785 | 0 | 46.1 min | 4 |

**i = 88.0° was chosen**: the most conservative non-transiting option. It is only
1.8° from the other planets, so the system stays very nearly coplanar and no result
depends on unusual geometry.

This mirrors reality. Non-transiting planets *are* discovered this way — Kepler-19c
was identified purely from the transit timing variations it induced in Kepler-19b.

## 6. The adopted candidate

```
mass                 40.0 M⊕
semi-major axis      0.22 au
period               35.80 d
eccentricity         0.0
mean anomaly         2.09 rad
inclination          88.0°   (b = 1.39 — does not transit)
```

Properties:

- **9.7 mutual Hill radii** from its nearest neighbour (vs 5.16 for the real g–h pair)
- **stable** over the full baseline, ΔE/E ~ 10⁻⁸
- **detectable in 4 visible planets**
- **peak-to-peak O−C of 23 minutes**, against published timing precision of 147–1210 s
- **does not transit**

Its signature on the visible planets, at the 1400-day baseline:

| planet | transits | O−C amplitude [min] | O−C RMS [s] | timing σ [s] |
|---|---|---|---|---|
| b | 200 | 0.67 | 6.9 | 146.9 |
| c | 160 | 0.59 | 6.6 | 198.7 |
| i | 97 | 3.18 | 55.9 | 414.7 |
| **d** | **24** | **27.68** | **415.5** | **362.9** |
| e | 16 | 11.24 | 172.3 | 544.3 |
| f | 11 | 2.74 | 55.1 | 1209.6 |

Note what this table says honestly: **planet d carries the detection**. Its O−C RMS
(415 s) exceeds its timing uncertainty (363 s); every other planet's signal sits below
its own noise. The detection is nevertheless solid because it accumulates over many
transits — the total is Δχ² ≈ 55, about 7σ — but it rests mainly on one planet, and
the published timing precision for this system is poor. That is a real limitation of
the experiment and is reflected in the identifiability results.
