# Identifiability — how well is Planet X actually determined?

**Phase 11 deliverable.** Reproduce with:

```
uv run --python 3.12 --extra dev python scripts/run_identifiability.py
```

The question is not "what is the best fit?" but "what do these observations
actually constrain, and what is degenerate with what?" Where the answer is
"not much", that is reported as the result.

## Summary

| Parameter | Constraint | Verdict |
|---|---|---|
| **Period / semi-major axis** | 1σ width **below the 0.0004 au grid resolution** (< 0.2%) | **Sharply determined** |
| **Orbital phase** | recovered to ~0.08 rad | **Well determined** |
| **Mass** | 1σ width **20%**, with a noise-dependent bias of the same order | **Weakly determined** |

The detection itself is solid: **Δχ² = 52.9 over the no-planet hypothesis, ≈ 7.3σ.**

## 1. Period is extremely well constrained — and that is why the search is hard

The χ² profile in semi-major axis is narrower than the 0.0004 au sampling grid.
A direct scan measured the shape:

```
a = 0.2195  →  χ² = 627.6
a = 0.2200  →  χ² = 516.8      ← true value
a = 0.2205  →  χ² = 564.7
a = 0.2210  →  χ² = 603.3
```

The half-width is ~0.0005 au, about **0.25% in a**. The physical reason is that the
TTV signal is quasi-periodic: an error in the perturber's period accumulates a phase
error ≈ 2π(T/P)(δP/P) across the baseline T, so the model decorrelates from the data
once δP/P exceeds ≈ P/(2πT) ≈ 0.4%.

This has a direct methodological consequence, learned the hard way: **a coarse grid
search cannot solve this problem.** An early coarse-to-fine search with 0.03 au grid
spacing — 60× wider than the minimum — converged to a spurious solution whose χ² was
*worse* than the true parameters'. The fix was a periodogram-style dense scan; see
`docs/INFERENCE.md`.

## 2. Mass is the weakly constrained parameter

The χ² curve in mass is shallow:

| mass [M⊕] | best χ² over phase |
|---|---|
| 10 | 540.6 |
| 20 | 532.7 |
| 30 | 522.9 |
| **40 (true)** | **517.0** |
| 50 | 514.9 |
| 60 | 516.0 |
| 80 | 526.9 |
| 100 | 548.6 |

The minimum sits at 50 M⊕ against a true value of 40 M⊕, and the 1σ interval
(Δχ² < 1) spans **45–55 M⊕**, a relative width of 20%.

**The true mass lies outside that particular interval.** This is reported rather than
smoothed over, and it is worth understanding: the profile above holds semi-major axis
and phase *fixed at their true values* while varying mass. The blind search, which
varies all three, found mass = 35 M⊕ with a 1σ interval of 32.5–40.0 M⊕ that *does*
contain the truth — because small compensating shifts in a and phase absorb part of
the mass error.

Two honest conclusions follow:

1. A 1σ interval from a one-dimensional χ² profile **understates** the real
   uncertainty when parameters are correlated.
2. With a single noise realisation, the recovered mass carries a bias comparable to
   its formal uncertainty. Mass should be quoted as *"tens of Earth masses"*, not as
   a precise value.

## 3. Dependence on observation duration

| baseline [days] | Δχ² vs null | significance |
|---|---|---|
| 400 | 12.3 | 3.5σ |
| 700 | 11.0 | 3.3σ |
| 1000 | 30.6 | 5.5σ |
| 1400 | 52.9 | 7.3σ |

Note the **non-monotonicity** between 400 and 700 days. Significance does not simply
accumulate with more data, because the TTV signal has its own long "super-period"
set by the proximity to resonance; a baseline that ends on an unfavourable part of
that cycle gains little. This is a real feature of TTV observations, not noise in the
measurement, and it means "observe longer" is not automatically "detect better".

A useful practical statement: **roughly 1000 days are needed** before this signal
becomes convincing (> 5σ).

## 4. Dependence on measurement precision

| noise scale | Δχ² vs null | significance |
|---|---|---|
| 0.25× | 637.1 | 25.2σ |
| 0.5× | 176.7 | 13.3σ |
| **1.0× (published precision)** | **52.9** | **7.3σ** |
| 2.0× | 17.6 | 4.2σ |

Significance scales approximately inversely with timing precision, as expected. At
twice the real uncertainties the detection would drop to a marginal 4σ.

This matters because the 1.0× row uses **the actual published transit-epoch
uncertainties for Kepler-90** (71–1210 s), not an invented noise level. The
experiment is therefore calibrated to the precision that genuinely exists for this
system.

## 5. Where the detection comes from

The signal is not spread evenly across the visible planets:

| planet | O−C RMS induced [s] | timing σ [s] | ratio |
|---|---|---|---|
| b | 6.9 | 146.9 | 0.05 |
| c | 6.6 | 198.7 | 0.03 |
| i | 55.9 | 414.7 | 0.13 |
| **d** | **415.5** | **362.9** | **1.15** |
| e | 172.3 | 544.3 | 0.32 |
| f | 55.1 | 1209.6 | 0.05 |

**Planet d carries the detection.** Only its induced signal exceeds its own timing
uncertainty. The other planets contribute through sheer transit count (planet b has
200 transits), but individually sit below their noise.

This is a genuine fragility of the experiment: the result rests largely on one
planet. If planet d's transit times were systematically biased, the inferred Planet X
would be biased with them.

## 6. What this means for the claim

The experiment supports:

- **An unseen body is required by the data** (Δχ² ≈ 53, ~7σ).
- **Its orbital period is tightly determined** (< 0.2%).
- **Its mass is constrained only to within tens of percent**, with a bias comparable
  to that width for any single noise realisation.

The experiment does **not** support:

- A precise mass measurement.
- The claim that the solution is unique across the full parameter space without the
  nonlinear verification step — several spurious periodogram peaks scored *better*
  than the truth under the linearised model.
- Any statement about the real Kepler-90 system. Planet X is synthetic.
