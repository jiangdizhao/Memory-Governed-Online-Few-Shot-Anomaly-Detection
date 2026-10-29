# Sun-CS Algorithm 3 — conservative endpoint candidate v2

**Primary research candidate:** `SUN_CS_A3_CONSERVATIVE_EXTENDED_REAL`

**Sensitivity version:** `SUN_CS_A3_EMPIRICAL_EXTREMA`

**Implementation version:** `B6-SUN-CS-A3-CONSERVATIVE-EXTENDED-REAL-v2.0.0`

**Primary status:** `PAPER_EQUATIONS_WITH_DECLARED_ENDPOINT_CONVENTIONS`.
There is no verified author-code trace parity and no claim that the paper's
theorems transfer to physical-defect labels, selective feedback, dependent
scores, or scores changed by dynamic Greedy memory.

## Confirmed endpoint pathology

For the registered `p=.99`, `delta=.01` setting, Equation (2) produces a
confidence radius larger than `.495` for the first two samples. Equation (1)
therefore clips the requested lower and upper quantile levels to `0` and `1`
for each one- or two-score prefix.

For distinct finite `a < b`, the empirical-extrema convention yields:

```text
C([a])    = [a, a]
C([a,b])  = [a, b]
C([b])    = [b, b]
```

The forward prefix intersection is `{a}` and the backward suffix
intersection is `{b}`. Algorithm 2 therefore reports a change after two
different observations even when both are ordinary draws from a stationary
continuous distribution. Algorithm 3 abstains at that event, resets, and
repeats the behavior on the next pair. This mechanism was reproduced by the
production Algorithm 2 and Algorithm 3 and by the independent reference path.

The conservative extended-real convention instead gives:

```text
C([a]) = C([a,b]) = C([b]) = [-infinity, +infinity]
```

for those clipped endpoints. Both Algorithm 2 implementations report no
change for the two-score trace. This removes the endpoint-induced false reset
in the minimal fixture. It does not make the method useful at class-length
streams: the upper endpoint remains unbounded at the registered configuration
through lengths of 3,815.

## Exact implementation conventions

The paper equations, Algorithm 3 reset rule, decision-before-update order,
and abstention behavior are unchanged. At current event `t`, Algorithm 2 sees
only scores before `t`. A detected change abstains at `t`, resets the history,
and appends the current score as the first score of the next segment after the
decision. An empty prefix emits `ABSTAIN_COLD_START`.

| Case | Conservative primary | Empirical sensitivity |
|---|---|---|
| Clipped `q=0` | `-infinity` | Sample minimum |
| Clipped `q=1` | `+infinity` | Sample maximum |
| Interior midpoint ranks | One-based floor/ceiling ranks; floor zero extends to rank one | Same |
| Empty past | `ABSTAIN_COLD_START` before current score is observed | Same |
| Reset event | `ABSTAIN_CHANGE_DETECTED`, reset, then append current score | Same |

These endpoint choices are explicit implementation conventions, not author
code. The empirical-extrema implementation remains available for numerical
sensitivity analysis, but is not the primary research candidate.

## Finite-stream identifiability

The raw radius is not clamped. At `n=64, 256, 1,050, 3,815`, the requested
levels are:

| `n` | `u_n(.01)` | lower level | upper level | lower endpoint for sample `1,...,n` | empirical upper endpoint | conservative upper endpoint |
|---:|---:|---:|---:|---:|---:|---|
| 64 | 0.35610462193394704 | 0.2777907561321059 | 1 | 17.5 | 64 | `+infinity` |
| 256 | 0.17992873144050392 | 0.6301425371189922 | 1 | 161.5 | 256 | `+infinity` |
| 1,050 | 0.08959651974489437 | 0.8108069605102113 | 1 | 851.5 | 1,050 | `+infinity` |
| 3,815 | 0.04730615890487752 | 0.8953876821902449 | 1 | 3,415.5 | 3,815 | `+infinity` |

The lower endpoint values use the exact declared midpoint order-statistic
rule on the illustrative sample `[1,...,n]`. For arbitrary samples, use the
midpoint of their one-based floor/ceiling order statistics at the displayed
lower level. The empirical upper endpoint is the sample maximum; the
conservative upper endpoint is positive infinity.

For every finite score, `score > +infinity` is false. Thus at these sample
sizes the conservative primary cannot issue an anomaly prediction. It may
issue a benign prediction below the finite lower endpoint or abstain. This is
the direct mathematical consequence of the registered confidence levels,
not a software defect and not an estimate of Real-IAD performance.

The first prefix length where `.99 + 2*u_n(.01) < 1` is `353,018`; at
`n=353,017` the upper level is still clipped to one, and at `n=353,018` it is
approximately `0.99999998684`. An anomaly prediction is not identifiable by
this endpoint rule before such a segment length is reached.

## Synthetic applicability protocol

The versioned diagnostic suite uses one registered seed per scenario and
does not search seeds or parameters:

| Scenario | Generator | Length | Target quantile |
|---|---|---:|---|
| Stationary IID | `Uniform(0,1)` | 512 | `.99` |
| Stationary heavy-tailed | `LogNormal(0,1)` | 512 | `exp(Phi^-1(.99))` |
| Piecewise stationary | `Uniform(0,1)` then `Uniform(2,3)` at `t=256` | 512 | `.99` before, `2.99` after shift |
| Class-scale bounded diagnostic | Seeded `Uniform(0,1)` | 3,815 | `.99` |

Every current score is predicted before its score is appended. The online
controller receives only the causal scalar prefix. The exact synthetic
distribution and its true quantile are held by the offline evaluator and are
used only after prediction to measure source-target classification and
confidence-interval behavior. These are quantile-target diagnostics, not
physical-defect FPR/FNR.

For each policy/scenario pair, record event and prediction counts, first
binary-decision position, coverage, reset count, false resets under stationary
streams, post-shift detection and latency, source-target confusion among
covered events, true-quantile confidence-set coverage, untraced runtime and
amortized runtime, separately traced Python peak allocation, sampled process
RSS when available, random seed, policy/version, and source fingerprints.
Class-scale runtime is measured without `tracemalloc`; memory tracing is a
separate diagnostic pass.

## Complexity and guarantee boundary

Algorithm 2 remains the exact persistent order-statistic implementation with
independent sort-based trace reference. No rolling window, prefix truncation,
approximate quantile, or alternate drift criterion is introduced. Algorithm
3 recomputes Algorithm 2 on each observed prefix: worst-case cumulative time
is `O(T^2 log T)` and peak temporary auxiliary space is `O(T log T)` for a
no-reset segment. No runtime optimization is claimed by this version.

The conservative endpoint policy addresses the confirmed false-reset
pathology by expressing unbounded endpoint uncertainty. That same uncertainty
precludes anomaly decisions at realistic class lengths under the registered
`p` and `delta`. Consequently Sun-CS is retained as a scientifically
transparent source-task reference and sensitivity study, not asserted as a
useful physical-defect calibrator in this regime. Neither endpoint policy
inherits source guarantees for a dynamic-memory-dependent score stream.
