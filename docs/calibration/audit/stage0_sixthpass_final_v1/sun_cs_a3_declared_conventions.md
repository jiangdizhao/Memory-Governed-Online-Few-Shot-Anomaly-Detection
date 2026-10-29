# Sun-CS Algorithm 3 — declared implementation conventions v1

**Registration:** `B6_SUN_CS_A3_PAPER_EQUATIONS_WITH_DECLARED_CONVENTIONS`

**Status:** `PAPER_EQUATIONS_WITH_DECLARED_ENDPOINT_CONVENTIONS` for the
primary executable; `CONSERVATIVE_ENDPOINT_SENSITIVITY` for the separate
extended-real diagnostic. No verified author implementation was located, so
author-code trace parity is not claimed. No theorem transfer to our
memory-dependent physical-defect setting is claimed.

## Source target and guarantee boundary

Sun, Sankararaman and Narayanaswamy, “Online Adaptive Anomaly Thresholding
with Confidence Sequences,” ICML 2024, [PMLR 235](https://proceedings.mlr.press/v235/sun24h.html).
The source target is `1[S_t > Q_Ft(p)]`, an anomaly event defined relative to
the score-stream quantile. It is not the Real-IAD physical-defect label. The
source method uses the full unlabeled score stream and does not update from
queried physical labels or treat labeled-normal C64 as offline matching data.

The paper's guarantees depend on its stated IID score assumptions. Theorem 4
also uses a piecewise-stationarity/change-separation condition. Our shared
score can depend on a changing Greedy memory and can be temporally dependent.
Therefore no Sun-CS finite-time, anytime, physical-defect FPR/FNR, or
memory-dependent score guarantee is inferred here.

## Implemented source equations

For `n >= 1`, the implementation uses the paper's confidence radius without
clamping:

```text
u_0(delta) = 1
u_n(delta) = 0.85/sqrt(n) * sqrt(log(log(e*n)) + 0.8*log(1612/delta))
```

For the sorted score sample `y_(1) <= ... <= y_(n)`, an interior empirical
quantile is the paper's midpoint of the one-based order statistics at ranks
`floor(q*n)` and `ceil(q*n)`. Equation (1) clips requested levels to `[0,1]`.
Algorithm 1 predicts anomaly only for a score strictly above the confidence
set's upper endpoint, benign only strictly below its lower endpoint, and
abstains on the inclusive confidence-set interval.

Algorithm 2 intersects all forward prefix confidence sets and all backward
suffix confidence sets. A change is detected if either intersection is empty
or the two nonempty intersections are disjoint. Algorithm 3 applies this test
only to scores observed before the current event. A detection abstains at the
current event and resets the past segment. Otherwise Algorithm 1 predicts
against the past-only confidence set. The current score is appended after its
decision in either case.

## Primary finite-sample conventions

The paper does not define all finite-sample endpoint ranks, the empty-prefix
confidence set, or the first Algorithm 3 prediction. The following explicit
implementation choices fill only those undefined cases:

| Case | Primary convention |
|---|---|
| Exact `q=0` | Select sample minimum (empirical-extrema extension). |
| Exact `q=1` | Select sample maximum, consistent with the source order-statistic endpoint. |
| Interior `q` with `floor(q*n)=0` | Extend both midpoint rank lookups to the first order statistic. |
| Empty historical prefix | Emit `ABSTAIN_COLD_START`; do not use the current score before prediction. |
| Change at zero-based engine event `t` | Emit `ABSTAIN_CHANGE_DETECTED`; set the new segment start to `t`; append current score as its first new-segment value after the decision. This is paper `tau=t-1` in one-based event indexing. |
| Reset event's threshold metadata | No confidence interval is asserted for that event; prediction threshold is unavailable. |

These are programming conventions, not claims that the paper explicitly
specifies them or that its proofs automatically cover them.

## Separate endpoint sensitivity

The secondary diagnostic variant uses `q=0 -> -infinity` and `q=1 ->
+infinity` when Equation (1) clips to an endpoint. It is not silently combined
with the primary implementation and is never selected from observed
Real-IAD outcomes. Infinity is represented in JSON as
`{"kind":"negative_infinity"}` or `{"kind":"positive_infinity"}` rather
than a nonstandard JSON number. This is a conservative endpoint sensitivity,
not an author-code reproduction.

## Registration and numerical configuration

Stage 0 comparison configuration: `p=0.99`, `delta=0.01`. This is a locked
Stage 0 choice, not a universal paper default. The paper reports `p=0.99`,
`delta=0.001` for one experiment and `p=1-10^-6`, `delta=10^-6` for its
real-cloud example; those remain source examples, not a parameter search.
Algorithm 4/5 offline data matching is not registered. C64 is protected,
known-normal calibration data and is not assumed to be independent,
distribution-matched unlabeled offline data.

## Independent reference, optimized calculation, and complexity

`algorithm2_reference` directly sorts every prefix and suffix for small
reference sequences. `algorithm2_optimized` builds exact persistent
order-statistic prefix roots over the currently observed score coordinates;
the two implementations are compared event-by-event in tests. There is no
truncation, approximate change detector, or rolling window.

For a segment of length `n`, the optimized Algorithm 2 check is
`O(n log n)` time and `O(n log n)` temporary persistent-tree nodes (plus
`O(n)` scores and intervals). Algorithm 3 checks Algorithm 2 on each event;
its worst-case cumulative time over a length-`T` segment is `O(T^2 log T)`.
The active history is `O(T)`, while peak temporary order-statistic storage is
`O(T log T)`. Frequent exact resets can reduce the observed cost, but no
worst-case bound is inferred from that behavior. The independent direct
reference is deliberately for small traces and is not used for class-length
runtime claims.

The registered p/delta imply that the upper level is clipped to one at
short-stream sizes. For scale, the lower-level and upper-level requests from
Equation (1) are:

| n | raw `u_n(.01)` | lower requested level | upper requested level |
|---:|---:|---:|---:|
| 64 | 0.35610462193394704 | 0.2777907561321059 | 1 |
| 256 | 0.17992873144050392 | 0.6301425371189922 | 1 |
| 1,050 | 0.08959651974489437 | 0.8108069605102113 | 1 |
| 3,815 | 0.04730615890487752 | 0.8953876821902449 | 1 |

The upper endpoint therefore uses the primary sample maximum at these sizes;
the sensitivity variant uses positive infinity. This follows from the
registered equation and is not tuned from defect outcomes.

## Reference evidence and interpretation

Tests cover radius values above one, interior and endpoint empirical
quantiles, both endpoint policies, Equation (1), Algorithm 1 strictness and
ties, forward/backward intersections, change detection, stationary and
piecewise deterministic score sequences, reset timing, current-event
abstention, Algorithm 2 optimized/reference parity, Algorithm 3 event-trace
parity, and event-boundary resume.

Synthetic prediction/abstention coverage is reported as a mechanism count on
the named deterministic fixtures only. It is not a coverage estimate for
Real-IAD images, physical defects, memory-dependent scores, or future streams.
