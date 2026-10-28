# Sun et al. confidence-sequence method: source-fidelity specification

**Reference:** Sophia Huiwen Sun, Abishek Sankararaman, and Balakrishnan Murali
Narayanaswamy, “Online Adaptive Anomaly Thresholding with Confidence
Sequences,” ICML 2024, PMLR 235, pp. 47105–47132. Primary paper and appendix:
https://proceedings.mlr.press/v235/sun24h.html and
https://raw.githubusercontent.com/mlresearch/v235/main/assets/sun24h/sun24h.pdf.

**Status: `BLOCKED_FIDELITY`.** This document records the source method before
any B6 implementation. It does not certify that its assumptions hold for
VisionAD scores, Real-IAD labels, or a changing Greedy bank. It is not a
registered Stage 1 comparator.

## Observation model and information at decision time

The paper (Sections 2.1–2.5) receives an unlabeled scalar anomaly score
`S_t` at each time. The online algorithm knows the target quantile
`p in (0,1)` and confidence parameter `delta in (0,1)`, but not the score
distribution, true quantile, change points, segment distributions, or labels.
At decision time it may use the current `S_t`, the past score prefix, and,
for Algorithms 4–5, independent unlabeled offline datasets. It emits
`{benign, anomaly, abstain}`. The source anomaly target is
`Y_t = 1[S_t > Q_t(p)]`, where `Q_t(p)` is the current score distribution's
true pth quantile. It is not a Real-IAD defect label.

The online stream is independent across time, scalar-valued, and has a
continuous distribution with no atoms. Algorithm 3 allows piecewise
stationary segment distributions. The paper's observations are not scores
fed back into an adaptive feature memory. The methods do not read selective
human feedback or labels to update a threshold.

## Empirical quantile and confidence set

For sorted observations `y_(1) <= ... <= y_(n)`, the paper defines its
empirical quantile as the midpoint of the order statistics at ranks
`floor(q*n)` and `ceil(q*n)` (Section 4; the paper's order-statistic indexing
is one-based). For `n >= 1` and interior ranks, this is

```text
Qhat(q; y_1:n) = (y_(floor(q*n)) + y_(ceil(q*n))) / 2.
```

The source does not state a complete finite-sample convention for ranks at
the endpoints `q=0` or `q=1`, or for a zero-length sequence. Equation (1)
clips its requested levels to `[0,1]`, so these cases occur, including during
the cold start. The paper states `u_0(delta)=1` but that does not define an
empirical quantile of an empty sequence or resolve endpoint order statistics.
No endpoint or empty-prefix convention is supplied here.

For `t >= 1`, Equation (2) defines

```text
u_t(delta) = 0.85 / sqrt(t) * sqrt(
    log(log(e*t)) + 0.8 * log(1612/delta)
)
```

and `u_0(delta)=1`. Equation (1) defines the quantile confidence set

```text
C(p, delta, y_1:n) = [
    Qhat(max(p - 2*u_n(delta), 0); y_1:n),
    Qhat(min(p + 2*u_n(delta), 1); y_1:n)
].
```

Theorem 1 cites the Howard–Ramdas quantile confidence sequence and states
time-uniform coverage for an IID score sequence. Algorithm 1 labels a score
an anomaly only when it is strictly greater than `max(C)`, abstains when it
belongs to `C` (including an endpoint), and otherwise labels it benign.

The paper does not estimate a free threshold by minimizing a binary loss.
The reported decision boundary is the upper edge of the current confidence
set; the lower edge creates a benign region and the set itself is an
abstention region.

## Change detection, reset, and offline matching

Algorithm 2 (Section 5) constructs

```text
C_forward = intersection over s=1..n of C(p, delta, S_1:s)
C_backward = intersection over s=1..n of C(p, delta, S_s:n)
change = 1 iff C_forward intersection C_backward is empty.
```

Algorithm 3 (Section 5) has `t0=0`. Before deciding at time `t`, it applies
Algorithm 2 to the post-reset substream `S_(t0+1):(t-1)`. If a change is
detected, it abstains at `t` and resets `t0=t-1`. Otherwise, it applies
Algorithm 1 to `S_t` and the confidence set of that substream's past scores.
This is a one-event abstention at a detected change followed by a fresh
confidence sequence.

Algorithm 4 (Section 6) uses offline data only when exactly one offline
dataset's confidence set intersects the online-prefix confidence set. It
then forms a confidence set on that offline dataset pooled with the online
prefix and applies Algorithm 1. With zero or multiple matches it uses only
the online prefix. Algorithm 5 combines Algorithm 2's reset with Algorithm
4. The paper assumes independent offline samples; all observations in each
offline dataset share that dataset's pth quantile. It does not define C64 as
normal-only calibration data or as labeled input.

The empty-prefix ambiguity also applies to Algorithms 3–5 at their first
decision. It prevents a source-defined decision for the first event absent an
additional convention.

## Assumptions and guarantee target

The algorithm targets cumulative false positives, false negatives, and
abstentions relative to the paper's score-quantile anomaly label. Theorem 2
gives zero mistakes and an order-`sqrt(T log log T)` abstention bound for a
stationary IID stream with confidence parameter `delta`. Theorem 4 adds
piecewise stationarity under Assumption 5.1: change points are sufficiently
separated relative to quantile-shift-dependent detection delays. The paper
explicitly says this separation assumption is not assumed in its experiments.
Theorem 5 uses stationary offline data under Assumption 6.1 (offline
confidence sets sufficiently separated, with the stated quantile-match or
quantile-shift conditions). Theorem 6 combines the piecewise-online and
offline assumptions. Its error and abstention bounds are high-probability
bounds on the paper's labels and abstention objective, not class-conditional
Real-IAD FPR/FNR guarantees.

These guarantees require the stated independence, continuity/no-atom,
piecewise-stationarity and (where relevant) spacing, detectability and
offline-separation assumptions. A shared score process whose distribution
changes when the calibration Greedy memory changes is not shown to satisfy
them. Image streams may also be temporally dependent. Defect labels are not
used by the paper and cannot repair this mismatch. Neither an ACI guarantee
nor a Sun et al. finite-time/anytime guarantee is claimed for selective
feedback or dynamic memory.

## Parameters and computational cost

Required method parameters are quantile level `p` and confidence error
`delta`; `p=0.99, delta=0.001` is reported for paper experiments, while the
real-cloud example uses `p=1-10^-6, delta=10^-6`. These are source experiment
settings, not chosen Stage 0 defaults. No B6 target or parameter grid is
registered by this specification. A B6 implementation must freeze its
target and abstention accounting before any comparator execution.

With all past scores retained, direct recomputation of every prefix/suffix
confidence set for Algorithm 2 costs `O(n^2 log n)` per check due to sorting
each subsequence, and `O(n)` storage. Algorithm 4 additionally costs the
offline confidence-set calculations and `O(K)` interval intersection checks
per decision. Incremental data structures could reduce the cost, but would
need separate numerical-parity tests.

## Source implementation search and unresolved items

The PMLR page supplies the paper and PDF but no code repository link. The
paper's Appendix G says the implementation will be open sourced. Searches of
GitHub by exact paper title, `sun24h`, and method/author names, plus the
authors' publication pages, did not locate a verified author implementation
or a commit that can be attributed to this paper. No third-party source is
treated as author code.

Before B6 can leave `BLOCKED_FIDELITY`, the project needs a source-grounded
resolution for (1) endpoint empirical quantiles when Equation (1) clips to
0/1, (2) empty-prefix/cold-start behavior at the first online decision, and
(3) a faithful reference trace for confidence-set construction, Algorithm 2
intersections, reset timing, and abstention. Until then, do not register a
controller, select an inferred all-real startup set, add C64 as offline data,
or call a dynamic-memory adaptation Sun-CS. Paper-defined interval decision
semantics can be unit-tested independently once a valid confidence set is
given, but they do not by themselves complete B6.
