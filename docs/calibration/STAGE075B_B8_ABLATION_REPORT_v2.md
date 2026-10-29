# Stage 0.75-B B8 mechanism ablation report

**Status:** synthetic-only ablation; no method is promoted to the Stage 1 registry.
**Results:** [`STAGE075B_SYNTHETIC_RESULTS_REPORT.md`](STAGE075B_SYNTHETIC_RESULTS_REPORT.md).

## Fairness

B3, B7, B3+Offset, B8-M, B8-S, B8-P, and B8-Full were driven by one shared prequential engine. They received the same Q-A-TS tape, common score vector, controller-independent queried-normal writes, and current-version synthetic evidence snapshots. The strong simple B3+Offset control and B8-Full used the same distinct queried normal and defect groups and source-exclusion rules. All moving arms used the same `0.01` raw threshold movement cap. Query selection and memory writes never used a controller threshold.

The versioned synthetic authority excludes an admitted normal source from all three fixture layers. This demonstrates the protocol and source ledger mechanics only; it does not establish native V0.2/Greedy LSO parity.

## Aggregate comparison

Unweighted mean of per-seed BA across the 42 defined scenario/seed rows:

| Arm | Mean BA | Paired difference vs B3 | Role |
|---|---:|---:|---|
| B3 | 0.746193 | reference | Actual `StateConsistentQuantile`, unchanged. |
| B7 | 0.750430 | +0.004237 | Actual `SCDACGController`; original count gates and descriptive CP behavior retained. |
| B3+Offset | 0.760482 | +0.014289 | Raw midpoint of queried class medians, bounded by the same move cap. |
| B8-M | 0.744957 | −0.001236 | Paired C64 median transport; weak under zero-median tail changes. |
| B8-S | 0.746193 | 0.000000 | Standardized residual; its trajectory matched B3 in this matrix. |
| B8-P | 0.743525 | −0.002668 | Percentile representation with a simple defect-median target and no full evidence gate. |
| B8-Full | 0.746193 | 0.000000 | Full evidence-conditioned candidate v2; stayed exactly at B3. |

Among the 42 defined paired rows, B8-Full versus B3+Offset was higher in 7, lower in 35, and tied in 0; the mean paired difference was `−0.014289`. This pattern is descriptive and generator-dependent. It does not establish B3+Offset as a generally superior method.

## What each ablation says

### B8-M: median transport

The zero-median tail fixture changed the C64 higher-q90 from `57.0` to `57.5` while the median remained fixed. Median-only transport cannot represent that tail change. Across the full suite, B8-M mean BA was slightly below B3. The synthetic result argues against treating median shift as a sufficient memory-state coordinate.

### B8-S: standardized residual

The standardized residual provides a scale-aware representation, but it did not move the decision away from B3 in the full matrix. Its mean BA exactly matched B3. No independent value from the residual representation was observed under these settings.

### B8-P: percentile-only

B8-P can choose a threshold based on the queried-defect median without the full two-class reliability machinery. It changed some thresholds, but its mean BA was below both B3 and B3+Offset. With C64's 64 ranks, different raw thresholds can share a percentile coordinate while producing different tie decisions. The implementation always logs and applies a raw threshold; this avoids silently changing the equality convention, but does not remove rank-resolution limits.

### B8-Full: evidence gate and baseline-relative regularization

The full candidate had both queried classes in 41 of 48 final snapshots. Its candidate-search reliability heuristic averaged `1.2222`; its maximum observed empirical delta-J at those final snapshots was `0.6310`. No full penalized objective was positive. Its raw threshold trajectory matched B3 in every scenario/seed row, so the percentile coordinate, evidence gate, and baseline-relative regularization did not produce an incremental action under the frozen v2 defaults.

This is not a reason to tune the heuristic after outcome inspection. Any changed penalty, candidate count rule, or evidence threshold is a new method version and requires a separately authorized, preregistered experiment.

## Complexity decision

The synthetic evidence does not justify B8-Full's added machinery over actual B3 or the simple raw-offset comparator. The most defensible current reading is `SIMPLER_BASELINE_SUFFICIENT` for this synthetic development suite, with the B8 mechanism classified `SYNTHETIC_MECHANISM_NOT_SUPPORTED`. This is a research decision candidate, not evidence that a simple offset is the correct Real-IAD method.
