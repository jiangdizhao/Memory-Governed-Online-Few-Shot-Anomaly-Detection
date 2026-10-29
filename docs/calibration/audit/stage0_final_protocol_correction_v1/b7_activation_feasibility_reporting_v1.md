# B7 sparse-feedback activation and feasibility reporting v1

**Scope:** Stage 1 reporting specification only. This document does not change
SCDAC-G v0.1, its numerical defaults, threshold objective, memory policy,
query policy, or feedback eligibility. Stage 1 remains unauthorized until the
protocol receives independent researcher approval.

## Registered B7 settings relevant to feasibility

| Setting | Value |
|---|---:|
| Minimum eligible current-memory normal anchors | 16 |
| Minimum eligible queried-defect anchors | 8 |
| One-sided Clopper-Pearson descriptive delta | 0.05 |
| Descriptive FPR target | 0.20 |
| Descriptive FNR target | 0.20 |

The bounds remain descriptive. Selective disclosure, time dependence, repeated
memory rescoring and post-selection prevent an automatic population-risk
interpretation.

## Evidence availability and descriptive feasibility are separate

For each prequential event `t`, report both:

```text
TWO_SIDED_EVIDENCE_AVAILABLE_t
  = (valid_current_memory_normal_anchor_count_t >= 16)
    and (valid_queried_defect_anchor_count_t >= 8)

DESCRIPTIVE_TWO_SIDED_CONSTRAINT_FEASIBLE_t
  = TWO_SIDED_EVIDENCE_AVAILABLE_t
    and (number of candidate thresholds with U_FPR <= .20
         and U_FNR <= .20 is greater than zero)
```

The first is an activation-count condition. The second is a candidate-grid
result using the current eligible evidence and current B7 confidence-bound
calculation. A `BOTH` state only indicates that evidence counts meet their
minimums; it does not say a feasible threshold exists. A subsequent
`CONFLICT` can mean the descriptive constraints are infeasible.

For the zero-error one-sided endpoint,

```text
U(0,n) = 1 - 0.05^(1/n).
```

At eight defects, `U(0,8) = 0.312343978...`, above the `.20` FNR target. At
thirteen it is approximately `.2059`, also above target; at fourteen it is
approximately `.1926`. Solving `1 - .05^(1/n) <= .20` gives minimum `n=14`.
Therefore, eight defects allow nominal two-sided processing but do not imply
that the descriptive FNR constraint can be satisfied, even with zero observed
errors. Fourteen is necessary only in this best-case zero-error calculation;
nonzero errors require more evidence.

## Required event and stream summaries

Report image-level queried normal and defect counts separately from current
eligible anchor counts. The normal anchor set includes C64 anchors plus only
queried verified normals rescored at the current memory state with source-aware
LSO where applicable. Defect anchors include only queried verified defects
rescored at the current memory state. Never infer class evidence from
unqueried labels.

| Metric | Definition / required fields |
|---|---|
| Query composition | Queried normal image count, queried defect image count, total queries, image fraction, official-sample-group fraction. |
| Valid evidence counts | Per-event current-memory normal anchor count and queried-defect anchor count; minimum-validity flags. |
| First state events | First zero-based event in `NORMAL_ONLY`, `BOTH`, and `CONFLICT`; emit null if never reached. Also preserve `NONE` if it occurs. |
| Evidence-state occupancy | Event count and fraction for `NONE`, `NORMAL_ONLY`, `DEFECT_ONLY`, `BOTH`, and `CONFLICT`. |
| Activation and feasibility | First event with `TWO_SIDED_EVIDENCE_AVAILABLE`; first event with `DESCRIPTIVE_TWO_SIDED_CONSTRAINT_FEASIBLE`; stream counts and conditional/unconditional feasibility rates. |
| HOLD behavior | Count/fraction of explicit `HOLD_*` decisions; separately count/fraction of events with no threshold movement. |
| Threshold path | Actual nonzero threshold update count; first event with nonzero movement; signed and absolute movements; maximum movement; threshold values before/after each update. |
| Search work | Candidate evaluations and CP risk-bound evaluations, both event-level and cumulative. |
| Conflict diagnostics | Counts by `INSUFFICIENT_PRECISION`, `RISK_CONSTRAINT_INCOMPATIBILITY`, `UNIDENTIFIABLE_SCORE_DISTRIBUTIONS`, and any other explicit source value. Preserve `DESCRIPTIVE_FEASIBLE_SET` separately. |
| Memory and anchors | Memory version/content changes, current-memory rescore revisions, C64 refresh count, queried-normal LSO-excluded rows, valid anchor additions/removals/revisions, and checkpoint/resume identity. |

`INSUFFICIENT_PRECISION` means at least one candidate satisfies the empirical
support limits but none satisfies both descriptive upper-endpoint limits.
`RISK_CONSTRAINT_INCOMPATIBILITY` means no candidate satisfies the empirical
support limits and the empirical class distributions are not identical.
`UNIDENTIFIABLE_SCORE_DISTRIBUTIONS` means the empirical normal and defect
score distributions are identical under the implementation's exact mass
comparison. These labels describe the registered implementation diagnostics;
they are not population conclusions.

Feasibility rates must use explicit denominators:

```text
conditional feasibility rate = feasible events / two-sided-evidence-available events
stream feasibility rate       = feasible events / all stream events
```

If a denominator is zero, report null/undefined. Do not substitute zero.
Additionally report first evidence availability and first actual threshold
movement independently; an available and feasible candidate can still leave
the clipped threshold unchanged.

## Interpretation boundary

The eight-defect threshold is not a precision claim. A descriptive feasible
set is not certified online risk control. Query counts and anchor counts are
not interchangeable: current-memory eligibility, source exclusion, duplicate
identity rules and memory revisions can change the number of usable anchors.
The diagnostic record must permit a reviewer to reconstruct why B7 was in a
state, whether a threshold could satisfy both descriptive limits, and whether
any applied threshold move remained feasible after movement clipping.

## Stage 0 metadata-only query-budget sensitivity

The frozen metadata preflight contains 15,295 development-stream images. The
3% half-up Q-A-TS budgets are 114 Audiojack, 116 Regulator, 115 Switch, and
114 Rolled Strip Base queries, for 459 total. The tape selects one queried
view per official sample group (459 of 3,059 groups, about 15.0%); these are
image-query counts and must not be confused with the percentage of groups. No
test labels were read to compute these budgets.

The following are hypothetical sensitivity calculations. Within each
scenario, each queried label is modeled as an independent Bernoulli outcome
with the stated common defect fraction, independently across classes. This is
an explicit mathematical assumption, not a model learned from Real-IAD labels
and not an estimate of actual defect prevalence.

| Hypothetical queried defect fraction | Expected classes reaching 8 defects | Probability all 4 reach 8 | Expected classes reaching 14 | Probability all 4 reach 14 | Per-class probability of fewer than 8 defects |
|---:|---:|---:|---:|---:|---:|
| 5% | 0.866 | 0.0022 | 0.007 | <0.00000000002 | 77.5%–78.9% |
| 10% | 3.589 | 0.6481 | 1.025 | 0.0043 | 9.6%–10.7% |
| 20% | 4.000 | 0.9999 | 3.959 | 0.9599 | 0.0022%–0.0030% |
| 30% | 4.000 | 1.0000 | 4.000 | 1.0000 | <0.00000003% |

Under the 5% hypothetical, each class is more likely than not to stay below
the eight-defect evidence minimum and remain in `NORMAL_ONLY`/HOLD. At 10%,
the activation minimum is likely in most classes but reaching fourteen
zero-error defects in all four is unlikely under this model. The Stage 1
result must use actual queried labels and preserve the registered query tape;
it must not tune the query rate or infer any class outcome from this
sensitivity table.
