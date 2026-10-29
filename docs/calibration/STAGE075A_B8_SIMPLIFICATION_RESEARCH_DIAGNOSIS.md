# Research Diagnosis: Can Simple Local Threshold Feedback Address the Post-B3 Mismatch?

This is an offline audit of the four exposed development classes. It does not implement or tune B8. Retrospective thresholds use the same evaluation cohort labels and are not online performance estimates.

## Findings

| Question | Evidence-based answer |
|---|---|
| 1. Does actual B3 materially lag current q90? | On full-stream equal-class macro BA, B3 is 0.73615 and read-only q90 is 0.73712; B3 minus q90 is -0.00097. Mean absolute lag per category is below 0.0004. Rare excursions reach 0.0491 and lower-bound updates can persist, but aggregate BA impact is small in these trajectories. |
| 2. How much residual headroom exists beyond q90? | It is category- and cohort-dependent. For all-remaining fixed snapshots, mean retrospective `H_beyond_q90` is audiojack 0.0188, regulator 0.0650, switch 0.0293, and rolled_strip_base 0.0202 BA. `H_q90` remains unclipped and may be negative. |
| 3. Which classes show the largest local diagnostic opportunity? | Regulator has the largest average all-remaining `H_beyond_q90` (0.0650) and local-grid gain (0.0431). Switch is next by local-grid gain (0.0236). These are in-sample diagnostics, not deployable gains. |
| 4. Can one-step local offsets capture much of the Oracle gap? | Mean within-step capture across defined all-remaining rows is regulator 0.257, switch 0.663, rolled_strip_base 0.538, audiojack 0.549. Wider grid capture is also shown in the table below. It is still label-selected on the same cohort. |
| 5. Are beneficial local actions predominantly lower-threshold actions? | Across 120 all-remaining class/snapshot/seed rows, the retrospective best grid action is `LOWER` 88 times, `KEEP` 12 times, and `RAISE` 20 times. This frequency is not an online policy recommendation. |
| 6. Is Regulator materially different from Rolled Strip Base? | Yes descriptively: Regulator’s all-remaining mean `H_beyond_q90` is 0.0650 versus 0.0202 for Rolled Strip Base; their Greedy AUROC means are 0.779 and 0.968. Rolled Strip Base has much stronger ranking overall, though late fixed cohorts can still have large threshold headroom. |
| 7. How much is retrospective only? | Every Oracle and best local action uses future evaluator labels from the same fixed cohort. No train/validation split or independent future test is created here. All reported Oracle gains are in-sample; no amount is certified online. |
| 8. Are legal queried labels and current-state anchor scores sufficient to identify useful actions? | Queried label groups accumulate under the frozen 3% tape; at `t=400` counts are audiojack: 0 defect groups / 14 normal groups; regulator: 2 defect groups / 11 normal groups; switch: 9 defect groups / 5 normal groups; rolled_strip_base: 11 defect groups / 3 normal groups. Audiojack has no queried defect group before `t=1600` (first queried defect at position 1305). However, no query-anchor score under each later snapshot memory, no source-excluded rescoring, and no versioned query-anchor lineage exist. Therefore online local gain is not identifiable from the existing artifacts. |
| 9. Is additional percentile-transport complexity supported? | No evidence supports adding percentile transport to solve Type 1 lag: actual B3 tracks current q90 closely in aggregate. Type 2 operating-point mismatch remains in some classes, but its sparse-feedback direction is not currently identifiable from legal current-state anchors. Further transport complexity is therefore not justified by this audit. |
| 10. What is the narrowest scientifically justified Recovery hypothesis? | A prospective **measurement hypothesis**, not an implemented method: record version-addressed, source-excluded scores for already queried anchors under each frozen current memory and determine whether their sparse labels identify a local direction around actual B3. Only after a separate preregistered study establishes identifiability should a single scalar correction be compared. No offset, update rule, or B8 hyperparameter is selected here. |

## Snapshot summary

`H` and capture values are means across defined seed/cohort rows for all-remaining snapshots; next256 values are shown separately. Best actions count all six snapshots per seed and are diagnostic only.

| Class | H_total | H_q90 | H_beyond_q90 | Within-step capture | Full-grid capture | Local-grid BA gain | Next256 H_beyond_q90 | Next256 local BA gain | All-remaining best actions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| audiojack | 0.0188 | 0.0000 | 0.0188 | 0.549 | 0.641 | 0.0144 | 0.0287 | 0.0210 | KEEP:4, LOWER:23, RAISE:3 |
| regulator | 0.0652 | 0.0002 | 0.0650 | 0.257 | 0.660 | 0.0431 | 0.0591 | 0.0381 | LOWER:23, RAISE:7 |
| switch | 0.0299 | 0.0006 | 0.0293 | 0.663 | 0.793 | 0.0236 | 0.0315 | 0.0247 | LOWER:27, RAISE:3 |
| rolled_strip_base | 0.0200 | -0.0001 | 0.0202 | 0.538 | 0.589 | 0.0154 | 0.0267 | 0.0191 | KEEP:8, LOWER:15, RAISE:7 |

## Interpretation

The three mechanisms remain distinct. **Type 1, threshold-state lag:** small on average but with rare clipped episodes. **Type 2, decision-point mismatch:** clear in Regulator and present to varying degrees elsewhere because current-normal q90 is not the retrospective BA optimum. **Type 3, score-ranking limitation:** more relevant in Audiojack and Regulator than in Rolled Strip Base, as shown by their lower shared-score AUROC; a threshold alone cannot remove overlap. Snapshot oracles can overfit and class/time cohorts are dependent.

The query-anchor rescoring gap is decisive for feasibility: the offline grid can reveal where a scalar threshold would help on the observed cohort, but the recorded stream cannot tell an online controller which direction to move at those later memory states. These findings justify collecting the missing legal evidence before investigating an adaptive Recovery rule. They do not establish B8 feasibility or improvement.
