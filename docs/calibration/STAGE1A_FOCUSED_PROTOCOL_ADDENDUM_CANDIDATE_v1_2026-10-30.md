# Stage 1-A focused protocol addendum — candidate

**Status:** `CANDIDATE; RESEARCHER_APPROVAL_REQUIRED; EXECUTION_NOT_AUTHORIZED`

This proposal does not replace or amend `STAGE1_EVALUATION_PROTOCOL_APPROVED_DESIGN_v3.md`. Stage 0 decision `STAGE1_EXECUTION_GO` is still required. No Stage 1-A evaluation was run in preparing this addendum.

## 1. Proposed scope

If separately approved, compare the following on the four development classes (`audiojack`, `regulator`, `switch`, `rolled_strip_base`), five support seeds `(0,1,2,3,4)`, and the locked Q-A-TS-v1 3% tape (seed 37):

| Proposed row | Role | Scientific identity and reporting rule |
|---|---|---|
| B0 | Frozen-Theta control | Fixed D2 `theta0`; binary decisions on the shared dynamic native score stream. |
| B1 | Sparse-feedback calibration | Rolling q90 of at most the last 256 queried verified-normal stale scores; 16 minimum; requires causal queried labels. |
| B2 | Historical v22 sidecar | Adapted `GuardedDualAnchorThresholdController`; C64 source-native initialization, queried labels only, source normal-only fallback disabled. Do not claim historical full-loop parity. |
| B3 | State-Consistent Quantile | Actual registered B3; protected C64 current-memory q90 and 0.01 movement clip. |
| B4 | Adapted Dual-ACI | Queried error feedback plus current-memory queried-anchor bounds; no ACI guarantee transfer. |
| B5a | SPOT source-native | Paper-equation implementation; 1,000 startup observations remain unavailable. Report source-native coverage and undefined full-stream BA. |
| B5a-Frozen | SPOT + Frozen-Theta | Separate deployment hybrid. Record each original-method and fallback decision; never call it source-native SPOT. |
| B5b | dSPOT source-native | Paper-equation implementation; 50 local observations plus 1,000 residual warm-up observations remain unavailable. Report source-native coverage and undefined full-stream BA. |
| B5b-Frozen | dSPOT + Frozen-Theta | Separate deployment hybrid with explicit original/fallback decision fractions. |
| B7 | SCDAC-G v0.1 | Preserve all registered count gates, descriptive CP bounds, HOLD/fallback behavior, and 0.01 movement cap. No hyperparameter changes. |
| Stage075BOffset | B3+Offset candidate | Distinct raw-median-offset adaptation from Stage 0.75-B, versioned separately. Its synthetic result is not evidence of Real-IAD benefit or approval. |
| B6 | Sun-CS applicability-only | Retain as an unsupervised score-quantile reference. Candidate scope omits a new full-stream run and reports its registered target, endpoint/applicability limitations, and existing provenance only. This is a proposed departure from approved v3 and requires explicit approval. |
| B8 | Historical negative control only | Keep existing Stage 0.75-B synthetic results as historical context. No new B8 variant or Stage 1-A action. |

The focused core sparse-feedback set is B0/B1/B2/B3/B4/B7 plus the separately identified Stage075BOffset candidate. B5a/B5b are unsupervised score-history references, each with a separate operational fallback row. B6 remains registered as an applicability reference. B8 is not a new comparator.

## 2. Locked inputs and causal order

Preserve the approved v3 candidate's four classes, five support seeds, four deployed normal supports plus the additional protected C64=64 normals, frozen Native Scorer, shared controller-independent K=256 Greedy bank, Q-A-TS 3% seed 37, and common horizon beginning at zero-based `t=1050`. Confirm support/C64/train/test disjointness and exact Seed-0 compatibility in the calibration namespace before any execution.

Use one shared score and memory trajectory per class × support seed × query profile. The order stays: score with pre-event memory; predict; apply the precommitted label-blind query tape; disclose only a queried verified label; run the independent shared writer only for a queried verified normal; refresh C64 and eligible query-anchor evidence after memory/evidence revision; update controllers for the next event. No controller may affect score, query, or writer decisions. Queried normals use all-layer source-group leave-out; defects are never written. One query anchor maximum per official sample group.

No labels, masks, future images, unqueried labels, hidden class counts, or confirmation-class outcomes may enter a controller. Keep labels and feature caches in private evaluator/checkpoint storage.

## 3. Required reporting

For every approved binary-output row, report paired per-class × seed full-stream TP/TN/FP/FN, BA/FPR/FNR with null denominators preserved, defect recall, and worst defined non-overlapping 256-image-window FPR/FNR/BA. Also report threshold movement and steps, query images and official groups/fractions, memory writes, C64 and query-anchor rescoring counts, evidence revisions, runtime/peak memory, and method-specific state counts.

For every row, report full-stream and common-horizon overall/normal/defect coverage, abstention, startup-unavailable, fallback counts and rates, plus selective metrics with denominators. Source-native B5a/B5b full-stream BA remains undefined because startup outputs are missing; do not impute fallback predictions into those rows. Report fallback hybrids separately with source-native/fallback decision fractions. Use the fixed common horizon `t=1050` without moving it after results.

All threshold-only rows sharing the same score and memory path must have identical score-based AUROC/AUPRC. A difference is a protocol failure. Report B7 `NORMAL_ONLY`/`BOTH`/`CONFLICT`, HOLD reasons, two-sided evidence separately from feasible constraints, count/first-event timelines, and descriptive CP labels. Do not interpret CP endpoints as certified online risk.

For B6, report the source target `1[S_t > Q_Ft(p)]` separately from the physical-defect target, registered conservative extended-real primary separately from empirical-extrema sensitivity, native abstention separately from operational Frozen-Theta fallback, source assumptions, author-code parity status, and lack of theorem transfer. Preserve the preregistered applicability rule: show actual overall/class-conditional coverage and undefined metrics where appropriate; do not introduce a numerical coverage cutoff after outcomes. A fallback hybrid is not evidence that Sun-CS itself is competitive.

## 4. B7 activation and feasibility interpretation

Keep B7 v0.1 unchanged: 16 queried normals, 8 queried defects, delta 0.05, descriptive support FPR/FNR 0.20, and 0.01 maximum threshold step. Eight defects allow nominal two-sided processing, not a guarantee that a feasible threshold exists. With zero errors, at least 14 queried defects are needed for the one-sided descriptive FNR upper endpoint to be at most 0.20; errors may require more. Sparse selection, dependence, repeated rescoring, and memory changes preclude a certified online population-risk interpretation.

If B7 does not reach BOTH or never finds a feasible threshold, report that as a legitimate result. Do not lower counts, change delta, enlarge the query budget, use unqueried labels, tune SCQ, or alter HOLD/fallback behavior.

## 5. Preflight blockers and approval items

The reviewed Stage 0.75-A artifacts support the common event score/memory path and C64 transport, but not full B4/B7/Stage075BOffset anchor evidence. The full event current-memory query-anchor score stream must be generated with a version-addressed native scorer/feature path after approval. A private causal query-label linkage must also be generated for queried-feedback methods. The original Stage 0.75-A G2 remains BLOCKED; the independent package-integrity certificate is PASS pending researcher review and does not prove the exact executed runner bytes.

Require independent approval of the focused method registry (including Stage075BOffset and B6 scope), the private label/evidence pipeline, exact feature cache and compute/storage budget, original G2 treatment, and a separate `STAGE1_EXECUTION_GO`. This addendum itself is not that approval.
