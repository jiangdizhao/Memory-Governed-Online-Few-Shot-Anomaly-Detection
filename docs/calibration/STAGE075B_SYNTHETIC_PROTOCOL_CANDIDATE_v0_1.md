# Stage 0.75-B preparation — B8 EC-MSC synthetic mechanism protocol candidate (v0.1)

**Status: CANDIDATE / IMPLEMENTATION NOT AUTHORIZED BY THIS DOCUMENT.**  
**Recorded:** 2026-10-29. **Parent:** 0.75-A exploration, commit `5e855c5be402f3dfd70a0294f894a5846b4f235b`.  
**Companion:** [B8 v0.2 mathematical design](STAGE075_B8_ECMSC_RESEARCH_DESIGN_v0_2.md). Preserves [B8 v0.1 historical proposal](STAGE075_B8_MTEAC_RESEARCH_DESIGN_v0_1.md).

## 1. Objective and nonobjective

Stage 0.75-B, **if separately approved by the researcher**, will test whether Memory-State Normalization, defect-informed decision state, and conservative baseline-aware updates are *mechanistically meaningful, causal, numerically stable and potentially useful beyond B3*, **before** full matched Real-IAD comparisons. Goal is a principled, reproducible implementable B8 **candidate**, not a claimed SOTA detector or a BA result on exposed classes.

Not allowed without a separate authorization: executing 0.75-B now, changing the Stage 1 method registry, modifying B3/B7/native Greedy/protected v22 code, running Real-IAD BA tuning or hyperparameter sweeps, inspecting Mint/PCB/masks, rewriting historical results, or moving from synthetic 0.75-B to 0.75-C/Stage 1 without a new decision.

Use offline synthetic score/memory trajectories under explicit generative assumptions. Reuse source-faithful *real B3 controller* as a comparator in synthetic driver. Store all diagnostic outputs in a new separate `stage075b` namespace if implementation is later authorized. No multiple controller-specific native writers: each controller sees identical synthetic score/memory/C64 and query evidence events.

## 2. Scientific contract / non-negotiable properties

- The **primary eventual benchmark** is class-macro paired BA(B8) − BA(B3), not best time-window or B8 versus weaker frozen theta alone. Some normal-only stretches may tie or lose to B3 if net macro improvement is real; no result is promised.
- In synthetic event order: score/predict first, query decision already frozen, label reveal only if queried, update shared normal-only memory, rescore legal references with exact content-version linkage, then update threshold for *following* events. Same group identity cannot contribute multiple independent labels.
- B3 shadow uses exact code, `Δ=0.01`, timing, scores and C64; *unclipped direct q90 from 0.75-A is diagnostic only*.
- Every B8 action must have B3-relative proposal/actual position, current-memory source hashes, known label counts, identity/LSO validity, update trigger and constraint projections.
- Input gate cannot read full synthetic future labels even though the synthetic evaluator knows them, nor derive acceptance from hidden true BA.
- Distinguish **reference-normal C64 empirical distribution** from *actual stream-normal* FPR; observe any bias in evaluator, but never use it in the controller. Defect rescoring is not a new label.
- No unqualified "LCB", "conformal coverage", "valid risk bound" or guaranteed no-worse-than-B3 label for a heuristic uncertainty score. If a theorem is proposed, state its assumptions/proof separately from synthetic evidence.
- All compared threshold-only methods run the same score trajectory; AUROC/AUPRC identity is a hard invariant.

## 3. Mathematical decisions to freeze in a future versioned implementation contract

These are **open, not quietly instantiated defaults**:
1. Finite-64 C64 CDF and percentile-to-raw-score inversion; equality condition score≥theta, order-statistic conventions, tie/saturation/zero-IQR behavior and float precision.
2. The B3 shadow synchronization mechanism and exact no-deviation fallback test; safe return-to-B3 after previous deviation without violating B8's own step limit.
3. The simple B3+raw defect-offset ablation; standardized-IQR residual and percentile-ranked candidate as equal-information baselines.
4. Which legitimate normal evidence estimate participates in J-gain: C64 alone, queried-normal LSO, or separate terms with mismatch diagnostic; never interpret C64 q90 as production FPR 10% by fiat.
5. Defect-anchor weights, age and effective-source-count handling; stale score versus stale concept; unbiasedness assumptions/limits for label-blind query schedules.
6. Finite candidates, tie-break and movement limits; no backprop and no extra pseudo-labels.
7. Conservative uncertainty penalty and confidence-based shrinkage/gating; how candidate-search optimism is handled. Do not conflate empirical penalty with a formal lower bound.
8. Explicit conflict/fail-closed response, controller audit states and checkpoints, and no-threshold-update on unchanged memory/evidence.
9. Lightweight computational path: C64 caching, rerank only when content changes, legal defect/normal anchor rescores and bounded runtime.
10. Experimental design for *hyperparameter fixing without retrospective Real-IAD label fitting*, or explicit documentation if eventually used as exploratory development tuning.

An implementation agent should first produce `STAGE075B_IMPLEMENTATION_CONTRACT_CANDIDATE_v1.md` and demonstrate every open item is resolved/frozen **for synthetic experiments only**, or mark BLOCKED if unresolved; do not silently choose coefficients for favorable development BA. No automatic real-data promotion.

## 4. Synthetic falsification matrix (predeclare inputs, observed outputs, failure diagnostics)

| ID | Predeclared synthetic regime | Expected diagnostic / possible falsification |
|---|---|---|
| S0 | B8 unactivated, B3 valid, no B8 divergence | Exact prediction/threshold agreement with *real* B3 where feasible; record event timing. |
| S1 | Shared positive shift, positive affine, strictly increasing nonlinear transform of all C64/evaluation scores | Finite CDF rank invariance subject to ties; distinguish from additive-only median transport. |
| S2 | Sparse C64 upper-tail modifications with `median(d)=0`, q90 changed | Median-only transport can fail; percentile/direct q90 and shape-sensitive components are checked. |
| S3 | C64 and defect scores change by different transformations | Normal rank invariance does not imply defect-preserving or BA-preserving control; expose failure. |
| S4 | Long delayed/no queried defect feedback with legal queried-normal Memory writes, followed by burst | B8 stays operational near B3, no future label leakage; test gradual adaptation after valid defect evidence. |
| S5 | Defect-only / no verified-normal Memory admission | No imaginary C64 transport, no spurious threshold oscillation; labels may alter defect evidence after legal query. |
| S6 | 1–7 defect labels, 8+, 14+, severe group imbalance and duplicated rescoring | Effective evidence does not equal repeated score count; absence of a hard 8 threshold does not confer risk certainty. |
| S7 | Memory-score staleness vs concept-age drift with immutable labels | Refresh current scores; age/reliability effects acknowledged separately. |
| S8 | Protected C64 distribution differs from deployment normal distribution | Reference-only empirical FPR differs from real FPR; demonstrate conditional failure/penalty behavior. |
| S9 | B8 departed from B3; evidence disappears; move limit binds | Baseline rejoin moves **gradually** as allowed or uses a separately documented justified exception. |
| S10 | Identical scores, ties at candidate threshold, C64 extrema, zero IQR, missing class or division-by-zero | Deterministic tie/inverse behavior, finite safe outputs, unavailable conditional risk marked undefined. |
| S11 | Same-group cross-view repetition, prior admitted normal and LSO, defect label repetition | No illicit self-scoring or treating multiple views/same label rescoring as independent evidence. |
| S12 | Multiple threshold-only controllers on a common frozen dynamic score tape | Exact AUROC/AUPRC equality; differences in BA arise only from thresholds. |
| S13 | Snapshot/checkpoint/resume from every regime transition | Exact state, threshold, evidence ledger, audit and digest reproducibility. |
| S14 | All-class oracle says no improvement over B3 or real defect conditional distribution changes invisibly to normals | B8 should acknowledge no identifiable gain; no algorithm can guarantee FNR/BA in this adversarial case. |

For each test: include the generative signal, known unqueried evaluator-only labels, sample-group identity, legal query schedule, expected decision trace, negative controls and deterministic assertions. Do not optimize the controller on those labels. Retain unfavorable results.

## 5. Comparison and selection discipline

Proposed synthetic arms: actual B3; B7 as applicable; B3+simple defect offset; B8-M paired median; B8-S normalized residual; B8-P percentile-only; B8-Full. Include B0 for diagnostic interpretation where useful. All use the same legal score and query events. No artificial advantage to B8 through a larger reference bank, more defect labels, a different threshold movement budget or an altered writer.

Avoid declaring the most promising arm using *only* scenarios constructed to favor it. Mix identity/no-shift, score-only shift, normal-tail drift, defect-only shift, sparse-label bursts, score overlap, and reference-mismatch negative controls. Report class-conditional errors, BA and uncertainty where both classes exist; one-class evaluation windows have **undefined BA/AUROC**. Report runtime and evidence availability even when B8 loses.

**Go/No-Go recommendation after synthetic 0.75-B** requires (a) correct causal implementation and safe failure behavior, (b) incremental mechanism over trivial B3+offset in at least some neutral scenarios, and (c) transparent failure boundaries. Passing these **does not mean** B8 already beats B3 on Real-IAD or authorizes 0.75-C.

## 6. Proposed later artifacts, tests and review gates (not executed now)

If approved, produce exact source/config fingerprints, a bounded synthetic experiment driver, a versioned frozen synthetic implementation spec, compact synthetic tests, numerical invariance proofs/counterexamples, no-leakage test traces, checkpoint parity, and a candidate research decision report. Proposed paths are under `experiments/calibration/`, `tests/calibration/`, and `docs/calibration/audit/stage075b_synthetic_v1/`, but nothing is created by this document alone.

Publish only after synthetic tests and an independent integrity check. **Explicit new researcher approval** is required for the future Codex implementation prompt, for any real development-set B8 comparison (0.75-C), and separately for Stage 1/Stage 2 or Mint/PCB. Stage 1/2 remain **HOLD**.

## 7. Record integrity and lineage

Earlier B8 v0.1 and Stage 0.75-A data/results remain unchanged. v0.2 is a candidate based partly on **exposed development labels**; it is not an independent pre-experiment discovery or a verified theoretical contribution. The most critical falsification is simple: B8 may beat B3 by using more defect feedback, yet not beat B3+raw-offset; in that case the elaborate state-coordinate hypothesis lacks independent support and must be simplified/rejected rather than relabeled novel.
