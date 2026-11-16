# N-shadow + Independent B3/C64 Supplementary Experiment

## Scope and status

This is a post-hoc exploratory supplementary analysis over the already sealed final-benchmark trajectories: Real-IAD 30 categories × P0/P1/P2, and VisA 12 categories × P0. It does not alter the principal four arms, their query decisions, existing scores, thresholds, feedback state, or historical outputs. The primary comparison is `BER(N-shadow+B3) − BER(F-B)`; positive values favor F-B. FPR and FNR differences are reported separately. F-M is descriptive because it has a different query and feedback history.

## N-shadow scoring and legal feedback

N-shadow retains the original complete state: F-B shared Greedy normal memory plus an independent normal-only native C1 correction codebook. It receives the same already executed F-B query opportunities and no independent queries. It accepts only previously purchased F-B feedback. A verified defect is ignored by N-shadow C1; a verified normal is applied using the original R2 `_capture_update` and the original F-B pre-feedback native C0 score, Top-32 patches, event position, source group and C0 threshold. F-B Correction and the original `common_c0_greedy_writer_action` are applied in their original order. No unqueried TEST outcome or mask enters Stage A.

## Independent B3 and protected C64

The new controller is a separate `StateConsistentQuantile` instance. It uses only the existing 64 protected TRAIN-normal C64 images for the exact category and Support condition. Initial and refreshed C64 score vectors use the existing native R1 scorer, R7 `_score_c64_base` cache, and R9 `_score_c64_for_codebook` adapter. R7 `_initialize_b3` and `_refresh_b3` govern evidence versions, state identity, update order and the 0.01 maximum movement per state transition. The quantile is q=0.90 with method `higher`, with the existing minimum of 16 protected normal anchors. No TEST outcome tunes this controller.

The threshold is refreshed after each actual N-shadow scoring-state change. A normal C1 change and a subsequent shared Greedy change can therefore cause two distinct refreshes at one feedback boundary. The changed state and threshold are used only by the next atomic unit. An F-B defect-side correction change by itself is excluded from the N-shadow state digest and causes no N-shadow B3 refresh.

## Query-event-only exact reconstruction

For each original sealed source group or VisA atomic unit, the supplementary runner:

1. Verifies original event identities, order, group/unit seals, completion seal, score-row hash and state digest/version.
2. Reuses each original sealed raw N-shadow score for the prediction record.
3. Re-encodes only an event where F-B had already executed a purchased query, then recomputes native F-B and N-shadow scores for parity against the original journal (absolute tolerance 1e-5).
4. Writes and fsyncs the label-blind supplementary prediction envelope for the entire atomic unit.
5. Only after that durable write reads the matching already purchased feedback records; only the F-B label is applied to the replayed shadow/F-B state.
6. Checks reconstructed N-shadow scoring-state digest and version at every subsequent image event, and updates C64/B3 only when that digest changes.

This avoids a redundant four-arm inference replay while preserving the exact legal update history. If any source state, event, query receipt, native query score, or feedback order fails parity, the affected trajectory remains INCOMPLETE and Stage B does not start.

## Stage B and statistics

Official image-level outcomes are projected only after every one of the 102 supplementary Stage A seals passes. Real-IAD uses registered JSON image-level `anomaly_class` relative to official `normal_class`; VisA uses the registered official image-level split table. Pixel masks are never opened. Existing purchased F-B feedback is checked against official image outcomes after sealing.

Metrics include AUROC, AP, BER, FPR, FNR and balanced accuracy for N-shadow+B3, with original F-B and F-M comparisons, query composition, first queried defect position, threshold history and temporal segments. Undefined denominators remain undefined. Real-IAD P0/P1/P2 are averaged within category before equal weighting across 30 categories. VisA is aggregated separately across 12 categories. Paired percentile bootstrap intervals use 10,000 category resamples with fixed, recorded seeds. Real-IAD and VisA are not pooled.

## Claims and limitations

This analysis reports realized post-hoc operating behavior under the existing F-B query tape. It does not identify counterfactual value of unqueried labels, establish that defect-side correction alone caused any difference, validate an independent threshold on a new held-out set, or establish VisA physical independence from Real-IAD. No tuning or method choice follows the Stage B outcomes.
