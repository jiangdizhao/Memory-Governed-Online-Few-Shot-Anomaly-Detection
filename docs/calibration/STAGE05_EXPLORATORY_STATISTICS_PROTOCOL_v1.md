# Stage 0.5 — Real-IAD Exploratory Statistical Audit Protocol v1

**Status:** authorized exploratory development-data audit; not Stage 1 validation and not Stage 1 execution approval. The Stage 1/2 hold in `STAGE0_RESEARCH_DECISION_2026-10-29.md` remains in force.

## Authorization and information boundary

The user explicitly authorized test-label-based exploratory statistics for only the four development categories `audiojack`, `regulator`, `switch`, and `rolled_strip_base`. This does not authorize access to Mint, PCB, or any other confirmation or development category. Segmentation masks are out of scope and must not be opened.

Once full development-stream labels have been inspected, those outcomes are exposed exploratory information. They can no longer be described as unseen for future method development. Any later design or method decision informed by this audit must disclose that exposure. Existing Stage 0 results, Gate decisions, and approved Stage 1 design remain unchanged. Stage 1 remains on HOLD.

This audit may not optimize or revise an algorithm, query rule, threshold, support selection rule, memory policy, or hyperparameter. It must not choose the best query seed, alter the 3% main tape, tune on any outcome, or select an algorithm based on the findings. It changes no controller source or controller parameter.

## Preregistered data and ordering

Only the official JSON files for the four authorized classes are used. The test stream follows the JSON record order consumed by the existing calibration preflight. The expected stream lengths are 3,815 Audiojack, 3,855 Regulator, 3,820 Switch, and 3,805 Rolled Strip Base images. The pinned Stage 0 metadata preflight supplies the class JSON SHA-256 and seed-37 Q-A-TS digest. A mismatch fails closed.

The first process parses the test JSON arrays with `anomaly_class` and `mask_path` discarded before any records are returned. It selects support and C64 identities from training-normal metadata, constructs every 3%, 5%, and 7% tape for all 200 preregistered seeds, checks nesting, quotas, group limits, and seed-37 identity, then writes and hashes the complete freeze manifest. No test-label value enters selection or tape construction.

The fixed query seeds are `[37] + list(range(1000,1199))`; seed 37 is the primary registered schedule. Query budget is `floor(rate*N + 0.5)`. Q-A-TS uses the registered 16 temporal bins, official sample key grouping, one queried camera view per official group, and the existing label-blind hash selection. The 3% seed-37 tape must match the existing Q-A-TS-v1 digest and positions exactly. The separately named `Q-A-TS-BudgetExtension-v1` adds deterministic group-limited choices while preserving parent positions and quotas. Failure to satisfy nested per-bin quotas blocks the extensions; the base 3% tape is never revised.

Only after the entire tape archive and support/C64 selections are checksummed does a separate evaluator phase join test labels for query-composition statistics. It records the full-stream class composition, queried normal/defect counts, camera and group-view composition, temporal bins, first 400 events, first queried defect positions for 1/8/14 defects, longest no-defect image span, and sliding 256-image query-defect counts. Across the 200 fixed seeds, summaries are finite-seed sensitivity results, not guarantees or probabilities over all possible tapes. No independence is assumed across camera views or official sample groups.

## Frozen support and native static scoring

The audit evaluates support seeds `(0,1,2,3,4)`. Every seed has exactly four distinct training-normal official sample groups and uses the historical seed-plus-891 camera-view choice. The same historical 64 additional protected C64 normal groups/views are selected with seed 190021 and the historical per-group view seeds. C64 remains additional to the four deployed supports, disjoint from each support seed, and read-only. Seed 0 must match the historical preflight exactly. Cross-seed support overlap is permitted.

For each class, the five strict-four-support banks are built from the native P0/v22/VisionAD path and fixed scoring configuration. Native dynamic Greedy memory is empty, set to K=256, and disabled throughout. There are no online writes, queries, threshold updates, or feedback. Each C64 and test image is feature-encoded once and scored against all five banks. The original DINOv2 checkpoint/source and configuration fingerprints, fixed-bank hashes, support/C64 identity digests, score-tape hashes, and bounded direct-native versus P0 scoring parity are recorded. No generic nearest-neighbor substitute or label-driven feature/cache choice is allowed.

The complete per-class/per-seed C64 and test score arrays are written to isolated local output, hashed, and frozen before the evaluator opens labels for score metrics. Full score tapes and feature tensors remain local and are not committed. A feature cache may be reused only if source, configuration, and identity hashes prove exact equivalence; this implementation uses one in-process feature extraction per image across the five banks and does not persist a full feature cache.

## Registered metrics and exact conventions

Query-composition counts are image-label counts. An official group identity is not interpreted as a proven physical manufactured-item identity. Group-level reporting is aggregated by group view multiplicity; it does not infer independence between views.

For score metrics, physical defect is positive and the frozen binary decision is `score >= theta0`. `theta0` is the D2 higher empirical 0.90 quantile of the 64 C64 normal scores: zero-based rank `ceil(0.90*(64-1)) = 57`. Confusion counts, BA, FPR, FNR, defect recall, AUROC, and average precision are reported with class prevalence. Undefined denominators are NULL.

The retrospective threshold result is labeled `IN_SAMPLE_RETROSPECTIVE_STATIC_THRESHOLD_ORACLE`. It searches the exact unique score thresholds plus the all-defect and none-defect boundaries under `score >= threshold`. Balanced accuracy is compared by exact rational arithmetic. If thresholds tie on BA, choose minimum absolute movement from theta0, then the higher threshold. Report oracle confusion metrics and movement from theta0. This is an optimistic in-sample diagnostic, not an executable online rule, an unbiased estimate, an upper bound for adaptive scoring, or a controller input.

Query-feedback windows use every stride-one 256-image window in default JSON order. For each start position, report the seed-37 queried-defect count and min/5th/25th/median/75th/95th/max/mean/sample-SD across the 200 fixed schedules. The output contains aggregated counts, not image-level labels. Static-score temporal evaluation uses the same windows; a window lacking either class has undefined BA, and class-specific rates with missing denominators are NULL. Report every score window's confusion counts/rates, the worst defined BA window, undefined-window counts, and the longest consecutive false-negative run.

`FROZEN_STATIC_NEAR_CEILING` is an exploratory screen only: frozen BA ≥ 0.95 and oracle headroom ≤ 0.01. It is neither a significance criterion nor a calibrated probability that the task is solved. High static performance does not establish that online calibration is unnecessary when dynamic memory changes score state.

## B7 interpretation and scientific limits

The frozen SCDAC-G v0.1 values remain `min_normals=16`, `min_defects=8`, descriptive `delta=0.05`, and FPR/FNR targets 0.20/0.20. Eight queried defects can satisfy the nominal two-sided minimum count. Eight zero-error defects still have a one-sided 95% Clopper–Pearson upper endpoint above 0.20. At least 14 zero-error defect anchors are needed for this specific fixed-time descriptive endpoint to be ≤0.20; any observed errors may require more. Query counts alone cannot show that a feasible risk-constrained threshold exists because that also depends on current-memory scores and class overlap. None of these empirical calculations certify online population risk under adaptive selection, temporal dependence, repeated rescoring, or memory-dependent scores.

## Outputs, reproducibility, and stop conditions

Execution phases are isolated and resumable under `outputs/calibration/stage05_exploratory_20261029/`. Query tapes, support identities, score tapes, and manifests are SHA-256 protected. A phase never overwrites a completed artifact; a partial run retains `INCOMPLETE` and exact blockers. Only compact aggregate reports, manifests with hashes, exposure records, and checksums are published under `docs/calibration/audit/stage05_exploratory_v1/`.

The audit ends after publication. No Stage 1/2 experiment, confirmation outcome access, method selection, controller modification, or Stage 1 authorization follows from this protocol.
