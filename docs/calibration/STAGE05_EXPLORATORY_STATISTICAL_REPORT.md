# Stage 0.5 exploratory statistical report

**Status: COMPLETE as exploratory development-data reconnaissance. This is not Stage 1 validation or execution approval.** The four authorized development test-label streams were inspected; they are exposed information and must not be described as unseen in later method development. Stage 1 and Stage 2 remain on HOLD. No controller or controller parameter was changed, and no confirmation-class outcome or segmentation mask was accessed.

## Scope and reproducibility

The audit covers `audiojack`, `regulator`, `switch`, and `rolled_strip_base` only, in the official category JSON order used by the calibration preflight. Expected stream lengths matched the frozen metadata. Query tapes were frozen before test-label joins; score tapes were frozen and checksummed before the independent score evaluator joined test labels. The 3% seed-37 tape matched the Stage 0 Q-A-TS-v1 positions and digest for every class. All three rates were built as nested schedules for 200 fixed seeds (`37` and `1000` through `1198`), preserving 16 temporal bins and at most one queried view per official group.

The audit used support seeds 0-4, four deployed training-normal supports per seed, and the same additional protected 64 C64 normal images selected with historical seed 190021. Seed 0 matched the historical preflight exactly. Native v22/P0/VisionAD score and memory parity path was retained; dynamic Greedy memory was empty, K=256 and disabled. This is a static M0 diagnostic, with no writes, threshold adaptation, controller queries, or feedback adaptation.

Local resumable artifacts are under `outputs/calibration/stage05_exploratory_20261029/`. The compact, aggregate-only public package is `audit/stage05_exploratory_v1/`. It includes SHA-256 fingerprints for all 2,400 class/rate/seed tapes without selected positions or identities, and excludes full per-image label and score tapes, images, masks, weights, and feature caches. See the package provenance and checksums for source, model, configuration, identity, and tape fingerprints.

## Full-stream composition

| Class | Images | Normal | Defect | Defect fraction | Official groups | Images per group |
|---|---:|---:|---:|---:|---:|---:|
| audiojack | 3,815 | 2,519 | 1,296 | 0.3397 | 763 | 5 |
| regulator | 3,855 | 2,775 | 1,080 | 0.2802 | 771 | 5 |
| switch | 3,820 | 1,826 | 1,994 | 0.5220 | 764 | 5 |
| rolled_strip_base | 3,805 | 1,280 | 2,525 | 0.6636 | 761 | 5 |

All groups in these streams had five views. A 3% image query budget therefore covers roughly 15% of the official groups when one view is selected per group. Group identity is not evidence that views or groups are independent physical manufactured items.

## Seed-37 query composition and arrival

The image-query budget uses `floor(rate * N + 0.5)`. The combined query totals across the four classes are 459 at 3%, 765 at 5%, and 1,070 at 7%. The 5% and 7% schedules are deterministic nested extensions; they are analysis conditions and do not revise the frozen Stage 1 design.

| Class | Rate | Queries | Normal | Defect | Queried defect fraction | First defect | First 8 defects | First 14 defects | Longest no-defect span |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| audiojack | 3% | 114 | 74 | 40 | 0.3509 | 1,305 | 1,732 | 2,023 | 1,305 |
| audiojack | 5% | 191 | 124 | 67 | 0.3508 | 1,305 | 1,672 | 1,881 | 1,305 |
| audiojack | 7% | 267 | 171 | 96 | 0.3596 | 1,305 | 1,531 | 1,766 | 1,305 |
| regulator | 3% | 116 | 81 | 35 | 0.3017 | 116 | 1,007 | 1,504 | 937 |
| regulator | 5% | 193 | 130 | 63 | 0.3264 | 29 | 532 | 759 | 937 |
| regulator | 7% | 270 | 178 | 92 | 0.3407 | 10 | 309 | 488 | 937 |
| switch | 3% | 115 | 54 | 61 | 0.5304 | 55 | 375 | 631 | 914 |
| switch | 5% | 191 | 90 | 101 | 0.5288 | 10 | 250 | 395 | 914 |
| switch | 7% | 267 | 132 | 135 | 0.5056 | 10 | 162 | 362 | 914 |
| rolled_strip_base | 3% | 114 | 33 | 81 | 0.7105 | 35 | 297 | 501 | 887 |
| rolled_strip_base | 5% | 190 | 58 | 132 | 0.6947 | 8 | 159 | 349 | 887 |
| rolled_strip_base | 7% | 266 | 83 | 183 | 0.6880 | 8 | 96 | 232 | 887 |

Positions are zero-based image-stream positions; the gap includes leading and trailing no-defect-query runs. At 3%, Audiojack has no defect query among the first 400 events; the first defect arrives at 1,305. Regulator has 2 queried defects among its first 400 seed-37 events; Rolled Strip Base has 11 and Switch has 9. Audiojack reaches 8 and 14 defect anchors only at 1,732 and 2,023. Higher nested budgets add queries but cannot move the original seed-37 Audiojack first-defect position.

## Across-seed query sensitivity

The following are descriptive summaries over 200 preregistered, label-blind schedules per cell. Quantiles use linear interpolation; SD is the sample standard deviation. `F(>=8)` and `F(>=14)` are finite-set frequencies, not probabilities over all schedules or guarantees.

| Class | Rate | Defects min | P05 | P25 | Median | P75 | P95 | Max | Mean +/- SD | Min normals | F(>=8) | F(>=14) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| audiojack | 3% | 29 | 31.95 | 35 | 38.5 | 41 | 45 | 50 | 38.24 +/- 4.14 | 64 | 1.000 | 1.000 |
| audiojack | 5% | 51 | 56.95 | 61 | 65 | 68 | 74 | 79 | 64.78 +/- 5.36 | 112 | 1.000 | 1.000 |
| audiojack | 7% | 73 | 81.95 | 87 | 91 | 95 | 102 | 110 | 91.39 +/- 6.09 | 157 | 1.000 | 1.000 |
| regulator | 3% | 21 | 25 | 29 | 32 | 35 | 39.05 | 44 | 32.03 +/- 4.35 | 72 | 1.000 | 1.000 |
| regulator | 5% | 41 | 45 | 50 | 54 | 58 | 62 | 68 | 53.85 +/- 5.37 | 125 | 1.000 | 1.000 |
| regulator | 7% | 60 | 64.95 | 71 | 75 | 79 | 86.05 | 92 | 75.46 +/- 6.41 | 178 | 1.000 | 1.000 |
| switch | 3% | 50 | 53.95 | 57 | 60 | 64 | 67 | 71 | 60.48 +/- 4.10 | 44 | 1.000 | 1.000 |
| switch | 5% | 82 | 91.90 | 96 | 100 | 103 | 109 | 112 | 99.79 +/- 5.50 | 79 | 1.000 | 1.000 |
| switch | 7% | 122 | 128.95 | 134 | 139 | 144 | 149 | 152 | 139.19 +/- 6.26 | 115 | 1.000 | 1.000 |
| rolled_strip_base | 3% | 66 | 69.95 | 74 | 76 | 78 | 81 | 86 | 75.63 +/- 3.44 | 28 | 1.000 | 1.000 |
| rolled_strip_base | 5% | 116 | 120.95 | 124 | 127 | 129 | 134 | 141 | 126.62 +/- 3.95 | 49 | 1.000 | 1.000 |
| rolled_strip_base | 7% | 166 | 169 | 174 | 177 | 179 | 184 | 193 | 176.69 +/- 4.49 | 73 | 1.000 | 1.000 |

All 12 cells reached 8 and 14 queried defects in all 200 schedules, and every cell also had at least 16 queried normals (the smallest across-seed minimum was 28). Thus every finite schedule met the count-only nominal two-sided minimum; this does not establish a feasible SCDAC-G threshold. First-activation quantiles for every class/rate and all camera, bin, group-view, first-400, paired-budget, and sliding-window aggregates are in `query_seed_sensitivity.json`. For example, at 3% the median first-14 position (P05-P95) is 2,186 (1,925-2,470) for Audiojack, 1,173 (837-1,612) for Regulator, 496 (405-623) for Rolled Strip Base, and 596 (471-809) for Switch.

Every 200-schedule sensitivity distribution in each class/rate cell contains a schedule with at least one 256-image window with zero queried defects. The per-window seed-37 count and min/P05/median/P95/max/mean/SD over seeds are in `query_window_sensitivity.csv.gz`. These long gaps show that sufficient total feedback does not imply early or locally available defect feedback.

## B7 evidence-count feasibility

The frozen SCDAC-G v0.1 settings remain `min_normals=16`, `min_defects=8`, descriptive `delta=0.05`, and FPR/FNR targets 0.20/0.20. Eight queried defects permit nominal two-sided evidence processing only when at least 16 queried normals are also present. Eight zero-error defect anchors do not meet the stated descriptive FNR endpoint: the one-sided 95% Clopper-Pearson upper endpoint is about 0.312. At least 14 zero-error anchors are needed for this fixed-time endpoint to be no greater than 0.20 (about 0.193); any observed errors can require more anchors.

The finite schedule set had at least 14 defects in every cell, so the count-only minimum was attained in this sensitivity set. This is not a future guarantee. It also does not establish that the feasible risk-constrained threshold set exists: score ordering, current memory, normal/defect overlap, and the normal-side constraint matter. Adaptive selection, temporal dependence, repeated rescoring, and memory-dependent scores prevent interpreting these descriptive empirical endpoints as certified online population risk. B7 was not run or modified in this audit.

## Five-seed static native scorer

Each summary is mean +/- sample SD over support seeds 0-4. The oracle is `IN_SAMPLE_RETROSPECTIVE_STATIC_THRESHOLD_ORACLE`, using the same frozen score vector and labels unavailable to any controller. Threshold decisions use `score >= threshold`; BA ties are resolved by minimum movement from theta0, then higher threshold.

| Class | AUROC | AUPRC (prevalence varies by class) | Frozen theta0 BA | Oracle BA | Oracle - Frozen BA | FROZEN_STATIC_NEAR_CEILING |
|---|---:|---:|---:|---:|---:|---:|
| audiojack | 0.6226 +/- 0.0470 | 0.4605 +/- 0.0694 | 0.5488 +/- 0.0402 | 0.5902 +/- 0.0322 | 0.0413 +/- 0.0132 | 0/5 |
| regulator | 0.6255 +/- 0.0568 | 0.3839 +/- 0.0894 | 0.5349 +/- 0.0524 | 0.6134 +/- 0.0298 | 0.0785 +/- 0.0426 | 0/5 |
| rolled_strip_base | 0.9109 +/- 0.0127 | 0.9541 +/- 0.0074 | 0.8233 +/- 0.0205 | 0.8294 +/- 0.0169 | 0.0061 +/- 0.0050 | 0/5 |
| switch | 0.7353 +/- 0.0566 | 0.7766 +/- 0.0521 | 0.6474 +/- 0.0498 | 0.6701 +/- 0.0504 | 0.0228 +/- 0.0118 | 0/5 |

Across all 20 class-seed configurations, zero met the exploratory flag rule `Frozen BA >= 0.95 AND oracle headroom <= 0.01`. Rolled Strip Base had high rank discrimination and low retrospective headroom but its Frozen BA remained below 0.95; it is therefore not flagged. Audiojack and Regulator had much weaker fixed operating points and larger threshold headroom. The oracle is optimistic and in-sample: it is neither a usable controller threshold nor an unbiased estimate or upper bound for adaptive scoring.

Exact per-seed AUROC, AUPRC, prevalence, theta0, confusion counts, FPR/FNR/recall, oracle threshold and metrics, movement, overlap/tie values, and near-ceiling flag are in `static_score_seed_summary.csv`. Normal/defect score quantiles, overlap intervals, and cross-class tie counts are in `static_score_distribution_seed_summary.csv`. Per-window confusion outcomes are in `static_score_windows.csv.gz`.

## Temporal operating behavior

| Class | Undefined single-class windows (out of all seed-windows) | Worst defined 256-image Frozen BA | Longest FN run across support seeds |
|---|---:|---:|---:|
| audiojack | 4,880 | 0.4340 | 7-16 |
| regulator | 3,210 | 0.4196 | 7-8 |
| rolled_strip_base | 3,095 | 0.4232 | 6-10 |
| switch | 3,000 | 0.4325 | 7-10 |

Undefined window BA is kept undefined when one class is absent. Overall performance obscures local operating failures: the worst defined windows are around 0.42-0.43 BA, and false-negative runs of 6-16 consecutive images occur. These are static M0 results, not an online deployment evaluation.

## Interpretation, exposure, and limits

The query-selection algorithm was label-blind, but full test labels for all four authorized classes were then accessed for query composition and score evaluation after tapes were frozen. These outcomes are now exposed exploratory information. Future design decisions informed by these results must say so. The audit did not access masks, Mint/PCB, other categories, or any Stage 1/2 outcomes. It made no algorithm choice, query-budget revision, controller change, or protocol revision.

The scorer used native v22/P0/VisionAD code with the frozen DINOv2 configuration, support/C64 identities, and checksummed score tapes. Direct native-to-P0 parity was exact on bounded parity fixtures for all 20 class-seed configurations (maximum score delta 0). Test-stream scoring runtime, measured after support/C64 setup, was 635.60 s for Audiojack, 396.84 s for Regulator, 403.79 s for Rolled Strip Base, and 398.48 s for Switch. Peak CUDA allocation in those loops was 1.80 GB per class. One feature extraction was shared per C64+test image across the five support banks; 83 support/C64 feature records were encoded per class.

The static audit fixes support and C64, leaves dynamic memory disabled, and does not test common-score causal feedback, shared memory writes, adaptive thresholds, controller benefit, order stress, or generalization beyond these four exposed streams and finite query schedules. It does not show that online calibration is unnecessary, establish empirical superiority, or provide a real-world risk guarantee.

## Decisions intentionally deferred

No Stage 1 experiment was started. The approved 3% main schedule and frozen protocol remain unchanged; no algorithm was selected. Any use of these exposed statistics to change the design requires a separately recorded scientific rationale and authorization. Stage 1 and Stage 2 remain on HOLD.
