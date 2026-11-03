# R7 Four-Class Fixed-Shuffle Correction Necessity Experiment

## Classification and protocol

**Classification:** `RETROSPECTIVE_DEVELOPMENT_DIAGNOSTIC`. This is not independent confirmation or a blind evaluation. Historical development outcomes may have been observed before this experiment. The four category streams were permuted by Source Group (seed 0); original camera-view order was kept within each group. Reported positions are positions in that shuffled stream, not factory chronology. This source-group shuffle is not an exact reproduction of v22's legacy image-level shuffle.

All six decisions at an event and its fixed Q-A-TS query decision were durably sealed before any queried label was disclosed. Stage A read only the exact query labels. Stage B opened image-level outcomes only after all four full Stage A seals and did not open masks. Greedy and C1 updates used the same legal queried feedback. Threshold variants changed decisions only; the shared Greedy writer and Correction eligibility used the pre-event G/C0 score and frozen C64 q90.

C1-Full used the frozen native SignedCorrection formula and state with K=1024; Greedy used K=1024. No FPSA, BEA, hierarchy, parameter search, or formula changes were used. Q-A-TS-v1 used rate 3%, Seed 37, 16 bins, half-up rounding, with at most one queried view per Source Group.
The exact manifest-bound P0 scoring values, profile digest, C1-Full configuration, Greedy writer settings and capacity overrides are recorded in the private experiment identity.

## Execution receipts

- Branch and parent commit: `calibration-stage0-contract-20261029` / `276746b01fba47a7d41fa3ceb01b40e961fee341`.
- Design identity SHA-256: `7b90eeb1baa1481e4c54222be473d0417d5c09fe0c9f72e4f42e4136b5c3c5bc`.
- Stage A status: `SEALED_COMPLETE`; 15,295 shuffled events and 459 legal queries.
- GPU smoke: `PASS` on audiojack, 512 events and 17 queried events; Greedy state changes 12, C1 state changes 17. The smoke is excluded from Stage B metrics.
- Stage B status: `SEALED_COMPLETE`; outcome projection opened after the four category seals. Windows Job Object result: `COMPLETED`.
- Independent post-seal CPU reconciliation: PASS. AUROC/AP were recomputed from the sealed score rows and the Stage B `anomaly_class` projection; all six prediction confusion matrices per category were recomputed from sealed predictions. Reconciliation covered all 15,295 rows and verified the Stage A row hash, Stage A seal hash, Stage B seal hash and aggregate hash.
- Every full category encoded each scored TEST event once. Audiojack reused the checksum-verified sealed R5 C64 cache directly; regulator, switch, and rolled_strip_base encoded each of their 64 C64 references once.
- R5 cache lineage: the pinned R5 report records executed runner `69d34f74dd81e43834a08028407211ae48e27451c2c388da88fc86618608c939` and committed runner `bb07c3a9bef677c3321a4c384af04d459185a5ee801d91f51aabe9058d6d73cb`; it records that only a trailing blank line changed. R7 verified both hashes, the report digest `b3888fe53316dd7beaf169b18476a2ac03b77c9425a9e980e56ff589b583f9f3`, R5 seals, cache manifest/file checksums, C64 group/image identities, and pixel equality. The later R6 cache report contains an invalid 63-character executed-runner digest, so R7 does not rely on the R6 copy.
- Runtime ceiling was 120 minutes per full category child and 15 minutes for smoke. Each child was supervised in a Windows Job Object. No restart or resume was used.
- Output controls: fresh private run namespace, 4 GiB pre-write OutputBudget per smoke/full directory, 64 KiB reserved failure metadata, and at least 10 GiB free space before launch. This is an application-level bound, not an OS-level hard storage quota.

## Results by category and arm

AUROC/AP rank the sealed scorer trajectory; controller thresholds do not alter those scores. Confusion is TN/FP/FN/TP. Worst-window values use every stride-1 complete 512-event window in shuffled order. Query count/rate is shared across the six arms.

### audiojack

Events 3,815; Source Groups 763; outcomes normal 2,519, defect 1,296; queries 114 (2.988%; normal feedback 75, defect feedback 39).

| Arm | AUROC | AP | BA | FPR | FNR | Accuracy | TN / FP / FN / TP | Coverage | Queries | Worst 512 BA | Worst 512 FPR | Worst 512 FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G-Frozen | 0.7393 | 0.6067 | 0.5421 | 0.0083 | 0.9074 | 0.6862 | 2498 / 21 / 1176 / 120 | 1.0000 | 114 | 0.5188 | 0.0444 | 0.9624 |
| G-B1 | 0.7393 | 0.6067 | 0.5916 | 0.0183 | 0.7986 | 0.7166 | 2473 / 46 / 1035 / 261 | 1.0000 | 114 | 0.5580 | 0.0467 | 0.8703 |
| G-B3 | 0.7393 | 0.6067 | 0.6743 | 0.1020 | 0.5494 | 0.7460 | 2262 / 257 / 712 / 584 | 1.0000 | 114 | 0.5842 | 0.1438 | 0.7191 |
| GC-Frozen | 0.7727 | 0.6605 | 0.5847 | 0.0127 | 0.8179 | 0.7138 | 2487 / 32 / 1060 / 236 | 1.0000 | 114 | 0.5539 | 0.0533 | 0.8860 |
| GC-B1 | 0.7727 | 0.6605 | 0.6156 | 0.0258 | 0.7431 | 0.7305 | 2454 / 65 / 963 / 333 | 1.0000 | 114 | 0.5688 | 0.0533 | 0.8168 |
| GC-B3 | 0.7727 | 0.6605 | 0.6893 | 0.1453 | 0.4761 | 0.7423 | 2153 / 366 / 617 / 679 | 1.0000 | 114 | 0.5725 | 0.2256 | 0.7600 |

Paired GC−G contrasts and Source Group bootstrap intervals:

| Contrast | Estimate | 95% Source Group bootstrap interval |
|---|---:|---:|
| auroc_GC_minus_G | 0.0334 | [0.0218, 0.0452] |
| average_precision_GC_minus_G | 0.0538 | [0.0379, 0.0698] |
| balanced_accuracy_GC_minus_G_B1 | 0.0240 | [0.0144, 0.0335] |
| balanced_accuracy_GC_minus_G_B3 | 0.0150 | [0.0004, 0.0300] |
| balanced_accuracy_GC_minus_G_Frozen | 0.0426 | [0.0346, 0.0511] |
| fnr_GC_minus_G_B1 | -0.0556 | [-0.0735, -0.0366] |
| fnr_GC_minus_G_B3 | -0.0733 | [-0.0989, -0.0484] |
| fnr_GC_minus_G_Frozen | -0.0895 | [-0.1065, -0.0738] |
| fpr_GC_minus_G_B1 | 0.0075 | [0.0012, 0.0146] |
| fpr_GC_minus_G_B3 | 0.0433 | [0.0298, 0.0562] |
| fpr_GC_minus_G_Frozen | 0.0044 | [0.0000, 0.0089] |

Mechanism and shuffled-region diagnostics:

- First defect position: 3; first queried defect position: 116.
- Cumulative query coverage at 25/50/75/100% of the shuffled stream: 25%: 3.145% (30/954); 50%: 3.040% (58/1,908); 75%: 3.005% (86/2,862); 100%: 2.988% (114/3,815).
- Correction score delta GC−G: `{"count": 3815, "max": 0.06699377298355103, "mean": 0.016581333535058158, "median": 0.01908448338508606, "min": -0.03355392813682556, "negative_fraction": 0.20340760157273918, "nonzero_fraction": 0.9895150720838795, "positive_fraction": 0.7861074705111403, "q05": -0.01694190800189972, "q95": 0.0449854254722595}`; activation fraction 0.9895.
- Greedy native update events: 75; actual shared-bank mutations: 75; accepted patches 616; skipped patches 0.
- C1 queried updates: 114; scorer-state changes 113; final occupancy 1024/1024; first full capacity position 2023.
- C1 native patch decisions (inserted / kept / merged / replaced / skipped): 1,024 / 3,648 / 7 / 866 / 1,751.
- B1 updates/threshold movements: `{"G-B1": {"final_threshold": 0.5345697402954102, "queried_events_observed": 114, "threshold_movements": 9, "verified_normal_updates": 75}, "GC-B1": {"final_threshold": 0.5481221675872803, "queried_events_observed": 114, "threshold_movements": 8, "verified_normal_updates": 75}}`; B3 complete-state updates/threshold movements: `{"G-B3": {"complete_scorer_state_updates": 75, "final_threshold": 0.48162195086479187, "threshold_movements": 35}, "GC-B3": {"complete_scorer_state_updates": 113, "final_threshold": 0.5046878457069397, "threshold_movements": 94}}`.
- Query-time Greedy and Correction occupancy traces are included as anonymous shuffled positions and counts in the aggregate; no sample identity is included.
- Mutable tensor bytes at completion: `{"combined_mutable_tensor_bytes": 21884928, "correction_allocated_metadata_bytes": 61440, "correction_allocated_prototype_bytes": 16777216, "correction_capacity": 1024, "correction_negative_occupancy": 592, "correction_occupancy": 1024, "correction_positive_occupancy": 432, "correction_zero_sign_occupancy": 0, "greedy_capacity_by_layer": [1024], "greedy_mutable_metadata_bytes": 0, "greedy_occupancy_by_layer": [616], "greedy_used_prototype_bytes": 5046272}`.
- C64 rescoring rounds: `{"G": 76, "GC": 115}`; score calls: `{"G": 4864, "GC": 7360}`; seconds: `{"G": 38.066534500801936, "GC": 33.92698560026474}`.

| Shuffled region | Events | Queries (normal / defect) | G-Frozen AUROC | GC-Frozen AUROC | GC−G AUROC | G-Frozen BA | GC-Frozen BA | GC−G BA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| early | 1271 | 38 (30 / 8) | 0.6960 | 0.7109 | 0.0149 | 0.5493 | 0.5747 | 0.0254 |
| middle | 1272 | 39 (20 / 19) | 0.7713 | 0.7831 | 0.0118 | 0.5373 | 0.5745 | 0.0372 |
| late | 1272 | 37 (25 / 12) | 0.8388 | 0.8439 | 0.0051 | 0.5409 | 0.6057 | 0.0648 |


### regulator

Events 3,855; Source Groups 771; outcomes normal 2,775, defect 1,080; queries 116 (3.009%; normal feedback 87, defect feedback 29).

| Arm | AUROC | AP | BA | FPR | FNR | Accuracy | TN / FP / FN / TP | Coverage | Queries | Worst 512 BA | Worst 512 FPR | Worst 512 FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G-Frozen | 0.7667 | 0.5853 | 0.5113 | 0.0032 | 0.9741 | 0.7248 | 2766 / 9 / 1052 / 28 | 1.0000 | 116 | 0.4977 | 0.0190 | 1.0000 |
| G-B1 | 0.7667 | 0.5853 | 0.6041 | 0.0094 | 0.7824 | 0.7741 | 2749 / 26 / 845 / 235 | 1.0000 | 116 | 0.4977 | 0.0190 | 0.9860 |
| G-B3 | 0.7667 | 0.5853 | 0.7127 | 0.0912 | 0.4833 | 0.7990 | 2522 / 253 / 522 / 558 | 1.0000 | 116 | 0.5599 | 0.1500 | 0.7589 |
| GC-Frozen | 0.8131 | 0.7107 | 0.5879 | 0.0058 | 0.8185 | 0.7665 | 2759 / 16 / 884 / 196 | 1.0000 | 116 | 0.5384 | 0.0299 | 0.9091 |
| GC-B1 | 0.8131 | 0.7107 | 0.6916 | 0.0187 | 0.5981 | 0.8189 | 2723 / 52 / 646 / 434 | 1.0000 | 116 | 0.5384 | 0.0364 | 0.8958 |
| GC-B3 | 0.8131 | 0.7107 | 0.7449 | 0.0843 | 0.4259 | 0.8200 | 2541 / 234 / 460 / 620 | 1.0000 | 116 | 0.5756 | 0.1405 | 0.7292 |

Paired GC−G contrasts and Source Group bootstrap intervals:

| Contrast | Estimate | 95% Source Group bootstrap interval |
|---|---:|---:|
| auroc_GC_minus_G | 0.0464 | [0.0359, 0.0564] |
| average_precision_GC_minus_G | 0.1255 | [0.1057, 0.1456] |
| balanced_accuracy_GC_minus_G_B1 | 0.0874 | [0.0735, 0.1020] |
| balanced_accuracy_GC_minus_G_B3 | 0.0321 | [0.0198, 0.0458] |
| balanced_accuracy_GC_minus_G_Frozen | 0.0765 | [0.0632, 0.0903] |
| fnr_GC_minus_G_B1 | -0.1843 | [-0.2132, -0.1576] |
| fnr_GC_minus_G_B3 | -0.0574 | [-0.0828, -0.0352] |
| fnr_GC_minus_G_Frozen | -0.1556 | [-0.1828, -0.1294] |
| fpr_GC_minus_G_B1 | 0.0094 | [0.0044, 0.0144] |
| fpr_GC_minus_G_B3 | -0.0068 | [-0.0186, 0.0043] |
| fpr_GC_minus_G_Frozen | 0.0025 | [0.0007, 0.0048] |

Mechanism and shuffled-region diagnostics:

- First defect position: 0; first queried defect position: 55.
- Cumulative query coverage at 25/50/75/100% of the shuffled stream: 25%: 3.008% (29/964); 50%: 3.008% (58/1,928); 75%: 3.008% (87/2,892); 100%: 3.009% (116/3,855).
- Correction score delta GC−G: `{"count": 3855, "max": 0.12409317493438721, "mean": 0.021188907995916682, "median": 0.02059715986251831, "min": -0.05836844444274902, "negative_fraction": 0.26718547341115434, "nonzero_fraction": 0.9885862516212711, "positive_fraction": 0.7214007782101167, "q05": -0.026978084444999693, "q95": 0.07314852476119993}`; activation fraction 0.9886.
- Greedy native update events: 87; actual shared-bank mutations: 87; accepted patches 712; skipped patches 0.
- C1 queried updates: 116; scorer-state changes 113; final occupancy 1024/1024; first full capacity position 1305.
- C1 native patch decisions (inserted / kept / merged / replaced / skipped): 1,024 / 3,712 / 106 / 1,532 / 1,050.
- B1 updates/threshold movements: `{"G-B1": {"final_threshold": 0.477100670337677, "queried_events_observed": 116, "threshold_movements": 8, "verified_normal_updates": 87}, "GC-B1": {"final_threshold": 0.4966999292373657, "queried_events_observed": 116, "threshold_movements": 8, "verified_normal_updates": 87}}`; B3 complete-state updates/threshold movements: `{"G-B3": {"complete_scorer_state_updates": 87, "final_threshold": 0.42196470499038696, "threshold_movements": 31}, "GC-B3": {"complete_scorer_state_updates": 114, "final_threshold": 0.46015486121177673, "threshold_movements": 101}}`.
- Query-time Greedy and Correction occupancy traces are included as anonymous shuffled positions and counts in the aggregate; no sample identity is included.
- Mutable tensor bytes at completion: `{"combined_mutable_tensor_bytes": 22671360, "correction_allocated_metadata_bytes": 61440, "correction_allocated_prototype_bytes": 16777216, "correction_capacity": 1024, "correction_negative_occupancy": 766, "correction_occupancy": 1024, "correction_positive_occupancy": 258, "correction_zero_sign_occupancy": 0, "greedy_capacity_by_layer": [1024], "greedy_mutable_metadata_bytes": 0, "greedy_occupancy_by_layer": [712], "greedy_used_prototype_bytes": 5832704}`.
- C64 rescoring rounds: `{"G": 88, "GC": 116}`; score calls: `{"G": 5632, "GC": 7424}`; seconds: `{"G": 45.46050709998235, "GC": 39.483596599893644}`.

| Shuffled region | Events | Queries (normal / defect) | G-Frozen AUROC | GC-Frozen AUROC | GC−G AUROC | G-Frozen BA | GC-Frozen BA | GC−G BA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| early | 1285 | 38 (28 / 10) | 0.7355 | 0.7610 | 0.0255 | 0.5171 | 0.5710 | 0.0539 |
| middle | 1285 | 40 (31 / 9) | 0.7938 | 0.8319 | 0.0380 | 0.5107 | 0.5690 | 0.0583 |
| late | 1285 | 38 (28 / 10) | 0.8231 | 0.8490 | 0.0259 | 0.5066 | 0.6207 | 0.1142 |


### switch

Events 3,820; Source Groups 764; outcomes normal 1,826, defect 1,994; queries 115 (3.010%; normal feedback 59, defect feedback 56).

| Arm | AUROC | AP | BA | FPR | FNR | Accuracy | TN / FP / FN / TP | Coverage | Queries | Worst 512 BA | Worst 512 FPR | Worst 512 FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G-Frozen | 0.8686 | 0.8900 | 0.6733 | 0.0110 | 0.6424 | 0.6594 | 1806 / 20 / 1281 / 713 | 1.0000 | 115 | 0.6206 | 0.0309 | 0.7546 |
| G-B1 | 0.8686 | 0.8900 | 0.7426 | 0.0318 | 0.4829 | 0.7327 | 1768 / 58 / 963 / 1031 | 1.0000 | 115 | 0.6701 | 0.0628 | 0.6337 |
| G-B3 | 0.8686 | 0.8900 | 0.7983 | 0.1046 | 0.2989 | 0.7940 | 1635 / 191 / 596 / 1398 | 1.0000 | 115 | 0.7431 | 0.1571 | 0.4183 |
| GC-Frozen | 0.8849 | 0.9062 | 0.7417 | 0.0307 | 0.4860 | 0.7317 | 1770 / 56 / 969 / 1025 | 1.0000 | 115 | 0.6974 | 0.0702 | 0.5823 |
| GC-B1 | 0.8849 | 0.9062 | 0.7755 | 0.0438 | 0.4052 | 0.7675 | 1746 / 80 / 808 / 1186 | 1.0000 | 115 | 0.7102 | 0.0702 | 0.5299 |
| GC-B3 | 0.8849 | 0.9062 | 0.8063 | 0.0926 | 0.2949 | 0.8018 | 1657 / 169 / 588 / 1406 | 1.0000 | 115 | 0.7428 | 0.1341 | 0.4263 |

Paired GC−G contrasts and Source Group bootstrap intervals:

| Contrast | Estimate | 95% Source Group bootstrap interval |
|---|---:|---:|
| auroc_GC_minus_G | 0.0163 | [0.0099, 0.0226] |
| average_precision_GC_minus_G | 0.0161 | [0.0108, 0.0214] |
| balanced_accuracy_GC_minus_G_B1 | 0.0328 | [0.0233, 0.0417] |
| balanced_accuracy_GC_minus_G_B3 | 0.0080 | [-0.0013, 0.0171] |
| balanced_accuracy_GC_minus_G_Frozen | 0.0684 | [0.0591, 0.0791] |
| fnr_GC_minus_G_B1 | -0.0777 | [-0.0939, -0.0613] |
| fnr_GC_minus_G_B3 | -0.0040 | [-0.0186, 0.0101] |
| fnr_GC_minus_G_Frozen | -0.1565 | [-0.1761, -0.1388] |
| fpr_GC_minus_G_B1 | 0.0120 | [0.0049, 0.0197] |
| fpr_GC_minus_G_B3 | -0.0120 | [-0.0232, -0.0005] |
| fpr_GC_minus_G_Frozen | 0.0197 | [0.0125, 0.0276] |

Mechanism and shuffled-region diagnostics:

- First defect position: 0; first queried defect position: 55.
- Cumulative query coverage at 25/50/75/100% of the shuffled stream: 25%: 3.141% (30/955); 50%: 3.089% (59/1,910); 75%: 3.037% (87/2,865); 100%: 3.010% (115/3,820).
- Correction score delta GC−G: `{"count": 3820, "max": 0.06926175951957703, "mean": 0.01677486040673331, "median": 0.020819634199142456, "min": -0.03290647268295288, "negative_fraction": 0.16073298429319371, "nonzero_fraction": 0.9895287958115183, "positive_fraction": 0.8287958115183246, "q05": -0.015665258467197418, "q95": 0.0386282354593277}`; activation fraction 0.9895.
- Greedy native update events: 59; actual shared-bank mutations: 59; accepted patches 488; skipped patches 0.
- C1 queried updates: 115; scorer-state changes 111; final occupancy 1024/1024; first full capacity position 2139.
- C1 native patch decisions (inserted / kept / merged / replaced / skipped): 1,024 / 3,680 / 24 / 841 / 1,791.
- B1 updates/threshold movements: `{"G-B1": {"final_threshold": 0.4835672378540039, "queried_events_observed": 115, "threshold_movements": 6, "verified_normal_updates": 59}, "GC-B1": {"final_threshold": 0.4996565282344818, "queried_events_observed": 115, "threshold_movements": 7, "verified_normal_updates": 59}}`; B3 complete-state updates/threshold movements: `{"G-B3": {"complete_scorer_state_updates": 59, "final_threshold": 0.4571969211101532, "threshold_movements": 17}, "GC-B3": {"complete_scorer_state_updates": 113, "final_threshold": 0.490955650806427, "threshold_movements": 103}}`.
- Query-time Greedy and Correction occupancy traces are included as anonymous shuffled positions and counts in the aggregate; no sample identity is included.
- Mutable tensor bytes at completion: `{"combined_mutable_tensor_bytes": 20836352, "correction_allocated_metadata_bytes": 61440, "correction_allocated_prototype_bytes": 16777216, "correction_capacity": 1024, "correction_negative_occupancy": 410, "correction_occupancy": 1024, "correction_positive_occupancy": 614, "correction_zero_sign_occupancy": 0, "greedy_capacity_by_layer": [1024], "greedy_mutable_metadata_bytes": 0, "greedy_occupancy_by_layer": [488], "greedy_used_prototype_bytes": 3997696}`.
- C64 rescoring rounds: `{"G": 60, "GC": 115}`; score calls: `{"G": 3840, "GC": 7360}`; seconds: `{"G": 29.894571498967707, "GC": 33.78236069926061}`.

| Shuffled region | Events | Queries (normal / defect) | G-Frozen AUROC | GC-Frozen AUROC | GC−G AUROC | G-Frozen BA | GC-Frozen BA | GC−G BA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| early | 1273 | 39 (20 / 19) | 0.8528 | 0.8554 | 0.0026 | 0.7007 | 0.7441 | 0.0434 |
| middle | 1273 | 39 (20 / 19) | 0.8887 | 0.8919 | 0.0033 | 0.6689 | 0.7341 | 0.0652 |
| late | 1274 | 37 (19 / 18) | 0.9029 | 0.9158 | 0.0130 | 0.6499 | 0.7469 | 0.0970 |


### rolled_strip_base

Events 3,805; Source Groups 761; outcomes normal 1,280, defect 2,525; queries 114 (2.996%; normal feedback 37, defect feedback 77).

| Arm | AUROC | AP | BA | FPR | FNR | Accuracy | TN / FP / FN / TP | Coverage | Queries | Worst 512 BA | Worst 512 FPR | Worst 512 FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G-Frozen | 0.9534 | 0.9734 | 0.8175 | 0.0375 | 0.3275 | 0.7700 | 1232 / 48 / 827 / 1698 | 1.0000 | 114 | 0.7813 | 0.1404 | 0.4373 |
| G-B1 | 0.9534 | 0.9734 | 0.8512 | 0.0437 | 0.2539 | 0.8168 | 1224 / 56 / 641 / 1884 | 1.0000 | 114 | 0.7934 | 0.1404 | 0.3384 |
| G-B3 | 0.9534 | 0.9734 | 0.8867 | 0.1359 | 0.0907 | 0.8941 | 1106 / 174 / 229 / 2296 | 1.0000 | 114 | 0.8048 | 0.1844 | 0.2528 |
| GC-Frozen | 0.9568 | 0.9760 | 0.8779 | 0.1023 | 0.1418 | 0.8715 | 1149 / 131 / 358 / 2167 | 1.0000 | 114 | 0.7776 | 0.3397 | 0.2333 |
| GC-B1 | 0.9568 | 0.9760 | 0.8694 | 0.0945 | 0.1667 | 0.8576 | 1159 / 121 / 421 / 2104 | 1.0000 | 114 | 0.7776 | 0.3397 | 0.2691 |
| GC-B3 | 0.9568 | 0.9760 | 0.8945 | 0.1195 | 0.0915 | 0.8991 | 1127 / 153 / 231 / 2294 | 1.0000 | 114 | 0.8031 | 0.1776 | 0.2812 |

Paired GC−G contrasts and Source Group bootstrap intervals:

| Contrast | Estimate | 95% Source Group bootstrap interval |
|---|---:|---:|
| auroc_GC_minus_G | 0.0035 | [0.0015, 0.0056] |
| average_precision_GC_minus_G | 0.0026 | [0.0012, 0.0041] |
| balanced_accuracy_GC_minus_G_B1 | 0.0182 | [0.0075, 0.0289] |
| balanced_accuracy_GC_minus_G_B3 | 0.0078 | [0.0017, 0.0140] |
| balanced_accuracy_GC_minus_G_Frozen | 0.0604 | [0.0477, 0.0727] |
| fnr_GC_minus_G_B1 | -0.0871 | [-0.1030, -0.0721] |
| fnr_GC_minus_G_B3 | 0.0008 | [-0.0048, 0.0070] |
| fnr_GC_minus_G_Frozen | -0.1857 | [-0.2055, -0.1669] |
| fpr_GC_minus_G_B1 | 0.0508 | [0.0371, 0.0648] |
| fpr_GC_minus_G_B3 | -0.0164 | [-0.0270, -0.0058] |
| fpr_GC_minus_G_Frozen | 0.0648 | [0.0503, 0.0807] |

Mechanism and shuffled-region diagnostics:

- First defect position: 0; first queried defect position: 35.
- Cumulative query coverage at 25/50/75/100% of the shuffled stream: 25%: 3.151% (30/952); 50%: 3.048% (58/1,903); 75%: 3.013% (86/2,854); 100%: 2.996% (114/3,805).
- Correction score delta GC−G: `{"count": 3805, "max": 0.07725244760513306, "mean": 0.026242793914366333, "median": 0.025767385959625244, "min": -0.024933069944381714, "negative_fraction": 0.023915900131406046, "nonzero_fraction": 0.9894875164257556, "positive_fraction": 0.9655716162943495, "q05": 0.004074311256408692, "q95": 0.0506475806236267}`; activation fraction 0.9895.
- Greedy native update events: 37; actual shared-bank mutations: 37; accepted patches 312; skipped patches 0.
- C1 queried updates: 114; scorer-state changes 109; final occupancy 1024/1024; first full capacity position 2181.
- C1 native patch decisions (inserted / kept / merged / replaced / skipped): 1,024 / 3,648 / 47 / 541 / 2,036.
- B1 updates/threshold movements: `{"G-B1": {"final_threshold": 0.4593237340450287, "queried_events_observed": 114, "threshold_movements": 3, "verified_normal_updates": 37}, "GC-B1": {"final_threshold": 0.48636624217033386, "queried_events_observed": 114, "threshold_movements": 3, "verified_normal_updates": 37}}`; B3 complete-state updates/threshold movements: `{"G-B3": {"complete_scorer_state_updates": 37, "final_threshold": 0.4073565602302551, "threshold_movements": 18}, "GC-B3": {"complete_scorer_state_updates": 109, "final_threshold": 0.43289870023727417, "threshold_movements": 80}}`.
- Query-time Greedy and Correction occupancy traces are included as anonymous shuffled positions and counts in the aggregate; no sample identity is included.
- Mutable tensor bytes at completion: `{"combined_mutable_tensor_bytes": 19394560, "correction_allocated_metadata_bytes": 61440, "correction_allocated_prototype_bytes": 16777216, "correction_capacity": 1024, "correction_negative_occupancy": 266, "correction_occupancy": 1024, "correction_positive_occupancy": 758, "correction_zero_sign_occupancy": 0, "greedy_capacity_by_layer": [1024], "greedy_mutable_metadata_bytes": 0, "greedy_occupancy_by_layer": [312], "greedy_used_prototype_bytes": 2555904}`.
- C64 rescoring rounds: `{"G": 38, "GC": 111}`; score calls: `{"G": 2432, "GC": 7104}`; seconds: `{"G": 19.208171400241554, "GC": 32.46720559918322}`.

| Shuffled region | Events | Queries (normal / defect) | G-Frozen AUROC | GC-Frozen AUROC | GC−G AUROC | G-Frozen BA | GC-Frozen BA | GC−G BA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| early | 1268 | 38 (11 / 27) | 0.9189 | 0.9192 | 0.0004 | 0.8144 | 0.8211 | 0.0067 |
| middle | 1268 | 39 (12 / 27) | 0.9769 | 0.9793 | 0.0025 | 0.8256 | 0.9135 | 0.0879 |
| late | 1269 | 37 (14 / 23) | 0.9796 | 0.9823 | 0.0027 | 0.8105 | 0.8965 | 0.0860 |


## Equal-category macro results

| Arm | AUROC | AP | BA | FPR | FNR | Accuracy | Queries |
|---|---:|---:|---:|---:|---:|---:|---:|
| G-Frozen | 0.8320 | 0.7638 | 0.6361 | 0.0150 | 0.7129 | 0.7101 | 459 |
| G-B1 | 0.8320 | 0.7638 | 0.6974 | 0.0258 | 0.5795 | 0.7601 | 459 |
| G-B3 | 0.8320 | 0.7638 | 0.7680 | 0.1084 | 0.3556 | 0.8083 | 459 |
| GC-Frozen | 0.8569 | 0.8133 | 0.6980 | 0.0379 | 0.5660 | 0.7709 | 459 |
| GC-B1 | 0.8569 | 0.8133 | 0.7380 | 0.0457 | 0.4783 | 0.7936 | 459 |
| GC-B3 | 0.8569 | 0.8133 | 0.7837 | 0.1104 | 0.3221 | 0.8158 | 459 |

Paired macro GC−G effects and intervals:

| Contrast | Estimate | 95% Source Group bootstrap interval |
|---|---:|---:|
| auroc_GC_minus_G | 0.0249 | [0.0208, 0.0292] |
| average_precision_GC_minus_G | 0.0495 | [0.0429, 0.0559] |
| balanced_accuracy_GC_minus_G_B1 | 0.0406 | [0.0353, 0.0463] |
| balanced_accuracy_GC_minus_G_B3 | 0.0157 | [0.0102, 0.0215] |
| balanced_accuracy_GC_minus_G_Frozen | 0.0620 | [0.0561, 0.0678] |
| fnr_GC_minus_G_B1 | -0.1012 | [-0.1114, -0.0914] |
| fnr_GC_minus_G_B3 | -0.0335 | [-0.0433, -0.0243] |
| fnr_GC_minus_G_Frozen | -0.1468 | [-0.1574, -0.1362] |
| fpr_GC_minus_G_B1 | 0.0199 | [0.0155, 0.0244] |
| fpr_GC_minus_G_B3 | 0.0020 | [-0.0041, 0.0078] |
| fpr_GC_minus_G_Frozen | 0.0229 | [0.0186, 0.0275] |

Category sign counts for each GC−G contrast:

| Contrast | Positive categories | Negative categories | Zero | Available |
|---|---:|---:|---:|---:|
| auroc_GC_minus_G | 4 | 0 | 0 | 4 |
| average_precision_GC_minus_G | 4 | 0 | 0 | 4 |
| balanced_accuracy_GC_minus_G_B1 | 4 | 0 | 0 | 4 |
| balanced_accuracy_GC_minus_G_B3 | 4 | 0 | 0 | 4 |
| balanced_accuracy_GC_minus_G_Frozen | 4 | 0 | 0 | 4 |
| fnr_GC_minus_G_B1 | 0 | 4 | 0 | 4 |
| fnr_GC_minus_G_B3 | 1 | 3 | 0 | 4 |
| fnr_GC_minus_G_Frozen | 0 | 4 | 0 | 4 |
| fpr_GC_minus_G_B1 | 4 | 0 | 0 | 4 |
| fpr_GC_minus_G_B3 | 1 | 3 | 0 | 4 |
| fpr_GC_minus_G_Frozen | 4 | 0 | 0 | 4 |

## Interpretation

Correction improved image-level ranking over the shared Greedy-only trajectory in all four exposed development categories: equal-category macro GC?G AUROC was +0.0249 (95% Source Group bootstrap interval [0.0208, 0.0292]) and AP was +0.0495 [0.0429, 0.0559]. These results support incremental ranking value for this fixed-shuffle configuration only; they do not establish independent generalization.

Operational effects were positive for macro balanced accuracy under all three threshold controllers, with GC?G effects of +0.0620 Frozen, +0.0406 B1 and +0.0157 B3. The trade-off remains visible in the error rates: macro FNR fell by 0.1468, 0.1012 and 0.0335 respectively, while macro FPR rose by 0.0229, 0.0199 and 0.0020. For B3, the paired FPR interval includes zero ([-0.0041, 0.0078]); category effects were mixed, including an audiojack FPR increase and decreases in the other three categories. B1/Frozen FPR increases occurred in all four categories. Thus the ranking benefit is consistent, while operating-point benefit comes with FPR cost that varies by controller and category.

The existing simple calibration controls also improved Greedy-only BA over Frozen calibration (equal-category macro G-B1 0.6974 and G-B3 0.7680 versus G-Frozen 0.6361). Calibration reduces Correction's incremental BA margin from +0.0620 under Frozen to +0.0157 under B3. Under this registered comparison, Correction adds ranking signal and a smaller residual BA gain after B3, but it is not free of false-positive trade-offs. G and GC also have different total mutable memory footprints; this was a conditional incremental-value experiment, not a total-memory-byte-matched comparison.

## Runtime fingerprints and audit limits

The exact runtime profile and DINO hub source, checkpoint, and loaded-state fingerprints are recorded in the aggregate by category. The source code fingerprint set is recorded below. No source fingerprint was used to change the treatment.

Actual execution resources (decimal seconds as reported by the child receipts; memory shown in GiB):

| Stage | Events encoded | Child wall seconds | Peak CPU RSS | Peak CUDA allocated | Peak CUDA reserved |
|---|---:|---:|---:|---:|---:|
| GPU smoke (audiojack) | 512 | 52.844 | 3.687 | 1.464 | 13.303 |
| Stage A audiojack | 3,815 | 420.578 | 3.687 | 1.464 | 13.303 |
| Stage A regulator | 3,855 | 485.266 | 3.688 | 1.465 | 13.303 |
| Stage A switch | 3,820 | 436.813 | 3.686 | 1.464 | 13.303 |
| Stage A rolled_strip_base | 3,805 | 438.297 | 3.689 | 1.464 | 13.305 |
| CPU Stage B | ? | 20.484 | Not recorded | CUDA hidden | CUDA hidden |

The smoke supervisor wall time was 65.515 seconds (including child setup/teardown); Stage B ran with CUDA hidden. The four Stage A category children completed sequentially. Their per-category C64 rescoring calls and times are listed above.

| Source | SHA-256 |
|---|---|
| `experiments/calibration/r7_shuffled_correction_4class.py` | Launch `73a0d0c564500b565e58ea6c45e052f4b944b6a0ba38f24ad8ded8310c5545e9`; committed `0e2b0bc468cdcc1a8d72082487b1618817a45342fe93b99a00052ecb50f85b3a` |
| `experiments/calibration/r3_fullstream_2x2.py` | `0a67d893df6fe73399726f4cc60caa5967a6e3c2eb93535f23f7089350986741` |
| `experiments/calibration/r6_capacity1024_audiojack_runner.py` | `f9176d50420a6fc5474ebccb0cb2dd640cf0f3152911ea6c4c9736bad85d0165` |
| `experiments/calibration/r5_audiojack_b3_fpsa_runner.py` | `bb07c3a9bef677c3321a4c384af04d459185a5ee801d91f51aabe9058d6d73cb` |
| `experiments/calibration/query_tape.py` | `ed6f77c2a4de1bd05969ae3c09f2c98c13a744481ca436f80e4633bd6c412c6d` |
| `experiments/calibration/stage1a_p12_label_provider.py` | `4e51255ae894c2d6ffcc7142e8fd59d9714df94b18ed26ac996528adb163ceaf` |
| `experiments/calibration/controllers.py` | `25fbdfc9a92e8bfb6fe983185dca3cdd5ed24d804c28838b4797e0ae8670cf7d` |
| `experiments/calibration/stage05.py` | `4b8aae252f5f1e096b127906794064da269aa79f7cd5514f6f738cda585c017e` |
| `experiments/calibration/profiles.py` | `7ff59b6a3783bf4061ab75c6b0782b23a0f6e456b40457a022e5d0de13d6f7a5` |
| `experiments/calibration/native.py` | `3a347d65aaf4bfd6b41eff2ea7339d878065bb2e08ed64795f57b0458135dd45` |
| `experiments/calibration/r1_matched_correction_pilot.py` | `3208e2c52277ad76351eb86968e362aa315effe9e3c734bfe2b6ff7bab0db448` |
| `experiments/calibration/r2_feedback_repair.py` | `92fe290c4c7ea5cbb4da53b925b97ae58b8fa6c66e0ab98ca86895bee48fb0f2` |
| `experiments/calibration/r2_exp2a_c1_strength.py` | `4b798dcb1472e6dd630f58b3e62f3782dfb302f7579e942fcdb399885a5f33d3` |
| `experiments/calibration/r1_runtime_supervisor.py` | `e6254a82d8dd24156cea65380b41b04cb1372744da64a085043aa6dd74df5bab` |
| `experiments/risk_safe/native_bridge.py` | `9575214ea8d14ebe32a537fa15afa332446b41d66c1c8c85f890d2251eaef8f7` |
| `experiments/p0_protocol_audit.py` | `b83918ec5aceb3ef63cca942fc9fb0e241559d9672d20bed422b15fcd48135e7` |
| `experiments/v02_memory.py` | `3c4e247e3a41dc9acc8147a35aa892cf3f3571aca85beb4169809bd70ad48ae0` |
| `vmb_visionad_new.py` | `bd6545bf5ff63acce01792e5d42aa964c88e73fb59670febeb55f2d1139a409a` |
| `test_phase3_AIF_simplified_v22.py` | `bfc871d5a9a32b2f5bdb9d7aab5177e702dec2e1110a6632f20bd13c20fcba30` |

The R7 runner was bound to the parent source commit and its SHA-256 is recorded in the run manifest and aggregate identity. After execution, only trailing blank lines were removed from the runner to satisfy `git diff --cached --check`; the executable source lines are unchanged. The launch-time digest remains bound by the run manifest and is reported alongside the committed file digest. Fixed Support digests were independently checked against the frozen structured receipt for every category. C64 support is additional to the four fixed Support images.

This is retrospective evidence from four previously exposed development categories. It cannot establish independent generalization, blind confirmation, or physical time-order behavior. Per-event identities, labels and scores remain only in the private run directory; this report and its aggregate contain no sample identities, feature weights, event-level labels, or raw event-level score vectors.

## Reproducibility

Private output: `outputs/calibration/r7_shuffled_correction_4class_20261103_a4/` (smoke and full children are separate, non-resumable directories). Public tracked deliverables are this report and `aggregate_results.json`. Stage A seal SHA-256: `0f96772ddafcb6321918fee061c7aa43007a5fc83cc14a45fd031137665b2ad7`. Stage B aggregate SHA-256: `92b18f96613f37e6bd9cd60c95d5b5fc7ee913f2949d9c06c4047908c048ac2d`.
