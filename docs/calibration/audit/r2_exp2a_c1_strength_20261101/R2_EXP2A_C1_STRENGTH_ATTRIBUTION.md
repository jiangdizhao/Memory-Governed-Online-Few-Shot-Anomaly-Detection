# R2-EXP2A — Causal Attribution of C1 Correction Strength

**Status:** NO_CLEAR_INCREMENTAL_BENEFIT
**Coverage:** four historically exposed development categories, 512 events per category (2048 total).
**Independent confirmation:** not provided.

## Protocol and provenance

- Branch and parent commit: calibration-stage0-contract-20261029 at 4db0a1b701830339019c944d6862d7b25b6e7d14.
- Stage 05 frozen inputs SHA-256: 48b80202e4b5e02a85b8a950431b386199ee59046d220956062eef25804d0166.
- Q-A-TS binding and verification SHA-256: dbe72a2f989ee314becc07e0c8b8c7a8353ffacabbfc93863d5666cff9d28183 and ffbedc32edbd8f6c9b79d0d5b0fe3663d4cfb315d20273815405b84a486cc991.
- DINO checkpoint SHA-256: 36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51.
- Support regime: 4 deployed seed-0 supports plus 64 additional C64 training normals; C64 score tapes were reused and not recomputed.
- One shared native feature capture per category fed C0, four independent C1 Codebooks, and C2(a=0.02/0.20). Every query image was encoded once; each of four support images contributed four frozen augmentation views.
- DINO feature encodings across all categories: 2112. No per-arm feature extraction was performed.
- Stage A GPU: NVIDIA GeForce RTX 3090 Ti; 8078.766 seconds. Peak CUDA allocated/reserved: 1573543424/14285799424 bytes.
- Completed Stage B CPU child: 28.047 seconds; combined Stage A plus successful Stage B child time: 8106.813 seconds. The initial combined child failure is recorded separately below.
- Run output size at report generation: 1098143560 bytes of 2147483648; prelaunch free space was 298187132928 bytes.
- Query labels were disclosed online only at frozen Q-A-TS positions after all seven current-event scores were sealed. Stage B read the first 512 matching outcomes only after the complete Stage A seal. Masks were not opened.
- C2(a=0.02) was selected after switch development exposure. C2(a=0.20) is an optional historical engineering reference.
- Initial combined child: FAILED_NONZERO_EXIT, exit 2, after a complete Stage A seal. It stopped in the Stage B summary because the all-normal audiojack horizon has an undefined ranking result; the Q6 aggregator assumed that key existed.
- Repaired Stage B continuation: COMPLETED, exit 0, 28.047 seconds in a Windows Job Object. Stage A scores and failure records were preserved; Stage A was not rerun.
- The coverage-aware report keeps the strict four-category macro undefined where audiojack has no defects, and separately reports descriptive estimates over the three categories with both classes.

## C1-Full switch reference parity

- Result: PASS; 512 events and 17 query events checked against R1-S4B.
- Checks included exact C0 maps and Top-32 indices, C1-Full dense-map deltas, C1 score and state hashes, native update counts, common C0 memory state, query decisions, queried labels, and C2(0.20) deltas.
- Historical switch C1 AUROC was 0.843083; the recomputed value is 0.843083.

## Image AUROC

| Category | C0 | C1-Full | No outcome weight | No margin | No both | C2 0.02 | C2 0.20 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| switch | 0.807035 | 0.843083 | 0.855426 | 0.855517 | 0.852495 | 0.841025 | 0.752198 |
| audiojack | undefined | undefined | undefined | undefined | undefined | undefined | undefined |
| regulator | 0.687158 | 0.694623 | 0.687720 | 0.712958 | 0.718031 | 0.704128 | 0.597781 |
| rolled_strip_base | 0.898310 | 0.910798 | 0.907509 | 0.907956 | 0.906870 | 0.904954 | 0.828814 |
| Category-equal macro | undefined | undefined | undefined | undefined | undefined | undefined | undefined |
| Defined-category macro (3/4; descriptive) | 0.797501 | 0.816168 | 0.816885 | 0.825477 | 0.825799 | 0.816702 | 0.726265 |

## Factorial contrasts

Index 1 is margin adaptation; index 2 is outcome-strength weighting. F11 is C1-Full, F10 removes outcome weights, F01 fixes base beta to 0.02, and F00 removes both components.

| Contrast | Strict 4-category mean | Defined-category mean | Defined n | Positive / negative | Defined range |
|---|---:|---:|---:|---:|---:|
| outcome_effect_margin_enabled | undefined | -0.000717 | 3/4 | 2 / 1 | 0.019247 |
| outcome_effect_margin_disabled | undefined | -0.000322 | 3/4 | 2 / 1 | 0.008095 |
| margin_effect_outcome_enabled | undefined | -0.009309 | 3/4 | 1 / 2 | 0.021177 |
| margin_effect_outcome_disabled | undefined | -0.008914 | 3/4 | 2 / 1 | 0.033242 |
| interaction | undefined | -0.000395 | 3/4 | 2 / 1 | 0.027343 |

## Operating metrics by category and arm

| Category | Arm | AUROC | AUPRC | Prevalence | FPR | FNR | Balanced accuracy | TP/TN/FP/FN | Queried normals/defects |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| switch | c0 | 0.807035 | 0.915183 | 0.7012 | 0.0261 | 0.6017 | 0.6861 | 143/149/4/216 | 7/10 |
| switch | c1_full | 0.843083 | 0.932168 | 0.7012 | 0.0654 | 0.4318 | 0.7514 | 204/143/10/155 | 7/10 |
| switch | c1_no_outcome_weight | 0.855426 | 0.936804 | 0.7012 | 0.0719 | 0.4735 | 0.7273 | 189/142/11/170 | 7/10 |
| switch | c1_no_margin | 0.855517 | 0.938133 | 0.7012 | 0.0523 | 0.4624 | 0.7427 | 193/145/8/166 | 7/10 |
| switch | c1_no_both | 0.852495 | 0.936922 | 0.7012 | 0.0588 | 0.4791 | 0.7310 | 187/144/9/172 | 7/10 |
| switch | c2_a002 | 0.841025 | 0.930454 | 0.7012 | 0.0588 | 0.4958 | 0.7227 | 181/144/9/178 | 7/10 |
| switch | c2_a020 | 0.752198 | 0.892874 | 0.7012 | 0.7255 | 0.1226 | 0.5760 | 315/42/111/44 | 7/10 |
| audiojack | c0 | undefined | undefined | 0.0000 | 0.0059 | undefined | undefined | 0/509/3/0 | 17/0 |
| audiojack | c1_full | undefined | undefined | 0.0000 | 0.0039 | undefined | undefined | 0/510/2/0 | 17/0 |
| audiojack | c1_no_outcome_weight | undefined | undefined | 0.0000 | 0.0039 | undefined | undefined | 0/510/2/0 | 17/0 |
| audiojack | c1_no_margin | undefined | undefined | 0.0000 | 0.0039 | undefined | undefined | 0/510/2/0 | 17/0 |
| audiojack | c1_no_both | undefined | undefined | 0.0000 | 0.0039 | undefined | undefined | 0/510/2/0 | 17/0 |
| audiojack | c2_a002 | undefined | undefined | 0.0000 | 0.0000 | undefined | undefined | 0/512/0/0 | 17/0 |
| audiojack | c2_a020 | undefined | undefined | 0.0000 | 0.0000 | undefined | undefined | 0/512/0/0 | 17/0 |
| regulator | c0 | 0.687158 | 0.588829 | 0.3887 | 0.0096 | 0.9246 | 0.5329 | 15/310/3/184 | 13/3 |
| regulator | c1_full | 0.694623 | 0.621926 | 0.3887 | 0.0128 | 0.8492 | 0.5690 | 30/309/4/169 | 13/3 |
| regulator | c1_no_outcome_weight | 0.687720 | 0.609472 | 0.3887 | 0.0128 | 0.8995 | 0.5439 | 20/309/4/179 | 13/3 |
| regulator | c1_no_margin | 0.712958 | 0.633661 | 0.3887 | 0.0096 | 0.9246 | 0.5329 | 15/310/3/184 | 13/3 |
| regulator | c1_no_both | 0.718031 | 0.638728 | 0.3887 | 0.0096 | 0.9347 | 0.5279 | 13/310/3/186 | 13/3 |
| regulator | c2_a002 | 0.704128 | 0.608921 | 0.3887 | 0.0064 | 0.9447 | 0.5244 | 11/311/2/188 | 13/3 |
| regulator | c2_a020 | 0.597781 | 0.487596 | 0.3887 | 0.0990 | 0.8693 | 0.5158 | 26/282/31/173 | 13/3 |
| rolled_strip_base | c0 | 0.898310 | 0.982437 | 0.8613 | 0.0704 | 0.3129 | 0.8083 | 303/66/5/138 | 3/14 |
| rolled_strip_base | c1_full | 0.910798 | 0.984914 | 0.8613 | 0.1831 | 0.1383 | 0.8393 | 380/58/13/61 | 3/14 |
| rolled_strip_base | c1_no_outcome_weight | 0.907509 | 0.984280 | 0.8613 | 0.1972 | 0.1542 | 0.8243 | 373/57/14/68 | 3/14 |
| rolled_strip_base | c1_no_margin | 0.907956 | 0.984374 | 0.8613 | 0.1831 | 0.1519 | 0.8325 | 374/58/13/67 | 3/14 |
| rolled_strip_base | c1_no_both | 0.906870 | 0.984077 | 0.8613 | 0.1831 | 0.1701 | 0.8234 | 366/58/13/75 | 3/14 |
| rolled_strip_base | c2_a002 | 0.904954 | 0.983556 | 0.8613 | 0.1549 | 0.1950 | 0.8250 | 355/60/11/86 | 3/14 |
| rolled_strip_base | c2_a020 | 0.828814 | 0.970508 | 0.8613 | 0.9155 | 0.0249 | 0.5298 | 430/6/65/11 | 3/14 |

## Mechanism, ranking, and uncertainty

The sanitized JSON aggregate contains per-category Source Group cluster bootstrap intervals, beta distributions before and after each original clipping stage, patch multipliers, dN admission diagnostics, codebook insert/merge/replace/skip counts, occupancy by time, anchor signs, source-group exclusions, ablation activation, state and score divergence, prediction flips, false-positive/false-negative repairs and harm, and cross-source-group normal–defect pair repair/damage counts.

The four categories are four historically exposed development observations, not independent confirmation. Bootstrap intervals condition on each fixed online trajectory and do not quantify regenerated-stream uncertainty. A null contrast under weak activation or heavy clipping is not evidence that the mechanism is unnecessary.

audiojack contains 512 normal and 0 defect outcomes in the authorized horizon. Its AUROC, AUPRC, FNR, balanced accuracy, factorial effects, and cross-group normal-defect ranking diagnostics are undefined for that denominator; no replacement value is imputed.

## Scientific interpretation

- Q1 margin effect with outcome weights enabled: -0.009309 across 3/4 evaluable categories; the strict four-category macro is undefined.
- Q2 outcome-weight effect with margin enabled: -0.000717 across 3/4 evaluable categories.
- Q3 factorial interaction: -0.000395 across 3/4 evaluable categories.
- Q5 C1-Full minus C2(a=0.02), defined-category mean: -0.000534 over 3/4 categories; positive/negative 2/1.
- Q7: not established from four historically exposed development categories; describe mechanism only at the strength supported by the observed factorial effects.
- Experiment 2B remains separate and was not run. These results do not authorize Gaussian-protocol changes or further evaluation.

## Resource controls and reproducibility

- Windows Job Object hard deadline: 180 minutes; output budget: 2147483648 bytes with 65536 bytes reserved for failure metadata; minimum prelaunch free space: 10 GiB.
- The output budget is application level; OS-level storage isolation is not provided.
- Stage A runner SHA-256: 5f36962307f27c6a28cd97d3e9312a2b984da90cdd0cdf0e1bc17d0f33f6d840.
- Stage B metrics runner SHA-256: 1433d54e0083af9858d967267c4a60be964509c2d0ddb4706b906da4f8056c21.
- Coverage-aware reporting source SHA-256: 4b798dcb1472e6dd630f58b3e62f3782dfb302f7579e942fcdb399885a5f33d3.
- Source hash manifest: {"experiments/calibration/native.py":"3a347d65aaf4bfd6b41eff2ea7339d878065bb2e08ed64795f57b0458135dd45","experiments/calibration/profiles.py":"7ff59b6a3783bf4061ab75c6b0782b23a0f6e456b40457a022e5d0de13d6f7a5","experiments/calibration/query_tape.py":"ed6f77c2a4de1bd05969ae3c09f2c98c13a744481ca436f80e4633bd6c412c6d","experiments/calibration/r1_matched_correction_pilot.py":"3208e2c52277ad76351eb86968e362aa315effe9e3c734bfe2b6ff7bab0db448","experiments/calibration/r1_runtime_supervisor.py":"e6254a82d8dd24156cea65380b41b04cb1372744da64a085043aa6dd74df5bab","experiments/calibration/stage05.py":"4b8aae252f5f1e096b127906794064da269aa79f7cd5514f6f738cda585c017e","experiments/calibration/stage05_runner.py":"1d110e1ca813a064767ebeab597b609b3d863f8afd4c8085b06948ad88347e14","experiments/calibration/stage1a_p11_production.py":"77e84fa11c83eca683b0b95c4849b175053c0b24298e78dfe6482c91bdf6ea7d","experiments/calibration/stage1a_p12_label_provider.py":"4e51255ae894c2d6ffcc7142e8fd59d9714df94b18ed26ac996528adb163ceaf","experiments/p0_protocol_audit.py":"b83918ec5aceb3ef63cca942fc9fb0e241559d9672d20bed422b15fcd48135e7","experiments/risk_safe/native_bridge.py":"9575214ea8d14ebe32a537fa15afa332446b41d66c1c8c85f890d2251eaef8f7","experiments/v02_memory.py":"3c4e247e3a41dc9acc8147a35aa892cf3f3571aca85beb4169809bd70ad48ae0","test_phase3_AIF_simplified_v22.py":"bfc871d5a9a32b2f5bdb9d7aab5177e702dec2e1110a6632f20bd13c20fcba30","vmb_visionad_new.py":"bd6545bf5ff63acce01792e5d42aa964c88e73fb59670febeb55f2d1139a409a"}.
- Machine-readable sanitized aggregate: E:/Memory-Governed-Online-Few-Shot-Anomaly-Detection/docs/calibration/audit/r2_exp2a_c1_strength_20261101/sanitized_aggregate.json.
- Focused CPU regression: 21 passed; Python compilation passed. Tests include coverage with an undefined single-class category and defined/undefined cross-group ranking summaries.
- CPU synthetic tests cover the original v22 update path, beta clipping and patch weighting, independent Codebooks and sign-aware retention, source-group exclusion, causal query timing, dense-map maxima, reference parity, category aggregation, and incomplete Stage B gating.
