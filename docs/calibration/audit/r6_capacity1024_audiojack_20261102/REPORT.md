# R6-CAPACITY1024-AUDIOJACK — Results

**Status:** COMPLETE. Retrospective exposed audiojack development feasibility test; not independent confirmation, a blind evaluation, production certification, or an exact reproduction of the original Hierarchy system.

## Primary endpoint

Prequential Balanced Accuracy over all 3,815 audiojack TEST events in frozen JSON order. Physical chronology is not asserted. Query tape: 114 events (2.9882%); first anomalous queried feedback at position 1305. Four normal support images were used; C64 contains 64 additional protected TRAIN normals.

| Arm | BA | FPR | FNR | Accuracy | TP | TN | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| G1024-Frozen | 0.537974 | 0.001985 | 0.922068 | 0.685452 | 101 | 2514 | 5 | 1195 |
| GC1024-Frozen | 0.585315 | 0.009925 | 0.819444 | 0.715072 | 234 | 2494 | 25 | 1062 |
| FPSA1024-Frozen | 0.588215 | 0.009528 | 0.814043 | 0.717169 | 241 | 2495 | 24 | 1055 |
| G1024-B3 | 0.759226 | 0.111949 | 0.369599 | 0.800524 | 817 | 2237 | 282 | 479 |
| GC1024-B3 | 0.758144 | 0.175069 | 0.308642 | 0.779554 | 896 | 2078 | 441 | 400 |
| FPSA1024-B3 | 0.760317 | 0.171497 | 0.307870 | 0.782176 | 897 | 2087 | 432 | 399 |

Prediction coverage was 3,815/3,815 events for each of the six arms.

The 20% FNR and 20% FPR criteria are illustrative research screens only. They are not factory-approved tolerances or production safety limits.

## K=256 historical comparison

| Historical arm | BA | FPR | FNR | Accuracy | TP | TN | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| G256-B3 | 0.710291 | 0.134974 | 0.444444 | 0.759895 | 720 | 2179 | 340 | 576 |
| GC256-B3 | 0.717037 | 0.135371 | 0.430556 | 0.764351 | 738 | 2178 | 341 | 558 |
| FPSA256-B3 | 0.683282 | 0.148869 | 0.484568 | 0.737090 | 668 | 2144 | 375 | 628 |

R5 used the same frozen event identities/query protocol with K=256. Its metrics were re-evaluated from its sealed predictions under the post-seal evaluator and matched its aggregate.

## Required paired contrasts

Positive differences mean the left arm is higher for BA, FNR, or FPR. Intervals are descriptive source-group-paired BA bootstrap intervals (1,000 resamples).

| Contrast | Δ BA | Δ FNR | Δ FPR | paired source-group BA 95% interval |
|---|---:|---:|---:|---|
| FPSA1024-B3 minus R5-FPSA256-B3 | +0.077035 | -0.176698 | +0.022628 | [+0.062033, +0.092786] |
| FPSA1024-B3 minus G1024-B3 | +0.001090 | -0.061728 | +0.059547 | [-0.013268, +0.015449] |
| FPSA1024-B3 minus GC1024-B3 | +0.002172 | -0.000772 | -0.003573 | [-0.003667, +0.008193] |
| G1024-B3 minus R5-G256-B3 | +0.048935 | -0.074846 | -0.023025 | [+0.034837, +0.061513] |
| GC1024-B3 minus G1024-B3 | -0.001082 | -0.060957 | +0.063120 | [-0.015623, +0.012512] |
| GC1024-B3 minus R5-GC256-B3 | +0.041108 | -0.121914 | +0.039698 | [+0.024412, +0.057317] |

## Worst complete 512-event window

| Arm | Worst complete 512-event BA | Window | Normal/anomaly events |
|---|---:|---|---:|
| G1024-Frozen | 0.531350 | [2048, 2560) | 271/241 |
| GC1024-Frozen | 0.549515 | [1024, 1536) | 376/136 |
| FPSA1024-Frozen | 0.549515 | [1024, 1536) | 376/136 |
| G1024-B3 | 0.723715 | [1536, 2048) | 246/266 |
| GC1024-B3 | 0.691897 | [1536, 2048) | 246/266 |
| FPSA1024-B3 | 0.691897 | [1536, 2048) | 246/266 |

## Capacity and first divergence

- Initial native dynamic Greedy capacities: `[1024]`; `MemoryArm.flat` and native dynamic Codebooks were 1024.
- Fresh native GC and FPSA Correction Codebooks each allocated K=1024; prototype and all metadata tensors had 1024 rows. Both CorrectionConfig values were 1024. The frozen K=256 R3 recipe guard passed before allocation.
- Final Greedy occupancy by layer: `[608]`; GC sign occupancy: `{'negative': 691, 'positive': 333, 'total': 1024, 'zero': 0}`; FPSA sign occupancy: `{'negative': 661, 'positive': 363, 'total': 1024, 'zero': 0}`.
- First capacity-dependent divergence:
- `G`: update at position 1007; first subsequent score at 1008; Greedy occupancy [256] -> [264]. R6 admitted a prototype beyond the historical full-bank boundary.
- `GC`: update at position 859; first subsequent score at 860; GC Correction bank occupancy 225 -> 257; shared Greedy occupancy [224] -> [232]. Native update admitted a prototype beyond the historical full-bank boundary.
- `GC-FPSA`: update at position 859; first subsequent score at 860; FPSA Correction bank occupancy 225 -> 257; shared Greedy occupancy [224] -> [232]. Native update admitted a prototype beyond the historical full-bank boundary.
- R5 score parity was enforced while the corresponding states remained capacity-equivalent. Initial G/GC/FPSA scores matched. Later K-dependent score differences were not treated as implementation failures. R5 did not store per-event features, so bytewise stream feature equality is unavailable; same image bytes, DINO sources/weights/runtime, reused C64 cache, and pre-divergence scores were verified.

## Correction and FPSA

- Final FPSA: queried normals/anomalies `74/40`; positive/negative occupancy `363/661`; target capacities `359/665`; borrow opportunities/successes/rejections `702/34/668`.
- FPSA mechanism activated: `True`. No bank was artificially filled and no gates were changed.
- Query-level Greedy, GC/FPSA occupancy and FPSA target history is in the aggregate without event labels, paths, source-group identities or raw scores.

## B3 calibration and runtime

| Controller | Initial θ | Final θ | State updates | Threshold movements | Final state version |
|---|---:|---:|---:|---:|
| FPSA1024-B3 | 0.566524 | 0.520968 | 105 | 77 | 105 |
| G1024-B3 | 0.566524 | 0.488322 | 74 | 29 | 74 |
| GC1024-B3 | 0.566524 | 0.517747 | 104 | 93 | 104 |

- Unchanged `StateConsistentQuantile`: delta=0.01, min_normals=16; three independent complete scorer-state streams. Six decisions were durably written before queried-label disclosure. B3 stayed decoupled from frozen C64 writer/correction thresholds.
- C64 features were copied from the checksum-verified R5 cache; no C64 feature was re-encoded. C64 scoring time by scorer was `{'G': 37.71035979967564, 'GC': 30.938320600194857, 'GC-FPSA': 30.37578939832747}` seconds; calls and rounds are recorded in the aggregate.
- Stage A: 495.3s; Stage B: 5.0s; overall through Stage B: 584.7s.
- Peak CUDA allocated/reserved: 1,572,342,784/14,285,799,424 bytes; peak process RSS: 3,954,745,344 bytes.
- Source-group paired bootstrap: seed 20261102, 1000 replicates, descriptive only.

## FNR interpretation

- G1024-B3: FNR 36.9599% (+6.10 percentage points lower than the best R5 K=256 B3 FNR, 43.0556%); FPR 11.1949%; illustrative 20%/20% screen pass `False`.
- GC1024-B3: FNR 30.8642% (+12.19 percentage points lower than the best R5 K=256 B3 FNR, 43.0556%); FPR 17.5069%; illustrative 20%/20% screen pass `False`.
- FPSA1024-B3: FNR 30.7870% (+12.27 percentage points lower than the best R5 K=256 B3 FNR, 43.0556%); FPR 17.1497%; illustrative 20%/20% screen pass `False`.

## Capacity feasibility conclusion

The best R6 B3 arm was `FPSA1024-B3` with FNR 30.7870%, a 12.27 percentage-point reduction from the historical R5 best B3 FNR of 43.0556%. This is a material improvement on this retrospective audiojack stream, with FPR 17.1497%. All three full-stream B3 arms still fail the illustrative combined 20% FNR/20% FPR screen because their FNR exceeds 20%. The result is not production readiness or independent confirmation.

GC1024-B3 reduced FNR relative to G1024-B3 while raising FPR and slightly lowering BA. FPSA1024-B3 and GC1024-B3 were close overall despite FPSA borrowing being activated. Treat these as single-class feasibility results.

These are same-class retrospective results. Interpret FNR with FPR and BA. This experiment does not establish generality beyond audiojack or production readiness.

## Post-1305 operating slice

This slice begins at event position 1306, after the first anomalous queried feedback at 1305. Correction B3 arms have post-slice FPR above the illustrative 20% screen; their full-stream FPR is lower. This is a visible false-positive trade-off, not a deployment pass.

| Arm | BA | FNR | FPR | Events |
|---|---:|---:|---:|---:|
| G1024-Frozen | 0.538232 | 0.921924 | 0.001612 | 2509 |
| GC1024-Frozen | 0.582610 | 0.816246 | 0.018533 | 2509 |
| FPSA1024-Frozen | 0.585773 | 0.810726 | 0.017728 | 2509 |
| G1024-B3 | 0.750556 | 0.365931 | 0.132957 | 2509 |
| GC1024-B3 | 0.716832 | 0.302839 | 0.263497 | 2509 |
| FPSA1024-B3 | 0.720852 | 0.302050 | 0.256245 | 2509 |

## Protocol and provenance

- Stage A and Stage B: `SEALED_COMPLETE`; exactly 3,815 events and 114 queried feedback events. Stage B ran CPU-only after Stage A sealing. No masks or unqueried TEST labels entered Stage A.
- R5 queried feedback receipts matched all R6 query positions, event identities and labels after R6 Stage A ended.
- Parent `c993cdc1ec933e0be0224dd72b18d39435b6aaf1`; branch `calibration-stage0-contract-20261029`.
- Executed R6 runner SHA-256: `3d63520f2462f525372358b935e06d3395b77e687fb149ba696e4ed94e7958e`; executed CPU test SHA-256: `5184f49ed64f3663f59cb58496300fa1bc77ce92850fc02de4075b7683df56b8`.
- Final committed R6 runner SHA-256: `4949460429133debe518e7a0ceba3d741cdb2da572632a31ed960ce5b756df6e`; final committed CPU test SHA-256: `6480f4463c6531d92eac6f6966c45a4d80fae98bbb9aeed67d4d74179f2e24da`. The post-experiment runner change corrects report-only capacity-divergence wording and adds the feasibility conclusion; it does not change scoring, state updates or Stage B metrics. One associated synthetic formatter regression was added.
- CPU tests before execution: `PASS: 41 passed, 4 warnings in 5.27s`; final post-execution regression: `PASS: 42 passed, 4 warnings in 6.20s`.
- 256-event GPU smoke: `GPU_SMOKE_PASS_AFTER_PRESERVED_FAILURES` in 3 attempt(s); full run: `STAGE_A_AND_B_COMPLETE`.
- Smoke-attempt record: The two earlier failed attempts remain immutable, and both supervisors confirmed child-tree exit. Attempt 1 failed before model/scorer initialization because the parent process had hidden CUDA visibility from the child. Attempt 2 failed before stream event scoring or label disclosure because the runner incorrectly expected a 224x224 C64 RGB cache while the sealed R5 cache stores 1024x1024 RGB inputs. The corrected third attempt passed all smoke checks.
- Source fingerprints:
- `experiments/calibration/controllers.py`: `25fbdfc9a92e8bfb6fe983185dca3cdd5ed24d804c28838b4797e0ae8670cf7d`
- `experiments/calibration/profiles.py`: `7ff59b6a3783bf4061ab75c6b0782b23a0f6e456b40457a022e5d0de13d6f7a5`
- `experiments/calibration/protocol.py`: `16b39ead3975bf442047737bc5746814ef9697cd0a9962464ee388bbfc52da60`
- `experiments/calibration/query_tape.py`: `ed6f77c2a4de1bd05969ae3c09f2c98c13a744481ca436f80e4633bd6c412c6d`
- `experiments/calibration/r1_matched_correction_pilot.py`: `3208e2c52277ad76351eb86968e362aa315effe9e3c734bfe2b6ff7bab0db448`
- `experiments/calibration/r1_runtime_supervisor.py`: `e6254a82d8dd24156cea65380b41b04cb1372744da64a085043aa6dd74df5bab`
- `experiments/calibration/r2_exp2a_c1_strength.py`: `4b798dcb1472e6dd630f58b3e62f3782dfb302f7579e942fcdb399885a5f33d3`
- `experiments/calibration/r2_feedback_repair.py`: `92fe290c4c7ea5cbb4da53b925b97ae58b8fa6c66e0ab98ca86895bee48fb0f2`
- `experiments/calibration/r3_fullstream_2x2.py`: `0a67d893df6fe73399726f4cc60caa5967a6e3c2eb93535f23f7089350986741`
- `experiments/calibration/r4_fpsa_audiojack_runner.py`: `c7878683439e50a2ffe9f1bd386fa2c29d1b93f01b214e3d3bd8fbc00053a610`
- `experiments/calibration/r4_fpsa_codebook.py`: `9bc513fd3e3553c3f4693da69096a497b7b07c47e841f82618f3c72b019355f1`
- `experiments/calibration/r5_audiojack_b3_fpsa_runner.py`: `bb07c3a9bef677c3321a4c384af04d459185a5ee801d91f51aabe9058d6d73cb`
- `experiments/calibration/stage05.py`: `4b8aae252f5f1e096b127906794064da269aa79f7cd5514f6f738cda585c017e`
- `experiments/calibration/stage1a_p12_label_provider.py`: `4e51255ae894c2d6ffcc7142e8fd59d9714df94b18ed26ac996528adb163ceaf`
- `experiments/p0_protocol_audit.py`: `b83918ec5aceb3ef63cca942fc9fb0e241559d9672d20bed422b15fcd48135e7`
- `experiments/risk_safe/native_bridge.py`: `9575214ea8d14ebe32a537fa15afa332446b41d66c1c8c85f890d2251eaef8f7`
- `experiments/v02_memory.py`: `3c4e247e3a41dc9acc8147a35aa892cf3f3571aca85beb4169809bd70ad48ae0`
- `test_phase3_AIF_simplified_v22.py`: `bfc871d5a9a32b2f5bdb9d7aab5177e702dec2e1110a6632f20bd13c20fcba30`
- `vmb_visionad_new.py`: `bd6545bf5ff63acce01792e5d42aa964c88e73fb59670febeb55f2d1139a409a`
- Runtime profile matched R5: `stage05_set_seed_0_math_sdp_v1`; DINO checkpoint, loaded model, hub source tree and explicit SDP settings matched the sealed R5 receipt.
- Private scores, labels, caches and manifests remain untracked under `outputs/calibration/r6_capacity1024_audiojack_20261102_a1/`.
- Output limit: 4,294,967,296 bytes per run subdirectory, with 65,536 bytes reserved for failure metadata. Automatic restart is disabled.

One local commit contains only the R6 runner, focused tests, this report and the safe aggregate. No push performed.
