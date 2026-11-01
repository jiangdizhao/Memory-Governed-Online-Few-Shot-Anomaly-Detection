# R5-AUDIOJACK-B3-FPSA — Results

**Status:** COMPLETE. Retrospective exposed development-class validation; not independent validation or production-readiness evidence.

## Primary endpoint

Prequential Balanced Accuracy across all 3,815 audiojack events in frozen TEST JSON order. Physical chronology is not asserted. Query tape: 114 events (2.9882%); first anomalous queried feedback at position 1305.

| Arm | BA | FPR | FNR | Accuracy | TP | TN | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| G-Frozen | 0.541820 | 0.002779 | 0.913580 | 0.687811 | 112 | 2512 | 7 | 1184 |
| GC-Frozen | 0.537400 | 0.001588 | 0.923611 | 0.685190 | 99 | 2515 | 4 | 1197 |
| FPSA-Frozen | 0.593792 | 0.010719 | 0.801698 | 0.720577 | 257 | 2492 | 27 | 1039 |
| G-B3 | 0.710291 | 0.134974 | 0.444444 | 0.759895 | 720 | 2179 | 340 | 576 |
| GC-B3 | 0.717037 | 0.135371 | 0.430556 | 0.764351 | 738 | 2178 | 341 | 558 |
| FPSA-B3 | 0.683282 | 0.148869 | 0.484568 | 0.737090 | 668 | 2144 | 375 | 628 |

## Required decision contrasts

| Contrast | Δ BA | Δ FPR | Δ FNR |
|---|---:|---:|---:|
| FPSA-B3 − FPSA-Frozen | +0.089490 | +0.138150 | -0.317130 |
| FPSA-B3 − GC-B3 | -0.033755 | +0.013497 | +0.054012 |
| FPSA-B3 − G-B3 | -0.027009 | +0.013894 | +0.040123 |
| GC-B3 − G-B3 | +0.006746 | +0.000397 | -0.013889 |

All differences are left arm minus right arm. The query tape and outcomes are retrospective development data. Thresholds do not enter Greedy memory or correction updates.

Source Group paired 95% descriptive bootstrap intervals for ΔBA (1,000 resamples): FPSA-B3−FPSA-Frozen `[+0.074599, +0.105133]`; FPSA-B3−GC-B3 `[-0.047827, -0.020021]`; FPSA-B3−G-B3 `[-0.041251, -0.011970]`; GC-B3−G-B3 `[-0.002277, +0.015156]`. These intervals are not independent confirmation evidence.

## Temporal windows and post-1305

Worst defined window across the seven complete 512 event windows and the partial 231 event tail. Here the worst defined window for each arm is a complete window:

| Arm | Worst complete 512 event BA | Window | Class counts (normal/anomaly) |
|---|---:|---|---|
| G-Frozen | 0.527201 | [2048, 2560) | 271/241 |
| GC-Frozen | 0.524896 | [2048, 2560) | 271/241 |
| FPSA-Frozen | 0.561572 | [1536, 2048) | 246/266 |
| G-B3 | 0.671013 | [2048, 2560) | 271/241 |
| GC-B3 | 0.665240 | [2048, 2560) | 271/241 |
| FPSA-B3 | 0.635539 | [3072, 3584) | 240/272 |

The tail `[3584, 3815)` contains 113 normal and 118 anomalous events; its BA is 0.533898 (G-Frozen), 0.533898 (GC-Frozen), 0.627119 (FPSA-Frozen), 0.711789 (G-B3), 0.742763 (GC-B3), and 0.675529 (FPSA-B3). Any single-class window has undefined BA. The aggregate includes all eight windows and all post-1305 metrics. Post-1305 BA/FPR/FNR are: G-Frozen 0.541764/0.003223/0.913249; GC-Frozen 0.537838/0.001612/0.922713; FPSA-Frozen 0.590874/0.020145/0.798107; G-B3 0.692942/0.174053/0.440063; GC-B3 0.696422/0.180500/0.426656; FPSA-B3 0.655121/0.207897/0.481861.

## B3 thresholds and state updates

| Arm | Initial threshold | Final threshold | State updates | Threshold movements |
|---|---:|---:|---:|---:|
| G-B3 | 0.566524 | 0.507892 | 74 | 36 |
| GC-B3 | 0.566524 | 0.485110 | 79 | 71 |
| FPSA-B3 | 0.566524 | 0.548802 | 111 | 86 |

## C64 rescoring, runtime, and native state provenance

- Initial C64 parity: all 64 native scores matched the frozen receipt for G, GC, and GC-FPSA; max absolute error was 0 for each.
- Changed-state cache/native parity: one representative protected C64 reference was rescored per changed arm at the first changed state; max absolute error was 0 for G, GC, and GC-FPSA. All 64 encoded feature caches also passed byte-exact round-trip checks.
- C64 rescore calls / rounds / seconds: G 4,738 / 74 / 65.841; GC 5,058 / 79 / 46.025; GC-FPSA 7,106 / 111 / 63.113.
- B3 state update opportunities were 74 / 79 / 111 for G / GC / GC-FPSA; threshold movements were 36 / 71 / 86.
- Full Stage A runtime: 677.719 seconds within the supervised child (686.890 seconds including supervisor overhead). CPU Stage B child runtime: 4.688 seconds. The successful 256-event smoke child took 133.968 seconds.
- Runtime profile: DINOv2 on NVIDIA GeForce RTX 3090 Ti, seed 0, deterministic algorithms on, TF32 off, Flash and memory-efficient SDP off, math SDP on. Checkpoint SHA-256: `36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51`; loaded model state SHA-256: `1f26464ba4055403e550f1d5c1333fb012631d98661df6993cabea730d17e3f9`; hub source tree SHA-256: `b7c0037f90671c4932393bd56de8c8a205b29a6eb22dea563b6c8a88c471da15`.
- Final fixed Support digest: `837fa5153e852b617e1d0ecf563509bca037f5c0cab29c48b06befd21ce02673`. Final full scorer-state digests: G `fc768e2ba7306abd008efaff89d225666c35d1f9373dc313f513f3c99e30de28`; GC `d49ee6d4f6bd1ba2ddee36290f2a2ae1754e0f3fb4d3c9786f9c6c62ddc3a0ad`; GC-FPSA `2d0bd5e5ad857aec683fe21e4d19c166f17a92d5ac41b76e4f43d0a09a65ca89`.

## Parity and work performed

- R4 G/GC/GC-FPSA numerical parity: 3815 events; maximum absolute errors G=0, GC=0, FPSA=0.
- R3 G-only common Greedy writer parity: 74 write events, 458 accepted patches, and 150 skipped patches; all matched the frozen reference. Final GC occupancy was 1 positive / 255 negative; GC-FPSA was 91 / 165. FPSA quota totals were 365 borrow opportunities, 93 cross-sign replacements, and 272 rejections.
- Initial 64-reference C64 parity and changed-state cached/native representative parity passed. C64 rescore calls, rounds, elapsed time, and B3 state-update counts are recorded in the aggregate.
- Stage A score decisions were sealed before offline outcomes were read. Queried labels were disclosed only after each current score row flushed. No TEST masks were opened.
- B3 implementation: unchanged `StateConsistentQuantile`, delta=0.01, min_normals=16; three independent complete scorer-state streams.
- Source-group paired bootstrap intervals are descriptive only. Seed=20261102; 1000 resamples.

## Deployment interpretation

The hypothetical 1% FPR / 5% FNR scenario is illustrative and is not a factory-approved risk budget:

| Arm | Meets hypothetical FPR ≤1% | Meets hypothetical FNR ≤5% |
|---|---|---|
| G-Frozen | Yes | No |
| GC-Frozen | Yes | No |
| FPSA-Frozen | No | No |
| G-B3 | No | No |
| GC-B3 | No | No |
| FPSA-B3 | No | No |

B3 substantially improves the frozen FPSA BA but increases FPR; FPSA-B3 is below both G-B3 and GC-B3 in BA, with higher FPR and FNR than those two alternatives. This run uses one already-exposed development category, an artificial fixed label-blind 3% query tape, frozen JSON ordering, four deployed supports plus 64 additional protected TRAIN normals, and no independent production-domain data. It does not establish factory readiness or reproduce the original paper's 30-class controller protocol.

## Provenance

- Parent source commit: `dea81e097ec0bd5994eb5f59a2009c3936f320b6`
- Branch: `calibration-stage0-contract-20261029`
- Executed R5 runner SHA-256 (bound by the run manifest): `69d34f74dd81e43834a08028407211ae48e27451c2c388da88fc86618608c939`
- Committed R5 runner SHA-256: `bb07c3a9bef677c3321a4c384af04d459185a5ee801d91f51aabe9058d6d73cb`. After execution, only the extra blank line at EOF was removed to make `git diff --cached --check` clean; no executable code changed.
- CPU test file SHA-256: `cceb73be2ec9bba6da5ea2bad2b14828cf792abed9650f2431a52faba49b0b62`
- CPU tests: `PASS: 25 passed, 4 warnings in 21.38s before execution; final audit rerun 25 passed, 4 warnings in 21.90s`
- 256 event GPU smoke: `GPU_SMOKE_PASS_NOT_EVALUATED`; output `r5_audiojack_b3_fpsa_20261102_a1_smoke_retry1`
- Failed smoke attempt `r5_audiojack_b3_fpsa_20261102_a1_smoke` retained unchanged after the first query-history append contract error; retry used the fresh namespace `r5_audiojack_b3_fpsa_20261102_a1_smoke_retry1`.
- Full output namespace: `outputs/calibration/r5_audiojack_b3_fpsa_20261102_a1`
- Full score rows SHA-256: `7592cc0e08f506fdb565885afa40466935254148459ecaa5429750ca45724945`
- Sealed output aggregate SHA-256: `f8de502f5b7b35e97835df3efd36431ebeeabe5ab09a57db87eb023877d01480`
- Stage A seal SHA-256: `56cdfb2ff9c20f3a8fa8b10ef370e821c34cdfeb54c632983ffbed69f2dd60ec`
- Complete run SHA-256: `39b23ad23d81ee89a28ab5c863d8a44b685f8f8ef6c759c381d3f790ad979178`
- Output byte budget: 4,294,967,296 bytes with 65,536 bytes reserved for failure metadata.

Local commit contains this report, aggregate, runner, and focused tests only. No push performed.
