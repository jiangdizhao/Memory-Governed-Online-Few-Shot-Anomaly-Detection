# R3-FULL-2X2 — Full-Stream Dual-Memory Complementarity Evaluation

## Classification and status

**Status: COMPLETE — `RETROSPECTIVE_FULL_STREAM_DEVELOPMENT_EVALUATION`.** Earlier development outcome summaries were already known. These results are not independent, blind, or confirmation evidence. The four arms, memory rules, correction policy, query tape and evaluation contrasts were frozen for this run; no post-result parameter search was performed.

The Stage A online pass sealed all 15,295 event scores before Stage B read the full TEST outcome projection. Stage A decoded only the current legal Q-A-TS queried label after fsync of all four current event scores. TEST masks were not opened. The order is the frozen prequential JSON order; timestamps are retained as metadata and are not asserted to represent physical chronology.

## Run identity and runtime

- Run ID: `r3_fullstream_2x2_20261102_a7`
- Source parent: `b23f6956b6dd3e8359d27050f3fe27713ef3a69e` on `calibration-stage0-contract-20261029`
- Local commit: `same scoped commit as this report`
- GPU: NVIDIA GeForce RTX 3090 Ti; CUDA runtime 11.7.
- Runtime controls: `{"cublas_workspace_config": ":4096:8", "cuda_device_index": 0, "cuda_device_name": "NVIDIA GeForce RTX 3090 Ti", "cuda_matmul_allow_tf32": false, "cuda_runtime_version": "11.7", "cudnn_allow_tf32": false, "cudnn_benchmark": false, "cudnn_deterministic": true, "deterministic_algorithms": true, "profile_id": "stage05_set_seed_0_math_sdp_v1", "python_version": "3.11.14", "sdp": {"flash_enabled": false, "math_enabled": true, "memory_efficient_enabled": false}, "seed": 0, "torch_version": "2.0.1+cu117"}`.
- Stage A stream encodings: 15,295 — exactly one encoder invocation per TEST event.
- Support augmentation encodings: 64 total across the four categories.
- Supervised Stage A wall time: 2223.4 seconds; Stage B CPU time: 16.4 seconds.
- Peak CUDA allocated/reserved: 1,714,062,336 / 14,287,896,576 bytes.
- Local private output: `outputs/calibration/r3_fullstream_2x2_20261102_a7/` (15,570,740 bytes; 4 GiB application bound; 64 KiB failure reserve).
- Recovery history: Final run `r3_fullstream_2x2_20261102_a7` passed a fresh 128-event GPU smoke, reused all four completed category trajectories from `a3`, and completed CPU Stage B. Earlier `a1`/`a2` smokes stopped before DINO/event scoring after a CPU-only dependency hid CUDA; `a3` sealed all categories but exposed a final-record whitespace error in Stage B; `a4` was an isolated CPU-only label-reader diagnosis using copied sealed category receipts and wrote no aggregate; `a5` exposed an overly strict runner-identity comparison; `a6` completed Stage B but hit a parent wall-time bookkeeping error. These namespaces were preserved. No full-stream category was re-encoded or partially resumed in the final run.
- Stage A trajectory runner SHA-256: `06df65ef04d2c63468becd3fb11939f9859d4a327b5f381c22c68c6b3d26c991` (`a3`); Stage B and finalizer runner SHA-256: `5506530f5be1e2768e0426c735543954f39a8de1b16c063ad462b490197aad8a` (`a7`). The exact prior runner SHA was accepted only after all frozen dependencies, experiment identity fields other than that SHA, event counts, score rows, query positions and feedback receipts matched. Stage A scoring logic and its sealed score rows did not change.
- The 120-minute Windows Job Object supervisor was used separately for each category child and Stage B; the parent enforced an eight-hour total ceiling including tests, smoke and reused categories (2287.2 seconds charged). No automatic restart of an incomplete category occurred.
- The application byte accountant provides no OS/filesystem hard quota. Its pre-write accounting, unique namespace and disk-space preflight were active.

## Frozen inputs and method configuration

- DINO source tree SHA-256: `b7c0037f90671c4932393bd56de8c8a205b29a6eb22dea563b6c8a88c471da15`; checkpoint SHA-256: `36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51`; loaded state SHA-256: `1f26464ba4055403e550f1d5c1333fb012631d98661df6993cabea730d17e3f9`.
- Support seed: 0; four deployed Support images with four augmentation views each. The Support bank used the native structured fixed-bank digest; C64's 64 disjoint normal TRAIN references supplied only each category's initial threshold and sigma.
- Fixed-bank structured SHA-256 by category: `{"audiojack": "837fa5153e852b617e1d0ecf563509bca037f5c0cab29c48b06befd21ce02673", "regulator": "b162fe852cdc4c7dd5c5a25e3560263427cde588d42f12364bac0d3c79df48bb", "rolled_strip_base": "12c1dcacba90b1f0c576480b0afddaabf2d80466845ac61f4346aebfd968d4f3", "switch": "8a710d028509350568e976bc3ede66aa59603499cbea525bdd8579322b1a1558"}`.
- Support image and Source Group identity SHA-256 by category: `{"audiojack": "0f5de47d2465fac92317207399e607bbb08990615407198d76814a8a98b24f11", "regulator": "4a3a651ce886b217bb64e40b74719a156775a239eeb950c33dbc2dae0e3c1f55", "rolled_strip_base": "c9b3e99d9c31a7d335a6497c1a5a79c2b4310a5b235c66ae2da05d55d6267bc5", "switch": "6f831e8245fa4d78569c53b4c95624c420aa50dc5511c061eefdf3643409127e"}` / `{"audiojack": "d6f69aea82610f47a6a27d51653c4ef52673b33c4e16c5dd6c86f184ff208c06", "regulator": "263491265a0a825f96be4dd99b0f988df7f09140c98ce21545125881326a1225", "rolled_strip_base": "78eaf67949733f3601ea675769652b42b96da01cda5bac67c12e71d0845dbc38", "switch": "6bbcff1cb06314d26ffcfe9e3717bc4c27c5e735521e8ae196c3a8d7fca055cb"}`.
- C64 image and Source Group identity SHA-256 by category: `{"audiojack": "8093ef50e911bea3a5ff8e6e2375b188b829c561c29c614895de20a8f1600809", "regulator": "70870d11e1e7a3d51c7658748fdf3efb4481cab990af56cd9b71146f69a3fcbe", "rolled_strip_base": "62693ed27c1c607be5691ab423f95e739ff6adb40c550b4fd1d60af04979ea1c", "switch": "561164a4f42bf00394539ebcd021e3d62fc490a7698633b8008d273b5cf7fcbc"}` / `{"audiojack": "88f61b52b1678d3f154907b65115896632e577c61c3da32992c4fcd20405b321", "regulator": "b67740f3dd88aff13d0615a6da58e0ef0db11a606bed8f3af4a60787e9fcc27d", "rolled_strip_base": "1cd93b91415ba4fb582d2057bf42d0efaab84a5b2273c7f02559b7ffd9f65dc8", "switch": "b6d27657b707f22a43ee68ee63e3cb61676ba21c31a2f007d2feadbbf965719f"}`; C64 score tape SHA-256: `{"audiojack": "ad6edc2677809f1f0c455f80ea8aa002e9123fd9f19711936b5c8501696fdd93", "regulator": "0f6fdb241810bb3943ffd93d8fa5b9e2005c2eb6a1e6e9d53d99dcde5235f9f2", "rolled_strip_base": "774c02efb5744337fc881e3f1469943e8780fb68dbccedac161c6bfa8a5bddc6", "switch": "91fefe78fe54a2d933770b120df08894b4c8e6f69297d0cd2d2ac7b950e0aa9e"}`.
- Existing Stage 0.5 Q-A-TS-v1 tape: 3% rate, seed 37, 16 bins, official Source Group unit. Tape SHA-256 by category: `{"audiojack": "cb1b8b4342b23dc520a45ef72c014ceb3ffa1e976046aef726fcc0aaa45504d5", "regulator": "fe4dae9b355bad1ec7cd836293ee445b353f1a6168fb85bac997601355139ca5", "rolled_strip_base": "c5068752da92dfe17ad198d1103e5b18f8433d6bed057007a13f712ccef3e8a2", "switch": "b48928b1b547d68f2978769191662362c8fd5f57bd40d39b0126e5800648cc16"}`.
- First-512 EXP2A score reference SHA-256: `d6c59dfef488d2a0a9becea6f77713c75bd60390d19b61e3b9509b93a7e0dfec`. Parity was checked on G vs EXP2A C0 and GC vs EXP2A C1-Full at atol=1e-05, rtol=0.0001.
- S/SC share only the immutable initial Support representation; SC correction state is independent. G/GC share the same Greedy normal-memory trajectory; GC correction never gates memory writes. C1-Full uses each branch's own base map, Top-32, native dN, base outcome and beta update.
- No new correction kernel, beta policy, admission gate, query controller, threshold sidecar, label-based query selection, or online threshold adaptation was introduced.
- Exact frozen C1-Full configuration: `{"beta_max": 0.2, "beta_min": 0.02, "beta_scale": 0.5, "capacity": 256, "corr_replace_margin": 0.0, "corr_weight": 1.0, "implementation": "frozen v22 SignedCorrectionCodebook / C1-Full", "kappa0": 1.0, "kappa_inc": 0.1, "kappa_max": 5.0, "keep_defect_bank": false, "merge_eps": 0.05, "min_dN_for_store_fn": 0.12, "min_dN_for_store_fp": 0.1, "outcome_weights": {"FN": 1.2, "FP": 1.3, "TN": 0.6, "TP": 0.8}, "patch_beta_cap": 1.25, "patch_beta_floor": 0.75, "patch_beta_power": 1.0, "patch_weighted_beta": true, "radius0": 0.1, "radius_max": 0.25, "radius_min": 0.05, "sign_aware_retention": true, "sign_aware_write": true}`; each arm uses its own codebook, source-group history, native Top-32 and pre-event outcome.

### Historical fingerprint reconciliation

Stage 0.5's historical score manifest records DINO source-tree SHA `1de67887ca330bdfaad8ccd45128d2b9313d63aa707bbe8e4f38310e52005b24`; the later verified R1/EXP2A local source used here records `b7c0037f90671c4932393bd56de8c8a205b29a6eb22dea563b6c8a88c471da15`. The local R1 receipt and checkpoint were verified, and the actual rebuilt structured fixed-support digests matched the Stage 0.5 receipts for all four categories. Stage 0.5 source hashes for `stage05.py` and `stage05_runner.py` differ from current files; the archived seed-37 tape digest and event/source-group identities were validated directly, and no new tape was sampled.

Exact Stage 0.5 source hash mismatches (frozen -> current):

- `experiments/calibration/stage05.py`: `af1ef435addfe2f08cb2e40a1d141b617fb15c852ed469c8aa7062663ccf8626` -> `4b8aae252f5f1e096b127906794064da269aa79f7cd5514f6f738cda585c017e`
- `experiments/calibration/stage05_runner.py`: `1e40d9c5e48c252eef3d96686d2a315c4a87f1f325130a6c2961f7cc41cafaea` -> `1d110e1ca813a064767ebeab597b609b3d863f8afd4c8085b06948ad88347e14`

## Full-stream event and query coverage

| Category | Events / encodings | GPU seconds | Queries | Normal query feedback | Anomalous query feedback | First anomaly | First queried anomaly | Greedy writes | SC/GC final occupancy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| audiojack | 3815 / 3815 | 537.9 | 114 | 74 | 40 | 1231 | 1305 | 74 | 256/256 |
| regulator | 3855 / 3855 | 549.9 | 116 | 81 | 35 | 1 | 116 | 81 | 256/256 |
| switch | 3820 / 3820 | 546.1 | 115 | 54 | 61 | 0 | 55 | 54 | 256/256 |
| rolled_strip_base | 3805 / 3805 | 588.8 | 114 | 33 | 81 | 0 | 35 | 33 | 256/256 |

Total query feedback was 459 events (242 normal, 217 anomalous). Each query was one unique Source Group and was delivered identically to all four arms.

## Primary metrics and factorial effects

| Category | Arm | AUROC | AP | FPR | FNR | Balanced accuracy | TN/FP/FN/TP |
|---|---|---:|---:|---:|---:|---:|---:|
| audiojack | S | 0.618178 | 0.459562 | 0.119095 | 0.780093 | 0.550406 | 2219/300/1011/285 |
| audiojack | SC | 0.652784 | 0.515684 | 0.023819 | 0.870370 | 0.552905 | 2459/60/1128/168 |
| audiojack | G | 0.727906 | 0.616682 | 0.002779 | 0.913580 | 0.541820 | 2512/7/1184/112 |
| audiojack | GC | 0.716921 | 0.608758 | 0.001588 | 0.923611 | 0.537400 | 2515/4/1197/99 |
| audiojack | AUROC GC−G | -0.010984 | — | — | — | — | CI [-0.017904, -0.004737] |
| audiojack | AUROC GC−SC | 0.064137 | — | — | — | — | CI [0.042506, 0.085864] |
| audiojack | AUROC SC−S | 0.034606 | — | — | — | — | CI [0.024685, 0.044687] |
| audiojack | AUROC G−S | 0.109728 | — | — | — | — | CI [0.088420, 0.131334] |
| audiojack | AUROC interaction I | -0.045590 | — | — | — | — | CI [-0.057759, -0.033450] |
| regulator | S | 0.602720 | 0.330169 | 0.122523 | 0.916667 | 0.480405 | 2435/340/990/90 |
| regulator | SC | 0.679796 | 0.568592 | 0.051532 | 0.650000 | 0.649234 | 2632/143/702/378 |
| regulator | G | 0.784193 | 0.650541 | 0.001441 | 0.953704 | 0.522427 | 2771/4/1030/50 |
| regulator | GC | 0.742988 | 0.651214 | 0.005766 | 0.761111 | 0.616562 | 2759/16/822/258 |
| regulator | AUROC GC−G | -0.041205 | — | — | — | — | CI [-0.059062, -0.023690] |
| regulator | AUROC GC−SC | 0.063191 | — | — | — | — | CI [0.045039, 0.081048] |
| regulator | AUROC SC−S | 0.077076 | — | — | — | — | CI [0.057965, 0.095922] |
| regulator | AUROC G−S | 0.181473 | — | — | — | — | CI [0.166492, 0.195420] |
| regulator | AUROC interaction I | -0.118281 | — | — | — | — | CI [-0.138341, -0.098345] |
| switch | S | 0.793275 | 0.832200 | 0.063527 | 0.552658 | 0.691908 | 1710/116/1102/892 |
| switch | SC | 0.807340 | 0.845623 | 0.215225 | 0.328987 | 0.727894 | 1433/393/656/1338 |
| switch | G | 0.887668 | 0.902096 | 0.013143 | 0.619860 | 0.683498 | 1802/24/1236/758 |
| switch | GC | 0.873280 | 0.895306 | 0.080504 | 0.380642 | 0.769427 | 1679/147/759/1235 |
| switch | AUROC GC−G | -0.014388 | — | — | — | — | CI [-0.021035, -0.008424] |
| switch | AUROC GC−SC | 0.065940 | — | — | — | — | CI [0.055215, 0.076641] |
| switch | AUROC SC−S | 0.014065 | — | — | — | — | CI [0.006156, 0.022220] |
| switch | AUROC G−S | 0.094393 | — | — | — | — | CI [0.084237, 0.105313] |
| switch | AUROC interaction I | -0.028453 | — | — | — | — | CI [-0.036516, -0.020702] |
| rolled_strip_base | S | 0.900086 | 0.946924 | 0.130469 | 0.258614 | 0.805459 | 1113/167/653/1872 |
| rolled_strip_base | SC | 0.934755 | 0.967814 | 0.228125 | 0.089109 | 0.841383 | 988/292/225/2300 |
| rolled_strip_base | G | 0.971073 | 0.984724 | 0.017188 | 0.314455 | 0.834179 | 1258/22/794/1731 |
| rolled_strip_base | GC | 0.970704 | 0.984699 | 0.082031 | 0.103762 | 0.907103 | 1175/105/262/2263 |
| rolled_strip_base | AUROC GC−G | -0.000369 | — | — | — | — | CI [-0.002628, 0.001754] |
| rolled_strip_base | AUROC GC−SC | 0.035949 | — | — | — | — | CI [0.029062, 0.043770] |
| rolled_strip_base | AUROC SC−S | 0.034669 | — | — | — | — | CI [0.029457, 0.039980] |
| rolled_strip_base | AUROC G−S | 0.070987 | — | — | — | — | CI [0.061559, 0.081518] |
| rolled_strip_base | AUROC interaction I | -0.035038 | — | — | — | — | CI [-0.040181, -0.029892] |

| Category | Metric | GC−G | GC−SC | SC−S | G−S | Interaction I |
|---|---|---:|---:|---:|---:|---:|
| audiojack | auroc | -0.010984 | 0.064137 | 0.034606 | 0.109728 | -0.045590 |
| audiojack | average_precision | -0.007924 | 0.093073 | 0.056122 | 0.157120 | -0.064046 |
| audiojack | balanced_accuracy | -0.004420 | -0.015505 | 0.002499 | -0.008586 | -0.006919 |
| regulator | auroc | -0.041205 | 0.063191 | 0.077076 | 0.181473 | -0.118281 |
| regulator | average_precision | 0.000673 | 0.082622 | 0.238422 | 0.320372 | -0.237749 |
| regulator | balanced_accuracy | 0.094134 | -0.032673 | 0.168829 | 0.042022 | -0.074695 |
| switch | auroc | -0.014388 | 0.065940 | 0.014065 | 0.094393 | -0.028453 |
| switch | average_precision | -0.006790 | 0.049683 | 0.013424 | 0.069897 | -0.020214 |
| switch | balanced_accuracy | 0.085929 | 0.041533 | 0.035987 | -0.008409 | 0.049942 |
| rolled_strip_base | auroc | -0.000369 | 0.035949 | 0.034669 | 0.070987 | -0.035038 |
| rolled_strip_base | average_precision | -0.000025 | 0.016885 | 0.020890 | 0.037800 | -0.020916 |
| rolled_strip_base | balanced_accuracy | 0.072925 | 0.065720 | 0.035924 | 0.028720 | 0.037000 |

All five requested AUROC contrasts are in the machine-readable aggregate: GC−G, GC−SC, SC−S, G−S and factorial interaction I. The same contrasts for AP and balanced accuracy are reported as point estimates; the aggregate also carries paired intervals for them. The interaction was calculated inside each of 1,000 paired Source Group bootstrap replicates, with the same sampled groups reused across all four arms. These intervals are conditional on the four fixed development streams; they do not make the categories independent samples.

## First-512 EXP2A parity

| Category | Events checked | G vs C0 max absolute error | GC vs C1-Full max absolute error | Numeric mismatches |
|---|---:|---:|---:|---:|
| audiojack | 512 | 0 | 0 | 0 |
| regulator | 512 | 0 | 0 | 0 |
| switch | 512 | 0 | 0 | 0 |
| rolled_strip_base | 512 | 0 | 0 | 0 |

## Error repair, harm and ranking

For both GC-vs-G and GC-vs-SC, the aggregate includes FN→TP and FP→TN repairs, TP→FN and TN→FP harms, the base/candidate confusion matrices, and verified transition reconstruction. It separately reports combination-only repairs and harms by normal/anomalous label. Ranking diagnostics count normal/defect pair credit changes for all pairs and Source Group-disjoint pairs; ties receive 0.5 credit, and denominators are explicit.

### Transition counts

| Category | Comparison | FN→TP | FP→TN | TP→FN | TN→FP |
|---|---|---:|---:|---:|---:|
| audiojack | GC_vs_G | 9 | 4 | 22 | 1 |
| audiojack | GC_vs_SC | 9 | 57 | 78 | 1 |
| regulator | GC_vs_G | 211 | 1 | 3 | 13 |
| regulator | GC_vs_SC | 21 | 134 | 141 | 7 |
| switch | GC_vs_G | 480 | 2 | 3 | 125 |
| switch | GC_vs_SC | 69 | 289 | 172 | 43 |
| rolled_strip_base | GC_vs_G | 533 | 1 | 1 | 84 |
| rolled_strip_base | GC_vs_SC | 10 | 199 | 47 | 12 |

| Category | Outcome group | GC-only combination repair | GC-only combination harm |
|---|---|---:|---:|
| audiojack | normal | 0 | 1 |
| audiojack | anomalous | 5 | 18 |
| regulator | normal | 0 | 6 |
| regulator | anomalous | 21 | 2 |
| switch | normal | 0 | 42 |
| switch | anomalous | 62 | 0 |
| rolled_strip_base | normal | 0 | 12 |
| rolled_strip_base | anomalous | 9 | 1 |

### Fixed-stream quartiles

| Category | Quartile | Events | Anomaly proportion | Queries | Queried normal/anomaly | GC−G mean score | GC−SC mean score | G/GC AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| audiojack | 1 | 953 | 0.000000 | 30 | 30/0 | -0.016477 | -0.026755 | undefined/undefined |
| audiojack | 2 | 954 | 0.343816 | 28 | 17/11 | -0.018968 | -0.037483 | 0.785334/0.792817 |
| audiojack | 3 | 954 | 0.483229 | 28 | 15/13 | -0.019923 | -0.039677 | 0.745386/0.749288 |
| audiojack | 4 | 954 | 0.531447 | 28 | 12/16 | -0.021803 | -0.046777 | 0.781167/0.800052 |
| regulator | 1 | 963 | 0.389408 | 29 | 22/7 | 0.009791 | -0.019957 | 0.718317/0.713224 |
| regulator | 2 | 964 | 0.349585 | 29 | 15/14 | 0.041554 | -0.027342 | 0.811703/0.793875 |
| regulator | 3 | 964 | 0.359959 | 29 | 16/13 | 0.048436 | -0.028935 | 0.773030/0.750522 |
| regulator | 4 | 964 | 0.021784 | 29 | 28/1 | 0.046088 | -0.041593 | 0.710852/0.783922 |
| switch | 1 | 955 | 0.707853 | 30 | 8/22 | 0.016410 | -0.003373 | 0.808498/0.833747 |
| switch | 2 | 955 | 0.683770 | 29 | 11/18 | 0.024598 | -0.006752 | 0.835360/0.841922 |
| switch | 3 | 955 | 0.646073 | 28 | 9/19 | 0.029044 | -0.009843 | 0.891357/0.869429 |
| switch | 4 | 955 | 0.050262 | 28 | 26/2 | 0.032370 | -0.025455 | 0.885658/0.862459 |
| rolled_strip_base | 1 | 951 | 0.864353 | 30 | 3/27 | 0.026385 | -0.000864 | 0.920085/0.925404 |
| rolled_strip_base | 2 | 951 | 0.845426 | 28 | 3/25 | 0.027847 | -0.003587 | 0.950503/0.951704 |
| rolled_strip_base | 3 | 951 | 0.876972 | 28 | 2/26 | 0.026465 | -0.006226 | 0.966355/0.968456 |
| rolled_strip_base | 4 | 952 | 0.068277 | 28 | 25/3 | 0.025646 | -0.032270 | 0.892984/0.894077 |

## Interpretation

A four-category GC AUROC mean is defined at 0.825973. Category-specific conditional gains, intervals, false-positive/false-negative trade-offs, repair/harm ledgers and score-by-quartile effects govern the scientific reading. Complementarity and positive factorial interaction are evaluated separately; no positive interaction is required, and no synergy claim is made unless its paired interval excludes zero.

Any AUROC gain accompanied by higher FPR is a ranking improvement with an operating-point cost. Results describe only these four previously exposed development streams and do not establish independent confirmation, population risk, or transfer of a statistical guarantee.

## Provenance, tests and limitations

- Focused CPU tests: 30 passed, 0 failed, 0 skipped under the Windows Job Object with a 180-second ceiling (6.3 seconds). The integrated pre-smoke suite at the execution runner revision also passed all 27 then-current tests; the final three tests cover AST test-counting, supervised test-summary behavior, and conclusion wording.
- Source parent SHA: `b23f6956b6dd3e8359d27050f3fe27713ef3a69e`; Stage A trajectory runner SHA-256: `06df65ef04d2c63468becd3fb11939f9859d4a327b5f381c22c68c6b3d26c991`; Stage B execution runner SHA-256: `5506530f5be1e2768e0426c735543954f39a8de1b16c063ad462b490197aad8a`; execution test SHA-256: `f90181125f49739d1b920b878e3e73cdbc80c704931bec72e3b1d51292cf957b`.
- Final committed report/test-harness source SHA-256: runner `0a67d893df6fe73399726f4cc60caa5967a6e3c2eb93535f23f7089350986741`; tests `66677f60a991066c1de1c42b42feedbdfd91e728d22fffaf0cae1775e070f6ec`. Post-run edits only corrected report confidence-interval coverage, conclusion wording and test-count/provenance bookkeeping; they did not alter the sealed scores or outcome calculations.
- Stage A score seal SHA-256: `7314306b65173bcd707e9f371b0ff61d15951fd399754d9891bc9b33e637c324`; Stage B seal SHA-256: `6a84e0393142083bfeed02d9bb9e736a6a3396fc9d4718b582ebe51447faac3a`; public aggregate SHA-256: `040839b1472426e77bfe731e98c7710ffc559ff2eb56ad8802fe3b28894336a4`.
- Stage A used the restricted queried-position label reader only after the four scores were fsynced. Stage B used the restricted label scanner after complete score sealing; it skipped mask fields and did not open mask images.
- The 4 GiB OutputBudget is an application-level pre-write bound with a reserved failure record and unique output directory. Windows does not provide OS hard storage isolation here; unrelated same-user processes could bypass the accountant.
- DINO/Support numerical fidelity was checked by the structured native fixed-bank digest and first-512 reference parity. Later-stream errors can reflect the frozen prequential state trajectory and are not independent trajectories.
- No second algorithm search was performed. No GPU fallback model, label-driven query selection, per-category filtering, future feedback access or mask access was used.

## Scientific conclusion

The four fixed streams show Case D: GC improved AUROC over SC in every category, but GC AUROC was below G in every category (with the paired interval crossing zero only for `rolled_strip_base`). Thus Greedy contributes beyond Static when Correction is enabled, while Correction has no positive AUROC increment beyond Greedy in these streams. The GC-SC intervals are positive in all four categories. The factorial interaction estimates are negative, and every paired 95% interval is wholly below zero; these results do not support positive interaction.

Greedy-only and Static-plus-Correction each improved AUROC over Static across all four categories. Operating-point costs remain: GC had higher FPR than G for regulator, switch and rolled_strip_base, while GC AUROC did not exceed G in any category. The aggregate records all five paired AUROC intervals, the corresponding AP and balanced-accuracy contrasts, confusion transitions, combination-only repairs/harms, pair-ranking diagnostics, and the four fixed-stream quartiles. These are retrospective development results, not independent confirmation.

See `aggregate_results.json` for the complete per-category metrics, confidence intervals, transitions, paired ranking diagnostics, quartiles and update telemetry.
