# R1-S2 — Bounded C64 Native Score Parity Report

**Date:** 2026-10-31
**Main status:** `C64_PARITY_PASS`
**Scope:** one bounded `switch`, support-seed-0 native C64 parity replay

## 1. Executive result

The rebuilt native fixed support bank matched the historical Stage 0.5 structured digest. The current frozen DINOv2/VisionAD scorer then reproduced all 64 Stage 0.5 C64 scores in the same frozen order.

- `BITWISE_SCORE_PARITY`: **true**
- `NUMERICAL_SCORE_PARITY`: **true**
- `SCORE_PARITY_FAILED`: **false**
- Main status: **`C64_PARITY_PASS`**

This establishes numerical compatibility for this fixed support bank, scorer profile, runtime, and 64-image C64 vector. It does not reproduce the full historical Stage 0.5 experiment or establish anomaly-detection performance.

## 2. Git source identity

- Required branch: `calibration-stage0-contract-20261029`
- Starting HEAD verified before implementation: `cf1bf2b72381633663b5b81fe0966c0f0487eda3`
- Implementation commit used for GPU inference: `0bb29966d5b74cda67b5b81fe0966c0f0487eda3`
- The committed tracked worktree was clean for the run.
- GPU output namespace: `outputs/calibration/r1_s2_c64_parity_switch_s0_20261031_c64parity_20261031_a1/`
- External launcher exit code: `0`; no retry occurred.

The GPU execution manifest binds the run to the implementation commit above. The separate documentation commit is not part of the inference source identity.

## 3. Authorization boundaries

The replay used exactly the four frozen switch support images at seed 0 and the 64 registered additional normal C64 references. It used the existing frozen DINOv2 ViT-L/14 checkpoint and original `p0.encode_query()` then `p0.score_encoded()` scoring path. Dynamic Greedy memory was empty and disabled; no Greedy writer was instantiated. Correction memory was disabled and empty, and the defect bank was disabled.

Four support images and 64 C64 images were decoded. No stream image was decoded or scored. No query label, test label, mask, confirmation-class data, or confirmation outcome was accessed. No R1 512-event pilot, R1 dense-map smoke, or R2 evaluation was run.

## 4. CPU identity and input gates

The committed CPU preflight passed before GPU execution:

- Frozen support count/order: four unique support images and official groups; support/C64 groups disjoint.
- Frozen C64 count/order: 64 unique images and official groups. Image-to-group alignment was checked, and the frozen order was bound by SHA-256 `02e3652b5739cf62aaa110e90f4d46d54d0074ee809d4ad0b55b9d330acb4b57`.
- C64 image-set identity: `561164a4f42bf00394539ebcd021e3d62fc490a7698633b8008d273b5cf7fcbc`.
- Official C64 group identity: `b6d27657b707f22a43ee68ee63e3cb61676ba21c31a2f007d2feadbbf965719f`.
- All 68 authorized support/C64 reference paths were present.
- Frozen input checksum, score manifest checksum, score tape manifest row, and C64 score receipt checksum matched. Historical C64 receipt SHA-256: `91fefe78fe54a2d933770b120df08894b4c8e6f69297d0cd2d2ac7b950e0aa9e`.
- Historical scorer profile matched the current locked profile. Profile SHA-256: `1b8e9d96b121e3f1f6c1bbee13ab9f0063facee8fb08ad696dc5f9cb43dbd452`.
- The recorded R1-S1 support-only verification was `SUPPORT_ONLY_VERIFIED` for the same structured fixed-bank digest.

The S1 receipt, failed R1 run manifest and attempt history, and prior R1 engineering report were hash-checked before execution and remained unchanged afterward. The existing Stage 0.5 score receipt also retained its registered checksum.

## 5. Fixed support bank

| Digest | Algorithm | SHA-256 |
|---|---|---|
| Historical expected and rebuilt actual structured digest | `experiments.calibration.native._fixed_bank_digest.v1:path-shape-dtype-bytes` | `8a710d028509350568e976bc3ede66aa59603499cbea525bdd8579322b1a1558` |
| Rebuilt P0 raw prototype diagnostic | `experiments.p0_protocol_audit.fixed_bank_profile.v1:raw-prototype-bytes` | `087feded0d780bbc3a56d45404b4126991b1aeb54f1ae3724bce025caf0d380e` |

The structured digest gate passed before any C64 image was decoded. The single fixed layer contained 8,192 prototypes of shape `[8192, 4096]`, dtype `float16`, 67,108,864 prototype bytes, with configured capacity 25,000. The P0 raw digest was retained independently and was not compared with the structured receipt.

## 6. Checkpoint, source, and runtime identities

The original checkpoint SHA-256 matched the Stage 0.5 score provenance and the R1-S1 support-only receipt:

`36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51`

The current DINO hub Python-source tree matched the historical tree SHA-256 `1de67887ca330bdfaad8ccd45128d2b9313d63aa707bbe8e4f38310e52005b24` using the comparable `stage075a_runner._tree_sha256.v1` contract (sorted relative Python paths, path-length prefix, and file SHA-256 bytes). R1-S1 recorded `b7c0037f90671c4932393bd56de8c8a205b29a6eb22dea563b6c8a88c471da15` using a different JSON path/hash-list contract; those two values are not directly comparable and are not treated as a source mismatch.

The score-time and current fingerprints match for the four core scorer sources:

| Source | SHA-256 |
|---|---|
| `vmb_visionad_new.py` | `bd6545bf5ff63acce01792e5d42aa964c88e73fb59670febeb55f2d1139a409a` |
| `test_phase3_AIF_simplified_v22.py` | `bfc871d5a9a32b2f5bdb9d7aab5177e702dec2e1110a6632f20bd13c20fcba30` |
| `experiments/p0_protocol_audit.py` | `b83918ec5aceb3ef63cca942fc9fb0e241559d9672d20bed422b15fcd48135e7` |
| `experiments/calibration/native.py` | `3a347d65aaf4bfd6b41eff2ea7339d878065bb2e08ed64795f57b0458135dd45` |

The current runtime was Python `3.11.14`, Torch `2.0.1+cu117`, CUDA runtime `11.7`, on an NVIDIA GeForce RTX 3090 Ti, matching the recorded Python/Torch versions and GPU name. The Python executable SHA differs: current `3f3c33da4424d4c52eca5446d7867441a211b7e8f9d7c74e4ea5bc55319fb10a`; historical `725415adc61e93eb69322c8081ce71a2ca19ca28fbb00a50cfacc1697c76b506`. This runtime provenance difference remains recorded despite the matching version strings and exact score-vector parity.

The Stage 0.5 orchestration fingerprints also differ from the current published files: `stage05_runner.py` score-time `1e40d9c5e48c252eef3d96686d2a315c4a87f1f325130a6c2961f7cc41cafaea`, current `1d110e1ca813a064767ebeab597b609b3d863f8afd4c8085b06948ad88347e14`; `stage05.py` score-time `af1ef435addfe2f08cb2e40a1d141b617fb15c852ed469c8aa7062663ccf8626`, current `4b8aae252f5f1e096b127906794064da269aa79f7cd5514f6f738cda585c017e`. The surviving source history records post-score runner revisions for verification/reporting and a later core-source whitespace cleanup. The exact score-time orchestration source blobs are not independently recoverable here, so this remains a historical provenance limitation; it is not a mismatch in the four core scorer sources listed above.

The recorded runtime setup used seed 0, disabled Flash and memory-efficient SDP, enabled math SDP, and preserved v22 deterministic-algorithm, cuDNN, and TF32 settings. DINO loading emitted xFormers-unavailable warnings; source and checkpoint files were not modified.

## 7. C64 order and score comparison

The historical C64 vector had 64 finite `float64` values. Its set identity matched the frozen receipt, and the ordered input digest above was checked before comparison. Rebuilt features had shape `[784, 4096]` and dtype `torch.float32`.

The preregistered per-score criterion was `|new - historical| <= 1e-5 + 1e-4 * |historical|`. Maximum relative error used denominator floor `1e-12`.

| Metric | Result |
|---|---:|
| Paired scores | 64 |
| Maximum absolute error | 0.0 |
| Mean absolute error | 0.0 |
| Root mean squared error | 0.0 |
| Maximum relative error | 0.0 |
| Exact elementwise equality | 64 / 64 |
| Scores within numerical tolerance | 64 / 64 |
| `BITWISE_SCORE_PARITY` | true |
| `NUMERICAL_SCORE_PARITY` | true |
| `SCORE_PARITY_FAILED` | false |
| All values finite | true |

Historical float64 score-vector bytes SHA-256: `4801518f6c36c50e9bf5691808bc8d8916077b7d1793944289d3f8d9ce303403`. Rebuilt float64 score-vector bytes SHA-256: the same value. The maximum-error tie resolves to zero-based position 0; its opaque sample identity is `3091b8185d93a84e5f1f21298d63690244ef35f67b342d64d0e2c4f21ef8122b`.

## 8. Threshold compatibility diagnostics

The frozen higher-q90 threshold algorithm and sigma floor were applied independently to both vectors. These are compatibility diagnostics only; the historical threshold was not replaced or tuned.

| Quantity | Historical | Rebuilt | Absolute difference | Relative difference |
|---|---:|---:|---:|---:|
| `theta0` | 0.5165398120880127 | 0.5165398120880127 | 0.0 | 0.0 |
| `sigma0` | 0.021652716174389672 | 0.021652716174389672 | 0.0 | 0.0 |

## 9. Resource use

- External GPU-child wall time: **68.931566 seconds** of the 1,200-second ceiling.
- DINO model load: **9.644066 seconds**.
- Four-support encoding and bank construction: **35.144707 seconds**; K=256 bank isolation and digest: **0.543146 seconds**.
- C64 encoding: **7.167965 seconds**; C64 scoring: **0.661986 seconds**.
- Peak CUDA allocated: **1,572,285,440 bytes**; peak CUDA reserved: **14,283,702,272 bytes**.
- Python-native peak CPU RSS measurement was unavailable (`null`). The external Windows supervisor measured a peak process working set of **3,016,507,392 bytes**.
- Output namespace footprint at completion: **47,633 bytes**, below the 100 MiB ceiling.
- Actual DINO feature shape/dtype: **`[784, 4096]`, `torch.float32`**.

The external launcher recorded subprocess start/end, exit code 0, wall time, output bytes, and SHA-256 fingerprints for preserved stdout/stderr logs. It did not reach a resource limit.

## 10. Failure and stopping records

No stop condition occurred. The structured bank gate passed before C64 decoding; all 64 scores completed and passed both exact and numerical comparison. The process stopped after recording the C64 parity result. The existing failed R1 output was not resumed or modified.

## 11. Recommendation and limitations

**Reuse of the existing C64 score reference is supported for the R1 engineering pilot using this same frozen `switch` support-seed-0 profile.** The current scorer rebuilt the registered bank and reproduced every existing C64 float64 score byte-for-byte. This parity result does not itself authorize or execute R1, R1 dense-map parity, or R2.

Retain the Python executable hash and Stage 0.5 orchestration-source differences above as provenance caveats. Exact C64 score parity does not establish a complete historical experiment reproduction, nor any anomaly-detection performance claim.

## 12. Artifacts and verification

New score arrays and launcher logs remain only in the dedicated local output namespace listed in Section 2; none were added to Git. The implementation source commit contains the isolated runner, supervisor, and focused tests. The documentation report is committed separately.

CPU verification before GPU execution:

- Focused new S2 and existing R1 synthetic tests: **28 passed**.
- `compileall -q experiments/calibration tests/calibration`: **PASS**.
- PowerShell launcher syntax parse: **PASS**.
- Committed CPU preflight: **PREFLIGHT_PASS**.

Post-run CPU audit: the saved C64 array contains 64 finite values, matches the historical vector byte-for-byte, carries the verified receipt identity and order digest, and matches its recorded file SHA-256. All six protected prior R1/S1 files retained their pre-run SHA-256 values.

No source implementation was changed after inspecting GPU results. No push was made.
