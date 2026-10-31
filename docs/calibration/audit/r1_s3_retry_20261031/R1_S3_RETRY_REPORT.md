# R1-S3 Repaired Native Map Parity Retry Report

Date: 2026-10-31

## Execution identity

- Branch: `calibration-stage0-contract-20261029`
- Execution source commit: `b30a60181d75e9c780e069595a1cd93f9fada823`
- Run ID: `native-map-retry-20261031-b1`
- Authorization ID: `USER_AUTH_R1_S3_RETRY_20261031`
- Profile: `r1_s3_native_map_parity_v1`; runtime profile `stage05_set_seed_0_math_sdp_v1`
- Category / support seed: `switch` / `0`
- Support: four frozen references; stream: positions 0-15
- Output: `outputs/calibration/r1_s3_native_map_switch_s0_16_20261031_native-map-retry-20261031-b1/`
- GPU device: NVIDIA GeForce RTX 3090 Ti; PyTorch `2.0.1+cu117`; CUDA `11.7`
- Checkpoint SHA-256: `36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51`

The source-bound PowerShell supervisor was invoked once. Its 600-second time limit and 128 MiB output limit remained in force. No retry or implementation change occurred.

## Result

**Final status: `S3_NATIVE_MAP_PARITY_PASS`.** The run processed and sealed all 16 authorized events, positions 0 through 15.

| Gate | Result |
|---|---:|
| Dense-map exact element and byte parity | 16/16 |
| Maximum absolute dense-map difference | 0.0 |
| Ordered Top-32 agreement across adapter, `score_bank()`, and diagnostic interface | 16/16 |
| Scalar exact equality | 16/16 |
| Scalar tolerance | 16/16 |
| Maximum absolute scalar difference | 0.0 |
| Maximum per-event scalar tolerance | `6.9859007596969613e-05` |
| Exact adapter-score reconstruction from base-map maximum | 16/16 |

The adapter map was stored as 784 FP32 values. Native diagnostic `A_pre` was 28x28 FP32 and compared by row-major identity-grid flattening with no interpolation. Readback confirmed stored base maps as `[784] float32`, selected features as `[32,4096] float32`, and exact map, Top-32, scalar-maximum, scalar-tolerance, zero-correction, dynamic-memory, and feature-immutability counts of 16. The 16 event positions and unique event IDs were present; source-group and bank identities matched; all tape schema, shard, and array checksums passed. The independent CPU invocation of the existing verifier returned `readback_integrity_pass=true`, `events_present=16`, and `full_s3_event_count=true`.

## State and limits

- Structured fixed-support digest before and after: `8a710d028509350568e976bc3ede66aa59603499cbea525bdd8579322b1a1558` (unchanged and equal to expected receipt).
- Raw fixed-prototype digest before and after: `087feded0d780bbc3a56d45404b4126991b1aeb54f1ae3724bce025caf0d380e` (unchanged).
- Dynamic memory: count 0 before and after; digest `773a440dc3c6434857f581f81cec7a6796ff53ef20b1eb38e5c47442e39d9312` unchanged.
- Greedy writer invoked: false; normal-memory writes: 0. Correction/defect updates: 0. Correction and defect banks stayed disabled with zero anchors/prototypes.
- C64 images decoded or rescored: false. Prohibited payload list: empty. No queried feedback, query labels, or masks were accessed.
- Fixed, dynamic, correction, defect, and source-feature immutability checks passed.
- Root exception: none. Launcher child exit code: 0. Tape verifier exit code: 0.
- GPU child wall time: 65.901794 seconds (under 600 seconds).
- Audited output directory size: 8,630,546 bytes (under 134,217,728 bytes).
- Peak CUDA allocated: 1,572,285,440 bytes. Peak CUDA reserved: 14,283,702,272 bytes. Peak CPU working set: 3,010,777,088 bytes.

The prior failed S3 output was not modified. Its eight-file, 41,206-byte tree fingerprint was `64e74bf8c9c3d834f0c4f753df545074d0605285955b9f3d082cf09cacec32a2` both before and after this retry.

## Recommendation and scope

Recommend that the R1-S4 proposal may proceed to **technical protocol review** based on this fixed-support scoring and evidence-replay smoke test. This result does not authorize R1-S4 execution, R2, correction optimization, or any scientific performance claim; each requires its own authorization and review. No R1-S4 or R2 evaluation was run.
