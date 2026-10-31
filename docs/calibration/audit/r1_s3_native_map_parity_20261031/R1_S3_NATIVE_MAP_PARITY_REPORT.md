# R1-S3 Bounded Native Dense-Map Parity Smoke Report

Date: 2026-10-31
Branch: `calibration-stage0-contract-20261029`

## Executive decision

**Final S3 classification: `INCOMPLETE`.** The authorized GPU child ran once and stopped on the first stream position before any event was sealed. No map, Top-32, or image-score parity conclusion was obtained.

The run manifest records `TAPE_INTEGRITY_FAILED`, but the separate CPU verifier reports `readback_integrity_pass: true` for an empty tape with zero events. The supervisor used the full 16-event acceptance failure to assign its tape-integrity status. The tape files themselves were internally consistent; the required 16-event tape was not produced.

The immediate implementation defect is a return-arity mismatch. The S3 runner unpacks three values from `experiments.a0_a1_strict4_matched.score_bank()`, while that unchanged source function returns `(score, amap)`. The adapter scored the first image, then the native-reference call raised `ValueError: not enough values to unpack (expected 3, got 2)`. This report is based on that single authorized attempt. No implementation edits or repeat GPU execution followed it.

R1-S3 did **not** pass. This result does not establish native map, Top-32, or scalar parity and does not authorize R1-S4.

## Source identity and changed files

The source commit used for the GPU attempt was:

`9aab9094c04e6a720e8a4aa29ae31fff33e076d6`

Implementation files in that commit:

- `experiments/calibration/r1_s3_native_map_parity.py`
- `experiments/calibration/run_r1_s3_bounded.ps1`
- `tests/calibration/test_r1_s3_native_map_parity.py`

This report is committed separately. No historical R1, S1, S2, P0, Native Scorer, Greedy Memory, VisionAD, or v22 source was modified. No prior output was overwritten. The S3 output remains local and uncommitted at:

`outputs/calibration/r1_s3_native_map_switch_s0_16_20261031_native-map-20261031-a1/`

## CPU checks and preflight

Before the GPU attempt:

- Branch, expected implementation SHA, and clean tracked worktree checks passed.
- Metadata-only preflight validated the Stage 0.5, S1, and S2 receipts and source fingerprints.
- Preflight resolved four supports and stream positions 0–15 using path metadata and file existence checks. It did not read image bytes, decode images, load DINOv2, or instantiate a label provider.
- The CPU synthetic preflight harness passed with repeated camera views retaining the same opaque group identity.
- `python -m compileall -q experiments/calibration tests/calibration` passed.
- Focused tests passed: **37 passed** across the S3, S2 parity, and R1 pilot test modules.
- PowerShell parser check passed for `run_r1_s3_bounded.ps1`.
- `git diff --cached --check` passed before the implementation commit.

Test command:

```powershell
@'
import sys
sys.path.append(r'D:\anaconda3\Lib\site-packages')
import pytest
raise SystemExit(pytest.main([
    'tests/calibration/test_r1_s3_native_map_parity.py',
    'tests/calibration/test_r1_s2_c64_score_parity.py',
    'tests/calibration/test_r1_matched_correction_pilot.py',
    '-q', '-p', 'no:cacheprovider'
]))
'@ | & 'D:\anaconda3\envs\torch\python.exe' -
```

## Frozen bank, runtime, and provenance

The GPU attempt rebuilt the four frozen `switch / seed 0` supports using the existing P0 encoding and bank-construction functions.

- Expected structured digest: `8a710d028509350568e976bc3ede66aa59603499cbea525bdd8579322b1a1558`
- Actual structured digest: `8a710d028509350568e976bc3ede66aa59603499cbea525bdd8579322b1a1558`
- Structured digest algorithm: `experiments.calibration.native._fixed_bank_digest.v1:path-shape-dtype-bytes`
- P0 raw prototype digest: `087feded0d780bbc3a56d45404b4126991b1aeb54f1ae3724bce025caf0d380e`
- Runtime profile: `stage05_set_seed_0_math_sdp_v1` (seed 0; Flash and memory-efficient SDP disabled; math SDP enabled)
- Checkpoint SHA-256: `36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51`
- Historical DINO source-tree SHA-256: `1de67887ca330bdfaad8ccd45128d2b9313d63aa707bbe8e4f38310e52005b24`

The fixed structured and raw digests remained unchanged through the scoring calls that completed. Dynamic normal memory had the same empty-state digest before and after: `773a440dc3c6434857f581f81cec7a6796ff53ef20b1eb38e5c47442e39d9312`. Correction was disabled with zero anchors; the defect bank was disabled with zero prototypes.

The digest match verifies the rebuilt fixed bank against the historical receipt. It does not establish native map or image-score parity.

## GPU attempt and parity counts

The supervisor launched exactly one GPU child for category `switch`, support seed 0. The child encoded one stream image at position 0. The adapter call completed, but the original native reference call failed before the event was sealed.

| Measure | Result |
|---|---:|
| Stream positions authorized | 0–15 |
| Stream images decoded | 1 |
| Events processed and persisted | 0 / 16 |
| Native scalar comparisons | 0 / 16 |
| Dense-map comparisons | 0 / 16 |
| Top-32 sequence comparisons | 0 / 16 |
| Diagnostic `A_pre` maps captured | 0 |
| Event shards written | 0 |
| Readback positions | none |

Dense-map element equality, Top-32 equality, scalar exact equality, and scalar tolerance counts are **not evaluated**. The manifest counters are zero because the failure occurred before those comparisons; zero does not mean a measured mismatch.

The error came from this call shape in the S3 runner:

```python
native_score, native_amap, _ = score_bank(...)
```

The unchanged native function returns two items. The next bounded implementation should bind those two items and add a focused synthetic contract test before any new run is considered. This attempt is consumed; no automatic retry or repair run occurred.

## Evidence Tape readback

The independent CPU verifier exited successfully and returned:

- `readback_integrity_pass: true`
- `events_present: 0`
- `event_positions: []`
- `full_s3_event_count: false`
- dense-map, Top-32, scalar, zero-correction, dynamic-memory, and feature-immutability readback counts: all 0

Thus the empty partial tape had no missing or corrupt shards, but the run did not produce a usable 16-event evidence tape. The supervisor's `TAPE_INTEGRITY_FAILED` status is retained in the execution manifest; the scientific completion status in this report is `INCOMPLETE` because the replay stopped before its first persisted event.

## Resource use and information boundary

- GPU: NVIDIA GeForce RTX 3090 Ti
- GPU child wall time: 24.883433 seconds (600-second limit)
- Output size at completion: 41,206 bytes (128 MiB limit)
- Peak CPU working set: 3,002,982,400 bytes
- Peak CUDA allocated: 1,572,285,440 bytes
- Peak CUDA reserved: 14,283,702,272 bytes
- Model initialization: 4.660923 seconds
- Support encoding and bank construction: 11.285548 seconds
- One stream-image encoding: 0.083479 seconds
- Adapter scoring: 0.006004 seconds
- Native score-bank phase: did not complete; duration unavailable
- Diagnostic scoring and tape serialization: not reached; duration unavailable

No query labels, masks, confirmation data, or C64 images were accessed. The existing C64 score receipt was checksum-verified only; it was not decoded or rescored. No Greedy writer, normal-memory write, correction update, or defect-bank update ran. No scientific performance metric was computed.

## Recommendation and stopping boundary

**R1-S4 is blocked.** First resolve the two-value `score_bank()` interface mismatch in a separately reviewed, versioned implementation; rerun CPU checks and source-bound preflight; then obtain new explicit authorization for a fresh bounded S3 attempt. Do not use this empty tape or this incomplete run as parity evidence. This report makes no recommendation to start the 512-event R1 trajectory or R2.

Stop after this S3 report and wait for independent review and new authorization.
