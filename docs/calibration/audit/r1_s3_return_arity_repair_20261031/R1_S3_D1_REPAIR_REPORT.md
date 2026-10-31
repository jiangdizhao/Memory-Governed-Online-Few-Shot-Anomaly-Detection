# R1-S3-D1 Native Reference Return-Arity Repair

Date: 2026-10-31
Branch: `calibration-stage0-contract-20261029`
Starting HEAD: `b60844e96c6909c2c2952eab32428e4c10166ec3`

## Finding

The failed S3 attempt stopped at event 0 because `r1_s3_native_map_parity.py` unpacked three values from `experiments.a0_a1_strict4_matched.score_bank()`. That wrapper unpacks the bank's three-value `score_sample_multiview()` result internally and returns exactly `(float(score), amap)`. The exception was an API return-arity defect; no native parity comparison ran and no event was sealed. The historical empty tape passed its structural and checksum readback.

## Repair

- Added an isolated S3 native-reference helper that calls the real `score_bank()` wrapper with its two-value contract, extracts and validates native layer-0 Top-32 IDs, and classifies an API exception as `INCOMPLETE` while preserving its root exception type.
- Kept diagnostic-map scoring independent. It retains the original diagnostic metadata, validates finite FP32 `A_pre[28,28]`, and separately extracts/validates diagnostic Top-32 IDs. This preserves S3's map, Top-32, and scalar checks and its distinct scoring timings. The existing R1 helper was not used because its return value flattens the map and omits S3's grid/view/layer metadata.
- Centralized the existing S3 parity comparisons without changing their thresholds or semantics: exact map element/byte equality, ordered Top-32 agreement across all three paths, scalar tolerance (`1e-5 + 1e-4 * abs(native score)`), and exact adapter max reconstruction.
- Added root exception reporting to the run manifest for caught replay/API failures.
- Fixed the supervisor's status separation. A valid zero-event tape can report tape integrity `PASS`, `event_count_complete=false`, and overall execution `INCOMPLETE`. It becomes `TAPE_INTEGRITY_FAILED` only when the verifier reports a structural validation failure or invalid serialized JSON. A missing/unreadable verifier receipt remains `NOT_VERIFIED` and does not get reclassified as tape corruption. Specific replay parity failures remain their own statuses.

No Stage 0.5 source, P0 source, native scorer, Greedy code, prior S3 output, or historical report was changed.

## Verification

- Requested S3 pytest selection (`tests/calibration/test_r1_s3_native_map_parity.py -q`): **14 passed**.
- Focused R1 suite (`test_r1_s3_native_map_parity.py`, `test_r1_s2_c64_score_parity.py`, and `test_r1_matched_correction_pilot.py`): **42 passed**.
- Both test selections used the Torch interpreter with the pytest CLI loaded via `runpy` after appending `D:\anaconda3\Lib\site-packages` to `sys.path`. A direct `python -m pytest` attempt with that path in `PYTHONPATH` failed before collection because the base environment's `pathlib.py` shadowed the Torch interpreter's standard library; the in-process CLI invocation avoids that path-order issue.
- Python compilation for the S3 runner and its focused tests: **passed**.
- PowerShell parser validation for `run_r1_s3_bounded.ps1`: **passed**; the script was parsed only and not executed.
- Synthetic regression exercises the actual `score_bank()` wrapper over a fake bank returning three values, checks the wrapper's two-value return and S3 consumption, verifies separate map/Top-32 extraction, rejects malformed maps and missing IDs, records API exceptions as `INCOMPLETE`, and checks the existing map/index/scalar gates.

These CPU tests establish the corrected software interface and status handling only. They do not establish real-image, DINO, or native GPU parity.

## Status

- `SOURCE_INTERFACE_DEFECT`: **RESOLVED**.
- `OVERALL_EXECUTION_STATUS`: **INCOMPLETE until a separately authorized fresh S3 replay completes**.
- `TAPE_INTEGRITY_STATUS`: independent of overall execution; a valid empty/partial tape is `PASS` for integrity.
- `EVENT_COUNT_COMPLETE`: independent boolean; it is true only when both the replay manifest and verified tape contain all 16 events.
- `NATIVE_GPU_PARITY`: **NOT VERIFIED**.
- GPU execution, DINO loading, dataset image decoding, query-label access, C64 rescoring, stream replay, and retry: **NOT AUTHORIZED AND NOT PERFORMED**.

## Files changed

- `experiments/calibration/r1_s3_native_map_parity.py`
- `experiments/calibration/run_r1_s3_bounded.ps1`
- `tests/calibration/test_r1_s3_native_map_parity.py`
- `docs/calibration/audit/r1_s3_return_arity_repair_20261031/R1_S3_D1_REPAIR_REPORT.md`

## Conditions before a future S3 retry

A separately authorized run must use the corrected implementation commit as `ExpectedSourceCommit`, a fresh unique `RunId` (therefore a new output namespace), the same frozen 16-event input protocol, and a new authorization ID. The supervisor retains its 600-second limit and 128 MiB output ceiling, with no automatic retries. A completed replay must pass all 16 event-level map, ordered Top-32, scalar, zero-correction, memory-immutability, feature-immutability, and tape-integrity checks. No R1-S4 or R2 action follows this smoke test.

Proposed command (**NOT EXECUTED; a separate authorization is required**):

```powershell
& .\experiments\calibration\run_r1_s3_bounded.ps1 `
  -ExpectedSourceCommit '<NEW_IMPLEMENTATION_SHA>' `
  -RunId 'native-map-repair-20261031-<UNIQUE_SUFFIX>' `
  -AuthorizationId '<FUTURE_AUTHORIZATION_ID>'
```

The command uses the supervisor's configured 600-second and 128 MiB limits and creates a fresh run namespace. It was not run as part of this repair.
