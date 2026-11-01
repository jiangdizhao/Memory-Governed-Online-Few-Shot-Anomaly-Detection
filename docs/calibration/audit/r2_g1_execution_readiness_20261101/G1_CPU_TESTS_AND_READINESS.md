# R2-G1 CPU Tests and Readiness Validation

## Test provenance

Interpreter: `D:\anaconda3\envs\torch\python.exe`
Python: 3.11.14
pytest: 9.2.0
Mode: CPU-only synthetic tests.

Command:

```powershell
& 'D:\anaconda3\envs\torch\python.exe' -m pytest -q tests/calibration/test_r2_l2a_allclass_geometry.py tests/calibration/test_r2_l2b_role_certification.py tests/calibration/test_r2_l2d_data_role_adjudication.py tests/calibration/test_r2_l2e_partition_protocol.py
```

Result: **127 passed in 1.47s; 0 failed; 0 skipped.**

## Coverage exercised

- L2A G1 geometry exact blockwise cosine distances, source-group and same-image exclusion, nearest/fifth-nearest ties, strict radius boundaries, bounded blocks, and source-group-aware query bootstrap.
- G1 feature-cache/descriptor receipt rejection paths and category/seed-specific fixed-bank and M0 receipt contracts; missing external review leaves G1-B `NOT_YET_IDENTIFIABLE`.
- L2B and L2D role identity, overlap, approval, receipt, and fail-closed behavior using synthetic identity fixtures.
- L2E deterministic H-SOURCE proposal, grouping, exposure routing, and pending-vs-certified status using synthetic digests/counts.
- New assertion for the five R1 profile source hashes. The expected values were independently compared with canonical LF hashes of the pinned `c86d5b89fa0062b87d80adb324eaa6bc3c94448e` source blobs and current files.

No Real-IAD data, image, DINO checkpoint, model load, GPU operation, TEST metadata, labels, masks, private feedback, or Support/C64 identity was read.

## Readiness interpretation

No additional readiness CLI was added: the existing L2B certify_a_reference_availability(), G1 validate_feature_cache_receipt(), and g1b_identifiability_status() expose the necessary per-gate states. The synthetic tests exercise the pending and fail-closed paths. The synthetic tests show that the existing geometry and authorization/receipt validators fail closed under the tested cases. They do **not** certify any local A role, authorize image/feature access, validate a real cache, create a native M0/Top-32 receipt, or establish G1-A/G1-B execution readiness.

No G1-specific image-to-descriptor adapter currently connects the R1 feature path and P1 in-memory cache to the G1 `DescriptorBatch` and projected-cache receipts. A future CPU-only adapter/preflight test is required before any authorized image decode or DINO execution. A no-receipt read-only preflight invocation returned `PENDING_A_REFERENCE_RECEIPT` from `certify_a_reference_availability()` and `NOT_YET_IDENTIFIABLE` from `g1b_identifiability_status()`. It read no data paths. No G1 geometry was run in this task.
