# R2-G1 Readiness Report

## Decision

- Overall: **`G1_EXECUTION_CONTRACT_READY_FOR_RESEARCHER_APPROVAL`**
- G1-A contract: **`ENGINEERING_READY; PENDING_A_REFERENCE_RECEIPT; FEATURE_ACCESS_AUTHORIZATION_PENDING; FEATURE_CACHE_PENDING`**
- G1-B: **`SUPPORT_C64_IDENTITY_PENDING; NATIVE_M0_RECEIPT_PENDING; NOT_YET_IDENTIFIABLE; EXECUTION_NOT_AUTHORIZED`**
- G2: **`NOT_AUTHORIZED`**
- Real-data geometry, feature extraction, DINO load, and label access: **NOT RUN**

This is a preparation and contract report. It is not permission to read images, create descriptors, select Supports/C64, execute G1, or run G2.

## Scope and repository identity

Repository branch: `calibration-stage0-contract-20261029`
Expected parent and verified HEAD at task start: `59412b1f326a66196e21b64b9361e0510c79522f`
Tracked worktree at task start: clean. Existing untracked paths were preserved and not staged. No branch switch occurred.

The only source change is the correction of three stale hashes in the new G1 descriptor-profile metadata. One focused synthetic assertion was added. No geometry formula, scorer, DINO, P0/P1, native, vmb, v22, R1 runner, dataset, or historical output was changed.

## Source and model profile verification

The R1 profile declares source commit `c86d5b89fa0062b87d80adb324eaa6bc3c94448e`. I recomputed the profile's canonical LF source hashes from both that commit's blobs and the current files. The initial profile values for the P0, vmb, and v22 files were stale; corrected values now match both source commit and current files:

| File | Verified SHA-256 |
|---|---|
| `experiments/calibration/r1_matched_correction_pilot.py` | `3208e2c52277ad76351eb86968e362aa315effe9e3c734bfe2b6ff7bab0db448` |
| `experiments/calibration/native.py` | `3a347d65aaf4bfd6b41eff2ea7339d878065bb2e08ed64795f57b0458135dd45` |
| `experiments/p0_protocol_audit.py` | `03586ae40f79e9e49ad6545a4131c411a5f6adef9d085feae0b885f2f98966da` |
| `vmb_visionad_new.py` | `0430959e86ba75b029341aef003655e6a3ac7d08df22f3f48a04e0b2d7fc39c9` |
| `test_phase3_AIF_simplified_v22.py` | `a79b6a78ccf918c9314025ab0acef2d9e8d84cf44600615a4085cd82802bda4f` |

The profile also records checkpoint SHA-256 `36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51`, DINO hub source tree SHA-256 `b7c0037f90671c4932393bd56de8c8a205b29a6eb22dea563b6c8a88c471da15`, and loaded model state SHA-256 `1f26464ba4055403e550f1d5c1333fb012631d98661df6993cabea730d17e3f9`. These are historical intended receipt values. This task did not open the checkpoint or hub tree, instantiate DINO, or verify present GPU/runtime artifacts. A future run must verify actual weight/source/state before decoding images.

The historical R1 S4B report records a 512-event run, GPU wall time 1,365.719 seconds, peak CUDA allocated 1,572,285,440 bytes, peak CUDA reserved 14,283,702,272 bytes, and peak CPU RSS 2,893,500,416 bytes. These are prior R1 engineering measurements, not G1 measurements, guarantees, or authorization.

## Independent provenance gates

| Gate | Current evidence | Status |
|---|---|---|
| Public 30-category catalog | Existing pinned Tencent catalog receipt | Verified catalog only |
| L2C TRAIN metadata authorization | R2-L2D-R validated signed authorization digest `969fee786585fddfae88d1b3ecf19f2cd36a91e66ee614b6c0d9e9f57eee2587` | Valid for TRAIN metadata census only |
| Historical trust-root pin timing | Current trust digest matches researcher-provided pin, but no contemporaneous independent pin evidence | Unverified |
| Independent source-fingerprint baseline | Not established by two outputs of one Stage B run | Pending independent pin |
| H-SOURCE partition policy | L2E policy SHA `73afab2b6f306043094bd5e3c1643a8c004c04ba2237be6ba43b6eca25642719` | Proposal; researcher approval pending |
| A/B/C group membership | No certified group receipts or independent disjointness attestation | Pending |
| TRAIN-normal candidate availability | Stage B report says 7,293 groups, five views/group, minimum 167 groups/category | Candidate census only; no A availability receipt |
| Image/feature authorization | L2C authorization explicitly excluded image decoding/features and Support/C64 selection | Not authorized |
| G1-A execution | Exact geometry engine is tested; A availability receipt, image/feature scope, and bounded projected-cache adapter/receipt are absent | PENDING_A_REFERENCE_RECEIPT; FEATURE_ACCESS_AUTHORIZATION_PENDING; FEATURE_CACHE_PENDING |
| G1-B support/M0/Top-32 | Only retained fixed-bank digest is switch/seed-0; no externally verified C64-cohort Top-32/M0 receipt exists | SUPPORT_C64_IDENTITY_PENDING; NATIVE_M0_RECEIPT_PENDING; NOT_YET_IDENTIFIABLE |
| G2 | No G2 authorization | Not authorized |

The absence of a receipt is an authorization/provenance gap, not evidence that local data is missing. G1-A does not require B/C certification, C64, Support, Top-32, or G2. It does require an A-only normal availability receipt, independent source/cache verification, and a new image/feature authorization. The existing L2C census authorization cannot be reused for these operations.

For G1-B, the retained switch/seed-0 fixed-bank digest is `8a710d028509350568e976bc3ede66aa59603499cbea525bdd8579322b1a1558`. It only identifies that switch/seed-0 bank. It does not supply the M0 certificate or A-cohort native C0 Top-32 receipts, and it cannot be reused for another category or seed.

## Engineering/readiness findings

1. The existing G1 geometry implementation preserves exact blockwise float64 cosine distance, source-group/image exclusion before top-k, deterministic tie order, exact nearest and fifth-nearest values, strict radius comparison, and capped memory blocks.
2. The implementation is geometry-only. It accepts already validated in-memory descriptors and does not decode images, load metadata, or load DINO.
3. The existing P1 cache is in-memory and unbounded; the G1 receipt expects full maps, while G1 needs projected 49/32-row descriptors. A future G1 adapter must give full maps a bounded one-image lifetime and introduce a versioned projected-descriptor receipt before data execution.
4. The existing G1-PROTOCOL-MATCHED profile uses four Supports plus 64 additional C64 groups; the C64 cohort supplies matched geometry descriptors. Its 68-group planning count is below the Stage B aggregate minimum of 167, but that aggregate does not certify category-specific A availability.
5. The two-category pilot is proposed as `switch` then `audiojack`; each category needs its own A availability and, for G1-B, its own Support/M0/Top-32 receipts.

## CPU test and validation results

Interpreter: `D:\anaconda3\envs\torch\python.exe`, Python 3.11.14, pytest 9.2.0.

Command:

```powershell
& 'D:\anaconda3\envs\torch\python.exe' -m pytest -q tests/calibration/test_r2_l2a_allclass_geometry.py tests/calibration/test_r2_l2b_role_certification.py tests/calibration/test_r2_l2d_data_role_adjudication.py tests/calibration/test_r2_l2e_partition_protocol.py
```

Result: **127 passed in 1.47s**, no failures or skips. Fixtures were synthetic identities/features only. A separate receipt-empty status check returned `PENDING_A_REFERENCE_RECEIPT` for G1-A and `NOT_YET_IDENTIFIABLE` for G1-B. No dataset files, images, DINO weights, GPU, TEST metadata, labels, masks, or protected-category identities were read. The profile-hash assertion checks all five expected source hashes.

## Remaining blockers and classification

- **G1-A:** new researcher-signed TRAIN-normal image decode/feature authorization; approved deterministic camera/sample policy; certified A availability receipt; independent trust/source/cache fingerprint review; bounded adapter and projected-cache receipt; unique run manifest/output path.
- **G1-B:** common A identity/authorization/representation/cache gates for its C64 cohort, plus separately authorized actual four Support and 64 C64 selection under G1-PROTOCOL-MATCHED, a category/seed-specific fixed-bank receipt, empty-Greedy M0 state/certificate, native C0 Top-32 receipts, and external verification of all receipts. It does not require completing the distinct broad-population G1-A study.
- **A/B/C:** separate researcher approval of L2E; field-scoped TEST identity/exposure authorization; source snapshot and exposure registry; global role disjointness; independent reviewer attestation. These are not a prerequisite for A-only G1-A.
- **G2:** separate authorization and protocol; not part of this readiness work.

No new dataset records or artifacts were missing during this task because none were sought. The current blockers are new authorization, independent provenance/attestation, role receipts, and a small G1 adapter; they are not a demonstrated local-data absence.
