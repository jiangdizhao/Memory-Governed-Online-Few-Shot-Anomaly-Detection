# R2-L2C Metadata Census Readiness

**Date:** 2026-11-01
**Classification:** `R2_L2C_EXECUTOR_READY_AWAITING_AUTHORIZATION`
**Local data authorization:** `R2_L2C_LOCAL_ACCESS_NOT_AUTHORIZED`

## Decision

The default-deny metadata census executor and its synthetic CPU coverage are ready for a later, separately authorized run. This task was completed in Mode A only. No Real-IAD local metadata, test labels, masks, images, features, or geometry were accessed. The census is not complete, no real normal-reference counts are available, and no A/B/C partition is certified.

The Mode A CLI reported 30 public categories, zero locally checked categories, seven missing-approval items, and `local_data_io_performed: false`. The repository's approval template is default-deny and is not a signed authorization. No approved local-access manifest or out-of-band trust-directory pin was supplied.

## Source identity and provenance

- Repository branch: `calibration-stage0-contract-20261029`
- Starting HEAD (pre-commit): `335469e4a92c4bb5f0b43f243c0bd9d004fe7ec8`
- Initial tracked worktree and index: clean
- Pre-existing untracked paths: 84; preserved without modification
- Public catalog: Tencent `RealIAD-full/config.yaml`, pinned commit `98c28ffbe95b380d17a3777d77219152368c19e9`, SHA-256 `9c17d4ba78210ad65a288fedcdd6ea8a65d0926da05a960d3ed8dd1c3210552f`
- Identity version: `realid-offical-sample-key-v2`
- Primary grouping unit: `official_sample_key`
- R1 source-group digest: `sha256:official-sample-group-nul-prefix-utf8.v1`

SHA-256 fingerprints captured for this audit:

| File | SHA-256 |
|---|---|
| `experiments/calibration/r2_l2c_metadata_census.py` | `7768939117fc78bb5356a2de9bd6a1369f16447175f2a0ea8d556c7911a55f59` |
| `tests/calibration/test_r2_l2c_metadata_census.py` | `682edf01b6edcc2787ee2cecd6fbe1f4438fcdd57285f72043d928c85da21049` |
| `experiments/risk_safe/preflight.py` | `2070ba770ac36af06f688b0450c711d67b7bca526e3939178d559216535c25ba` |
| `experiments/calibration/r2_l2b_role_certification.py` | `1829ac180480aaaa24fe5b4a2dee48df55db7d3534c82a4d97753e7f51856d88` |
| `experiments/calibration/r2_l2a_allclass_inventory.py` | `6ce759ebeeefb069a1a834fad6a343436e9e828e0bbd45003153dc44d345bc24` |

These are source fingerprints, not fingerprints of any local dataset files.

## Implementation summary

`experiments/calibration/r2_l2c_metadata_census.py` adds a two-mode workflow:

- **Mode A preflight** reports the pinned public catalog and missing approvals without probing the dataset root. Authorization documents are also rejected lexically if their paths lie inside the dataset root; reparse-point paths are rejected before opening.
- **Mode B execute** returns `R2_L2C_LOCAL_ACCESS_NOT_AUTHORIZED` before any dataset/output path operation unless the signed authorization, pinned trust directory, exact operation, root, categories, individual Mint/PCB decisions, selected field permissions, output fields, and byte budget all validate.
- The command-line executor fixes the prospective data root to `G:\dataset\realiad`; it does not expose a dataset-root override.
- The projection reader reads only approved `meta.normal_class`, `meta.prefix`, TRAIN `image_path`, and TRAIN `anomaly_class` values. It structurally skips the co-located test value without materializing test labels or masks, and records whether test-section bytes were encountered. Unknown or duplicate top-level sections/keys fail closed.
- The frozen identity parser validates official sample keys and camera/session identity. The R1 digest helper groups camera views by `official_sample_key`; raw keys and image paths are excluded from output.
- Per-category outputs include authorization and availability states, denominators, normal image/group counts, camera-view distribution, malformed/duplicate/ambiguous counts, A-role candidate status, partition status, protected-category status, and limitations. Unavailable categories remain unknown rather than becoming zero counts.
- Mint and PCB files are not opened unless each has a separate explicit protected-category approval.
- A/B/C partition assessment accepts only separately reviewed receipts and a pinned reviewer trust root. Census execution never invents B or C groups; absent evidence produces `A_CANDIDATES_IDENTIFIED_PARTITION_NOT_CERTIFIED`, with an explicit incomplete-C status.
- Directory-serial overlap is an optional, separately authorized sensitivity result. It does not replace official sample-key grouping.
- The 64/68 distinction is enforced: 64 unique groups can form a candidate C64 pool; at least 68 distinct groups are necessary for four Supports plus separate C64. That threshold alone does not certify a particular selection. Census authorization rejects Support/C64 selection.
- The output directory must be new and outside the dataset tree. An exclusive writer lock, fixed file allowlist, pre-write byte accounting, and non-overwriting file creation guard the public census report and private group receipts. The public report is written last as the completion marker.

The executor preserves private group receipts only in the authorized local output directory. They contain source-group digests and view counts, not sample keys, image paths, labels, or masks. No production credentials or signatures were created.

## Validation

All tests used synthetic data under pytest temporary directories. No test accessed `G:\dataset\realiad`.

| Check | Result |
|---|---:|
| R2-L1 locality kernels | 44 passed |
| R2-L2A all-class geometry contracts | 38 passed |
| R2-L2B role certification | 40 passed |
| R2-L2C census and authorization | 34 passed |
| Total synthetic CPU tests | 156 passed |
| `python -m compileall -q experiments/calibration tests/calibration` | passed |
| Mode A CLI preflight | passed; 30 public categories, no local I/O, authorization absent |

The L2C tests cover default-deny access, signed permission checks, test-section projection, protected Mint/PCB exclusion, all-30 reporting, official digest parity, multiview grouping, malformed and duplicate identities, missing files, A/B/C evidence and trust checks, directory-serial sensitivity, 63/64/67/68 boundaries, Support/C64 overlap, output field restrictions, and pre-write budget refusal.

## Census status

- Public 30-category catalog: available from the pinned repository receipt.
- Local Real-IAD metadata access: **not authorized; not performed**.
- Authorized train records / normal images / source groups: **not counted**.
- A candidates: **not identified from local data**.
- A/B/C partition: **pending**.
- Directory-serial sensitivity: **not run on local data**.
- Support/C64 selection: **not performed**.
- Real-data distance statistics or G1/G2: **not run**.

## Authorization required for Mode B

The researcher must supply a separately signed `r2_l2c.local_access_authorization.v1` manifest and an independently provisioned reviewer trust directory with its SHA-256 pin supplied out of band. The authorization must approve the exact `G:\dataset\realiad` and metadata roots, exact category allowlist, individual Mint and PCB decisions if requested, physical JSON-byte access, possible encounter with co-located test bytes, each selected TRAIN and metadata field, official OK/NG/defect-token interpretation, private group-receipt output, the exact output-field allowlists, a new output directory, and its maximum byte budget. Test-section parsing, masks, feature extraction, image decoding, and Support/C64 selection must remain denied.

Even after an authorized census identifies A candidates, H-SOURCE certification still requires separately approved and independently reviewed A, B, and C identity receipts under a pinned trust root. Missing or unverified C evidence remains `C_IDENTITY_CERTIFICATE_INCOMPLETE`. Any later four-support plus C64 selection requires its own category/seed-bound, independently reviewed selection receipt; this census does not select it.

## Stop conditions and remaining limitations

No real-data run is authorized by this report. Application-level output accounting is not an OS/filesystem quota and does not create a physical storage boundary. A future execution must use the signed output budget and a fresh output directory; an external filesystem writer is outside the Python writer-lock guarantee. No DINO/GPU work, image decoding, label/mask access, local feature scan, geometry, tuning, or outcome evaluation was performed.
