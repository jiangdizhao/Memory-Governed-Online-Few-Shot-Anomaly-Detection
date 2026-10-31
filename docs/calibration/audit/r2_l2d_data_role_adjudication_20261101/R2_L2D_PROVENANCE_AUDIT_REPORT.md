# R2-L2D Provenance Audit Report

- **Date:** 2026-11-01
- **Overall state:** `R2_L2D_PROTOCOL_READY_CERTIFICATION_PENDING`
- **Artifact-pair scope:** `ARTIFACT_AUDIT_PASS` (internal consistency only)
- **Authentication/provenance:** `BLOCKED_INPUT`

## Scope and repository state

The audit inspected the existing R2-L2C output pair only. It did not access the dataset root or reopen source JSON. No labels, masks, features, images, or G1/G2 results were read or produced. The local report and private receipt files were treated as read-only.

At task start the repository was on `calibration-stage0-contract-20261029` at `f767ae45a0ece0530763a1b8dcdf6a716174e8a7`. The tracked tree and index were clean. Existing untracked audit/output paths were preserved. The L2C output directory contained exactly the two files expected by its executor; it had no standalone run manifest or checksum file. In L2C, `census_report.json` is the declared completion marker and is written last.

## Existing census artifact findings

The audited directory is:

`outputs/calibration/r2_l2c_hsource_all30_train_census_20261101_v1/`

| Artifact | Bytes | Finding |
|---|---:|---|
| `census_report.json` | 35,726 | Schema, completion status, current parser fingerprints, and aggregate fields validated. SHA-256: `8aa27fb725002abc26e2a9d257ca065b4f86e789db1552357a7849212cf7fcce`. |
| `private_source_group_receipts.json` | 848,612 | Schema, category coverage, source fingerprint cross-matches, group digest format, duplicate checks, view counts, and privacy flags validated in memory. Its file digest and private group identities are intentionally not included in this report or Git. |
| **Total** | **884,338** | Below the 64 MiB limit (`67,108,864` bytes). No other files were present in the output directory. |

The two artifacts report and internally agree on authorization ID `r2-l2c-hsource-all30-train-census-20261101-v1` and authorization SHA-256 `969fee786585fddfae88d1b3ecf19f2cd36a91e66ee614b6c0d9e9f57eee2587`. This digest is an identifier, not proof of a valid signature.

The L2D checker confirmed:

- 30/30 canonical categories, 36,465 eligible normal TRAIN images, and 7,293 unique source groups.
- All 30 categories contain at least 68 candidate groups; the minimum reported category count is 167. This is numerical feasibility only.
- All groups retain five camera views. Group identities are based on the complete `official_sample_key`, so camera views remain together.
- No malformed, duplicate, or ambiguous official group identities were found in the supplied receipt pair. Directory-serial overlap sensitivity was `NOT_AUTHORIZED` and remains unmeasured. No private group identifiers were emitted.
- The 30 reported source JSON SHA-256 values match their private receipt counterparts. Their parser source fingerprints match the current L2C parser and identity parser files.
- The frozen identity contract remains `realid-offical-sample-key-v2`, grouping unit `official_sample_key`, and algorithm `sha256:official-sample-group-nul-prefix-utf8.v1`, using SHA-256 over `official-sample-group\0` followed by the official sample key encoded as UTF-8.
- 90/90 A/B/C category-role slots remain uncertified. A and B are pending independent role receipts; C remains `C_IDENTITY_CERTIFICATE_INCOMPLETE`.
- The report says `test_section_bytes_encountered: true` and `test_section_parsed: false`. That is a record of the prior L2C operation; this audit did not read Test metadata or outcomes.
- Image decoding, feature extraction, and geometry are reported false. The output directory contains no G1/G2 feature or evaluation outputs.

## Provenance blockers

The artifacts do not contain the signed authorization payload or an independently pinned trust-root fingerprint. The earlier unsigned authorization draft is not validation evidence for this run. Cryptographic authorization therefore remains `BLOCKED_INPUT` despite the reported authorization ID and digest.

The tracked `R2_L2C_MODE_B_REAL_METADATA_CENSUS_REPORT.md` says that no census ran and that the output directory was absent. That historical report conflicts with the present local output pair. It has not been rewritten. No independent, previously recorded 30-category source-fingerprint baseline is available to resolve the conflict. The source metadata were not reopened, as required; source fingerprints can only be described as reported and internally cross-matched.

The public Tencent catalog receipt is pinned to commit `98c28ffbe95b380d17a3777d77219152368c19e9`, file `experiments/RealIAD-full/config.yaml`, SHA-256 `9c17d4ba78210ad65a288fedcdd6ea8a65d0926da05a960d3ed8dd1c3210552f`, size 5,574 bytes. This verifies the public category contract only; it does not authorize local data access.

## Verification record

Source fingerprint scope:

| Source | SHA-256 |
|---|---|
| Pinned public category config | `9c17d4ba78210ad65a288fedcdd6ea8a65d0926da05a960d3ed8dd1c3210552f` |
| R2-L2D validator (`r2_l2d_data_role_adjudication.py`) | `840600fabc0f0424afeab5438f291190b79a4667226d57ded665074f6924cab4` |
| R2-L2D tests (`test_r2_l2d_data_role_adjudication.py`) | `1b11f2ed4c9705f9f21a39c32557bb455cfd3cf660cdebaa9d7f245a8805286e` |
| L2A public catalog contract | `6ce759ebeeefb069a1a834fad6a343436e9e828e0bbd45003153dc44d345bc24` |
| L2A G1 geometry / bank-identity contract | `3fd3ce36a1317c7c408cdbdc829d87d238aefaab81e453ecf56c36a43aa2201f` |
| L2B role certification contract | `1829ac180480aaaa24fe5b4a2dee48df55db7d3534c82a4d97753e7f51856d88` |
| L2C metadata census contract | `7768939117fc78bb5356a2de9bd6a1369f16447175f2a0ea8d556c7911a55f59` |
| Frozen identity parser (`risk_safe/preflight.py`) | `2070ba770ac36af06f688b0450c711d67b7bca526e3939178d559216535c25ba` |

Commands run (PowerShell):

```powershell
& 'D:\anaconda3\envs\torch\python.exe' -m py_compile experiments/calibration/r2_l2d_data_role_adjudication.py tests/calibration/test_r2_l2d_data_role_adjudication.py
& 'D:\anaconda3\envs\torch\python.exe' -m pytest -q tests/calibration/test_r2_l1_locality_kernels.py tests/calibration/test_r2_l2a_allclass_geometry.py tests/calibration/test_r2_l2b_role_certification.py tests/calibration/test_r2_l2c_metadata_census.py tests/calibration/test_r2_l2d_data_role_adjudication.py
@'
import json
from experiments.calibration.r2_l2d_data_role_adjudication import audit_l2c_artifact_pair
result = audit_l2c_artifact_pair(r'E:\Memory-Governed-Online-Few-Shot-Anomaly-Detection\outputs\calibration\r2_l2c_hsource_all30_train_census_20261101_v1')
print(json.dumps(result, sort_keys=True, indent=2))
'@ | & 'D:\anaconda3\envs\torch\python.exe' -
```

The new L2D suite passed 22 tests. The combined R2-L1, L2A, L2B, L2C, and L2D suites passed 178 tests after finalization. The read-only artifact checker returned `ARTIFACT_AUDIT_PASS` for internal pair consistency and `BLOCKED_INPUT` for authentication/baseline provenance. Directory-serial sensitivity remained `NOT_AUTHORIZED_IN_CENSUS`. The checker result includes the public completion-report SHA above and intentionally omits the private receipt file hash.

## Classification

- **`ARTIFACT_AUDIT_PASS`** — only for file/schema, internal cross-file, aggregate, parser-fingerprint, privacy, and output-size consistency of the supplied pair.
- **`BLOCKED_INPUT`** — signed authorization/trust-root evidence and an independent prior source-fingerprint baseline are missing.
- **No local data inventory authorization is inferred from this public category or internal consistency audit.**
- **No A/B/C membership is certified.** See the role specification and decision record for exact gates.

The final local commit SHA is reported in the execution response. This report does not self-reference a commit hash because changing the file to include its own commit hash would create a different commit.
