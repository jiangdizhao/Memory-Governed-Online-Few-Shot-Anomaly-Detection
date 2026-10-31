# R2-L2D-R — Provenance Recovery and Gate Reconciliation

**Date:** 2026-11-01
**Repository HEAD:** `0bf74af6be9e6880f7d2db4c2708cb15491efe63`
**Branch:** `calibration-stage0-contract-20261029`
**Disposition:** `R2_L2D_PROTOCOL_READY_CERTIFICATION_PENDING`

## Scope and result

This reconciliation inspected only the three researcher-named authorization files, tracked L2C/L2D/L2B reports, and existing Stage B outputs. It did not scan the dataset root or unrelated private locations, inspect any private key, rerun the census, open TEST metadata, or access labels/masks. The original L2C preflight and authorization validator verified the signed manifest and scope without local data I/O.

The Stage B authorization digest, Ed25519 signature, trusted researcher public key, and TRAIN-census scope all verify. The researcher-supplied trust-directory digest matches the supplied directory now. Current materials do not prove that this hash was independently pinned before historical Stage B execution. Status: `L2C_AUTHORIZATION_SIGNATURE_SCOPE_VERIFIED; HISTORICAL_TRUST_PIN_TIMING_UNVERIFIED`.

## Stage A and Stage B are separate historical events

| Event | Evidence and SHA-256 | Verified status |
|---|---|---|
| Stage A authorization attempt | [`R2_L2C_MODE_B_REAL_METADATA_CENSUS_REPORT.md`](../R2_L2C_MODE_B_REAL_METADATA_CENSUS_REPORT.md), SHA-256 `c987cb6d7a81153fd7c50feac220d09f522d5c8de7d636e5e64c6bc0e649b390`; report says Stage A stopped before census/local I/O at expected HEAD `95a372af139f32ae1bdca52319b2b4ca8dd42143`. | `R2_L2C_LOCAL_ACCESS_NOT_AUTHORIZED` for that event. The historical report is unchanged. |
| Later Stage B public census report | `outputs/calibration/r2_l2c_hsource_all30_train_census_20261101_v1/census_report.json`, SHA-256 `8aa27fb725002abc26e2a9d257ca065b4f86e789db1552357a7849212cf7fcce`. | Present; passed L2D pair-internal schema, aggregate, parser-fingerprint, privacy, and output-size checks. It reports 30/30 categories, 36,465 normal images, 7,293 source groups, minimum 167 groups/category, five views/group, and 30/30 same-run source-fingerprint matches. |
| Later Stage B private source-group receipt | `outputs/calibration/r2_l2c_hsource_all30_train_census_20261101_v1/private_source_group_receipts.json`, SHA-256 `0cb850af9bcc6f246ef778c4c2081994ea89c754d1387e2151a829b4717873ab`. | Present; validated jointly with the public report. The receipt contains hashed group identities; this report publishes no group identities. Identity hashes are unique across the reported TRAIN-normal candidate pool. |

These are sequential events: the earlier Stage A attempt stopped before local I/O, and a later Stage B output pair was produced. The recovered manifest signature and scope now validate the authorization ID/digest recorded by Stage B. Stage A and Stage B outputs remain unchanged. The materials do not establish the Stage B operator or when the trust-directory hash was pinned; historical trust-pin timing remains unverified.

### Authorization evidence recovery

The original files were loaded with `load_authorization_and_trust()` and passed to `authorization_errors()` in `experiments.calibration.r2_l2c_metadata_census`; the existing module's `--mode preflight` was also run with the researcher-provided trust pin. Results: `authorization_errors=[]`, `missing_approvals=[]`, `local_access_authorization_status=VALIDATED_NOT_EXECUTED_MODE_A_ONLY`, and `local_data_io_performed=false`.

Invocation (preflight mode only):

```powershell
$authDir = Join-Path $env:USERPROFILE '.r2_l2c_authorization'
& 'D:\anaconda3\envs\torch\python.exe' -m experiments.calibration.r2_l2c_metadata_census --mode preflight --access-manifest (Join-Path $authDir 'authorization.signed.json') --trust-directory (Join-Path $authDir 'trust_directory.json') --expected-trust-directory-sha256 'b5e29d2d4efaa934054d36b6d7aa56ff122270c159ae6d7f4a7405a0a64c93c5'
```

| Material | Verification result |
|---|---|
| `authorization.signed.json` | File SHA-256 `c00824052c15e4fe24a0c55de478c8d4749f15cc3475e41a1a82a2f0a539a5c2`; schema `r2_l2c.local_access_authorization.v1`; ID `r2-l2c-hsource-all30-train-census-20261101-v1`. |
| Canonical signed payload | Recomputed SHA-256 `969fee786585fddfae88d1b3ecf19f2cd36a91e66ee614b6c0d9e9f57eee2587`; matches manifest `authorization_sha256` and Stage B report. |
| Ed25519 signature | Valid for the canonical payload under the public key trusted for `researcher-policy-approver-01` in the supplied trust directory. |
| `researcher_public_key.hex` | File SHA-256 `42fee2a44310f7a4c818b79b93ec9f0cf662c0f702c34be51466392e6567b928`; its public key bytes match the trust-directory entry. SHA-256 of decoded raw public-key bytes: `354197cf1e13dfa8b9b83f87921612b91f5004d00d39c2d80c224b550c827e14`. |
| `trust_directory.json` | File SHA-256 `c9e1706db193b65223cb7f45405953f207ea77ea6b0cd82593f848a23e58af76`; recomputed and embedded directory digests both match researcher-provided `b5e29d2d4efaa934054d36b6d7aa56ff122270c159ae6d7f4a7405a0a64c93c5`. |

**Historical pin limitation:** the researcher supplied the trust-directory digest for this verification, and it matches the directory bytes now. No contemporaneous Stage B run manifest or external record was located to prove that this exact digest had been pinned before Stage B ran. Present cryptographic verification therefore succeeds under the supplied trust root, but historical trust-root pin timing remains unverified.

The verified decision is `R2_L2C_AUTHORIZED_TRAIN_METADATA_CENSUS_V1`, for dataset root `G:\dataset\realiad`, metadata root `G:\dataset\realiad\jsons\realiad_jsons`, and output directory `outputs/calibration/r2_l2c_hsource_all30_train_census_20261101_v1/`. Scope covers the 30 canonical categories, explicitly including protected `mint` and `pcb`, for the TRAIN metadata census only. It permits physical JSON bytes; acknowledgment that co-located TEST-section bytes may be encountered; TRAIN `image_path` and `anomaly_class`; official sample-key condition tokens; `meta.normal_class` and `meta.prefix`; and private group-receipt output, with a 67,108,864-byte limit. The validator confirmed required public/private field sets and rejected no permission. It denies TEST-section parsing, directory-serial sensitivity, masks, image decoding, feature extraction, and Support/C64 selection. Stage B reports `test_section_bytes_encountered=true`, `test_section_parsed=false`, `feature_extraction_performed=false`, and `geometry_performed=false`, consistent with the verified scope. Validation did not access dataset records.

The available unsigned draft remains distinct: SHA-256 `241b2b2cae02739dc9b381202dea7318ab4eb43204364233acfb334b71dc8bfb`, `researcher_id: RESEARCHER_ID_TO_BE_SUPPLIED`; it is not the signed Stage B authorization.

## Provenance and role checks

- **Independent source-fingerprint baseline:** `NOT ESTABLISHED`. The 30 report/receipt source hashes agree within this Stage B output pair. No independent prior baseline was located. This gap leaves actual source-file provenance unverified but does not negate the measured cross-file agreement.
- **Candidate-pool identity-only check:** `PASS, scoped to supplied receipt`. Existing L2D audit verified 7,293 well-formed group digests are unique within and across the 30 reported TRAIN-normal candidate lists, with reported five-view groups. This verifies candidate-pool digest uniqueness only.
- **A/B/C disjointness:** `NOT TESTABLE / NOT CERTIFIED`. The census marks A candidates as identified, but those are not assigned A-role membership receipts. No B or C membership receipts exist. Its `cross_role_overlap_count: 0` is not proof of disjointness when all role statuses are pending. The candidate-pool uniqueness result does not certify any pairwise role intersection.
- **A-role design:** H-SOURCE defines A as the parameter-design reference and the L2C output reports normal TRAIN candidate availability in all 30 categories. This supports a design and a candidate-capacity claim only; normal-only A membership and independent A availability remain uncertified.
- **Support/C64:** output reports candidate pool capacity, but no support or C64 identities were selected. C64 is additional to four deployed supports. No native bank identity or category-specific Top-32 receipt was produced.

## Proposed deterministic A/B/C identity rule

Retain the existing H-SOURCE policy (`docs/calibration/audit/R2_L2B_PROPOSED_30_CLASS_ROLE_POLICY.md`, proposal SHA-256 `531a62e78519702f13dfc1fdb579e8476c29d3ff2993219605516d0ca7433324`): keep each complete `official_sample_key` group, including all camera views, in one role; require global A/B/C disjointness; keep Mint, PCB, and any other protected confirmation groups unassigned unless explicitly approved.

For a future policy revision, propose this deterministic assignment mechanism, without running it here:

1. Freeze the authorized group population, per-category counts for A/B/C, integer seed, role order A then B then C, all-category scope, and protected-group treatment in a researcher-approved versioned policy. The current proposal has no counts or seed; none are invented here.
2. Use only canonical group digests under identity contract `realid-offical-sample-key-v2`, grouping unit `official_sample_key`, and digest algorithm `sha256:official-sample-group-nul-prefix-utf8.v1`.
3. Within each category, sort groups by `(SHA256("sha256:r2-l2d-hsource-group-rank.v1\0" + policy_sha256 + "\0" + category_id + "\0" + seed + "\0" + source_group_sha256), source_group_sha256)` using ASCII encoding. Assign consecutive ranges to A, B, C in that fixed order using the frozen quotas. Reject duplicate inputs, insufficient/excess quota totals, malformed identities, or any global cross-role collision.
4. Keep A's normal-only evidence as a separate signed availability check. The hash assignment does not prove normal status, role purpose, source provenance, or authorization. Require B/C source populations and any TEST metadata in a separate approved access scope.

This is an algorithm proposal, not an approved role policy or independent identity certificate. No quotas are selected and no protected group is assigned. Required identity inputs are the approved policy digest, seed, per-category quota table, category IDs, source-population/split identity manifest, canonical source-group digests, complete group/view membership proof, protected-group list, and reviewer trust pin. Role receipts must bind these inputs and have independent attestations.

## Gate separation and what each next action requires

| Gate / operation | Status | Additional requirement |
|---|---|---|
| L2D engineering and protocol | `PASS_SCOPED_CPU` | Targeted synthetic contract tests passed (see below); no empirical/data execution is implied. |
| L2C Stage B authorization signature and scope | `VERIFIED` | Manifest digest, Ed25519 signature, trusted public key, required/denied permissions, and Stage B authorization ID/digest match all pass the existing validator. |
| Historical independent trust-pin timing | `UNVERIFIED` | Supplied digest matches the trust directory now; no contemporaneous evidence establishes that the pin predated Stage B execution. |
| Independent source-fingerprint baseline | `BLOCKED_INPUT` | Locate an independently pinned pre-run baseline. If source metadata must be reopened, obtain new access authorization first. |
| A/B/C policy approval | `PENDING_RESEARCHER_APPROVAL` | Researcher must approve the exact H-SOURCE policy or version a replacement; freeze quotas, seed, role/source populations, and protected-category handling. |
| A/B/C membership/disjointness certification | `PENDING_INDEPENDENT_REVIEW` | Requires authorized role identity manifests, normal-only A evidence, global pairwise checks, independent reviewer attestation, and separate C identity certification. Current candidate hashes alone are insufficient. |
| B/C identity access from TEST or other new records | `NOT_AUTHORIZED_HERE` | New signed data-access authorization must specify exact metadata fields and source scope. Label and mask access require separate approval. |
| Actual Support/C64 selection | `NOT_RUN` | Requires approved A membership, category/seed/source-bound selection protocol and authorization, actual four-plus-additional-64 identity receipts, and independent selection/bank-binding review. |
| G1-A / G1-B / G2 | `NOT_AUTHORIZED` | Each needs separate explicit execution authorization after its applicable provenance, partition, identity, native-bank, and protocol gates pass. |

The authorization signature and scope are verified. Remaining provenance limitations are the missing contemporaneous proof of trust-pin timing and the missing independent source-fingerprint baseline; neither indicates that local category data are missing or invalidates the verified signature/scope. Researcher role-policy approval and independent reviewer attestation remain separately pending, so L2D certification is not complete.

## Targeted CPU regression provenance

Command:

```powershell
& 'D:\anaconda3\envs\torch\python.exe' -m pytest -q tests/calibration/test_r2_l2b_role_certification.py tests/calibration/test_r2_l2c_metadata_census.py tests/calibration/test_r2_l2d_data_role_adjudication.py
```

Result for this closure run: **96 passed in 5.65s**. These are synthetic/contract tests; the Stage B artifact audit was a separate read-only operation on the existing output pair. No new dataset records were accessed.

Validator/test source SHA-256 values at this reconciliation:

| File | SHA-256 |
|---|---|
| `experiments/calibration/r2_l2b_role_certification.py` | `1829ac180480aaaa24fe5b4a2dee48df55db7d3534c82a4d97753e7f51856d88` |
| `experiments/calibration/r2_l2c_metadata_census.py` | `7768939117fc78bb5356a2de9bd6a1369f16447175f2a0ea8d556c7911a55f59` |
| `experiments/calibration/r2_l2d_data_role_adjudication.py` | `840600fabc0f0424afeab5438f291190b79a4667226d57ded665074f6924cab4` |
| `tests/calibration/test_r2_l2b_role_certification.py` | `ca909ab7ef4ed792dedb42f1ec27f1ac1b1cccd4e9772db863731546bd1c60d2` |
| `tests/calibration/test_r2_l2c_metadata_census.py` | `682edf01b6edcc2787ee2cecd6fbe1f4438fcdd57285f72043d928c85da21049` |
| `tests/calibration/test_r2_l2d_data_role_adjudication.py` | `1b11f2ed4c9705f9f21a39c32557bb455cfd3cf660cdebaa9d7f245a8805286e` |

## Final status

- `L2D_ENGINEERING_PROTOCOL`: `PASS_SCOPED_CPU`
- `L2C_STAGE_A_HISTORICAL_EVENT`: `AUTHORIZATION_FAILED_BEFORE_CENSUS` (historical; report preserved)
- `L2C_STAGE_B_OUTPUT_PAIR`: `INTERNAL_CONSISTENCY_PASS`
- `L2C_STAGE_B_AUTHORIZATION_SIGNATURE_AND_SCOPE`: `VERIFIED`
- `L2C_HISTORICAL_TRUST_PIN_TIMING`: `UNVERIFIED`
- `INDEPENDENT_SOURCE_BASELINE`: `NOT_ESTABLISHED`
- `A_B_C_ROLE_MEMBERSHIP`: `PENDING`
- `SUPPORT_C64_IDENTITY`: `PENDING_NOT_SELECTED`
- `G1_A_G1_B_G2`: `NOT_AUTHORIZED`
