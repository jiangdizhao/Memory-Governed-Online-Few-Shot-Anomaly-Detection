# R2-L2C Mode B Real Metadata Census — Stage A Authorization Gate

**Date:** 2026-11-01
**Current status:** `R2_L2C_LOCAL_ACCESS_NOT_AUTHORIZED`
**Census status:** Not run; stopped at Stage A as required

## Decision

The requested TRAIN metadata scope is explicit, but the task attachment does not include the required researcher-signed authorization manifest, researcher public policy-approver key, independently pinned trust directory, or out-of-band trust-directory SHA-256. The approval template located in `docs/calibration/audit` is default-deny; no signed L2C artifacts were supplied with this task. I therefore stopped before Stage B and did not run Mode B.

No path under `G:\dataset\realiad` was inspected. No local metadata, TEST contents, masks, images, feature caches, feedback ledgers, correction-anchor ledgers, or protected evaluation results were accessed. The suggested census output directory remains absent. The separate unsigned draft directory below is not a census output directory and must never be passed to `--mode execute`.

## Repository and source identity

- Branch: `calibration-stage0-contract-20261029`
- Expected and observed HEAD before this task: `95a372af139f32ae1bdca52319b2b4ca8dd42143`
- Tracked worktree and index at start: clean
- Pre-existing untracked work: preserved
- Current executor SHA-256: `7768939117fc78bb5356a2de9bd6a1369f16447175f2a0ea8d556c7911a55f59`
- Current executor test SHA-256: `682edf01b6edcc2787ee2cecd6fbe1f4438fcdd57285f72043d928c85da21049`
- Identity parser `experiments/risk_safe/preflight.py` SHA-256: `2070ba770ac36af06f688b0450c711d67b7bca526e3939178d559216535c25ba`
- Role and digest contract `experiments/calibration/r2_l2b_role_certification.py` SHA-256: `1829ac180480aaaa24fe5b4a2dee48df55db7d3534c82a4d97753e7f51856d88`
- Catalog contract `experiments/calibration/r2_l2a_allclass_inventory.py` SHA-256: `6ce759ebeeefb069a1a834fad6a343436e9e828e0bbd45003153dc44d345bc24`
- Prior readiness report SHA-256: `7abac407603af793ee7adc66abad8d9fccb5d1a1add5af2311a311ec6b07f495`
- Public catalog receipt SHA-256: `77f30230e3752a5921ea500695b7ad066103d07d870ac823b65ac5af310340bc`

The pinned public receipt `docs/calibration/audit/R2_L2A_R_OFFICIAL_PUBLIC_CATALOG_RECEIPT.json` passed the local catalog-header contract check. It has 30 distinct train category IDs and 30 distinct test category IDs, equal to `CANONICAL_CATEGORY_IDS`. Its pinned source is Tencent commit `98c28ffbe95b380d17a3777d77219152368c19e9`, `experiments/RealIAD-full/config.yaml`, SHA-256 `9c17d4ba78210ad65a288fedcdd6ea8a65d0926da05a960d3ed8dd1c3210552f`, size 5574 bytes. The pinned public config is [available at the cited commit](https://github.com/Tencent/AnomalyDetection_Real-IAD/blob/98c28ffbe95b380d17a3777d77219152368c19e9/experiments/RealIAD-full/config.yaml). The receipt and category inventory are public catalog evidence only; they do not authorize local data access.

The existing Mode A preflight returned `R2_L2C_EXECUTOR_READY_AWAITING_AUTHORIZATION`, `R2_L2C_LOCAL_ACCESS_NOT_AUTHORIZED`, 30 public categories, seven missing approval items, and `local_data_io_performed: false`.

## Unsigned authorization payload prepared

The unsigned canonical payload is at:

`outputs/calibration/r2_l2c_modeb_authorization_draft_20261101_v1/authorization_payload.unsigned.json`

Its digest is recorded separately in `authorization_payload.sha256`:

`241b2b2cae02739dc9b381202dea7318ab4eb43204364233acfb334b71dc8bfb`

This digest is for the current draft, which uses `RESEARCHER_ID_TO_BE_SUPPLIED`. It is not a valid authorization. The researcher must replace that placeholder with the ID whose public key appears in the trust directory, and recompute the digest before signing. The proposed authorization ID is `r2-l2c-hsource-all30-train-census-20261101-v1`; changing it also changes the digest.

The draft was generated from the executor's frozen constants and checked to contain exactly the 30 canonical categories, `mint` and `pcb` as individually authorized protected categories, the requested TRAIN/META permissions, all requested denials, exactly `REQUIRED_PUBLIC_OUTPUT_FIELDS` and `REQUIRED_PRIVATE_RECEIPT_FIELDS`, the suggested fresh census output path, and a 67,108,864-byte limit. Directory-serial sensitivity remains false. It contains no signature or private key.

### Required signed-manifest schema

The final manifest must contain exactly these keys (no extras):

`schema`, `authorization_id`, `researcher_id`, `decision`, `dataset_root`, `metadata_root`, `approved_categories`, `explicitly_authorized_protected_categories`, `allow_physical_json_file_bytes`, `allow_test_section_bytes_encountered`, `allow_train_image_path`, `allow_train_anomaly_class`, `allow_official_sample_key_condition_tokens`, `allow_meta_normal_class`, `allow_meta_prefix`, `allow_directory_serial_sensitivity`, `allow_private_group_receipt_output`, `allow_test_section_parsing`, `allow_mask_access`, `allow_feature_extraction`, `allow_image_decoding`, `allow_support_c64_selection`, `approved_output_fields`, `approved_private_receipt_fields`, `output_directory`, `maximum_output_bytes`, `signature_ed25519_hex`, and `authorization_sha256`.

The signature covers the ASCII bytes of the canonical payload: JSON with sorted keys, separators `(',', ':')`, `ensure_ascii=True`, and `allow_nan=False`, excluding `signature_ed25519_hex` and `authorization_sha256`. The `authorization_sha256` is SHA-256 of those same canonical bytes. The researcher must sign using their own Ed25519 key or signing device and add the lowercase hexadecimal signature. The private key must remain under researcher control; it must not be sent to Codex, placed in the repository, or written into the census output.

### ReviewerTrustDirectory construction

No trust directory was created because the researcher public key and identity are absent. Once the researcher supplies a 32-byte lowercase-hex Ed25519 public key, construct the access-approval trust directory with `make_reviewer_trust_directory` from `experiments.calibration.r2_l2b_role_certification`:

```python
trust = make_reviewer_trust_directory(
    directory_id="researcher-selected-r2-l2c-access-root-id",
    trusted_policy_approver_public_keys={researcher_id: researcher_public_key_hex},
    trusted_identity_source_sha256s=(),
)
```

The helper derives `trusted_policy_approver_ids` from that mapping; other reviewer, identity-source, and role-policy lists remain empty for this data-access decision. Serialize the complete `ReviewerTrustDirectory` dataclass, including its computed `directory_sha256`. The researcher must independently pin and provide that SHA-256 out of band. This access trust root does not certify A/B/C membership. Independent Reviewer certification of A/B/C identities is a later, separate process and is not required merely to count TRAIN metadata.

## Category coverage

No category JSON was opened. The following 30 categories are each `NOT_SCANNED_SIGNED_ARTIFACTS_MISSING`; this is an authorization stop, not evidence that a category file or normal reference is missing.

| Category | Census status |
|---|---|
| audiojack | NOT_SCANNED_SIGNED_ARTIFACTS_MISSING |
| bottle_cap | NOT_SCANNED_SIGNED_ARTIFACTS_MISSING |
| button_battery | NOT_SCANNED_SIGNED_ARTIFACTS_MISSING |
| end_cap | NOT_SCANNED_SIGNED_ARTIFACTS_MISSING |
| eraser | NOT_SCANNED_SIGNED_ARTIFACTS_MISSING |
| fire_hood | NOT_SCANNED_SIGNED_ARTIFACTS_MISSING |
| mint | NOT_SCANNED_SIGNED_ARTIFACTS_MISSING |
| mounts | NOT_SCANNED_SIGNED_ARTIFACTS_MISSING |
| pcb | NOT_SCANNED_SIGNED_ARTIFACTS_MISSING |
| phone_battery | NOT_SCANNED_SIGNED_ARTIFACTS_MISSING |
| plastic_nut | NOT_SCANNED_SIGNED_ARTIFACTS_MISSING |
| plastic_plug | NOT_SCANNED_SIGNED_ARTIFACTS_MISSING |
| porcelain_doll | NOT_SCANNED_SIGNED_ARTIFACTS_MISSING |
| regulator | NOT_SCANNED_SIGNED_ARTIFACTS_MISSING |
| rolled_strip_base | NOT_SCANNED_SIGNED_ARTIFACTS_MISSING |
| sim_card_set | NOT_SCANNED_SIGNED_ARTIFACTS_MISSING |
| switch | NOT_SCANNED_SIGNED_ARTIFACTS_MISSING |
| tape | NOT_SCANNED_SIGNED_ARTIFACTS_MISSING |
| terminalblock | NOT_SCANNED_SIGNED_ARTIFACTS_MISSING |
| toothbrush | NOT_SCANNED_SIGNED_ARTIFACTS_MISSING |
| toy | NOT_SCANNED_SIGNED_ARTIFACTS_MISSING |
| toy_brick | NOT_SCANNED_SIGNED_ARTIFACTS_MISSING |
| transistor1 | NOT_SCANNED_SIGNED_ARTIFACTS_MISSING |
| u_block | NOT_SCANNED_SIGNED_ARTIFACTS_MISSING |
| usb | NOT_SCANNED_SIGNED_ARTIFACTS_MISSING |
| usb_adaptor | NOT_SCANNED_SIGNED_ARTIFACTS_MISSING |
| vcpill | NOT_SCANNED_SIGNED_ARTIFACTS_MISSING |
| wooden_beads | NOT_SCANNED_SIGNED_ARTIFACTS_MISSING |
| woodstick | NOT_SCANNED_SIGNED_ARTIFACTS_MISSING |
| zipper | NOT_SCANNED_SIGNED_ARTIFACTS_MISSING |

Authorized categories scanned: **0/30**. Normal-image counts, source-group counts, camera-view distributions, 64/68 thresholds, and category file hashes are **not available**, not zero. A-role candidate status is **not assessed**; no A candidates were derived by this task. No A/B/C partition, independent C confirmation, or Support/C64 selection was attempted.

## Validation and outputs

- Repository branch and expected HEAD: verified.
- Tracked tree and index: clean at start.
- Public catalog receipt: validated against the pinned catalog contract and 30-category canonical inventory.
- Mode A preflight: passed, no local data I/O.
- Unsigned payload: canonical byte representation and SHA-256 verified; no signature present.
- Suggested census output directory `outputs/calibration/r2_l2c_hsource_all30_train_census_20261101_v1`: absent and not created.
- No category source-file SHA-256 values exist because no local JSON file was opened.
- No CUDA/GPU, DINO, image decoding, TEST parsing, mask parsing, feedback/anchor access, geometry, tuning, or evaluation occurred.

No independent partition evidence is a census blocker: if the signed access artifacts are later supplied and pass preflight, the expected census result remains `R2_L2C_METADATA_CENSUS_COMPLETE_PARTITION_PENDING` unless separately reviewed A/B/C receipts are supplied. The census itself does not select Supports or C64. For any future G1 extraction, first require separately certified, source-disjoint A/B/C identities; preserve G1-FULL-A and G1-PROTOCOL-MATCHED as distinct regimes; and verify any four-Support plus additional C64 selection as a separate, category- and seed-bound reviewed receipt.

## Missing artifacts and next authorized step

Required before Stage B:

1. Researcher-selected `researcher_id` and the completed, researcher-signed manifest matching the exact schema above.
2. The corresponding researcher public policy-approver key inside a complete `ReviewerTrustDirectory` file.
3. The trust-directory SHA-256 independently pinned and supplied out of band.
4. The local paths to those two files, both outside `G:\dataset\realiad`.

After receiving them, run Mode A preflight with the signed manifest, trust directory, and independent SHA-256 pin. Only if it validates should the prescribed CPU-only Mode B command run. It is **not run here**:

```powershell
$env:CUDA_VISIBLE_DEVICES = ''
$PY = 'D:\anaconda3\envs\torch\python.exe'
& $PY -m experiments.calibration.r2_l2c_metadata_census `
  --mode execute `
  --access-manifest '<RESEARCHER_SIGNED_MANIFEST_PATH>' `
  --trust-directory '<INDEPENDENTLY_PINNED_TRUST_DIRECTORY_PATH>' `
  --expected-trust-directory-sha256 '<RESEARCHER_SUPPLIED_SHA256>'
```

This report is intentionally left uncommitted for researcher review. The unsigned payload and report are local, untracked artifacts; no signed access material or private source-group receipts were added to Git.
