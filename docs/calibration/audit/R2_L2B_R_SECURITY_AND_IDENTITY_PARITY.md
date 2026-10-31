# R2-L2B-R Security and Identity Parity Audit

Date: 2026-11-01
Branch: `calibration-stage0-contract-20261029`
Base commit: `2afe966c5ae5693ba703cb29a4768caeead8608c`

## Result

`R2_L2B_R_CORRECTIONS_PASS`

The CPU-only access, identity, and readiness repairs passed the requested R2-L1, R2-L2A, and R2-L2B synthetic regression suites. This result does not authorize local Real-IAD access or G1/G2 execution. No local dataset JSON, image, label, or mask was opened, and no feature extraction or geometry evaluation was run.

## Corrections

### A. Label and metadata access gates

- Added `anomaly_class` to both train label-bearing and test outcome field sets. Train reads require `allow_train_normal_ok_status=True`; test reads require `allow_outcome_label_access=True`.
- Added explicit train and test known-field registries. Unknown names are rejected in both a scope allowlist and a future `require_metadata_scope()` request, even if the unknown name was placed in the allowlist.
- Kept segmentation fields under the independent `allow_mask_access` gate. The known set includes `mask`, `mask_path`, `gt_mask`, `gt_mask_path`, `anomaly_mask`, `anomaly_mask_path`, `segmentation`, and `segmentation_path`.
- Kept image identity and condition-bearing sample-key tokens under their existing separate permissions. The access guard still does not read data itself.

### B. Frozen source-group identity parity

`source_group_digest_from_key()` now returns the exact R1 and Stage 0.5 operational digest:

```text
sha256(("official-sample-group\0" + str(sample_key)).encode("utf-8")).hexdigest()
```

The algorithm is identified as `sha256:official-sample-group-nul-prefix-utf8.v1`. Role receipts now use schema `r2_l2b.role_certification.v2`, and both role receipts and reviewed identity manifests bind that exact algorithm identifier. The historical identity contract spelling remains `realid-offical-sample-key-v2`; the grouping unit remains `official_sample_key`. Receipts using another or mixed group digest algorithm fail closed. No production receipt migration was introduced.

The frozen R1 and Stage 0.5 sources were read-only and unchanged:

| Source | SHA-256 |
|---|---|
| `experiments/calibration/r1_matched_correction_pilot.py` | `3208e2c52277ad76351eb86968e362aa315effe9e3c734bfe2b6ff7bab0db448` |
| `experiments/calibration/stage05.py` | `4b8aae252f5f1e096b127906794064da269aa79f7cd5514f6f738cda585c017e` |

Synthetic C1/C2 path fixtures map to one group digest, and a direct expected-value assertion matches the frozen formula.

### C. Readiness state consistency

- Policy status now reports `RESEARCHER_POLICY_APPROVED` independently of local access and identity certification.
- Local status reports `LOCAL_SCOPE_AUTHORIZED_METADATA_ONLY` only when the supplied local scope is valid. Per-category rows also reflect their category-specific scope and protected-category authorization. A public catalog receipt still only establishes public category membership.
- Fully certified synthetic A/B/C role identities and A availability report `R2_L2B_ROLE_IDENTITIES_CERTIFIED` and the corresponding reference-pool readiness states.
- Protocol-matched pool readiness derives from verified evidence: role certification, A availability, and at least 64 authorized normal A images across 64 distinct A source groups per category. This reports candidate-pool availability only; C64 selection and separate G1 regime authorization remain pending.
- Geometry status remains `NOT_AUTHORIZED_OR_RUN` and `g1_geometry_complete` remains false. A pool authorization is not reported as completed geometry or actual distance statistics.

The fully satisfied readiness test uses ephemeral synthetic signing keys and synthetic category/role receipts only. It does not create production reviewer credentials, production group counts, or real certificates. Incomplete/default-deny evidence remains pending.

## Regression and provenance

Command:

```text
python -m pytest tests/calibration/test_r2_l1_locality_kernels.py tests/calibration/test_r2_l2a_allclass_geometry.py tests/calibration/test_r2_l2b_role_certification.py -q
```

Result: **122 passed** (R2-L1: 44; R2-L2A: 38; R2-L2B: 40).

Coverage includes train/test `anomaly_class` rejection without the corresponding authorization, separate mask-field authorization, unknown-field rejection, exact frozen group SHA-256 parity, multiview grouping, mixed-algorithm rejection, Ed25519 approval and identity-signature negative cases, complete and incomplete readiness states, and the existing exact G1 geometry tests.

The G1 geometry implementation was not edited. Its SHA-256 remains:

`experiments/calibration/r2_l2a_g1_geometry.py` - `3fd3ce36a1317c7c408cdbdc829d87d238aefaab81e453ecf56c36a43aa2201f`

The L2A exact blockwise cosine distance, strict radius comparison, source/image exclusions, nearest/fifth-nearest selection, tie ordering, and summary statistics therefore remain covered by the passing L2A suite without a source change.

## Remaining authorization and execution requirements

- The synthetic trust directory, Ed25519 keys, role identities, and A availability receipts are test-only. No production reviewer trust root, researcher approval, local data scope, or independently reviewed A/B/C manifests were supplied or created by this task.
- Local data inventory and role certification remain pending until real researcher authorization and independent source evidence are provided.
- Protocol-matched G1 still needs its separate signed regime authorization and source-identified C64 selection evidence.
- No local Real-IAD scan, JSON access, labels, masks, image decoding, DINO/feature extraction, G1 geometry run, G2 run, or scientific outcome evaluation was performed or authorized.
