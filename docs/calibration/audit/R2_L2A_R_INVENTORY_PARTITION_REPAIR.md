# R2-L2A-R — Official Inventory and Partition Contract Repair

**Status:** `R2_L2A_R_PUBLIC_INVENTORY_PASS`; `R2_L2A_R_PARTITION_PENDING`;
`R2_L2A_R_G1_NOT_AUTHORIZED`

## Scope and evidence boundary

This report completes the R2-L2A-R continuation without overwriting the prior
R2-L2A readiness report. It records the public inventory separately from local
partition authorization and geometry results.

The work was CPU-only. No local Real-IAD files, sample keys, protected group
identities, images, labels, masks, feature caches, DINOv2 weights, or GPU
execution were accessed. The public configuration below is metadata only; it
does not authorize local data access or establish G1 geometry.

## A. Pinned official 30-category inventory

The parser fetched the exact pinned public source
[`experiments/RealIAD-full/config.yaml` at commit `98c28ff`](https://github.com/Tencent/AnomalyDetection_Real-IAD/blob/98c28ffbe95b380d17a3777d77219152368c19e9/experiments/RealIAD-full/config.yaml)
from its matching raw URL. The downloaded file was 5,574 bytes with SHA-256
`9c17d4ba78210ad65a288fedcdd6ea8a65d0926da05a960d3ed8dd1c3210552f`. Its
dataset config version is `v1.0.0`. Extraction algorithm:
`strict_yaml_dataset_train_test_meta_file_v1`.

The parser reads only the `dataset.train.meta_file` and
`dataset.test.meta_file` lists, extracts category identifiers from the
expected public JSON filename format, and rejects malformed YAML list entries,
duplicate categories, a train/test set mismatch, an unexpected config version,
or any catalog that differs from the pinned 30-name cross-check. It does not
retain those list paths in the receipt. The receipt projects category names,
source commit/path, source byte digest, config version, and extraction
algorithm; it flags image-path and label/mask fields as absent.

Both lists contain the same 30 distinct categories:

`audiojack, bottle_cap, button_battery, end_cap, eraser, fire_hood, mint, mounts, pcb, phone_battery, plastic_nut, plastic_plug, porcelain_doll, regulator, rolled_strip_base, sim_card_set, switch, tape, terminalblock, toothbrush, toy, toy_brick, transistor1, u_block, usb, usb_adaptor, vcpill, wooden_beads, woodstick, zipper`.

The inventory result is `VERIFIED_30_CATEGORY_PUBLIC_INVENTORY`. Every local
category access status remains `PENDING_AUTHORIZATION`; a public catalog is
not a local data authorization, partition identity receipt, or geometry
result. Source validation accepts this exact pinned file only, not arbitrary
GitHub pages or mutable branch URLs.

The category-only serialized receipt is
[`R2_L2A_R_OFFICIAL_PUBLIC_CATALOG_RECEIPT.json`](R2_L2A_R_OFFICIAL_PUBLIC_CATALOG_RECEIPT.json).
It retains only public source provenance, category identifiers and verification
counts. It contains no source-list paths, local paths, image paths, labels, or
masks. Its authorization fields explicitly remain false.

## B. Explicit A/B/C role policy and identity preflight

`partition_preflight()` now requires a frozen policy to cover every category ×
role pair. Each entry must be `REQUIRED`, `NOT_APPLICABLE` with a written
rationale, or `PENDING_AUTHORIZATION`. The policy digest is canonicalized and
must be supplied as externally verified before a `NOT_APPLICABLE` entry can
count as resolved. Missing receipts for `REQUIRED` roles remain missing;
absence is never interpreted as an empty role. Group overlap, duplicate group
identity, invalid receipt provenance, and receipts against a `NOT_APPLICABLE`
entry fail closed. Unknown or unreviewed C identity remains blocked.

A G1 reference population additionally requires a separate externally reviewed
count-only A availability receipt for authorized normal references, with a
positive image count and source-group count consistent with the A identity
receipts. Per-category A authorization is reported as pending,
`NOT_AUTHORIZED_FOR_G1`, or `AUTHORIZED_FOR_G1_REFERENCE_POPULATION`. An
unauthorized A category cannot be used as a G1 reference population.

The preflight reports these states separately: public catalog completeness,
policy declaration, external policy review, role identity receipt completion,
external role disjointness certification, authorized A reference availability,
and G1 geometry completion. The current audit has no proposed role policy,
group identity receipts, C certificate, or A availability receipts. Current
partition/G1 state is therefore:

| State | Result |
|---|---|
| Public 30-category inventory | `VERIFIED_30_CATEGORY_PUBLIC_INVENTORY` |
| Role policy declared / externally reviewed | No / No |
| Role identity receipts complete | No |
| Role disjointness externally certified | No |
| Authorized A data available | No |
| G1 geometry complete | No; geometry was not run |

No categories or group digests were fabricated to fill these gaps.

## C. G1-B representation and fixed-support identity

`R1DescriptorProfile.representation_sha256` is now class-invariant: it binds
the DINO model/checkpoint, hub and loaded model states, hook layers, feature
fusion, preprocessing/view, feature shape/dtype, normalization, and
encoder/preprocess fingerprints. It excludes experiment-specific profile and
source-hash fields, the native scorer profile, and the fixed-support digest.
The native scorer profile and category-specific M0 state are bound separately.
The computed representation identity is
`06237800ef7044857bb617677625f9c2be79c0ce837ef83175ed915b23a91c28`.
Feature-cache compatibility uses this representation identity.

The frozen encoder fingerprint is
`859c5270f1b9e82888a29055273718ef35ad92b143c9f6362ec46dd8566eb5c0`; the
preprocessing fingerprint is
`ac0cf9ae583f0eeb766a39ad4c21049bbb4135b2d001c6dd43d6d35568d04a8d`. Both are
part of the class-invariant representation identity.

G1-B selection receipts bind the category to its own fixed-support category,
seed, structured digest algorithm, structured bank digest, and fixed-support
receipt digest. The support-receipt digest is content-bound to category, seed,
algorithm, and structured bank digest. The separate M0 scorer-state digest
binds the category, representation, scorer profile, category support receipt,
score-state memory, frozen M0 contract, and M0 certificate. Therefore, changing
a receipt's category while reusing switch's support receipt or M0 state fails
validation. The historical digest
`8a710d028509350568e976bc3ede66aa59603499cbea525bdd8579322b1a1558` is retained
for `switch`, seed 0 only; other categories cannot claim it. A switch/seed-0
receipt must match it. The validator requires external verification of the
Top-32 receipt, category-specific support receipt, and M0 certificate. Missing
or unverified inputs report `NOT_YET_IDENTIFIABLE`.

The streaming distance, source-disjoint exclusion, radius, neighbor, and
bootstrap mathematics were not changed. Tests use synthetic NumPy arrays only.
The G1 descriptor input now fails closed unless its access status is
`AUTHORIZED_FOR_G1_REFERENCE_POPULATION` and the A availability receipt digest
matches the externally reviewed receipt mapped to that category; the receipt
digest is retained in geometry output metadata.
No claim is made that any rebuilt bank matches a historical bank or that any
category's native Top-32 receipt has been verified. No real feature extraction
or scoring occurred; the feature-selection test uses a synthetic tensor.

## D. Geometry mathematics and results boundary

The diff leaves the distance/statistics implementation unchanged. Existing
tests verify exact blockwise cosine distance (`1 - clip(float64 dot, -1, 1)`),
strict `distance < radius`, same-source and same-image exclusion before
selection, exact nearest and fifth-nearest results, deterministic reference-row
tie breaking, category-macro and pooled summaries, worst-category and lower-tail
summaries, and query-source-group bootstrap behavior. No radius was selected
from data. Actual real-data distance statistics are **`NOT RUN`**.

## E. Final authorization and readiness states

| Area | State | Evidence |
|---|---|---|
| Public 30-category inventory | `R2_L2A_R_PUBLIC_INVENTORY_PASS` | Pinned public config SHA and matching 30-name train/test projections |
| Local data inventory | `PENDING` | No local dataset JSON, image, label, mask, or group identity access authorized or performed |
| A/B/C partition | `R2_L2A_R_PARTITION_PENDING` | No frozen externally verified policy, identity receipts, or disjointness certificate |
| G1-A descriptors | `PENDING` | No A population authorization receipt or descriptor access |
| G1-B native receipts | `PENDING` | Contract is category-bound; no native Top-32, fixed-bank, or M0 receipt was supplied or verified |
| Real-data distance statistics | `NOT RUN` | No local features or images scanned |
| Overall G1 authorization | `R2_L2A_R_G1_NOT_AUTHORIZED` | Local A authorization and native G1-B receipts remain absent |

This is not `ALL_30_GEOMETRY_PASS`. The public catalog does not imply local
data authorization.

## F. Verification

Commands executed with `D:\anaconda3\envs\torch\python.exe`:

```powershell
& 'D:\anaconda3\envs\torch\python.exe' -m compileall -q experiments/calibration tests/calibration
& 'D:\anaconda3\envs\torch\python.exe' -m pytest -q tests/calibration/test_r2_l1_locality_kernels.py tests/calibration/test_r2_l2a_allclass_geometry.py
```

Results: compilation passed; **82 tests passed**. A separate read-only source
verification command retrieved the pinned public config bytes, verified the
exact SHA-256 and extracted the same 30 categories from train and test; the
category-only receipt JSON was then checked against those bytes. Synthetic
tests cover strict source-URI and source-digest rejection, duplicate and inconsistent catalogs,
explicit policy coverage and review, missing-role handling, reviewed
`NOT_APPLICABLE` roles, unresolved C identity, role disjointness, A
authorization receipts and G1 access enforcement, class-invariant
representation identity, category-bound support and M0 state identities,
cross-category switch-bank rejection, and unchanged synthetic streaming
geometry contracts.

No GPU or DINOv2 model was loaded. No local dataset labels, masks, or protected
group identities were read. Thus actual partitions and all-class geometry
remain unverified.

Source and test SHA-256 at audit time:

| File | SHA-256 |
|---|---|
| `experiments/calibration/r2_l2a_allclass_inventory.py` | `6ce759ebeeefb069a1a834fad6a343436e9e828e0bbd45003153dc44d345bc24` |
| `experiments/calibration/r2_l2a_g1_geometry.py` | `3fd3ce36a1317c7c408cdbdc829d87d238aefaab81e453ecf56c36a43aa2201f` |
| `tests/calibration/test_r2_l2a_allclass_geometry.py` | `932329d67525884e14e2116b138ef628c8441e37aade1e22f5712b4575390a3e` |
| `docs/calibration/audit/R2_L2A_R_OFFICIAL_PUBLIC_CATALOG_RECEIPT.json` | `77f30230e3752a5921ea500695b7ad066103d07d870ac823b65ac5af310340bc` |

Branch and starting HEAD were `calibration-stage0-contract-20261029` and
`743d639468998e6e1a6ac598eb1251518e8facfe`, matching the supplied expected
identity. The worktree also contains numerous pre-existing untracked audit and
output directories; they were preserved. Only the scoped R2-L2A-R source,
tests, public category receipt, and this report are eligible for the local
commit. No push is authorized.

## Disposition

The public inventory pass is complete. Partition certification remains pending
an explicit externally reviewed role policy, identity-only A/B/C receipts,
disjointness certification, and C identity certification where C is required.
G1-A remains unauthorized until a reviewed A reference-availability receipt
exists. G1-B remains unidentified until category-specific native Top-32,
support-bank, and M0 scorer-state receipts are supplied and externally
verified. No protected data access, GPU work, or empirical evaluation is
authorized by this report.
