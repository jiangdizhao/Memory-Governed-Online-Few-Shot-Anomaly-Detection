# R2-G1 Execution Contract

Status: **PROPOSAL ONLY**. This document defines a future G1-A/G1-B contract. It grants no dataset, image, feature, Support/C64, G1, or G2 authorization.

## 1. Separate estimands

### G1-A: fixed spatial descriptors

For each image, retain the 49 R1 grid cell-center descriptors at rows and columns `{2, 6, 10, 14, 18, 22, 26}` on the 28x28 patch grid, row-major. This is `r1_grid28x28_cell_centers_7x7_v1`. It does not use anomaly scores, labels, Support/C64, or B/C membership.

### G1-B: native C0 Top-32 descriptors

For each query/reference image, select exactly the 32 patch indices actually emitted by the frozen native C0 scorer. Do not locally rerank patches or approximate C0. Each receipt is bound to the image, category, representation, native scorer profile, category/seed fixed-bank receipt, M0 score-state memory digest, baseline-memory certificate, source receipt, and selection rule.

M0 means that category's frozen fixed-support bank plus an empty Greedy bank under `R1_M0_FIXED_SUPPORT_PLUS_EMPTY_GREEDY_V1`. It is not the evolving R1 stream state. Support identities are four deployed training-normal groups; an optional C64 uses 64 **additional** distinct training-normal groups. C64 is never described as strict four-shot-total.

The primary mapping is G1-A on G1-FULL-A (49 fixed positions over the broad A population) and G1-B on G1-PROTOCOL-MATCHED (native Top-32 under the category-specific M0 state on 64 C64 images, with four Supports). Do not conflate patch selection with population differences. For a patch-selection contrast, compute the fixed-49 companion on those exact same 64 C64 images under the same M0 context; the four Supports construct the fixed bank and are excluded from geometry. Do not attribute differences between broad A and C64 populations to patch selection.

## 2. Frozen representation and source identity

Use the existing `R1_DESCRIPTOR_PROFILE` in `experiments/calibration/r2_l2a_g1_geometry.py`:

- DINOv2 ViT-L/14 with four register tokens; checkpoint `dinov2_vitl14_reg4_pretrain.pth`.
- Expected checkpoint SHA-256 `36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51`; hub-source tree SHA-256 `b7c0037f90671c4932393bd56de8c8a205b29a6eb22dea563b6c8a88c471da15`; loaded-model-state SHA-256 `1f26464ba4055403e550f1d5c1333fb012631d98661df6993cabea730d17e3f9`.
- Hook layers `[5, 11, 17, 23]`; concatenate four 1024D outputs in hook order, then L2-normalize; `float32`, shape `(784, 4096)`.
- RGB; resize 448x448; center crop 392x392; PIL bicubic; the R1 mean/std normalization; identity view `id`.
- Representation fingerprint `06237800ef7044857bb617677625f9c2be79c0ce837ef83175ed915b23a91c28`; encoder fingerprint `859c5270f1b9e82888a29055273718ef35ad92b143c9f6362ec46dd8566eb5c0`; preprocessing fingerprint `ac0cf9ae583f0eeb766a39ad4c21049bbb4135b2d001c6dd43d6d35568d04a8d`; native scorer profile SHA-256 `ba3fa736734c2c6c3789f485f3288c5ae77177c68b6b3e24de7e257e447ced87`.

These are intended receipt values, not live hardware/model verification. A future run must hash the actual weight file, DINO hub source, loaded state, and allowed source files before extraction; mismatch stops before image decode. Record both the R1 profile source commit `c86d5b89fa0062b87d80adb324eaa6bc3c94448e` and the actual execution commit/dirty-tree identity. The actual G1 source commit must bind the new adapter and geometry code as well as dependencies.

The five R1 source fingerprints in the profile now match the pinned source commit and current files. Three stale metadata entries were corrected in this task; see the readiness report. This metadata repair does not alter the representation fingerprint or scorer mathematics.

## 3. Population and deterministic planning proposal

The public identity unit is one complete `official_sample_key` group under contract `realid-offical-sample-key-v2` and group hash algorithm `sha256:official-sample-group-nul-prefix-utf8.v1`. All five camera views remain in the same role. For G1-FULL-A, propose camera ID 1 for every group and fail closed if absent. For G1-PROTOCOL-MATCHED, preserve the existing D2 seeded one-view-per-group selector. The matched fixed-49 run uses the exact same selected C64 images as Top-32. Approve these rules before access; do not choose views from scores or labels.

- **G1-A full population:** all independently authorized TRAIN-normal A groups in each category.
- **G1-FULL-A bounded G1-A pilot:** one category (`switch`) with 100 distinct A groups; later, a separately approved two-category extension (`switch`, `audiojack`) with 100 per category. Rank groups by ascending `SHA256("r2-g1-bounded-pilot-v1\0" + profile_id + "\0" + category_id + "\0" + sample_seed + "\0" + group_digest)`, with group digest as tie-break. Proposed `sample_seed=0`; it is distinct from the L2E role seed and Support seed. The 100-group value is a planning floor, not a mathematical requirement.
- **G1-PROTOCOL-MATCHED:** use four deployed Supports plus 64 separately identified, additional C64 normal groups from A, following the existing category/seed selector. The four Supports build the category-specific fixed bank; the 64 C64 images are the query and reference geometry set, with source-group self-exclusion. Compute both fixed-49 and native Top-32 on those same 64 images for a matched patch-selection diagnostic. Supports and C64 are disjoint. C64 remains calibration-only and is never inserted into the fixed bank or Greedy memory. C64 is not strict four-shot-total. Use Support Seed 0 and historical C64 history seed 190021 for the initial proposed pilot; seed 0 is not authority to reuse another category bank.
- The existing selector is `experiments.calibration.real_iad_preflight._select_support_and_c64()` over sorted verified-normal TRAIN groups: C64 uses history seed 190021 to sample 64 group keys, sorted before view choice; C64 view choices use `random.Random(891 + i)` for sorted key position i. Four Support group keys are sampled from the remaining groups with Support Seed 0; Support views use a `random.Random(891)` stream. This is one view per group, disjoint by construction, and preserves the historical D2 selection. Its selector source fingerprints at the inspected parent are `real_iad_preflight.py` SHA-256 `b3b75b395b7e7ca0734f5796aa8e2365d2acaaef50a942c22c32c4d502c03fd8` and `risk_safe/preflight.py` SHA-256 `65d62b73a2835ee9625a26dc140c1c5b062be6292bc0b0981cd53218a424e0bb`; future execution must rehash them at its exact commit. The L2C Stage B aggregate reported 7,293 candidate TRAIN groups and a minimum of 167/category. This exceeds the 68-group matched Support/C64 planning count and the 100-group G1-FULL-A planning floor numerically, but it is not an A-role certificate or an availability receipt for either pilot category.

The L2B `G1_FULL_A` regime records four Support and 64 C64 counts as protocol metadata, but fixed-spatial G1-A does not consume those identities; they are not G1-A execution prerequisites. The L2B `G1_PROTOCOL_MATCHED` regime uses those four Supports and 64 additional C64 groups, with C64 as its geometry cohort. These are the existing population regimes, not new replacements.

For an A-only pilot, source-disjointness is enforced between query and reference rows using the full group and image digests. Global A/B/C certification remains a separate claim. For the bounded G1-FULL-A pilot, the 100 selected groups provide both query and reference descriptors; the engine excludes every query group from its own reference candidates. For G1-PROTOCOL-MATCHED, the same 64 C64 groups provide both query and reference descriptors; the four Support groups are bank inputs only. The bounded A sample rule and camera rule are proposals requiring researcher approval. No identities are assigned here.

## 4. H-SOURCE role assignment proposal

Reuse the existing R2-L2E policy `r2_l2e.h_source_partition_policy.v1`, SHA-256 `73afab2b6f306043094bd5e3c1643a8c004c04ba2237be6ba43b6eca25642719`; do not generate memberships in this task.

- A: all authorized TRAIN-normal source groups.
- Use full `official_sample_key` groups, with all views together. A has priority on group overlap; a TEST group matching A is excluded from B/C.
- For ordinary categories, route verified prior-exposed TEST groups to B. Rank remaining fresh TEST group digests by the frozen L2E hash formula and seed `20261101`; assign `floor(3N/5)` to B and the remainder to C.
- The four previously exposed development categories (`audiojack`, `regulator`, `rolled_strip_base`, `switch`) remain exposed B; C is proposed NOT_APPLICABLE only with explicit researcher approval and independent empty-role review.
- `mint` and `pcb` remain protected; B is proposed NOT_APPLICABLE and eligible fresh TEST groups are C-only after method freeze and separate authorization.
- Reject missing identity receipts, unknown categories, mismatched source snapshots, and every cross-role or cross-category overlap. Missing receipts never certify an empty role.

The public catalog and L2C TRAIN census are not local feature authorization. An A-only G1 can proceed without B/C certification if it has its own reviewed A identity/availability receipts and feature-access approval. This must not be reported as certified A/B/C partitioning.

## 5. Exact geometry and reporting

Reuse `compare_source_disjoint()`; do not change its distance implementation:

- exact blockwise `1 - clip(float64_dot, -1, 1)`;
- exclude same source group and same image before nearest-neighbor selection;
- exact nearest and fifth-nearest distances, ties ordered by distance then original reference row;
- strict `distance < radius` boundary;
- default 64x512 blocks with a 65,536-pair cap; no dense query-by-reference matrix;
- report group-aware query bootstrap conditional on the fixed reference set, including its limitation that it omits reference-set uncertainty.

Use the registered 101-point grid from 0 through 2 for descriptive coverage. No radius is selected here. For each category/profile, emit: profile and regime IDs; eligible A source-group count; query/reference group and image counts; query/reference descriptor counts; nearest and fifth-nearest distance quantiles; radius grid and nearest/fifth-neighbor coverage curves; same-source and same-image exclusions; missing-neighbor counts; exact equal-distance boundary counts; query-group bootstrap intervals conditional on the fixed reference set; representation, role/inventory, cache, scorer, support/M0 receipt hashes as applicable; and CPU/GPU/resource counters. Preserve unavailable denominators as unavailable. For complete 30-category reporting, apply the existing summary across every registered radius and fail closed if any category is missing or undefined; show category-equal macro, descriptor-pooled, worst category, and the mean of the lowest `ceil(10% * 30)` category coverages. Pooled patch counts are descriptors, not independent manufacturing samples. These are descriptive geometry summaries, not confidence guarantees, defect-safety evidence, or model-performance metrics. They do not substitute for G2 queried-anchor density.

## 6. Cache, receipt, and output contract

The existing P1 `FeatureCache` is in-memory, keyed by image/encoder/preprocessing identity, and says feature payload is not serialized. It has no bounded eviction interface. The G1 geometry API accepts in-memory, provenance-validated descriptor arrays and does not load models, images, metadata, or cache files. `FeatureCacheReceipt` is metadata-only and expects full `(784,4096)` inputs.

Before execution, add and CPU-test a minimal G1 adapter/receipt version that:
1. processes one authorized image at a time through the exact R1 encoder path and bounded cache lifetime;
2. derives fixed G1-A indices and, separately, actual native C0 Top-32 indices;
3. retains only projected float32 descriptors for the geometry cohort;
4. emits a content digest and role/cache/inventory receipts bound to the source, image/group digests, representation, projection rule, and selected rows;
5. contains no raw keys, paths, labels, masks, private feedback, or correction anchors.

Do not keep all full maps in the current unbounded cache. Any persistent feature cache and all temporary files must be inside a new unique output directory and covered by a pre-write byte budget. The proposed maximum for the bounded one/two-category pilot is **1 GiB (1,073,741,824 bytes)**, including feature descriptors, manifests, receipts, logs, and temporary/atomic-write files. Require at least 10 GiB free before launch. A full 30-category persisted descriptor cache does not fit this pilot budget; it needs a distinct resource approval.

## 7. Required receipt gates

G1-A requires, per category: verified official catalog identity; A-only TRAIN-normal identity and availability receipt; signed image-decode/feature-extraction scope; independently reviewed source/cache/inventory fingerprints; exact R1 representation and execution-source identity; and source-disjoint query/reference descriptors. It does not require B/C roles, C64, Support, Top-32, G1-B, or G2.

G1-B requires the A identity, authorization, representation, and cache gates that apply to its exact 64-image C64 cohort; it does not require completion of the separate broad-population G1-A study. It additionally requires: separate approval for four Support and 64 additional C64 selection under the matched profile; four actual deployed Support identities; category- and seed-specific structured fixed-bank digest; category-specific M0 state/certificate; actual native C0 Top-32 receipt for every C64 image; and independent verification of the Top-32, support, and baseline certificate digests. Historical switch/seed-0 fixed-bank receipt `8a710d028509350568e976bc3ede66aa59603499cbea525bdd8579322b1a1558` is the only retained fixed-bank digest. It cannot certify another category or seed. Missing or unreviewed Top-32/support/M0 receipts remain `NOT_YET_IDENTIFIABLE`.

## 8. Stop rules

Stop before decode/extraction if authorization, A availability, profile/source/weight hash, view rule, output path, unique run ID, disk floor, or receipt gate fails. Do not rerun L2C Census. Do not access TEST metadata, labels, masks, Mint/PCB identities, private feedback, or G2. Do not select a radius, tune, or claim empirical superiority. A future G1 run is not covered by any previous R1 or L2C authorization.
