# R2-L2A All-Class Geometry Census and G1 Audit Readiness

**Classification: `R2_L2A_INVENTORY_OR_PARTITION_BLOCKED`**

The CPU-only inventory/partition contracts and exact streaming geometry engine are implemented and pass synthetic regression. A verified 30-category inventory, A/B/C identity certification, authorized A descriptors, and actual G1 estimates are still absent. This report does not claim `ALL_30_GEOMETRY_PASS` or authorize a scan.

## Repository identity and scope

- Repository: `E:\Memory-Governed-Online-Few-Shot-Anomaly-Detection`
- Branch: `calibration-stage0-contract-20261029`
- Required parent and task-start HEAD: `cc213c40ba067afb83dfab984379019a5ba08e5d`
- Tracked worktree at task start: clean. Pre-existing untracked audit and output paths were preserved and not staged.
- GPU/CUDA runtimes and DINOv2/VisionAD models or checkpoints were not loaded or executed. No Real-IAD image was decoded; no label, mask, feature-bearing shard, private feedback ledger, or correction-anchor ledger was opened.
- No Mint/PCB material, R1/R2-0 replay, R2-1 evaluation, radius selection, or G2 run was performed.

Only these four files are in scope for this task:

- `experiments/calibration/r2_l2a_allclass_inventory.py`
- `experiments/calibration/r2_l2a_g1_geometry.py`
- `tests/calibration/test_r2_l2a_allclass_geometry.py`
- `docs/calibration/audit/R2_L2A_ALLCLASS_GEOMETRY_READINESS.md`

## Official category inventory and A/B/C partition

The official CVPR paper identifies Real-IAD as a 30-class dataset, but its category chart is explicitly partial and does not enumerate all 30 canonical identifiers. The official Hugging Face card says there are 30 per-object result files and shows examples, while requiring acceptance of access conditions before its repository files/content can be read. Neither source, as safely available in this audit, provides a complete non-label-bearing canonical list. The gated dataset content was not accessed. See the [official CVPR 2024 paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_Real-IAD_A_Real-World_Multi-View_Dataset_for_Benchmarking_Versatile_Industrial_Anomaly_CVPR_2024_paper.pdf), the [official Hugging Face dataset card](https://huggingface.co/datasets/Real-IAD/Real-IAD), and the [official project page](https://realiad4ad.github.io/Real-IAD/).

The inventory preflight therefore reports 30 explicit census slots, all with `PENDING_AUTHORIZATION`; unresolved slots have `category_id: null` and are not represented as discovered category names. A partial list can never become a verified 30-category inventory. Per-category local group/image counts remain `null`.

The existing identity convention is preserved from `experiments/risk_safe/preflight.py`: primary grouping is the official `official_sample_key` (`<category>_<NNNN>_<OK|NG[_DEFECT]>`) under identity contract `realid-offical-sample-key-v2` (historical spelling retained). Camera views from a group stay together. New partition receipts carry only SHA-256 group identities and receipt digests; they reject raw sample keys, image paths, labels, masks, role overlap, duplicate identities, incompatible identity contracts, and unverified C receipts.

| Role | Intended use | Current state |
|---|---|---|
| A | Parameter-design reference | `PENDING_AUTHORIZATION`; no certified group receipt or descriptor inventory |
| B | Development evaluation | `PENDING_AUTHORIZATION`; no certified group receipt |
| C | Sealed confirmation | `PENDING_AUTHORIZATION`; no independently certified identity receipt |

The preflight refuses to label A/B/C disjointness as certified until every category has role receipts, the identity receipt digests have been externally reviewed, and each C group matches an independently reviewed C-certificate digest. SHA-256 fields alone do not authenticate a receipt; the implementation does not claim to verify an issuer signature. No overlap conclusion is made from filenames or local cache presence.

## R1 representation identity and cache feasibility

The frozen R1 source and its run manifest establish this descriptor profile. The manifest's raw `hook_layers` list is included in the encoder fingerprint rather than emitted as a separate field; hook positions below are derived from the source rule and ViT-L's 24 transformer blocks.

| Field | R1 profile |
|---|---|
| Encoder | DINOv2 ViT-L/14 register model, `dinov2_vitl14_reg` |
| Weights | `dinov2_vitl14_reg4_pretrain.pth`; SHA-256 `36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51` |
| Hook layers | `[5, 11, 17, 23]`, from the source's four-stage hook rule |
| Fusion | Concatenate four 1024-D layer outputs, then L2-normalize |
| Preprocessing | RGB; resize 448×448; center crop 392×392; BICUBIC; model mean/std normalization |
| View | `id` |
| Full feature tensor | 784 positions on a 28×28 patch grid × 4096 dimensions; float32 |
| Correction interface | Native C0 Top-32 feature rows, shape 32×4096, float32 and L2-normalized |
| Hub source tree | SHA-256 `b7c0037f90671c4932393bd56de8c8a205b29a6eb22dea563b6c8a88c471da15` |
| Loaded model state | SHA-256 `1f26464ba4055403e550f1d5c1333fb012631d98661df6993cabea730d17e3f9` |
| Encoder fingerprint | SHA-256 `859c5270f1b9e82888a29055273718ef35ad92b143c9f6362ec46dd8566eb5c0` |
| Preprocessing fingerprint | SHA-256 `ac0cf9ae583f0eeb766a39ad4c21049bbb4135b2d001c6dd43d6d35568d04a8d` |
| Native scorer profile | SHA-256 `ba3fa736734c2c6c3789f485f3288c5ae77177c68b6b3e24de7e257e447ced87` |
| Fixed support receipt | Structured `_fixed_bank_digest` SHA-256 `8a710d028509350568e976bc3ede66aa59603499cbea525bdd8579322b1a1558` |

R1 source commit: `c86d5b89fa0062b87d80adb324eaa6bc3c94448e`. Source fingerprints recorded in the R1 receipt include:

| Source | SHA-256 |
|---|---|
| `experiments/calibration/r1_matched_correction_pilot.py` | `3208e2c52277ad76351eb86968e362aa315effe9e3c734bfe2b6ff7bab0db448` |
| `experiments/calibration/native.py` | `3a347d65aaf4bfd6b41eff2ea7339d878065bb2e08ed64795f57b0458135dd45` |
| `experiments/p0_protocol_audit.py` | `b83918ec5aceb3ef63cca942fc9fb0e241559d9672d20bed422b15fcd48135e7` |
| `vmb_visionad_new.py` | `bd6545bf5ff63acce01792e5d42aa964c88e73fb59670febeb55f2d1139a409a` |
| `test_phase3_AIF_simplified_v22.py` | `bfc871d5a9a32b2f5bdb9d7aab5177e702dec2e1110a6632f20bd13c20fcba30` |

The metadata-only cache receipt validator requires exact model, checkpoint, hub-tree, encoder, preprocessing, scorer, shape, dtype, view, normalization, inventory, and official group-identity provenance. It also requires role A and a normal-reference-only population, rejects labels/masks, image paths, private feedback, or correction anchors, and remains pending until external reviewers verify both cache and inventory receipt digests. No existing cache receipt certifies an all-class A population. R1 is a single-category switch run and its profile receipt is not an all-class geometry cache. No feature tensor or event shard was inspected, so compatibility of any existing feature-bearing shard is **unverified**.

G1-A is preregistered as 49 deterministic cell-center patches on the 28×28 grid (`r1_grid28x28_cell_centers_7x7_v1`), independent of observed distances. G1-B is `NOT_YET_IDENTIFIABLE` at present. It requires native C0 Top-32 indices emitted under the locked scorer, the R1 structured fixed-support digest, a certified fixed-support-plus-empty-Greedy M0 state, and externally reviewed identity-only source and baseline receipts. The implementation accepts only those emitted indices; it never recomputes or distance-ranks a convenient Top-32 subset.

## G1 streaming geometry implementation

`r2_l2a_g1_geometry.py` implements:

- finite float32/float64, fixed-dimension, unit-norm descriptor validation and SHA-256 identity checks;
- cosine distance `1 - clip(float64_dot, -1, 1)`;
- exact blockwise nearest and fifth-nearest distances, with same-source-group and same-image exclusions applied before top-k selection;
- deterministic ties by `(distance, original reference row index)`;
- explicit no-reference and fewer-than-five states (`NaN` distance plus count/index status), never a fabricated fifth neighbor;
- a strictly increasing diagnostic radius grid spanning `[0,2]`; compact-support counts use only `distance < radius`, and exact equality is tallied separately;
- per-radius nearest coverage, ≥5-local-reference fraction, zero-local-reference fraction, local-reference/overlap multiplicity, boundary equality counts, and nearest/fifth distance quantiles;
- deterministic resource caps: default query/reference blocks 64×512 and a maximum 65,536 pair elements. The dense query×reference matrix is never allocated;
- source-group-cluster bootstrap intervals for query-side coverage, conditional on a fixed reference set. Fewer than two independent query groups returns an undefined interval. Reference-set uncertainty and adaptive selection are not covered.

Thirty-category summaries require all 30 expected category IDs and valid data for each. Otherwise macro, pooled, worst-category, and lower-tail values are `null`; the code does not silently change the denominator. Complete summaries distinguish category-equal macro coverage, query-descriptor-weighted pooled coverage, worst category, and the mean of the lowest three categories (10% lower tail). Pooled results are descriptive and do not treat descriptors as independent observations.

No radius, radius quantile, multiplier, or regularizer was selected.

## Synthetic validation

Commands run in the existing Windows Torch environment with CUDA hidden:

```powershell
$env:CUDA_VISIBLE_DEVICES=''
$PY = 'D:\anaconda3\envs\torch\python.exe'
& $PY -m py_compile experiments/calibration/r2_l2a_allclass_inventory.py experiments/calibration/r2_l2a_g1_geometry.py tests/calibration/test_r2_l2a_allclass_geometry.py
& $PY -m pytest -q tests/calibration/test_r2_l1_locality_kernels.py tests/calibration/test_r2_l2a_allclass_geometry.py
```

Final result: **69 passed** (44 existing R2-L1 locality tests plus 25 new R2-L2A tests); `py_compile` succeeded. Two preceding combined runs caught stale test expectations after the profile source list and external C-certificate contract were tightened; both assertions were corrected. An intermediate EOF-normalization command briefly introduced literal `\n` text and caused a compile failure; the file endings were repaired before the final compile/test run. Tests cover the 30-category synthetic universe, category-size imbalance, source exclusion/self-match, no and insufficient reference sets, ties, exact radius boundary equality, brute-force versus streaming parity, deterministic sampling and bootstrap, resource bounds, strict identity/provenance validation, cache mismatch, incomplete inventory, protected receipt contamination, A/B/C overlap, and macro/pooled/worst/lower-tail disagreement.

Test provenance is synthetic NumPy only. No CUDA, Torch model, DINO weights, images, labels, masks, or real feature arrays were loaded.

## Resource plan for a separately authorized scan

No A-role images are currently authorized or counted (`N_A = 0` for this audit); this is an authorization status, not a statement that no local dataset exists. Category-specific counts and cache sizes cannot be estimated until the 30-category inventory and A receipts are certified.

- G1-A stores 49×4096 float32 values per image: **802,816 bytes (0.766 MiB) per authorized image** before provenance/metadata.
- G1-B stores 32×4096 float32 values per image: **524,288 bytes (0.5 MiB)**, only after its native selection receipt is available.
- A full 784×4096 float32 R1 feature matrix is **12,845,056 bytes (12.25 MiB) per image** if extraction must retain the unsampled tensor.
- Exact CPU distance work is `N_query_descriptors × N_reference_descriptors × 4096` multiply-accumulate work per category. The default 64×512 float64 distance block is 0.25 MiB; converted feature blocks use up to 18 MiB combined, before BLAS scratch and result arrays. Result arrays add approximately 808 bytes per query descriptor for two int32 radius-count matrices on the default 101-point grid.
- Aggregate-only text output can be budgeted at **16 MiB** for 30 categories × 101 radii, uncertainty summaries, and manifests, provided per-event tensors/distances are not written. This is a planning cap, not an enforced run authorization. Feature-cache storage is additional and scales with `N_A` using the per-image figures above.
- GPU activation memory cannot be estimated from output descriptor sizes. No GPU memory measurement or model initialization was performed; hardware, batch size, and a separately authorized extraction profile are required.

## G2 dependency boundary

G1 describes normal/reference descriptor geometry only. G2 requires legally admitted historical C1/C2 anchors, frozen source-group/event-time provenance, actual future native Top-32 descriptors, pre-score memory-state identities, queried-feedback sign metadata under explicit authorization, and the same source-group and strict-temporal exclusions. G1 cannot infer feedback-anchor density, sign balance, age, memory-state transport, or correction locality. No R1 private feedback or correction-anchor ledger was used.

## Source integrity and audit note

The R2-L1 references were unchanged from their recorded hashes:

| Reference | SHA-256 |
|---|---|
| `docs/calibration/audit/R2_L1_LOCALITY_MATHEMATICAL_SPECIFICATION.md` | `f011043a46ad48124fe8e51cb887702ad38eed920a0b35e75ffa173a9084e1de` |
| `docs/calibration/audit/R2_L1_ALL_CLASS_DISTANCE_GEOMETRY_PROTOCOL.md` | `e49d76f29b24aec2491fe204b3e5f28e0eb359360bf8d6a9fd0fc7fe2aacf0f0` |
| `experiments/calibration/r2_l1_locality_kernels.py` | `44b7f57e5542aff647d64b3a62efcc130ddce8d8c6840d250508476aee39d64f` |
| `tests/calibration/test_r2_l1_locality_kernels.py` | `691130ca983aeed3d6a91bdbe3a9701f2cb31e67297df6cee372ef81979df1c5` |

New implementation/test source hashes at audit completion:

| New file | SHA-256 |
|---|---|
| `experiments/calibration/r2_l2a_allclass_inventory.py` | `53380fb3f63fab2b56ac44461f3706b77a86f19a7b648d6e4d55153d87ddc3a6` |
| `experiments/calibration/r2_l2a_g1_geometry.py` | `d037324136200424d56985356a53bd66276290559f0a2e72834997b4b5ae4929` |
| `tests/calibration/test_r2_l2a_allclass_geometry.py` | `f2f6e808e00499a670b2e0f1f98821f2a99967ce953b375cbb51d2b8f1ffa741` |

The R1 manifest was queried for representation/source fingerprint metadata. It is minified onto one line; the text search consequently emitted additional manifest summary fields beyond the intended profile fields. Those additional values were not used for this study. The private feedback and correction-anchor ledger files themselves were not opened, and no feature-bearing shard was opened. This audit records the overbroad manifest-search output for provenance transparency.

## Readiness decision and remaining authorization

**`R2_L2A_INVENTORY_OR_PARTITION_BLOCKED`**

Before G1 authorization, provide an official non-label-bearing catalog containing all 30 canonical names, category-level A/B/C role receipts with disjoint official source-group hashes, an independent C identity certificate, authorized A-role normal/reference image counts, and compatible feature-cache receipts (or a separately authorized extraction plan). G1-B additionally requires a frozen native C0 Top-32/M0 selection receipt. Actual distance scans and all G2 work remain outside this task.
