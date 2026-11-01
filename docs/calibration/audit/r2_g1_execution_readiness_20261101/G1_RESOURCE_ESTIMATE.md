# R2-G1 Resource Estimate

This is a planning estimate from the frozen descriptor dimensions, the existing L2C aggregate census, and the historical R1 S4B engineering report. It is not a profiler result or a resource authorization.

## Assumptions and per-image payload

Use one fixed camera view per official sample group, only after approval. The Stage B census reports five camera views per group, 7,293 groups total, and at least 167 candidate TRAIN-normal groups/category. Group-level counts are not certified A-role availability.

| Payload | Bytes/image | MiB/image |
|---|---:|---:|
| Full R1 map: 784 x 4096 x float32 | 12,845,056 | 12.252 |
| G1-A projection: 49 x 4096 x float32 | 802,816 | 0.766 |
| G1-B projection: 32 x 4096 x float32 | 524,288 | 0.500 |

At 100 groups/category, G1-FULL-A retains 76.56 MiB/category. In G1-PROTOCOL-MATCHED, the 64 C64 query/reference images retain 49 MiB/category for the matched fixed-49 view and 32 MiB/category for native Top-32; both views together use 81 MiB/category. The four Support images are bank inputs, not geometry rows. Across a two-category run, an upper bound covering a disjoint 100-group full-A cohort plus both matched C64 projections is 315.13 MiB when each cohort is stored once and shared by query/reference batches.

G1-PROTOCOL-MATCHED uses 68 distinct groups/category: four Supports plus 64 additional C64 groups. Keeping those 68 full maps in the current unbounded in-memory cache would occupy about 0.813 GiB/category. A separate 100-group G1-FULL-A cohort could raise the disjoint combined set to 168 maps (2.01 GiB/category; 4.02 GiB for two categories). Process one image at a time, project descriptors, and bound cache lifetime. The existing P1 FeatureCache has no eviction limit.

## Geometry work

For 100 query and 100 reference groups represented by one image/group, with source-group self-exclusion:

| Profile | Query/reference descriptors | Eligible descriptor pairs/category | Dot-product dimensions/category |
|---|---:|---:|---:|
| G1-FULL-A (49 patches, 100 groups) | 4,900 / 4,900 | 23,769,900 | 97,361,510,400 |
| G1-PROTOCOL-MATCHED fixed-49 view (64 C64 groups) | 3,136 / 3,136 | 9,680,832 | 39,652,687,872 |
| G1-PROTOCOL-MATCHED Top-32 (64 C64 groups) | 2,048 / 2,048 | 4,128,768 | 16,911,433,728 |

The last column is pair count x 4,096 descriptor dimensions; it is not a measured CPU time. At 30 categories with the same 100-group planning floor, G1-FULL-A is 713,097,000 pairs. If every category runs the matched 64-C64 comparison, fixed-49 is 290,424,960 pairs and Top-32 is 123,863,040 pairs. For the full 7,293-group population, exact work depends on unknown per-category counts (n_c):

- G1-A pairs: (sum_c ((49n_c)^2 - 49^2 n_c))
- G1-B pairs: (sum_c ((32n_c)^2 - 32^2 n_c))

At an illustrative perfectly balanced allocation of 7,293 groups across 30 categories, those formulas give about 4.24 billion G1-A pairs and 1.81 billion G1-B pairs. This is an arithmetic illustration, not the observed class distribution.

The geometry engine caps each default active pair block at 64 x 512 = 65,536 distances. Float64 query and reference blocks plus a similarity block total about 18.25 MiB, excluding masks, top-k scratch, descriptors, Python, and BLAS workspace. A two-category upper-bound pilot stores about 315 MiB of projected arrays when each image cohort is stored once and the same arrays back query/reference batches. Add about 18.25 MiB for the active float64 blocks and several MiB for per-row/radius statistics; plan for at least roughly 0.35 GiB incremental descriptor/geometry data memory, with additional process/runtime overhead.

## GPU and elapsed-time reference

R1 S4B reported peak CUDA allocated 1,572,285,440 bytes, peak CUDA reserved 14,283,702,272 bytes, peak CPU RSS 2,893,500,416 bytes, and 1,365.719 seconds of GPU wall time for 512 events. This is the only local measured reference used here. It includes native R1 event work and is neither a G1 benchmark nor a guaranteed upper bound.

Its simple throughput ratio is 2.667 seconds/event. At that ratio, 100 one-view geometry images are about 4.45 minutes; two categories are 8.89 minutes. A 68-image G1-PROTOCOL-MATCHED category is about 3.02 minutes; two categories about 6.05 minutes. If the 100-image G1-FULL-A cohort and 68-image matched cohort are disjoint, their combined upper-bound is 168 images/category (7.47 minutes; two categories 14.94 minutes). DINO encode/scoring and CPU geometry differ from the R1 event loop, so these figures are only scheduling proxies. No empirical G1 timing or live VRAM check was performed. A future run must verify hardware and set an explicit GPU memory cap before authorization.

At the same ratio, 3,000 images (100 groups x 30 categories) are about 2.22 hours, and 7,293 images are about 5.40 hours, before geometry time. A full all-category run therefore cannot be assumed to fit a 120-minute job. Use separately authorized bounded jobs if later approved.

## Disk and output proposal

For a one/two-category pilot only:

- Unique, non-overwriting output: `outputs/calibration/r2_g1_pilot/<run_id>`.
- Application pre-write output cap: **1 GiB (1,073,741,824 bytes)** covering projected features, receipts, manifests, reports, logs, and temporary/atomic-write files.
- Start only with at least **10 GiB free**; this is a proposed threshold, not a present disk check.
- The two-category upper-bound projected cache is about 315 MiB when each image cohort is stored once; an atomic temporary copy brings this to about 631 MiB. Keep one shared cohort for query and reference rather than writing duplicate arrays. A 1 GiB cap leaves room for manifests and temporary writes. Never copy source images, raw metadata, masks, labels, or full maps into the output.

Retaining projected descriptors for all 7,293 groups would require about 5.45 GiB for G1-A or 3.56 GiB for a separately authorized hypothetical all-A Top-32 profile under the one-view assumption; the latter is broader than the current G1-PROTOCOL-MATCHED C64 cohort. The current pilot cap does not cover these full-population caches. Full-population output retention requires a distinct approved budget and storage plan; it is outside this task.
