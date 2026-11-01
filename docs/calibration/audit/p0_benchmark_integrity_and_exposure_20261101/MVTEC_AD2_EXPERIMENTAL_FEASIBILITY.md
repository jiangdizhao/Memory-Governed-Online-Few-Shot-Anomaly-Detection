# MVTec AD 2 Experimental Feasibility for Local Correction

No model, feature extractor, image, test header, test label value, or mask pixel was used. This is a count/protocol assessment only.

## Support and calibration counts

| Category | Train normals | Four deployed supports | Four supports + 64 additional C64 normals from TRAIN |
|---|---:|---|---|
| can | 412 | `COUNT_FEASIBLE` | `COUNT_FEASIBLE` |
| fabric | 387 | `COUNT_FEASIBLE` | `COUNT_FEASIBLE` |
| fruit_jelly | 263 | `COUNT_FEASIBLE` | `COUNT_FEASIBLE` |
| rice | 313 | `COUNT_FEASIBLE` | `COUNT_FEASIBLE` |
| sheet_metal | 137 | `COUNT_FEASIBLE` | `COUNT_FEASIBLE` |
| vial | 291 | `COUNT_FEASIBLE` | `COUNT_FEASIBLE` |
| wallplugs | 293 | `COUNT_FEASIBLE` | `COUNT_FEASIBLE` |
| walnuts | 432 | `COUNT_FEASIBLE` | `COUNT_FEASIBLE` |

Every category has at least 68 TRAIN images, so the count requirement for four supports plus 64 *additional* C64 images can be met from TRAIN alone; Validation access is not needed for that count. No support or C64 identities were selected. Distinct filenames do not establish independent Source Groups, and no MVTec group/lighting correspondence contract was verified. Strict four-shot-total remains a separate protocol and must not be conflated with four plus C64.

## Public 3% query-budget scale

Budgets are floor/ceiling of `0.03 × T`; they are not future tape selections. The public normal/defect split counts are aggregate directory metadata only.

| Category | Public T | Public normal/defect | `floor(0.03T)` | `ceil(0.03T)` |
|---|---:|---:|---:|---:|
| can | 162 | 72/90 | 4 | 5 |
| fabric | 156 | 66/90 | 4 | 5 |
| fruit_jelly | 80 | 20/60 | 2 | 3 |
| rice | 132 | 42/90 | 3 | 4 |
| sheet_metal | 114 | 24/90 | 3 | 4 |
| vial | 140 | 35/105 | 4 | 5 |
| wallplugs | 150 | 60/90 | 4 | 5 |
| walnuts | 150 | 60/90 | 4 | 5 |
| **All eight, pooled only for scale** | **1,084** | **379/705** | **32** | **33** |

The prior G2 anchor target of at least 20 verified normal and 20 verified defect query groups requires at least 40 query disclosures. It is mathematically impossible at 3% for any one category (maximum ceiling 5) and also impossible across the entire public set when pooled (ceiling 33). Distinct physical groups could only reduce the available query count. Status: `FULL_G2_ANCHOR_TARGET_INFEASIBLE_AT_3_PERCENT`. This does not make MVTec AD 2 useless: a frozen external comparison, or a separately preregistered smaller online-feedback study, remains possible.

## Split-use matrix

| Split | Frozen inference | Local offline evaluation | Query-feedback simulation | Official server evaluation | Lighting-shift analysis |
|---|---|---|---|---|---|
| `test_public` | Yes, subject to a frozen preprocessing profile | Yes; local labels/masks are part of the public release. Aggregate label counts were inspected in this audit, but no example or mask pixels were opened. | Potentially, with strict prequential disclosure after queried events; 2–5 queries/category at 3% is too small for the prior G2 target. | Not required for local public evaluation. | Limited public-set diagnostics possible; no independent physical-group assumption. |
| `test_private` | Yes, image inference only | No local ground truth; paper says evaluation is server-side. | No local label-feedback simulation. Do not assume the server offers interactive feedback. | Yes, through MVTec's server. | Possible only via server metrics and a frozen protocol. |
| `test_private_mixed` | Yes, image inference only | No local ground truth; paper says evaluation is server-side. | No local label-feedback simulation. | Yes, through MVTec's server. | Yes as the mixed-lighting test, but it depicts the same scenes as `test_private`; do not count the two as independent Source Groups. |

The official page describes private ground truth as non-public and server-evaluated; the paper states that private-mixed contains the same scenes with seen/unseen lighting. See the [official dataset page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2) and [official paper](https://doi.org/10.1007/s11263-026-02743-0).

## Frozen scorer compatibility questions

The current Real-IAD scorer uses a DINOv2 ViT-L/14 profile involving a 448 resize and 392 center crop. The official MVTec AD 2 paper describes 2.6–5 MP images with varying aspect ratios, single-channel grayscale scenarios (Sheet Metal, Vial, Wall Plugs), and anomalies that can occur at image borders. No compatibility inference was run.

- Freeze how grayscale images become model input (for example, a documented one-channel-to-three-channel mapping) before any evaluation; do not infer it from test outcomes.
- Freeze aspect-ratio handling and interpolation. Direct resize can distort geometry; center cropping can remove border defects. Any padding/tiling or alternative resolution is a new, named profile requiring pre-performance specification.
- Preserve a consistent DINO patch grid and disclose how any resized/cropped/tiled map is mapped back to original coordinates. The 392-pixel crop yields a 28×28 grid for a 14-pixel patch, but it does not by itself guarantee suitable geometry across these sources.
- Image-level maximum aggregation is compatible as a frozen ranking measure only after protocol freeze; it does not validate pixel-level segmentation. Any mask comparison needs exact spatial transform, interpolation and alignment rules.
- Multi-megapixel inputs raise memory and latency costs. A later inference plan must budget these costs and specify tiling/resolution without tuning on public test performance.

No frozen scorer was modified. Dataset structure and count feasibility do not authorize G1/G2, support selection, query-tape selection, or model inference.
