# Experiment A4: Equal-Budget Normal Reference–Calibration Allocation

**Protocol:** `A4_EB68_v1`
**Study type:** Four-category development sensitivity study; no independent confirmation.
**Condition:** P0 only (Support seed 0, group-shuffle seed 0).
**Authorization:** [`EXPERIMENT_A4_EB68_AUTHORIZATION.md`](../../EXPERIMENT_A4_EB68_AUTHORIZATION.md)

## Question and fixed information budget

A4 asks whether 64 additional labelled normal commissioning images should remain available to calibrate an empirical q90 operating threshold, or whether some should instead strengthen the deployed fixed normal reference. Every category uses the exact A1 seed-0 four Support identities `S4` and exact A1 historical additional C64 identities. These are 68 distinct TRAIN-normal image identities from distinct official Source Groups and have no TEST Source Group overlap. A1 draw-0 C32 membership is preserved exactly; R32 is the complement within C64. The budget is **four Support images plus 64 additional normal commissioning images**, not strict four-shot total.

## Registered arms

| Arm | Fixed reference | B3 reference | Interpretation |
|---|---|---|---|
| A | Historical native Support-4 bank | C64 | Original resource allocation |
| B | Exact same bank and writer trajectory as A | C32 | C64 versus C32 readout on shared scores |
| C | A3b K-Center of the same Support-4 augmentation descriptors | C32 | Selector effect on original Support images |
| D | Same K-Center method on Support-4 plus complementary R32 | C32 | Expanded fixed reference under the 68-image budget |

All methods use frozen native DINOv2 ViT-L/14 register features (layers 5/11/17/23; 4096D concatenated descriptors; 448 resize, 392 center crop; 28×28 grid), the native identity-view scorer and maximum patch aggregation, native Greedy K=1024 memory, and the same Q-A-TS-v1 query tape. Each fixed bank contains exactly 8,192 × 4,096 fp16 prototypes = 67,108,864 bytes. Equal fixed-reference bytes do not imply equal total deployed memory. D's K-Center pool is balanced across its 36 eligible Source Groups, tie-broken by identity hash, capped at 65,536 descriptors, and selected by blockwise CUDA cosine K-Center without an N×N matrix.

Memory accounting distinguishes per-arm logical fixed-reference capacity from the three unique online fixed banks (A/B share one), the three static M0 scoring shadows, native Greedy state bytes, the physically shared C64 FP32 calibration-feature pool (C32 arms reference its subset), calibration source-image copies, Support augmentation features, serialized identity/initialization metadata, the single current TEST feature, and temporary CPU/CUDA K-Center candidate matrices. No full TEST feature cache is retained. Peak CUDA allocation is also recorded; equal fixed-bank capacity is not presented as equal total memory.

## Calibration, writer, and causal order

B3 initializes from higher empirical q90 and sample standard deviation (`ddof=1`, floor 0.02). On a genuine memory content change, it rescales protected C64 for A and C32 for B/C/D, then clips threshold movement to ±0.01. The new state is available only at the next event. A and B share one exact native score and writer/memory trajectory; their AUROC/AP must match exactly.

For each distinct fixed bank, a frozen writer threshold and scale are initialized from C32 at M0 and held fixed through the stream. The writer uses the current native C0 score and that frozen C32 reference; it never uses a live arm-specific B3 threshold. This is an A4-specific matched writer-reference rule, not a numerical reproduction of historical R9's C64 writer reference. Only a queried verified normal can enter Greedy memory. There is no pseudo-labeling, Correction, BEA, mask access, or online backbone training.

At each event the current image is encoded once; all static and online scores and predictions are formed from pre-feedback state; the complete label-blind row is fsynced; the registered query decision is applied; only then may a queried label be disclosed; legal native writes and B3 refreshes follow. All four category Stage A seals must pass before CPU-only Stage B reads unqueried TEST labels.

## Outcomes and estimands

Report static M0 and prequential AUROC, AP, BER, FPR, FNR, balanced accuracy, confusion counts, query/write/candidate counts, memory changes, B3 thresholds, and non-overlapping 512-event windows. Undefined single-class metrics remain null. Macro summaries weight the four categories equally; five camera views are not independent physical samples.

The registered contrasts are `B−A` (calibration readout on shared scores), `C−B` (K-Center versus native construction on the same four Support images), `D−C` (add R32 to the same K-Center construction), and principal total-budget `D−A`. `A−D` changes both reference representation and calibration allocation and cannot isolate B3. A materially competitive allocation requires at least 0.01 macro AUROC improvement and positive category deltas in at least three of four categories. Safety metrics remain separate; otherwise classify the result mixed or inconclusive. This is a pragmatic decision guide, not a significance theorem.

## Provenance and stopping

Source-freeze commit and exact source hashes are recorded before the run. Historical A1/A2/A3/R9/R10 sources and output artifacts remain read-only. Category execution is Windows Job Object supervised (120-minute maximum); campaign maximum is eight hours; minimum free disk is 30 GiB; retained experiment output maximum is 4 GiB. New output paths are isolated and non-overwriting. Any role, source, model, native parity, causal, seal, disk, or protected-file failure stops the affected run. Smoke and partial artifacts are marked `INCOMPLETE` and are not scientific evidence.

## Scope limitations

These four previously exposed development categories cannot establish independent confirmation, population risk guarantees, physical-object independence beyond the official Source Group proxy, or universal superiority. A/D is a whole-system allocation comparison. B/C/D consume only the information assigned to their arm despite sharing the permitted candidate universe.
