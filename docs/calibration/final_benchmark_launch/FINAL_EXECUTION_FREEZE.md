# Final Real-IAD Execution Freeze

**Readiness status:** `FINAL_BENCHMARK_EXECUTION_INTEGRATION_INCOMPLETE`
**Scientific design:** researcher-approved and frozen
**Implementation:** reviewed locally; source changes remain uncommitted
**Execution:** not authorized or performed

## Freeze identity

- Branch: `calibration-stage0-contract-20261029`
- Current committed review base: `7a0dde1dd677dca493168e21fc9b8d0d00c390dd`
- Final source commit: intentionally unset until independent review and commit
- Gate 0 scientific freeze commit: `a34b870fd4700bfe970e459d2963a602f04b1341`
- Researcher-approved R2 base: `2511d4f5b61efe869994b23afb398040eb49d270`
- Gate 0 data-protocol approval addendum SHA-256: `9d1a5e4f7d0f27e6e40743586675d868707a0afda236ec288c60e96a2527c6cb`

The execution-plan template is [FINAL_RUN_CONFIGURATION_TEMPLATE.json](FINAL_RUN_CONFIGURATION_TEMPLATE.json). It has 90 category-condition rows and is bound to the certified Gate 1 manifest and category bundles. Its `final_source_commit` and campaign run ID are null by design. The schedule checksum is `a475907f4a198a96b83dcdf5b42bc76ad10cc19c9dbb101bed735832c4cdfb76`. No execution authorization was generated.

## Frozen scientific design

The principal comparison is the four independently evolving arms F-B (Full Gaussian C1 + simplified BN+Band), K-B (Signed kNN + simplified BN+Band), F-M (Full Gaussian C1 + `Margin_rho`), and K-M (Signed kNN + `Margin_rho`). N-shadow is a passive, non-querying, normal-only correction control that inherits F-B's legally executed query and normal feedback. It cannot change F-B's trajectory and reports AUROC/AP without an independently invented threshold.

P0 is Support Seed 0 / Stream Seed 0; P1 is Support Seed 0 / Stream Seed 11; P2 is Support Seed 1 / Stream Seed 0. Each Real-IAD trajectory uses four deployed verified-normal Supports plus 64 separate C64 normals, group-atomic five-view feedback, a 3% category hard cap, no forced queries, and no top-ups. The `0.003` realized-rate tolerance qualifies only an additional approximately rate-matched description; equal caps do not imply equal realized expenditure.

Static-4+64 is registered for P0/P1/P2. Static-36+32 is registered for all 30 Real-IAD categories at P0 only, with the deterministic approved 36/32 identity allocation, A3b K-Center, an 8,192 x 4,096 FP16 fixed bank, and a C32 higher-q90 static threshold. It is a same-total-normal-information comparator, not a mechanism ablation or historical A4's dynamic D arm.

FOADS is the separate P0-only baseline `FOADS-KNG-PAPER-CONSISTENT-PROTOCOL-ADAPTED`: 68 registered TRAIN-normal images, ImageNet ResNet18 `IMAGENET1K_V1`, 3,136 target nodes, 10 K-NG fit epochs, age limit 25, 100 selected dimensions, two complete five-view groups before each native Batch-10 update, raw batch-independent image maxima, and zero paid labels. Its registered differences from the pinned upstream implementation are the paper-consistent Equation (11) Mahalanobis calculation, batch-independent raw image scoring, registered Real-IAD order/update timing, and matched 68-normal initialization (the upstream script initializes from 10 images). The upstream Klinks topology, pseudo-normal screening, thresholds, and `online_fit()` update are retained. It is not an exact reproduction of published results.

## Data, models, source and claim limits

- Real-IAD identity basis: researcher-approved local snapshot, not publisher-authenticated. Local snapshot SHA-256: `d44a059bf998231fcdc2a296b97ca45544a2554075e1f92b4dfc15c89a5fe4e2`; derived protocol release SHA-256: `52b045ea219eb270c427d30785da8e9b891c8eff2ba035b4430560bf17c962ae`.
- Identity manifest canonical SHA-256: `4d6841d048cb073d3b121ee025a3afd3619572b62afcf167081acbb6aa0f9912`; serialized file SHA-256: `e8f9126f9e3e367be6008537bc423b38c60d427c704aac01e88acdcdcf3f5349`.
- Annotation artifact-set SHA-256: `8652d842d26785fff780fa3ac429592a39da72675c8dacb5fe56eabf512ce420`.
- Principal encoder evidence: DINOv2 ViT-L/14 Reg4; checkpoint SHA-256 `36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51`; Gate 2 model-receipt SHA-256 `af2423c89d28d865550127a27173d3c37ddd7b3cd003f26da52d560fd92868e2`.
- FOADS weights: `ResNet18_Weights.IMAGENET1K_V1`, SHA-256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.
- Pinned K-NG source: detached checkout `D:\Whishing-K-NG-63dd8e7c9ade5619771d4b5544f032206818bb22-lf`, commit `63dd8e7c9ade5619771d4b5544f032206818bb22`. `verify_upstream_source()` confirmed `Klink.py` SHA-256 `402f9bd969693207aedc5286029b2e51f9bd0000016eeacfab3510cfbf76d440`, `Konline.py` SHA-256 `a2bd1823cc1bb1e45b3749108bc2865816852fa493ae32ad201fcb153fef72f1`, and `datasets/mvtec.py` SHA-256 `5a1598b202e5ed3e797126e08827ef0b5215e0d530dbc8d79514fc7cc84596b6`.

Real-IAD is a historically exposed benchmark reassessment, not independent confirmation. The Gate 0 addendum records the historical census parser's transient decoding of TEST outcome and mask metadata. The corrected identity projection did not retain those values; the current verification decoded no TEST labels and opened no masks. No claim is made that historical values were never decoded. The cross-category byte-identical TEST pair remains in both official streams. No Signed kNN non-inferiority claim is permitted without an approved positive `delta_K`; the `0.01` BER proposal is not a hard success criterion.

## Prior numerical evidence

- Gate 2 Attempt 04: `GATE2_BOUNDED_NUMERICAL_PREFLIGHT_PASS`, Real-IAD audiojack P0, 10 unlabelled TEST images, four native arms, N-shadow, and both static references. It recorded no Stage A seal or Stage B evaluation. It is bound to source commit `f6bee0c7cff785da1d11dd27629b28bd07ca0744`.
- FOADS Attempt 05: `FOADS_KNG_REAL_GPU_PREFLIGHT_PASS`, current committed base `7a0dde1dd677dca493168e21fc9b8d0d00c390dd`, 68 TRAIN-normal images and 10 unlabelled TEST images. It recorded 3,100 surviving nodes, one native update, 91.757 seconds total, 1.320 GB peak allocated GPU memory, 1.411 GB reserved, and 11.01 GB peak host RSS. No full category was scored.

The uncommitted Real-IAD path correction changes identity/path validation and its callers, not the native numeric scoring kernels. The current Gate 2 and FOADS receipts remain bounded historical evidence; they do not certify a full campaign or a final source commit.

## Execution integration R1

The new `experiments.calibration.final_benchmark.launch` command reuses the existing one-category Stage A API. It validates the complete checksum-bound schedule and current source commit without loading DINO, lists all 90 trajectories, runs one category only with exact separate authorization, and accepts a sequential 90-row schedule only under a separate explicit campaign authorization. Sequential launches stop on the first failure. Attempt directories are created exclusively after metadata, authorization and resource checks. The existing Stage A group journal, feedback ledger, category seal and report remain the scientific output path; a new fsynced attempt status records `RUNNING`, `INCOMPLETE` or `SEALED_COMPLETE`, including phase, count and source/configuration identity.

The separate `foads-run-one` command binds the exact registered P0 projection/role/stream, K-NG source hashes, ResNet18 weight file hash, private locator file and independent authorization. It wraps the original `FOADSKNGP0Runner`; numerical scoring and online update code were not changed. Each score record contains the model-state digest before scoring; every native Batch-10 update has its own group/event/state receipt. A full-category seal rejects missing/reordered scores, absent updates, partial groups, wrong update count, unaccounted final group or modified journal. Post-seal metrics remain separately authorized.

Resource stop rules use the measured bounded Gate 2 and FOADS receipts. Disk output is reserved at 10x the observed bytes per event plus a per-attempt fixed reserve, with 10 GiB left free. Host memory must remain above 1.25x the observed peak; pre-run GPU free memory must exceed the observed reserved peak plus 1 GiB; runtime GPU reservation is capped at 1.25x the observed peak. The final read-only snapshot on 2026-11-11 found 381,233,139,712 bytes free disk, 41,316,900,864 bytes available host RAM, and 23,894,949,888 bytes free on the single RTX 3090 Ti. The extrapolated full-campaign output guard is 23,221,070,420 bytes plus the 10 GiB reserve. Full-category output growth and runtime remain unmeasured, so this limit is a stop guard, not a resource guarantee.

The source-dependent FOADS Python regression file passed all 13 tests with `KNG_SOURCE_ROOT` explicitly set to the verified checkout above. The full focused final-benchmark pytest suite passed 67 tests, with no skips. These are source-fidelity and synthetic execution checks only; they do not execute full-category FOADS scoring.

## Freeze boundary

No Stage A, FOADS full-category scoring, Stage B, TEST label projection, or mask access occurred in this integration task. No algorithm, arm, query policy, data identity, model setting, result, checkpoint, or historical Gate 0 document was changed. Source/configuration changes remain uncommitted; the final source-bound plan and researcher authorizations must be issued after review and commit. The integration is CPU-verified; full-category resource growth and runtime remain unmeasured.
