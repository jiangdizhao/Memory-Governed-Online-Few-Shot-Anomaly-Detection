# FINAL-GATE1 to Gate 2 Handoff

**Gate 2 status:** not authorized. This handoff records prerequisites; it does not approve a scientific run.

## Required before any Gate 2 resource preflight

1. Researcher and independent review of the prior census access deviation: the Real-IAD JSON decoder transiently materialized TEST outcome/mask values, including protected categories, and the VisA CSV reader decoded whole rows containing TEST cells. See the corrected identity audit. The repaired projections were exercised only on synthetic data.
2. A researcher-approved release identity convention or authenticated Real-IAD release receipt, plus resolution of the recorded cross-category duplicate's provenance. The existing local snapshot hashes identify local bytes but do not authenticate a publisher release.
3. An explicit researcher decision on VisA image-level or exact-duplicate-cluster feedback. Variable-size units now execute in synthetic tests with unit-level seal-before-disclosure and one-query-per-unit enforcement. Real VisA remains disabled unless a researcher approval is recorded against the exact candidate and certified role/release manifests. No physical-object independence claim is supported.
4. Independent review of the new concrete F-B/K-B/F-M/K-M backend, N-shadow and query-only broker, followed by GPU numerical checks of independent native score/update, protected C64/B3 and shared-feature behavior. The completed CPU checks cover source primitives and synthetic integration, not the full GPU constructor.
5. GPU verification of the source-connected A4/A3b Static-36+32 candidate construction and tie behavior, 8,192 FP16 bank, native static scorer and C32 q90 threshold. The comparison is P0 only.
6. A separately authorized plan for incomplete category recovery: the new broker's in-memory consumed-query index is not reconstructed after a restart. Preserve incomplete journals and feedback.
7. Separate execution authorization naming the exact reviewed source, data manifest, category/condition, event bound, resource limits and label-projection scope. The new `entrypoint.py` rejects uncertified release identity and authorization mismatches, and Stage B invokes its label loader only after the seal-bound grant validates.

## Gate 2 measurements

The following are planning arithmetic from the frozen Gate0 contract, not measurements or guarantees. Device inspection and benchmarks were not performed during Gate 1.

| Component | Frozen shape-based planning value | Measurement still required |
|---|---:|---|
| One current DINO feature map | `(784,4096)` FP32 = 12.25 MiB | Encode latency, peak transient VRAM/RAM, shared-feature lifetime |
| One category C64 FP32 cache | `(64,784,4096)` = 784 MiB | Actual cache format, CPU/GPU placement, peak host and device allocation |
| One fixed reference bank | `8192 x 4096` FP16 = 64 MiB | Actual bank bytes and overlap/sharing across trajectories |
| R10 feature FIFO values | `1024 x 4096` FP32 = 16 MiB per FIFO | Full tensor/metadata bytes, temporary distance memory and update cost |
| Stream volume | Approximately 114,585 Real-IAD + 2,162 VisA images; 350,241 shared encodings across three paired conditions | Exact eligible counts from certified manifests, plus static baseline work |

Measure model load, per-image encode, each native arm score, N-shadow and static score, C64 rescoring, B3 refresh, Correction/Greedy/R10 update, group journal fsync, category sealing, CPU/GPU utilization and contention, peak allocated/reserved VRAM, host RSS, persistent/logical versus shared physical bytes, temporary buffers, bytes/event and full output scale, per-category disk peak/free-space reserve, and category throughput. Measure resource cleanup and state release between categories. Set ceilings only from those measurements.

Do not run a full scientific trajectory as a substitute for this representative preflight. Resource checks do not authorize TEST-outcome access, Stage A/B, or checkpoint cleanup.

## Bounded Gate 2 numerical preflight (not executed)

The new entrypoint supports one Real-IAD category and one registered condition at a time. Its numerical-only mode accepts exactly **5 or 10 complete TEST events** (one or two five-view groups), requires `label_access: NONE`, creates no Stage A seal, and never invokes the label broker or Stage B evaluator. It still prepares the registered four Support plus 64 C64 commissioning state and the condition's registered static arms. For P0, that includes the candidate Static-36+32 source path.

After a separate Gate 2 authorization and release/role certification, the bounded checks are:

1. Verify the existing model receipt, local source revision, preprocessing config and one-device CUDA visibility; encode the 68 registered commissioning images and at most 10 TEST images once each.
2. Construct the four independent F-B/K-B/F-M/K-M states. For each of the at most 10 TEST images, compare native score, threshold, prediction and state digest to the historical source adapters; check that score calls do not mutate any arm.
3. Verify C64 source identities and current-state C1/R10 score inputs, and use synthetic labels only for isolated feedback-transition checks. Do not query or read any real TEST label in this preflight.
4. Verify N-shadow score/state noninterference and its negative-only synthetic update ordering relative to F-B Greedy.
5. For P0, verify Static-4+64 digest/score and Static-36+32 selected candidate identities, K-Center output shape/dtype/capacity (8,192 x 4,096 FP16; 64 MiB), and C32 q90 threshold. For P1/P2, verify Static-4+64 only.
6. Record model load/encode/scorer/rescore latency, peak allocated and reserved VRAM, process RSS, shared/persistent tensor sizes, disk growth and cleanup. These are measurements, not ceilings or scientific outcomes.

The 68 commissioning images and 10-event limit bound the numerical preflight to one role allocation and one or two TEST groups; it is not a full 30-category Real-IAD or 12-category VisA benchmark. No Gate 2 authorization or real preflight execution is implied here.

## Reproducibility commands

The local census is already recorded in `attempt_01`; do not repeat it merely for redundancy. Its prior access descriptions are corrected in the identity audit. Preserve those historical files.

CPU synthetic verification:

```powershell
$env:CUDA_VISIBLE_DEVICES=''
& 'D:\anaconda3\envs\torch\python.exe' -m unittest discover -s tests/calibration/final_benchmark -p 'test*.py' -v
& 'D:\anaconda3\envs\torch\python.exe' -m compileall -q experiments\calibration\final_benchmark tests\calibration\final_benchmark
```

Do not overwrite `attempt_01`.
