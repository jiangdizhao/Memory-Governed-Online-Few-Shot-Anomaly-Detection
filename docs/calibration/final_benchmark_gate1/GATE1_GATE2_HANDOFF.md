# FINAL-GATE1 to Gate 2 Handoff

**Gate 2 status:** not authorized. This handoff records prerequisites; it does not approve a scientific run.

## Required before any Gate 2 resource preflight

1. Researcher and independent review of the prior census access deviation: the Real-IAD JSON decoder transiently materialized TEST outcome/mask values, including protected categories, and the VisA CSV reader decoded whole rows containing TEST cells. See the corrected identity audit. The repaired projections were exercised only on synthetic data.
2. A researcher-approved release identity convention or authenticated Real-IAD release receipt, plus resolution of the recorded cross-category duplicate's provenance. The existing local snapshot hashes identify local bytes but do not authenticate a publisher release.
3. An explicit VisA physical-group or image/duplicate-cluster decision. The candidate-unit adapter remains non-executable; the current online runner still needs a variable-size unit path before VisA could run.
4. Independent review of the new concrete F-B/K-B/F-M/K-M backend, N-shadow and query-only broker, followed by GPU numerical checks of independent native score/update, protected C64/B3 and shared-feature behavior. The completed CPU checks cover source primitives and synthetic integration, not the full GPU constructor.
5. GPU verification of the source-connected A4/A3b Static-36+32 candidate construction and tie behavior, 8,192 FP16 bank, native static scorer and C32 q90 threshold. The comparison is P0 only.
6. A separately authorized plan for incomplete category recovery: the new broker's in-memory consumed-query index is not reconstructed after a restart. Preserve incomplete journals and feedback.
7. Separate execution authorization naming the exact reviewed source, data manifest, categories/conditions, resource limits and label-projection scope.

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

## Reproducibility commands

The local census is already recorded in `attempt_01`; do not repeat it merely for redundancy. Its prior access descriptions are corrected in the identity audit. Preserve those historical files.

CPU synthetic verification:

```powershell
$env:CUDA_VISIBLE_DEVICES=''
& 'D:\anaconda3\envs\torch\python.exe' -m unittest discover -s tests/calibration/final_benchmark -p 'test*.py' -v
& 'D:\anaconda3\envs\torch\python.exe' -m compileall -q experiments\calibration\final_benchmark tests\calibration\final_benchmark
```

Do not overwrite `attempt_01`.
