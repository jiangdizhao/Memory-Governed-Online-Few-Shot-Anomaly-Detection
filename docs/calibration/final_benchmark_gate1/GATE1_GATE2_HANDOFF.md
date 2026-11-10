# FINAL-GATE1 to Gate 2 Handoff

**Gate 2 status:** not authorized. This handoff records prerequisites; it does not approve a scientific run.

## Required before any Gate 2 resource preflight

1. Independent review of the new Gate 1 code, tests, identity receipts and frozen Gate0 consistency.
2. Official release provenance SHA-256 receipt for Real-IAD and resolution of every local identity/camera/group/duplicate exception.
3. A documented official VisA physical-group key or a separate researcher decision to exclude the affected online condition; no identity-level grouping assumption is available from `1cls.csv`.
4. A concrete source-bound native backend with separate F-B/K-B/F-M/K-M Greedy, correction, threshold and controller state; validated R7/R9/R10 score/update parity; C64 write protection; and N-shadow read-only F-B Greedy plus negative-only correction behavior.
5. Static-36+32 candidate identity/tie construction verified against the existing A3b/A4 source contract, plus source-parity checks for its 8,192 FP16 bank and C32 threshold.
6. Separate execution authorization naming the exact reviewed source, data manifest, categories/conditions, resource limits and label-projection scope.

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

Identity-only local census (non-overwriting output):

```powershell
& 'D:\anaconda3\envs\torch\python.exe' -m experiments.calibration.final_benchmark.identity_cli `
  --realiad-root 'G:\dataset\realiad' `
  --realiad-json-root 'G:\dataset\realiad\jsons\realiad_jsons' `
  --visa-root 'G:\dataset\VisA' `
  --output-dir 'outputs\calibration\final_benchmark_gate1\attempt_02'
```

CPU synthetic verification:

```powershell
& 'D:\anaconda3\envs\torch\python.exe' -m unittest tests.calibration.final_benchmark.test_gate1_synthetic -v
& 'D:\anaconda3\envs\torch\python.exe' -m compileall -q experiments\calibration\final_benchmark tests\calibration\final_benchmark
```

`attempt_01` already contains the local Gate 1 census; do not rerun into or overwrite it. A new output directory is required for a repeat.
