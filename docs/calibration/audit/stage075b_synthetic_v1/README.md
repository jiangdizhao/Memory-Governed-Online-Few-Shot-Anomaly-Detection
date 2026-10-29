# Stage 0.75-B compact synthetic audit package v1

**Status:** immutable audit package for the synthetic-only Stage 0.75-B run.
**Parent:** `0afa20967c5ed88f804d87f380089b2ca543a9fb` on `calibration-stage0-contract-20261029`.
**Primary method contract:** candidate v2; candidate v1 is preserved as superseded.

## Contents

* `results_compact.json` — per-scenario/per-seed BA, FPR/FNR, confusion counts, paired differences, query/memory counts, score digest linkage, threshold movement and runtime summaries. No labels or event traces.
* `method_cards.json` — B3, B7, B3+Offset, and each B8 ablation's exact synthetic method identity and fidelity boundary.
* `gates.json` — machine-readable G1–G15 results from the one-command runner.
* `test_results.json` — exact test and compilation commands, counts, return codes and captured log hashes.
* `source_provenance.json` — parent, source/config/document fingerprints and output artifact linkage.
* `artifact_linkage.json` — relative path and checksums for the local detailed runner artifact.
* `v1_transition_note.md` — reason for the candidate v1 to v2 method revision and handling of its unarchived development smoke.
* `checksums.json` — SHA-256 for every other file in this package.

Detailed results, logs and checksums remain in the local output artifact `outputs/calibration/stage075b_synthetic_20261029/`. The audit package is self-contained for review of the compact evidence and includes a checksum link to that artifact. It contains no datasets, images, masks, model weights, feature tensors, checkpoints, or raw event traces.

## Verification

From the repository root:

```powershell
$PY = "D:\anaconda3\envs\torch\python.exe"
& $PY -m experiments.calibration.stage075b_runner --verify
```

Verify this package by recalculating SHA-256 for every listed file except `checksums.json` and comparing the keys and values exactly. The output artifact's independent `--verify` additionally checks source hashes against the current working tree.

## Scientific disposition

Engineering gates passed, but B8-Full matched actual B3 across all 48 scenario/seed trajectories, and B3+Offset had higher BA in most paired synthetic rows. The candidate decision is `SIMPLER_BASELINE_SUFFICIENT`; the mechanism status is `SYNTHETIC_MECHANISM_NOT_SUPPORTED`. No later stage is authorized by this package.
