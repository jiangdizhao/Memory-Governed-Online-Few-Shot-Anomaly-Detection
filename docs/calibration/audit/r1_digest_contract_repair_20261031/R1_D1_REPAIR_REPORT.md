# R1-D1 Fixed-Bank Digest Contract Repair

**Date:** 2026-10-31

**Branch:** `calibration-stage0-contract-20261029`

**Starting HEAD:** `4e0c6c24b161616d54fe0595e97cbc60b9c6c971`

## Status

- `DIGEST_COMPARISON_FIXED` — the isolated R1 runner now compares the Stage 0.5 receipt using the structured native digest. Focused CPU tests pass.
- `HISTORICAL_BANK_EQUIVALENCE_UNVERIFIED` — no support bank was rebuilt in this task; the historical tensor equivalence is still unknown.
- `GPU_VERIFICATION_NOT_AUTHORIZED` — no GPU execution, DINO loading, image decoding, label access, C64 rescoring, or stream experiment was performed.

**R1 is not declared an engineering pass.** The previous failed output at `outputs/calibration/r1_switch_s0_512_engineering_20261031/` was not read for writing, resumed, or modified.

## Repair details

The receipt gate now calls `experiments.calibration.native._fixed_bank_digest` through the isolated runner and compares that structured digest with the Stage 0.5 `fixed_support_sha256`. Its algorithm identifier is `experiments.calibration.native._fixed_bank_digest.v1:path-shape-dtype-bytes`.

The P0 raw-prototype digest remains an independent diagnostic, identified as `experiments.p0_protocol_audit.fixed_bank_profile.v1:raw-prototype-bytes`. Run manifests name both algorithms and record expected and actual structured hashes, the raw hash, and each layer’s explicit shape, dtype, prototype count, capacity, and byte count. The evidence is atomically written before a mismatch raises `R1BlockedInput`; the structured receipt gate remains mandatory.

The internal fixed-bank immutability checks now verify both structured and raw digests. Event and completion manifests use the structured digest for `fixed_support_sha256` and carry the raw digest under its separate field.

## Runtime and execution identity

Before DINO model construction, the runner applies `impl.set_seed(0)`, disables Flash and memory-efficient SDP, and enables math SDP. It checks the resulting SDP, deterministic-algorithm, cuDNN, and TF32 settings. It records the runtime profile, Python/Torch/CUDA versions, CUDA device name, CUBLAS workspace setting, source-file fingerprints (including the runner, Stage 0.5 runner, P0 and native digest module), DINO hub-source tree hash, checkpoint file hash, and—after model load—the loaded model-state hash. These records do not establish numerical equivalence until the actual structured bank digest is verified.

The stale hard-coded parent commit has been removed. A future execution must provide an exact full `--expected-source-commit`, one of the versioned R1 profile IDs, a unique `--run-id`, and an `--authorization-id`. The runner verifies the expected branch and exact HEAD, rejects tracked worktree changes and untracked Python sources that could shadow bound imports, and validates the locked P0/native profile. It derives a new namespace under `outputs/calibration/r1_digest_contract_v1_...`; existing namespaces fail closed. The execution entrypoint has no resume option, and cannot select the failed legacy output path.

The isolated runner also has a `--support-only` mode for a future separately authorized check. It builds/verifies the four-support bank and exits before stream scoring or label-reader creation. Its preflight may read the frozen schedule metadata and existing C64 receipt; it does not rescore C64 or decode stream images. The mode is unverified on GPU and has not been executed.

## Changed files

1. `experiments/calibration/r1_matched_correction_pilot.py`
2. `tests/calibration/test_r1_matched_correction_pilot.py`
3. `docs/calibration/audit/r1_digest_contract_repair_20261031/R1_D1_REPAIR_REPORT.md`

No Stage 0.5, P0, native scorer, Greedy, archived report, or output file was modified. At task start the tracked worktree was clean; 828 pre-existing untracked audit artifacts were present and preserved.

## Verification

| Check | Result |
|---|---|
| Python syntax compile for isolated runner and focused tests | PASS |
| `compileall -q experiments/calibration tests/calibration` | PASS |
| Focused R1 CPU suite | PASS — 19 passed |
| Broader `tests/calibration` suite | INCOMPLETE — started, then was interrupted during a long-running CPU test before pytest produced a summary; no full-suite pass is claimed |
| GPU/DINO/support-bank verification | NOT RUN — not authorized |

The documented torch interpreter was available at `D:\anaconda3\envs\torch\python.exe` (Python 3.11.14, Torch 2.0.1+cu117), but pytest is not installed in that environment. Tests ran with that interpreter and the already-installed base-environment pytest 7.4.0 on the import path, with plugin autoload disabled. No package was installed or changed.

The focused tests cover distinct raw/structured algorithms, deterministic structured hashing, mutation sensitivity, receipt comparison against the structured digest, fail-closed persistence of both digests and layer metadata, source/profile rejection, new output namespace separation, runtime setup, and all prior R1 synthetic tests.

## Proposed future support-only command

Run only after separate researcher authorization. Replace the commit and authorization placeholders with the exact reviewed values. The command creates a new output namespace from the run ID and cannot resume the failed R1 output.

```powershell
$PY = "D:\anaconda3\envs\torch\python.exe"
& $PY -m experiments.calibration.r1_matched_correction_pilot `
  --support-only `
  --expected-source-commit "<D1_COMMIT_SHA>" `
  --profile-id r1_switch_s0_support_bank_verify_v1 `
  --run-id supportcheck_20261101_a1 `
  --authorization-id "<RESEARCHER_AUTHORIZATION_REFERENCE>" `
  --dataset-root "G:\dataset\realiad" `
  --json-root "G:\dataset\realiad\jsons\realiad_jsons"
```

Even a matching support digest would verify only the fixed-bank tensor under the recorded inputs. It would not establish full historical score equivalence without reconciling query-encoder and runtime identity. Do not proceed to C64 rescoring, R1, or R2 under this report.
