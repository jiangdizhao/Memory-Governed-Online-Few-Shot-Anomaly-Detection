# R1-S4B Minimal Readiness Repair

Date: 2026-10-31

Branch: `calibration-stage0-contract-20261029`

Pre-repair HEAD: `e0e4f633b48669b096d17d63aa51105e8087eead`

## Status

**PASS — CPU readiness under the revised application-budget pilot protocol.**

The researcher instruction for this task defines and accepts the revised protocol for a future, separately authorized R1-S4B pilot. That acceptance is limited to the controls below. This task did not authorize or run R1-S4B, load DINO, or access real dataset labels.

**OS-level hard storage isolation is NOT provided.** No Windows quota, VHDX, or filesystem limit is part of this protocol. The 2 GiB application budget is an application control, not a physical disk cap. A future run must use this revised protocol and a separate explicit execution authorization.

## Implemented controls

- Preserved the Windows Job Object supervisor and its 120-minute deadline, suspended assignment, kill-on-close behavior, and termination confirmation.
- Kept the 2 GiB pre-write `OutputBudget` and its 64 KiB failure-metadata reserve. The supervisor’s final status and the child’s failure manifest use the reserved capacity.
- Removed ordinary-write fallbacks from the R1 JSONL, create-once, and atomic-manifest helpers. They now fail closed unless an active budget owns the path.
- Audited the R1 output routes: evidence tape manifests, event NPZ shards, public score and transition JSONL, private feedback, Correction Anchor events, failure manifest, and supervisor status all write through `OutputBudget`. NPZ serialization occurs in memory before the budgeted write. No other R1 experiment output writer was found in the isolated runner or supervisor.
- Rejects output directories that already exist, keeping run IDs unique and preventing historical output reuse or automatic resume. The legacy failed R1 output remains protected.
- Added a fail-closed 10 GiB free-space check before child launch and again in the child before model initialization. The observed preflight value is recorded in supervisor status; the child check is recorded in the run manifest.
- Rejects symlink/reparse-point ancestors, output subdirectories, and files; output-tree accounting walks entries without following links. Unaccounted files, non-regular files, hard-linked files, out-of-root writes, and concurrent writers are refused.
- Preserved source/branch/profile authorization and existing Query, C0/C1/C2, and Greedy Memory code and contracts. Frozen v22/VisionAD sources were not modified.

## Verification

- Starting source revision: `e0e4f633b48669b096d17d63aa51105e8087eead` on the required branch.
- Read-only local disk check: E: had 299,800,567,808 bytes free at audit time, above the 10 GiB launch threshold.
- Python compilation passed for the R1 runner, Windows supervisor, and containment test module.
- CPU regression command ran with `CUDA_VISIBLE_DEVICES=''` against:
  - `tests/test_r1_s4_cpu_protocol.py`
  - `tests/calibration/test_r1_matched_correction_pilot.py`
  - `tests/calibration/test_stage1a_p12_label_provider.py`
  - `tests/test_r1_s4a_resource_containment.py`
- Result: **63 passed**, with four existing torchvision/Pillow deprecation warnings. This includes synthetic Job Object timeout, descendant termination, partial-output retention, writer-budget, no-budget rejection, path containment, free-space threshold, and historical-output reuse checks.
- No GPU execution, DINO loading, C64 rescoring, stream experiment, image decoding, or real-label access was performed.

## Remaining risks and limits

- Application accounting cannot stop an unrelated or non-cooperative same-user process from writing into the output directory or consuming disk space. It also cannot enforce a physical allocation limit or reserve physical disk sectors for failure metadata.
- The 10 GiB check is a preflight observation; another process can consume space after the check. The second check narrows, but does not eliminate, that race.
- Reparse-point checks are application path validation and remain subject to filesystem races. They are not a security boundary against a hostile process running as the same user.
- The 120-minute supervisor and these synthetic regressions establish process/storage control behavior, not GPU runtime, real-bank parity, native numerical parity, or scientific validity.

This report records readiness only for the revised application-level pilot protocol. It does not authorize execution.
