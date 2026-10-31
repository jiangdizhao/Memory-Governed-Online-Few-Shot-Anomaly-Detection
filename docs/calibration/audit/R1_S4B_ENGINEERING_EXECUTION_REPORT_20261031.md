# R1-S4B Engineering Execution Report

Date: 2026-10-31

## Classification

**ENGINEERING_PASS** for the single authorized 512-event engineering pilot. This is not a scientific performance evaluation. No AUROC, AUPRC, balanced accuracy, FPR, or FNR was computed.

## Authorization and execution identity

- Authorization: `R1-S4B-20261031-USER-GO`, limited to this R1-S4B run.
- Parent/source SHA: `c86d5b89fa0062b87d80adb324eaa6bc3c94448e`.
- Branch: `calibration-stage0-contract-20261029`.
- Run ID: `r1s4b-switch-s0-512-20261031-a1`.
- Output directory: `outputs/calibration/r1_digest_contract_v1_engineering_r1s4b-switch-s0-512-20261031-a1`.
- Exact command:

```powershell
& "D:\anaconda3\envs\torch\python.exe" `
  -m experiments.calibration.r1_runtime_supervisor `
  --expected-source-commit c86d5b89fa0062b87d80adb324eaa6bc3c94448e `
  --profile-id r1_switch_s0_512_digest_contract_v1 `
  --run-id r1s4b-switch-s0-512-20261031-a1 `
  --authorization-id R1-S4B-20261031-USER-GO `
  --dataset-root G:\dataset\realiad `
  --json-root G:\dataset\realiad\jsons\realiad_jsons `
  --authorize-real-execution
```

## Preflight

- Required branch, HEAD, and clean tracked worktree checks passed. Existing untracked paths were preserved.
- Read-only `_preflight()` passed without invoking `run_pilot()` or decoding labels/masks.
- Frozen switch stream: 3,820 entries; authorized prefix: positions 0–511.
- Initial supports: 4; separate C64 references: 64. Frozen support/C64/stream source-group disjointness checks passed.
- Query binding SHA-256: `dbe72a2f989ee314becc07e0c8b8c7a8353ffacabbfc93863d5666cff9d28183`.
- Query verification SHA-256: `ffbedc32edbd8f6c9b79d0d5b0fe3663d4cfb315d20273815405b84a486cc991`.
- Query positions (17): `35, 55, 72, 116, 159, 162, 189, 268, 297, 302, 349, 367, 375, 395, 413, 488, 501`.
- Query positions SHA-256: `7f4aedf49f3e3bffd2482bd00d9e16512553b7cef8c0af7e002b47b60ddcb17f`.
- C64 score tape SHA-256: `91fefe78fe54a2d933770b120df08894b4c8e6f69297d0cd2d2ac7b950e0aa9e`.
- Output directory was absent. E: had 299,798,880,256 bytes free against the 10 GiB minimum.

## Execution and validation

- Supervisor: `COMPLETED`; child exit code: `0`; process-tree termination confirmed; zero active Job Object processes.
- Supervisor elapsed time: 1,382.156 seconds; pilot GPU wall time: 1,365.719 seconds, below the 120-minute limit.
- Processed, sealed, finalized, and unique stream positions: 512 each.
- Queried labels disclosed after score sealing: 17, in frozen position order. Private label values remain only in the local restricted feedback file.
- C0 native parity: 16 designated smoke events; 0 parity failures.
- Common Greedy memory: all 512 transitions used common C0 inputs. Seven legal normal-memory updates occurred; unqueried events performed no supervised write, and all 10 queried defects were excluded from normal memory. Greedy memory version ended at 7; consecutive state digests were continuous.
- Fixed support bank remained unchanged. Structured digest before/after and expected receipt: `8a710d028509350568e976bc3ede66aa59603499cbea525bdd8579322b1a1558`. Raw-prototype diagnostic before/after: `087feded0d780bbc3a56d45404b4126991b1aeb54f1ae3724bce025caf0d380e`.
- C1: 197 positive and 93 negative anchor writes; 194 positive and 62 negative anchors active at completion. All 512 source-group exclusion checks passed. C1 remained the adapted native Signed Correction profile.
- C2: 17 queried events added 32 patch rows each; final FIFO state held 256 anchors. Independent reconstruction from event shards matched all 512 pre-score eligibility and post-update C2 state digests, including the FIFO evictions.
- Causal state mismatches: 0.
- Evidence tape: independent readback passed for 512 sealed shards and 512 finalized transitions. Every shard hash, array hash, and reconstructed score passed. Sealed and transition position digest: `61f3f6fd3aa109e05aa31cc4d74f333d17c8f73ced22284db2209e96a39884af`.
- The 15 source-file hashes in the run manifest matched the checked-out files during independent validation. Model/runtime fingerprints are recorded in the manifest, including DINO hub source, loaded model state, encoder, and preprocessing hashes.
- Peak CUDA allocated: 1,572,285,440 bytes; peak CUDA reserved: 14,283,702,272 bytes; peak CPU RSS: 2,893,500,416 bytes.
- Manifest output size: 272,690,113 bytes; measured output directory including supervisor status: 272,690,643 bytes, below the 2 GiB application limit. The 64 KiB failure reserve remained configured.

## Risks and scope limits

- OS-level hard storage isolation was not provided. This single pilot used the explicitly accepted application-budget protocol. The application accountant cannot prevent a separate same-user process from writing into the directory or consuming disk space; the 10 GiB free-space check is a preflight observation and cannot eliminate later space races.
- C2 remains an engineering fixture (`k=5`, amplitude `0.20`, capacity `256`, FIFO); its scientific hyperparameters are not approved. C1 is an adapted profile, not a historical faithful replay.
- No R2, additional category, parameter tuning, confirmation data, masks, unqueried labels, or scientific outcome metrics were accessed or run.

The run output and private feedback remain local and uncommitted. No push was performed.
