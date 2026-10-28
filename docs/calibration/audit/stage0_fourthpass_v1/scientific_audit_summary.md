# Stage 0 compact scientific audit

Runner status: **BLOCKED_WIP**.
This package contains implementation evidence, not scientific validation.
Stage 1 and Stage 2 were not run; no confirmation-class outcomes were opened.
All Real-IAD scoring evidence is limited to the bounded development-class training-normal diagnostic.
Synthetic binary labels in native replay are marked synthetic and are not Real-IAD outcomes.

## Gate snapshot

- G0: PASS — Correct branch and no protected worktree changes.
- G1: PASS — Synthetic Q-A-TS/Q-B metadata tests passed.
- G1_real_data: PASS — Official sample identities, train-only support/C64 selection, and label-blind Q-A-TS/Q-B were checked using development-category JSON metadata; no image or mask was opened.
- G2: PASS — Actual development train-normal score parity passed across live native v22, untouched clone, and calibration adapter.
- G3: PASS — Synthetic native operator matrix passed APPEND/REPLACE/SKIP, repeated-slot source attribution, prototype-content audit, LSO filtering and exact checkpoint restore; bounded CUDA v22 smoke passed live-score and append parity.
- G4: PASS — Synthetic predict-before-disclosure and queried-only feedback tests passed.
- G5: PASS — Synthetic protected C64, current-version evidence, and all-layer LSO source attribution tests passed.
- G6: BLOCKED_FIDELITY — B5a SPOT and B5b dSPOT are executable paper-equation implementations with independent numerical/stream tests; B6 Sun-CS remains BLOCKED_FIDELITY; B7 remains provisional.
- G7: BLOCKED — B5a/B5b Grimshaw, POT, update/exclusion, warm-up and checkpoint tests passed; all-family reference acceptance remains blocked by B6 source-faithfulness and observation/abstention mapping.
- G8: PASS — Event-boundary checkpoint/resume is compared with an uninterrupted synthetic run, including memory, controller, trace, private-label identity, and Python/NumPy/Torch RNG restoration.
- G9: BLOCKED — Native core parity passed only if recorded by G9_native_core; full G9 remains blocked because B6 is unavailable.
- G9_all_controller_matrix: BLOCKED_FIDELITY — B6 Sun-CS cannot participate until its source-faithful implementation and reference tests are complete.
- G9_native_core: PASS — Bounded multi-event actual v22/DINO train-normal trajectory passed causal query/write, current-version C64 refresh, LSO, native checkpoint-resume equivalence and synthetic-label AUROC invariance.

Executed tests: 63; passed 63; failed 0; errors 0; skipped 0.
Synthetic native Greedy operator matrix: PASS.
Native event replay: PASS; all-controller matrix: BLOCKED_B6_FIDELITY.

## Provenance

The compact evidence files are content-hashed in `checksums.json`. `audit_manifest.json` records the SHA-256 of the full local runner checksum manifest and the full source artifact files, so reviewers can verify the redacted export's relationship to that run.

Every unavailable requirement remains BLOCKED, SKIPPED, or UNVERIFIED in the machine-readable gates.
