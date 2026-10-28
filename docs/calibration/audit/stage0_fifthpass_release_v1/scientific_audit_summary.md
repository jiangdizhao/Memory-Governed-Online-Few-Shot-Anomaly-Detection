# Stage 0 compact scientific audit

Runner status: **BLOCKED_WIP**.
This package contains implementation evidence, not scientific validation.
Stage 1 and Stage 2 were not run; no confirmation-class outcomes were opened.
All Real-IAD scoring evidence is limited to the bounded development-class training-normal diagnostic.
Synthetic binary labels in native replay are marked synthetic and are not Real-IAD outcomes.

## Gate snapshot

- G0: PASS — Transitive protected-source baseline and byte-level historical shadow_memory check passed on the authorized branch.
- G0_PROTECTED_SOURCE_INTEGRITY: PASS — Transitive protected-source baseline and byte-level historical shadow_memory check passed on the authorized branch.
- G1: PASS — Synthetic Q-A-TS/Q-B metadata tests passed.
- G1_QUERY_PROTOCOL: PASS — Synthetic Q-A-TS/Q-B metadata tests passed.
- G1_real_data: PASS — Official sample identities, train-only support/C64 selection, and label-blind Q-A-TS/Q-B were checked using development-category JSON metadata; no image or mask was opened.
- G2: PASS — Actual development train-normal score parity passed across live native v22, untouched clone, and calibration adapter.
- G2_NATIVE_SCORING: PASS — Actual development train-normal score parity passed across live native v22, untouched clone, and calibration adapter.
- G3: PASS — Synthetic native operator matrix passed APPEND/REPLACE/SKIP, repeated-slot source attribution, prototype-content audit, LSO filtering and exact checkpoint restore; bounded CUDA v22 smoke passed live-score and append parity.
- G3_NATIVE_GREEDY_OPERATOR: PASS — Synthetic native operator matrix passed APPEND/REPLACE/SKIP, repeated-slot source attribution, prototype-content audit, LSO filtering and exact checkpoint restore; bounded CUDA v22 smoke passed live-score and append parity.
- G4: PASS — Synthetic predict-before-disclosure and queried-only feedback tests passed.
- G4_CAUSAL_FIREWALL: PASS — Synthetic predict-before-disclosure and queried-only feedback tests passed.
- G5: PASS — Synthetic protected C64, current-version evidence, and all-layer LSO source attribution tests passed.
- G5_ANCHOR_AND_LSO: PASS — Synthetic protected C64, current-version evidence, and all-layer LSO source attribution tests passed.
- G6: BLOCKED_FIDELITY — B5a/B5b are paper-equation implementations with reference tests; B6 Sun-CS remains BLOCKED_FIDELITY; B7 remains provisional.
- G6_ALL_METHOD_FAMILIES: BLOCKED_FIDELITY — B5a/B5b are paper-equation implementations with reference tests; B6 Sun-CS remains BLOCKED_FIDELITY; B7 remains provisional.
- G7: BLOCKED — SPOT/dSPOT equation references pass, but author-code trace parity is not claimed; B6 source fidelity and source-defined startup remain unresolved.
- G7_EXTERNAL_SOURCE_FIDELITY: BLOCKED — SPOT/dSPOT equation references pass, but author-code trace parity is not claimed; B6 source fidelity and source-defined startup remain unresolved.
- G8: PASS — Event-boundary checkpoint/resume is compared with an uninterrupted synthetic run, including memory, controller, trace, private-label identity, and Python/NumPy/Torch RNG restoration.
- G8_REPRODUCIBILITY: PASS — Event-boundary checkpoint/resume is compared with an uninterrupted synthetic run, including memory, controller, trace, private-label identity, and Python/NumPy/Torch RNG restoration.
- G9: BLOCKED — Native core and controller activation are separately scoped; the complete Stage 0 matrix remains blocked by B6 fidelity.
- G9_CONTROLLER_ACTIVATION: PARTIAL — B0-B5 and B7 synthetic activation results are in controller_activation_matrix.json; B6 remains source-blocked.
- G9_FULL_MATRIX: BLOCKED_FIDELITY — Synthetic activation covers B0-B5 and B7; B6 remains blocked, so the complete native controller matrix is not run or claimed.
- G9_NATIVE_CORE: PASS — Bounded multi-event actual v22/DINO train-normal trajectory passed causal query/write, current-version C64 refresh, LSO, native checkpoint-resume equivalence and synthetic-label AUROC invariance.
- G9_all_controller_matrix: BLOCKED_FIDELITY — Synthetic activation covers B0-B5 and B7; B6 remains blocked, so the complete native controller matrix is not run or claimed.
- G9_controller_activation: PARTIAL — B0-B5 and B7 synthetic activation results are in controller_activation_matrix.json; B6 remains source-blocked.
- G9_native_core: PASS — Bounded multi-event actual v22/DINO train-normal trajectory passed causal query/write, current-version C64 refresh, LSO, native checkpoint-resume equivalence and synthetic-label AUROC invariance.

Executed tests: 76; passed 76; failed 0; errors 0; skipped 0.
Synthetic native Greedy operator matrix: PASS.
Native event replay: PASS; all-controller matrix: BLOCKED_B6_FIDELITY.

## Provenance

The compact evidence files are content-hashed in `checksums.json`. `audit_manifest.json` records the SHA-256 of the full local runner checksum manifest and the full source artifact files, so reviewers can verify the redacted export's relationship to that run.
Source fingerprint recomputation: **PASS**.
Protected historical-source verification: **PASS**; G0 gate consistency: **PASS**.

Every unavailable requirement remains BLOCKED, SKIPPED, or UNVERIFIED in the machine-readable gates.
