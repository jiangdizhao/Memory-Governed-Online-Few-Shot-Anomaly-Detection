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
- G6: BLOCKED_FIDELITY — Existing G6 acceptance remains in force pending researcher review of the proposed gate amendment: external author-code parity is not claimed for SPOT/dSPOT/Sun-CS, B2 is an explicitly adapted historical sidecar, and B7 remains provisional. B6 itself is registered as a paper-equation implementation with disclosed finite-sample conventions, not labeled BLOCKED solely for those conventions.
- G6_ALL_METHOD_FAMILIES: BLOCKED_FIDELITY — Existing G6 acceptance remains in force pending researcher review of the proposed gate amendment: external author-code parity is not claimed for SPOT/dSPOT/Sun-CS, B2 is an explicitly adapted historical sidecar, and B7 remains provisional. B6 itself is registered as a paper-equation implementation with disclosed finite-sample conventions, not labeled BLOCKED solely for those conventions.
- G6_CORE_METHODS: PASS — B0-B4 and provisional B7 registered paths are covered by the passing calibration suite; B7 remains explicitly provisional.
- G6_EXTERNAL_PAPER_EQUATION_IMPLEMENTATIONS: PASS — B5a/B5b and B6 paper-equation implementations have independent equation/stream references; no author-code parity is claimed.
- G6_OPERATIONAL_VARIANTS: PASS — SPOT, dSPOT, and Sun-CS Frozen-Theta wrappers are separate registered rows with startup/abstention mode accounting.
- G6_UNRESOLVED_METHODS: BLOCKED_RESEARCH_REVIEW — Unresolved limitations remain visible; no method row is removed.
- G7: BLOCKED — Existing G7 source-fidelity acceptance remains in force: mathematical reference tests are reported separately, while verified author-code trace parity is unavailable for the external implementations and theoretical assumptions/guarantee transfer are not established for this score, feedback, and memory process.
- G7_ASSUMPTION_COMPATIBILITY: NOT_ESTABLISHED — Score dependence, physical-defect target, selective feedback, and dynamic memory differ from external source assumptions; disclosures are in the frozen candidate.
- G7_EXTERNAL_SOURCE_FIDELITY: BLOCKED — Existing G7 source-fidelity acceptance remains in force: mathematical reference tests are reported separately, while verified author-code trace parity is unavailable for the external implementations and theoretical assumptions/guarantee transfer are not established for this score, feedback, and memory process.
- G7_MATHEMATICAL_REFERENCE_TESTS: PASS — Calibration reference suite status; includes independent Sun-CS A2/A3 reference comparisons and SPOT/dSPOT equations.
- G7_SOURCE_IMPLEMENTATION_PARITY: BLOCKED_UNAVAILABLE — No verified author implementation trace was available for external SPOT/dSPOT/Sun-CS parity; independent equation tests are not a substitute.
- G7_THEORETICAL_GUARANTEE_TRANSFER: NOT_ESTABLISHED — No transfer proof for physical-defect risk under sparse feedback and adaptively changing memory is claimed.
- G8: PASS — Event-boundary checkpoint/resume is compared with an uninterrupted synthetic run, including memory, controller, trace, private-label identity, and Python/NumPy/Torch RNG restoration.
- G8_REPRODUCIBILITY: PASS — Event-boundary checkpoint/resume is compared with an uninterrupted synthetic run, including memory, controller, trace, private-label identity, and Python/NumPy/Torch RNG restoration.
- G9: PASS — Stage 0 native score/memory/AUROC invariance, all-controller synthetic activation, and B7 full-engine resume/counterfactual checks passed. This establishes engineering parity only, not Real-IAD scientific validation.
- G9_ALL_REGISTERED_METHODS: PASS — Every original controller and separate fallback row participates in the full PrequentialEngine test; B7 is covered by both two-scenario full-engine integration and the complete method activation matrix.
- G9_CONTROLLER_ACTIVATION: PASS — All registered original methods and separate fallback variants are represented in the causal synthetic controller activation matrix; see per-method rows.
- G9_CONTROLLER_SYNTHETIC_ACTIVATION: PASS — Causal shared-engine controller activation is synthetic mechanism evidence only.
- G9_FULL_ENGINE_INTEGRATION: PASS — B7 two-scenario multi-event PrequentialEngine integration, deterministic resume, current-memory C64 refresh, queried-normal LSO, and score/query/memory counterfactual invariance; synthetic only.
- G9_FULL_MATRIX: PASS — Every registered original and operational controller participated in the same bounded native-v22 score/memory/query replay; this is an engineering diagnostic, not an empirical performance result.
- G9_NATIVE_CORE: PASS — Bounded multi-event actual v22/DINO train-normal trajectory passed causal query/write, current-version C64 refresh, LSO, native checkpoint-resume equivalence and synthetic-label AUROC invariance.
- G9_SCIENTIFIC_VALIDATION_NOT_STAGE0: OUT_OF_SCOPE_NOT_RUN — Stage 0 does not establish Real-IAD scientific performance or Stage 1/2 authorization.

Executed tests: 96; passed 96; failed 0; errors 0; skipped 0.
Synthetic native Greedy operator matrix: PASS.
Native event replay: PASS; all-controller matrix: PASS_BOUNDED_NATIVE_DIAGNOSTIC.

## Provenance

The compact evidence files are content-hashed in `checksums.json`. `audit_manifest.json` records the SHA-256 of the full local runner checksum manifest and the full source artifact files, so reviewers can verify the redacted export's relationship to that run.
Source fingerprint recomputation: **PASS**.
Protected historical-source verification: **PASS**; G0 gate consistency: **PASS**.

Every unavailable requirement remains BLOCKED, SKIPPED, or UNVERIFIED in the machine-readable gates.
