# Stage 0 implementation and audit report

**Status: `BLOCKED_WIP`: infrastructure mechanisms pass their synthetic tests, but Stage 0 acceptance is incomplete. This is not scientific validation and is not authorization to start Stage 1.**

## Scope and provenance

Work is on `calibration-stage0-contract-20261029`, whose synchronized parent was `83041f6506d40af9614de661c95a69e3481b07fe`. The required interpreter is `D:\anaconda3\envs\torch\python.exe` (Python 3.11, PyTorch 2.0.1+cu117, SciPy 1.11.2). Validation runs use CPU; CUDA availability is recorded but no GPU computation was needed. No package or environment was changed.

Historical v22, VisionAD, P0/P1, V0.2, native bridge and archived experiment sources were inspected read-only. The adapter continues to call `experiments/risk_safe/native_bridge.py` and `a0_a1_strict4_matched.score_bank` into the historical v22 scoring path; it has no nearest-neighbor fallback. The K=256 shared Greedy writer reuses the native candidate extraction, normalization and `v02_memory.greedy_slot` operator. Historical sources were not modified.

The SCDAC-G source proposal is `docs/Designing and Prioritizing Memory-Dependent _ State-Consistent Calibration for Dynamic-Memory Indust.pdf`, SHA-256 `5fe07256384351c1170d67183eae5a80e5aa289a03586ba767671f062f4fb977`. Proposal-open numerical choices are frozen only in `SCDAC_G_V0_METHOD_SPEC.md` as `PROVISIONAL_METHOD_SPEC`; they are not attributed to the proposal as mandated values or validated science.

The original SPOT/dSPOT paper and author code were inspected. Author repository `asiffer/libspot` was pinned for audit to commit `d57e921a08930b960d07adcbec3a839bd228fe94`; its `src/spot.c` blob SHA-1 is `fdebc8c216fffeea93afcc51efab83a2d6013351` and identifies as version 3.0.0. The pinned source has SPOT code but no dSPOT implementation. No baseline is claimed complete from this source inspection alone.

## Implementation

New calibration engine code is under `experiments/calibration/`; calibration tests are under `tests/calibration/`.

- **Protocol and query tapes:** locked CPU scorer/profile settings, the exact distinction between four deployed supports and 64 additional protected C64 normals, label-blind Q-A-TS/Q-B schedules and group limits, causal observations, and explicit query-only feedback.
- **Shared engine:** one score per event for all controllers; predict-before-label disclosure; query-independent reference write policy; evidence refresh after queried feedback or memory change; memory version/digest checks; offline-only metrics and unavailable risk handling.
- **Checkpointing:** event-boundary snapshots include cursor, score/prediction/trace history, controller state, named component states, last evidence, query/evaluator identity digests and Python/NumPy/Torch RNG state. Anchor checkpoints include private features; native Greedy checkpoints include per-layer tensors, source rows, version, capacity and fixed-support digest. Restore fails closed on source/config/runtime/controller/component identity mismatches.
- **Controller families:** B0 frozen theta; B1 rolling q90 over queried verified normals at arrival; B2 historical v22 sidecar adapter with its unlabeled-normal fallback disabled and C64 initialization recorded as an adaptation; B3 current-state protected-C64 q90; B4 queried-label class-conditional ACI adaptation with corrected normal-lower/defect-upper exact empirical boundaries; and B7 executable provisional SCDAC-G. B5a SPOT, B5b dSPOT, and B6 Sun-CS fail closed as `BLOCKED_FIDELITY`.
- **Runner:** `python -m experiments.calibration.runner --stage 0 --all` performs configured-environment checks, unit tests, deterministic synthetic mechanisms, metadata-path presence checks and manifest/checksum generation. It does not load Real-IAD files, run image-based parity, or launch Stage 1/2. Output is non-overwriting by default, content-keyed, checksummed and marked `INCOMPLETE` while mandatory gates remain open. Resume verifies completed outputs or rebuilds an interrupted deterministic synthetic artifact after source/config identity validation. Verify-only checks a specific output's hashes and does not rerun tests or empirical results.

## Method fidelity record

| Method | Status | Fidelity and unresolved issue |
|---|---|---|
| B0 | `IMPLEMENTED` | D2 higher q90 of 64 additional C64 scores at M0; rank index 57. |
| B1 | `IMPLEMENTED` | Stale-at-arrival q90 of queried, verified normals; W=256, minimum 16. |
| B2 | `IMPLEMENTED_ADAPTED` | Historical v22 `GuardedDualAnchorThresholdController`; disables the unavailable unlabeled-normal path and separately reports native NumPy C64 initialization. No full-loop parity claim. |
| B3 | `IMPLEMENTED_PROPOSED` | Protected-C64 current-memory higher q90, minimum 16, step clip 0.01; queried-normal LSO is not included in this primary arm. |
| B4 | `IMPLEMENTED_ADAPTED` | Queried-only class-conditional ACI-style updates (alpha/beta=.10, gamma=.005), exact empirical threshold geometry and current-state evidence. Sparse feedback/dynamic memory do not inherit ACI guarantees. |
| B5a | `BLOCKED_FIDELITY` | No exact KDD Grimshaw/source convention parity trace. No alternate EVT estimator is labeled SPOT. |
| B5b | `BLOCKED_FIDELITY` | Pinned author source lacks dSPOT; no faithful dSPOT implementation/reference test. |
| B6 | `BLOCKED_FIDELITY` | Full source-equation reference test, observation mapping and abstention-to-primary-metric policy remain incomplete. No generic confidence bound is labeled Sun-CS. |
| B7 | `IMPLEMENTED_PROVISIONAL` | Versioned SCDAC-G mechanism and synthetic reference tests are present. Numerical choices remain proposals for independent research review; CP endpoints are descriptive and no guarantee is claimed. |

## Validation evidence

The focused calibration suite was run with the required environment:

```powershell
& 'D:\anaconda3\envs\torch\python.exe' -m compileall -q experiments/calibration tests/calibration
& 'D:\anaconda3\envs\torch\python.exe' -m unittest discover -s tests/calibration -v
```

Latest recorded suite result: **37 tests, 37 passed, 0 failed, 0 errors, 0 skipped**. It covers the causal label firewall, unqueried-label swap invariance, Q-A-TS/Q-B determinism and group limits, missing-class metrics, B1/B2/B3/B4 behavior, B4 ties and conflicts, B7 evidence states and thresholds, C64/strict4 separation, all-layer LSO, anchor feature checkpoint roundtrip, disk checkpoint-resume equivalence against an uninterrupted run, all executable-controller state restore, RNG restoration, and source/method registry checks. The B2 import emits existing torchvision/Pillow deprecation warnings.

No Real-IAD images, features, masks, confirmation-class outcomes, or hidden class histograms were opened. The dataset metadata path check is reported separately as `SKIP`; it is not a data preflight pass. No Stage 1 or Stage 2 run occurred.

## Gate matrix

| Gate | Status | Evidence or blocker |
|---|---|---|
| G0 branch / protected worktree | `BLOCKED` | Correct requested branch and parent; pre-existing tracked deletions in `outputs/risk_safe_sprint_20261027/` remain in the shared worktree. They were not modified or staged. |
| G1 query identities | `PASS` synthetic; `SKIP` Real-IAD metadata | Q-A-TS/Q-B synthetic identity, one-query-per-group, 16-bin and label-blind tests pass. Dataset roots were only presence-checked. |
| G2 native scorer parity | `BLOCKED` | Direct-vs-shadow VisionAD/v22 scoring parity was not run on a native image/feature bundle. |
| G3 Greedy parity | `BLOCKED` | Greedy mechanism/source-map tests pass; actual native K=256 update, fp16 normalization, static-bank immutability and write parity were not executed. |
| G4 causal firewall | `PASS` synthetic | Predictions precede disclosure; only queried labels are disclosed; threshold controllers share one score/memory trajectory. |
| G5 evidence ledger | `PASS` synthetic | C64 protection, current-version invalidation, private feature restore and all-layer source exclusion are tested. Native score/write attribution remains under G3. |
| G6 controller families | `BLOCKED_FIDELITY` | B5a, B5b and B6 are blocked. B7 is executable only under its provisional specification. |
| G7 reference math | `BLOCKED` | Reference mechanics pass for implemented methods, but all required external baseline families do not have verified implementations. |
| G8 execution/reproducibility | `PASS` synthetic | Midstream engine resume matches uninterrupted score, prediction and trace outputs; RNG restore and private-label identity checks pass. Runner output is checksummed, source/config-addressed and non-overwriting by default. |
| G9 scientific parity | `BLOCKED` | Native shared-score/AUROC invariance across controllers is not verified without the actual native feature path. |

Overall status remains `BLOCKED_WIP`. Individual passing software gates are mechanism evidence only; they do not make the research scientifically validated.

## Remaining blockers

1. Resolve the pre-existing protected archive deletions before G0 can pass. This implementation does not restore or stage them.
2. Execute permitted non-scientific native CPU parity when an approved native support/C64 feature bundle is available: direct/shadow scores, fixed-support immutability, K=256 update parity, fp16 normalization and source attribution.
3. Implement and reference-test exact SPOT Grimshaw behavior and separate dSPOT drift semantics; verify source initialization, update and tie/edge conventions.
4. Read and implement the complete Sun-CS algorithm/reference sequence, and decide how its source tri-state abstentions map to the primary metric without disguising adaptation.
5. Obtain independent research review of SCDAC-G provisional defaults and conflict/missing-side rules. The code executes but is not an authoritative scientific translation.

The Stage 0 output directory and its run-specific hashes/checksums are emitted by the runner and captured in the execution result. No generated runner output or prior smoke output is treated as empirical evidence.
