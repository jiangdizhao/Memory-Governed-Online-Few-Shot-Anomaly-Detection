# FINAL-GATE1 Integrated Implementation Report

**Status:** `GATE1_PARTIAL_WITH_SPECIFIC_BLOCKERS`  
**Gate 0 base / repository HEAD:** `a34b870fd4700bfe970e459d2963a602f04b1341`  
**Gate 1 source identity:** local, uncommitted bundle `outputs/calibration/final_benchmark_gate1/attempt_01/gate1_source_identity.json`  
**Source identity SHA-256:** `eae2b30374cba8cb620efe123f9cde10398d083ab4bf9fcaec9668b856956671`  
**Branch:** `calibration-stage0-contract-20261029`

The implementation source and tests are new uncommitted files; the Git HEAD above identifies only the frozen Gate 0 base. The source identity receipt binds the 12 Gate 1 Python modules, the synthetic test file, all seven unchanged Gate 0 document hashes, and the private census/audit artifact hashes. It is not a commit or independent review.

## Scope and access boundary

The researcher-authorized FINAL-GATE1 task covers identity-only projections and implementation work, but does not authorize scientific scoring or Gate 2. This work implements Gate 1 identity-only projections, frozen role/query rules, group-atomic orchestration, outcome-free group seals, a separate Stage B interface, synthetic metrics, and source-bound adapters. No real image was decoded; no DINO inference, GPU operation, scientific Stage A/B, TEST outcome projection, mask access, or confirmation-data access was performed. Gate 0 documents were not changed.

## Implemented components

- `experiments/calibration/final_benchmark/identity.py` reads an allowlisted Real-IAD metadata projection and a VisA `1cls` projection. TEST labels and masks are removed/skipped. Image files are read only as opaque bytes for SHA-256. The manifest validator rejects restricted fields recursively and the writer refuses overwrite.
- `query.py` implements the frozen per-dataset category caps, `Margin_rho` identity-hashed cold start and tie lottery, past-only float64 linear quantiles, and BEA price accounting from final actions.
- `runner.py` enforces ordered five-view Real-IAD groups, freezes all four principal score states per group, makes pre-group proposals, separates group and cap suppression, requires a durable sink, and calls the query-only broker only after fsync of all five score/proposal rows. It also carries the passive F-B-matched N-shadow and the two static score interfaces.
- `sink.py` writes non-overwriting group journal envelopes and a category seal using flush plus `fsync` before returning a disclosure boundary.
- `source_adapters.py` binds Full C1, Signed kNN, B3 refresh, BN+Band D-band utility, and the higher empirical q90 to existing historical source functions. `static_baselines.py` binds prototype selection to the existing A3b CUDA K-Center and records the 8,192 x 4,096 FP16 / 64 MiB contract.
- `evaluation.py` requires a complete Stage A seal and a separate label-projection authorization. `metrics.py` implements undefined-class handling, AUROC/AP/BER/FPR/FNR, Real-IAD non-overlapping Worst-256 diagnostics, VisA `NOT_APPLICABLE`, category-equal macro logic, and the registered paired category bootstrap.
- `telemetry.py` provides timing/counter/byte interfaces. No Gate 2 hardware measurements are represented.

## Gate 1 requirement disposition

| Requirement | Disposition | Evidence / remaining boundary |
|---|---|---|
| Identity projection, role builders, private manifests | `IMPLEMENTED_AND_CPU_VERIFIED` | Synthetic role/hash tests pass; local manifests validate as identity-only. Frozen roles remain uncertified without authenticated release hashes. |
| Real-IAD local census | `DATASET_OR_PROVENANCE_BLOCKED` | 30/30 local categories and complete five-camera inventory; external release provenance and one cross-category duplicate's provenance remain unresolved. |
| VisA 1cls identity | `DATASET_OR_PROVENANCE_BLOCKED` | Local identity census completed; no official physical-group key is available, so online grouping and conditional Static-36+32 eligibility are blocked. |
| Group-atomic runner and durable seal-before-disclosure boundary | `IMPLEMENTED_AND_CPU_VERIFIED` | Synthetic five-view ordering, no within-group state transition, fsync-before-disclosure and completion-seal checks pass for the injected backend interface. |
| Concrete F-B/K-B/F-M/K-M native trajectories | `SOURCE_FIDELITY_BLOCKED` | No joined native DINO/R7/R9/R10 backend with four independent Greedy, correction, C64/B3 and query states is implemented or parity-verified. |
| N-shadow source-compatible F-B wiring | `SOURCE_FIDELITY_BLOCKED` | Callback contract and synthetic defect exclusion exist; inherited query/disclosure and read-only Greedy linkage are not established against a concrete Full C1 trajectory. |
| Static-4+64 | `SOURCE_FIDELITY_BLOCKED` | Static scoring callback contract exists, but complete native feature/scoring/threshold integration has not been verified. |
| Static-36+32 | `SOURCE_FIDELITY_BLOCKED` | Identity schedule and A3b K-Center source entry point are bound; candidate-ID/tie construction and bank/q90 parity remain unresolved. |
| Stage B interface, synthetic metrics and bootstrap | `IMPLEMENTED_AND_CPU_VERIFIED` | Seal-gated synthetic label projection, metric edge cases and deterministic bootstrap pass; no real outcome evaluation was performed. |
| Historical source adapters and interfaces | `IMPLEMENTED_SOURCE_REVIEW_PENDING` | Static signatures/hashes and a CPU R10 score interface were checked; numerical R7/R9/R10/C1/DINO parity remains pending. |
| Runtime and memory limits | `GATE2_NUMERICAL_OR_RESOURCE_VERIFICATION_PENDING` | Interfaces exist; no device, throughput, peak RAM/VRAM, I/O or disk measurements were made. |

This matrix describes code and evidence status only. It does not assert independent review, implementation freeze, data certification, or execution authorization.

## Important implementation limit

The group orchestrator currently requires an injected `ArmBackend` and has no complete concrete DINO/native R7 + R9 C1 + R10 + per-arm Greedy/C64/B3 backend. The adapters expose the source call boundaries, but do not create or independently verify the four native mutable trajectories. Consequently the four arms, N-shadow, Static-4+64 and Static-36+32 are **not yet end-to-end source-fidelity validated**. This is a specific remaining Gate 1 implementation blocker; synthetic callback tests do not establish native algorithm parity.

Static-36+32 uses the approved 4+32 reference / 32 C32 identity schedule and existing A3b K-Center call target. Candidate identity construction from the selected 36 source images still needs a source-fidelity decision against the repository's A4 candidate-ID convention. No alternative coreset was introduced, and no K-Center GPU call was made.

## Data identity status

See [GATE1_DATASET_IDENTITY_AUDIT.md](GATE1_DATASET_IDENTITY_AUDIT.md). A local identity census was generated from the registered roots. The local trees do not provide an externally authenticated official release SHA-256 receipt, so Gate0 role and event hashes remain withheld. VisA's approved split has no recognized official physical-group key, so online VisA remains blocked.

## Synthetic verification

Command:

```powershell
& 'D:\anaconda3\envs\torch\python.exe' -m unittest tests.calibration.final_benchmark.test_gate1_synthetic -v
& 'D:\anaconda3\envs\torch\python.exe' -m compileall -q experiments\calibration\final_benchmark tests\calibration\final_benchmark
git diff --check
```

The synthetic CPU suite covers deterministic role identities, role disjointness and Static-36+32 partitioning, identity-only TEST projections, five-camera ordering and completeness, cold-start and quantile query behavior, BEA price recurrence, cap and suppression accounting, durable seal-before-disclosure order, independent arm state, N-shadow's defect exclusion, Stage B seal authorization, undefined metrics, bootstrap replay, the real CPU R10 score interface, and historical function-signature presence. The NumPy version used by the implementation and tests was `1.24.1`; `Margin_rho` explicitly requests `method="linear"`. Results: **14 tests passed; 0 failed** (after the final rerun). No real-data outcome metrics were computed.

Source signature/hash inventory: `outputs/calibration/final_benchmark_gate1/attempt_01/source_interface_audit.json`; the final Gate 1 bundle and sidecar are `outputs/calibration/final_benchmark_gate1/attempt_01/gate1_source_identity.json` and `.sha256`. Static inspection resolved the existing R10 `group_sha256` scorer keyword, the R9 C1 scoring interface, R7 B3 refresh, E6 BEA components, Experiment C D-band utility, and the A3b K-Center entry points. Numerical R7/R9/R10/C1/DINO parity remains pending the source-bound backend and separately authorized Gate 2 checks.

## Remaining blockers and next-stage boundary

1. Obtain an official release/provenance receipt binding the local Real-IAD annotations and images to a release digest. Only then can exact frozen role and stream hashes be emitted.
2. A researcher-approved official VisA physical-group key is absent from `split_csv/1cls.csv`; obtain separately documented provenance or leave online VisA excluded under the Gate0 decision rule.
3. Complete the concrete source-bound four-arm backend, including independent Greedy, C1/Signed kNN, C64/B3 and N-shadow state; verify C64 protection and native score/update parity on synthetic/source-approved fixtures.
4. Resolve Static-36+32 candidate-ID construction against the A3b/A4 source convention and verify 8,192 FP16 output and C32 higher-q90 arithmetic on approved test fixtures.
5. Only after independent Gate 1 review, obtain a separate Gate 2 authorization for representative GPU/RAM/disk/runtime/resource preflight. No scientific execution is authorized here.

No supervisor sign-off is asserted. No experiment was run, and Gate 2 is not authorized.

The unmeasured shape-based planning estimates and required Gate 2 measurement list are in [GATE1_GATE2_HANDOFF.md](GATE1_GATE2_HANDOFF.md). They include 12.25 MiB per current `(784,4096)` FP32 feature map, 784 MiB for a full category C64 FP32 cache, and 64 MiB per 8,192 x 4,096 FP16 static reference bank; these are contract arithmetic, not measured allocations or resource guarantees.
