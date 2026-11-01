# R2-G1A Switch Development Runner — Implementation and Run Report

## Continuation update — 2026-11-01

The earlier stop was recorded before any cohort selection or pilot execution. The researcher has since reviewed the scope incident and explicitly directed continuation. The prior run created no selected cohort, so the out-of-scope identities did not influence sampling. The private all-category receipt was not reopened, copied, enumerated, inspected, or used. This run used only switch normal TRAIN metadata and the predetermined switch-only selection policy.

**S1: ENGINEERING_PASS_DEVELOPMENT_ONLY. S2: PILOT_COMPLETE_DEVELOPMENT_ONLY.** This is descriptive normal-reference geometry, not independent certification or a defect-safety result.

- Python: `D:\anaconda3\envs\torch\python.exe` (`3.11.14`); PyTorch `2.0.1+cu117`; pytest `9.2.0`; CUDA available.
- Device: NVIDIA GeForce RTX 3090 Ti, 24,563 MiB.
- Execution commit: `1591ae692f553c2b53489191774cf7891ae18a28`.
- Run ID: `r2_g1a_switch_20261101_dc4c43879f62`.
- Source fingerprint: `ddcd78f4355191e48e911522d61aa4a8ec830f0a9f88425ebe403e81249940eb`.
- Representation SHA-256: `06237800ef7044857bb617677625f9c2be79c0ce837ef83175ed915b23a91c28`.
- Frozen checkpoint SHA-256: `36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51`.
- Frozen DINO source-tree SHA-256: `b7c0037f90671c4932393bd56de8c8a205b29a6eb22dea563b6c8a88c471da15`.
- Loaded model-state SHA-256 matched the R1 profile: `1f26464ba4055403e550f1d5c1333fb012631d98661df6993cabea730d17e3f9`.
- Runtime profile: `stage05_set_seed_0_math_sdp_v1`; seed 0 and deterministic algorithms enabled, TF32 disabled, math SDP enabled with flash and memory-efficient SDP disabled.
- Switch metadata physical-file SHA-256: `b863b819fe7734a9c83acc26b25d419c8d8897a9ae93d5b8fd9d0e424ecf42d8`. Its TEST section was structurally skipped; no TEST record count was computed.
- Normal TRAIN projection: 1,230 images across 246 Source Groups, with zero non-normal rows; all 246 groups had camera 1. One image per group was selected for 100 groups by the frozen seed-0 rank. The private identity-manifest digest is `54504664ddd0a880324301ce992866e1b4b2e3aebb7c7384773e44781ff65eb6`; selected-order digest is `55da7ac7dc1d6ef6ea6b10142a8174f50a051a6a051f811ecc8150aabf01d9e8`.

### Geometry results

Each distance distribution covers the fixed 49 descriptors per image. Nearest and fifth-nearest values use the unchanged exact source-disjoint comparator. Query and reference use the same cohort, with same-group and same-image exclusions.

| Stage | Groups / images | Descriptors | Distance | q05 | q25 | Median | q75 | q95 | Eligible / excluded pairs | Queries with fewer than 5 references |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| S1 | 10 / 10 | 490 | Nearest | 0.003159 | 0.007353 | 0.051665 | 0.124242 | 0.248169 | 216,090 / 24,010 | 0 |
| S1 | 10 / 10 | 490 | Fifth-nearest | 0.007079 | 0.036975 | 0.183226 | 0.275033 | 0.409350 | 216,090 / 24,010 | 0 |
| S2 | 100 / 100 | 4,900 | Nearest | 0.001200 | 0.003317 | 0.021336 | 0.066668 | 0.122455 | 23,769,900 / 240,100 | 0 |
| S2 | 100 / 100 | 4,900 | Fifth-nearest | 0.002030 | 0.005936 | 0.041604 | 0.100293 | 0.187137 | 23,769,900 / 240,100 | 0 |

S2 nearest-distance coverage was 90.653% at radius 0.10 and 100% at radius 0.50. At those radii, the fraction with at least five local references was 74.878% and 99.980%, respectively. The full fixed radius curve from 0 to 2.0 in 0.02 increments was produced. These values are descriptive; no radius was selected or tuned.

### Runtime and storage

| Measure | Result |
|---|---:|
| Child wall time | 78.281 s |
| Supervised wall time | 84.235 s |
| Child CPU time | 107.656 s |
| Peak CUDA allocated / reserved | 1,223,153,664 / 1,268,776,960 bytes |
| Peak process working set | 3,244,429,312 bytes |
| Prelaunch free disk | 299,286,052,864 bytes |
| Final accounted output | 80,729,032 bytes of 1 GiB |
| Event shards | 100, each `(49, 4096)` float32; no full feature maps persisted |
| Supervisor | `COMPLETED`, child exit 0, termination confirmed, zero active Job Object processes |

The application-level byte budget passed. OS-level hard storage isolation is not provided. The run stayed within the 120-minute supervisor and 10-GiB free-space threshold.

### Correct-interpreter regressions

| Command | Result |
|---|---|
| `D:\anaconda3\envs\torch\python.exe -m py_compile experiments/calibration/r2_g1a_switch_pilot.py tests/calibration/test_r2_g1a_switch_pilot.py` | PASS |
| `D:\anaconda3\envs\torch\python.exe -m pytest -q tests/calibration/test_r2_g1a_switch_pilot.py` | 41 passed |
| `D:\anaconda3\envs\torch\python.exe -m pytest -q tests/calibration/test_r1_matched_correction_pilot.py` | 19 passed |
| `D:\anaconda3\envs\torch\python.exe -m pytest -q tests/calibration/test_r1_s2_c64_score_parity.py` | 9 passed |
| `D:\anaconda3\envs\torch\python.exe -m pytest -q tests/calibration/test_r1_s3_native_map_parity.py` | 14 passed |
| `D:\anaconda3\envs\torch\python.exe -m pytest -q tests/calibration/test_r2_l2b_role_certification.py` | 40 passed |

The earlier single failure was caused by invoking the base `D:\anaconda3` interpreter rather than the specified `torch` environment; it did not reproduce under the required executable.

### Scope and limitations

The completed run used only switch normal TRAIN identities. It did not access TEST records as objects, labels, masks, other categories, Mint/PCB, MVTec AD 2, G1-B, G2, Support/C64, or correction outcomes. The prior scope incident remains documented above and in the original implementation report. No independent A-role certification is claimed. Results describe normal-reference feature geometry only and do not establish correction performance, defect recall, or risk safety.

The run output remains in its unique, untracked output namespace and was not added to Git. The execution commit above binds the implementation used for the run; this report is a later audit update.

Date: 2026-11-01
Branch: `calibration-stage0-contract-20261029`
Expected parent: `78e56c0820e77ee456e9c1729c3b50ece0c93a02`

## Initial implementation report (before continuation)

**Runner implementation: CPU engineering complete, subject to the test limitation below.**
**S1: NOT RUN. S2: NOT RUN.**
**Development pilot status: STOPPED before dataset or image access.**

During an earlier broad repository search, a pre-existing all-category private source-group receipt under `outputs` was accidentally matched. The truncated search output exposed hashed group identities outside the authorized switch-only scope. Those identities were not used. No further output or dataset paths were searched, and no dataset JSON, image, DINO model, GPU, label, TEST record, or mask was accessed. This is a scope incident, so the real-data pilot is stopped; the implementation work below is synthetic-only.

The current Python environment also lacks PyTorch. The runner deliberately requires a committed, clean source tree, verified local DINO artifacts, and the Windows Job Object supervisor before it can start a worker. No attempt was made to bypass these gates.

## Implementation

Implementation changes are limited to the isolated G1-A switch adapter and its focused test file; this audit report is the only added document.

- Added a distinct `DEVELOPMENT_ONLY` permit and CLI. It requires the explicit `R2-G1A-SWITCH-DEV-RUN` authorization argument and cannot be obtained through the existing strict signed-certification path. The strict path remains unchanged.
- Added a two-pass, switch-only metadata projector. It reads the selected `switch.json` bytes once, decodes only `meta.normal_class`, `meta.prefix`, TRAIN anomaly classes, and normal TRAIN image paths. TEST values and non-normal TRAIN image paths are structurally skipped. The code does not enumerate categories.
- Added frozen camera-1 Source Group selection using the existing seed-0 rank, identity parser, group digest, and exact 100-group cap. The selected identity manifest is private and budgeted; the public summary rejects paths, sample keys, labels, and masks.
- Added an integrated S1-to-S2 worker path. S1 uses the first 10 ranked groups; S2 reuses those projections and processes the remaining 90 from the same frozen 100-group cohort. Both stages call the existing `compare_source_disjoint()` mathematics without modification.
- Added local-only DINO loading. The runner checks the pinned source/checkpoint fingerprint before metadata access, directs `torch.hub.load` to the single fingerprinted local tree, refuses non-profile weights, disables downloads, and checks the loaded model state before image decoding.
- Bound all run files to the existing 1 GiB `OutputBudget` and failure reserve, with a tight G1 allowlist for event shards, manifests, supervisor status, private identities, and sanitized public summary. The run namespace must be new and cannot resume. Prelaunch disk space must be at least 10 GiB.
- The parent launches the worker through the existing 120-minute Windows Job Object supervisor. A per-run nonce prevents direct worker invocation. S1/S2 share one execution commit, namespace, deadline, selected cohort, and output budget.
- Resource and geometry receipts distinguish development evidence from independent A-role certification. No claim of third-party certification, correction performance, defect recall, or risk safety is emitted.

The CLI entry point is:

```powershell
python -m experiments.calibration.r2_g1a_switch_pilot `
  --run-development-pilot `
  --authorize-development-only R2-G1A-SWITCH-DEV-RUN `
  --run-id r2_g1a_switch_<unique-id>
```

This command was not invoked during the initial implementation task. The continuation run above used this CLI with a unique run ID.

## Verification

| Check | Result |
|---|---|
| `python -m py_compile experiments/calibration/r2_g1a_switch_pilot.py tests/calibration/test_r2_g1a_switch_pilot.py` | PASS |
| `python -m pytest -q tests/calibration/test_r2_g1a_switch_pilot.py -k "not exact_r1_adapter_projects_one_image_and_releases_full_map"` | 40 passed, 1 deselected |
| Complete `tests/calibration/test_r2_g1a_switch_pilot.py` | 40 passed, 1 failed because `torch` is not installed; failing test is the existing real adapter tensor test |
| `python -m pytest -q tests/calibration/test_r1_matched_correction_pilot.py` | BLOCKED at collection because `torch` is not installed |
| Dataset / image / DINO / GPU pilot | NOT RUN |

The passing synthetic coverage includes TEST/non-normal path skipping, deterministic 100-group camera-1 selection, selected-cohort order checks for S1/S2, unique non-resumable output namespaces, writer allowlisting and pre-write overflow refusal, exact supervisor output-byte accounting, explicit parent authorization, worker nonce enforcement, wrong branch/parent rejection, and local-only DINO loader behavior.

The full G1 and R1 test suites cannot be claimed as passing. No test was weakened or changed to hide the missing dependency.

## Provenance and remaining blockers

- Baseline branch and parent were verified before implementation. The implementation commit is reported with this audit; it is a source commit only, not an execution commit.
- The reported CPU tests ran against the working tree based on the expected parent; the exact source commit binds the same reviewed adapter, tests, and report.
- At the time of the initial report, a live G1 source/model fingerprint had not been generated because the selected Python environment lacked PyTorch. The continuation update above records the verified fingerprint.
- Before continuation, no run namespace or experiment output had been created by this work. The continuation outputs are described above; historical and pre-existing outputs were not changed.
- The accidental exposure of hashed identities from an all-category private receipt remains documented. The researcher reviewed it and directed continuation; no cohort had previously been selected, and the receipt was not reused.
- At the time of the initial report, the prior R1 regression suite was unverified because the selected Python environment lacked PyTorch. The continuation results above resolve that environment issue.

The initial report had no S1/S2 measurements. This continuation report records the completed development-only switch geometry pilot. The results do not authorize G1-B, G2, other categories, or confirmation-data work.
