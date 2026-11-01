# R2-G1A Switch Development Runner — Implementation and Stop Report

Date: 2026-11-01
Branch: `calibration-stage0-contract-20261029`
Expected parent: `78e56c0820e77ee456e9c1729c3b50ece0c93a02`

## Decision

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

This command was not invoked.

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
- A live G1 source/model fingerprint was not generated in this environment because PyTorch is absent. It must be computed from the committed clean tree and locally cached R1 artifacts before any future run.
- No run namespace or experiment output was created by this task. Historical and pre-existing outputs were not changed.
- The accidental exposure of hashed identities from an all-category private receipt is the principle-level blocker for this run. Do not resume this authorization as if the switch-only boundary had remained intact; obtain fresh researcher direction before any data access.
- Independently of that incident, the complete prior R1 regression suite remains unverified until run in an environment with PyTorch.

No S1 measurements, S2 distance quantiles, radius coverage, feature counts, GPU/RAM use, or real output-byte totals exist. The runner is not a scientific result and this report does not authorize G1-B, G2, other categories, or any confirmation-data work.
