# Stage 0 fifth-pass implementation and verification report

**Disposition: `BLOCKED_WIP`; implementation evidence is not scientific validation.**

This report records the fifth-pass repair on branch
`calibration-stage0-contract-20261029`. The Runner source snapshot is the
pre-commit checkout `a083738556424563cba39d8df246623f96a74fd1`. The full,
checksummed local Runner output is
`outputs/calibration/stage0/stage0_fifthpass_release_20261029/`; the compact,
privacy-scanned package is
[`audit/stage0_fifthpass_release_v1`](audit/stage0_fifthpass_release_v1/).

Stage 1 and Stage 2 were not run. No confirmation-class images, masks,
metadata, labels, outcomes, or hidden histograms were opened. The native
diagnostics were bounded to development-class training normals; the four
class preflight used JSON metadata only.

## Fifth-pass repairs

- Restored `experiments/risk_safe/shadow_memory.py` byte-for-byte to the
  reviewed `9d2d20df0bd94629eaea0f8cbd6a6f84ebc28a44` source. Its raw SHA-256
  and the baseline SHA-256 both equal
  `70c2ba3d29b225f241d9ad147f5be168ee2e0613f452e1428cc585bbce3339b5`.
- Moved calibration content-based memory versioning to
  `experiments/calibration/memory_state.py`. Historical Greedy operation
  version, calibration prototype-content version, and source-map revision are
  separately recorded and tested.
- Added a static protected-source baseline over the resolved native
  dependency closure plus explicit historical inputs. All 13 protected files
  matched the reviewed baseline; the Runner fingerprints 61 relevant source,
  test, proposal, and contract files. A committed mutation is checked against
  fixed hashes, so G0 does not depend on an uncommitted-worktree check.
- Read and specified the entire Sun et al. ICML 2024 paper and appendices in
  [`SUN_CS_SOURCE_FIDELITY_SPEC.md`](SUN_CS_SOURCE_FIDELITY_SPEC.md). The
  reference module implements only individually defined paper-equation
  primitives and raises on undefined endpoint and empty-prefix behavior. No
  B6 controller is registered. The PMLR paper and full PDF are linked from the
  specification.
- Added a synthetic activation harness reusing `PrequentialEngine`. It tests
  shared scores, common query tape, threshold-independent memory writes,
  event checkpoint/resume, B5 warm-up/updates/alarm exclusions, and B7
  source-tagged evidence branches plus a C64 current-memory refresh.
- Expanded Runner gate aliases and artifacts, and strengthened the compact
  audit exporter to recompute the full-source fingerprint, require G0's
  protected-source evidence, preserve B6's blocked status, link exported files
  to the local Runner checksums, and emit an independent verification summary.

## Verification executed

All commands used the existing environment at
`D:\anaconda3\envs\torch\python.exe`; no dependency was installed or changed.

```powershell
& 'D:\anaconda3\envs\torch\python.exe' -m compileall -q experiments/calibration tests/calibration
& 'D:\anaconda3\envs\torch\python.exe' -m unittest discover -s tests/calibration -v
& 'D:\anaconda3\envs\torch\python.exe' -m experiments.calibration.runner --stage 0 --all --output-dir outputs/calibration/stage0/stage0_fifthpass_release_20261029
& 'D:\anaconda3\envs\torch\python.exe' -m experiments.calibration.runner --stage 0 --verify-only --output-dir outputs/calibration/stage0/stage0_fifthpass_release_20261029
& 'D:\anaconda3\envs\torch\python.exe' -m experiments.calibration.audit_export --export --source-output-dir outputs/calibration/stage0/stage0_fifthpass_release_20261029 --output-dir docs/calibration/audit/stage0_fifthpass_release_v1
& 'D:\anaconda3\envs\torch\python.exe' -m experiments.calibration.audit_export --verify --source-output-dir outputs/calibration/stage0/stage0_fifthpass_release_20261029 --output-dir docs/calibration/audit/stage0_fifthpass_release_v1
git diff --check
```

Compilation passed. The full suite passed **76/76**, with zero failures,
errors, or skips. Runner artifact verification passed checksums, source/config
match, and protected-source verification. Compact audit export and independent
source-linked verification both passed. `git diff --check` found no whitespace
errors; Git emitted only line-ending normalization notices for CRLF files.

## Controller activation

The 1,150-event synthetic tape is explicitly a mechanism diagnostic, not a
Real-IAD performance experiment. Full versus event-boundary-resumed execution
matched, and changing the paired frozen controller threshold left scores,
query identities, and the entire memory path unchanged.

| Method | Warm-up / initialization | Available predictions | Threshold updates | Operational status |
|---|---:|---:|---:|---|
| B0 | 64 additional C64 anchors before stream; 0 stream warm-up events | 1,150 | 0 | PASS |
| B1 | 64 additional C64 anchors before stream; 0 stream warm-up events | 1,150 | 2 | PASS |
| B2 | 64 additional C64 anchors before stream; 0 stream warm-up events | 1,150 | 2 | PASS |
| B3 | 64 additional C64 anchors before stream; 0 stream warm-up events | 1,150 | 18 | PASS |
| B4 | 64 additional C64 anchors before stream; 0 stream warm-up events | 1,150 | 18 | PASS |
| B5a SPOT | 1,000 score observations | 150 | 42 | PASS; 149 accepted-normal updates, 62 peak updates, 1 alarm excluded |
| B5b dSPOT | 50-observation local window + 1,000 residual warm-up observations | 100 | 99 | PASS; 99 accepted-normal updates, 26 peak updates, 1 alarm excluded from POT and local mean; shifted local mean exercised |
| B6 Sun-CS | Not run; no source-defined startup | 0 | 0 | BLOCKED_FIDELITY |
| B7 SCDAC-G | One synthetic evidence update; protected C64 + queried anchors | 1 | 1 | PASS for listed synthetic branches; method remains provisional |

The harness's aggregate status is `PARTIAL` because B6 is blocked. Checkpoint
equivalence and shared-score/memory invariance are true for implemented
activation arms. B7 branch coverage includes BOTH, NORMAL_ONLY, CONFLICT,
missing-side hold, a B7-versus-SCQ departure, feasible-candidate selection,
step clipping and logged executed feasibility, exclusion of unqueried defect
rows, current-memory C64 score refresh, queried-normal LSO tagging, and
checkpoint restoration. These are code-path checks, not guarantees or risk
estimates.

## Native and protocol gates

| Gate | Status | Scope |
|---|---|---|
| G0 / G0_PROTECTED_SOURCE_INTEGRITY | PASS | Authorized branch, 13 protected hashes, exact historical shadow-memory SHA |
| G1 / G1_QUERY_PROTOCOL | PASS | Synthetic query protocol and causal-label tests |
| G1_real_data | PASS | Four development categories; metadata only; support/C64 disjointness and query tapes validated |
| G2 / G2_NATIVE_SCORING | PASS | Bounded CUDA audiojack train-only live v22, untouched clone, and adapter scores agree exactly; no absolute scores emitted |
| G3 / G3_NATIVE_GREEDY_OPERATOR | PASS | CPU APPEND/REPLACE/SKIP and repeated-slot source attribution exact; bounded CUDA append parity exact |
| G4 / G4_CAUSAL_FIREWALL | PASS | Predict-before-disclosure, query-only feedback and threshold-isolated writer tests |
| G5 / G5_ANCHOR_AND_LSO | PASS | Protected additional C64, current-version refresh, all-layer source LSO and source revision tests |
| G6 / G6_ALL_METHOD_FAMILIES | BLOCKED_FIDELITY | B6 unavailable; B7 remains provisional |
| G7 / G7_EXTERNAL_SOURCE_FIDELITY | BLOCKED | Paper-equation reference tests are not external author-code trace parity; B6 remains blocked |
| G8 / G8_REPRODUCIBILITY | PASS | 76 tests; checkpoint/RNG, source/config fingerprints, Runner and audit checksums |
| G9_NATIVE_CORE | PASS | Bounded 34-event audiojack train-normal replay; native resume, memory, LSO and synthetic-label invariance checks |
| G9_CONTROLLER_ACTIVATION | PARTIAL | B0-B5 and B7 synthetic activation pass; B6 is blocked |
| G9_FULL_MATRIX | BLOCKED_FIDELITY | Complete controller matrix is not run or claimed without B6 |
| G9 aggregate | BLOCKED | Full Stage 0 acceptance is not met |

The CUDA smoke used the cached `dinov2_vitl14_reg` checkpoint and PyTorch
`2.0.1+cu117`. Live native versus untouched clone and native versus
calibration-adapter score deltas were both zero. The feature-native Greedy
matrix separately exercised append, saturated replacement, skip, repeated
replacement of the same slot, attribution revision, and checkpoint restore.
The 34-event replay used a synthetic binary label fixture solely to verify
score-vector invariance; it is not anomaly-performance evidence.

## Baseline fidelity and remaining blockers

- **B5a SPOT and B5b dSPOT:** status remains
  `IMPLEMENTED_PAPER_EQUATIONS_REFERENCE_TESTED`. Tests cover independent
  Grimshaw/GPD comparison, POT threshold and near-zero-shape limit,
  warm-up/missingness, accepted updates, alarm exclusion, drift-window behavior
  and checkpoint restoration. No compiled numerical trace parity with pinned
  `libspot` is claimed. The inspected pinned `libspot` source contains SPOT but
  no dSPOT implementation.
- **B6 Sun-CS:** remains `BLOCKED_FIDELITY`. Seven reference primitive tests
  pass. No verified author implementation was located. The paper leaves
  endpoint empirical-quantile behavior undefined when Equation (1) clips to
  0 or 1, and does not define the empty-prefix confidence set or first
  Algorithm 3 decision. A faithful independent Algorithm 2/3 reset trace is
  also absent. Dynamic-memory dependent scores do not inherit its guarantees.
- **B7 SCDAC-G:** remains `IMPLEMENTED_PROVISIONAL`; its descriptive
  constraints, selected defaults and bounds are not statistically certified.
- **Overall:** no empirical superiority, Real-IAD risk guarantee, Stage 1/2
  outcome, or scientific validation claim is made. Independent research
  review remains required.
