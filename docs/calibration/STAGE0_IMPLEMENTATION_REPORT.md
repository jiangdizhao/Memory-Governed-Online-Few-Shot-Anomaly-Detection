# Stage 0 implementation report

> **Historical report:** this records the prior fourth-pass state and is
> retained as an archive. The current fifth-pass implementation, evidence,
> and blockers are in
> [`STAGE0_FIFTHPASS_IMPLEMENTATION_REPORT.md`](STAGE0_FIFTHPASS_IMPLEMENTATION_REPORT.md).

## Current fourth-pass status (authoritative)

**Overall status: `BLOCKED_WIP`; Stage 0 is not scientifically validated.**
The implementation snapshot is on `calibration-stage0-contract-20261029`,
based on source commit `9d2d20df0bd94629eaea0f8cbd6a6f84ebc28a44`. The complete
working-tree source fingerprint is `693b17200af009da4bf682ea0f25827e02d36d82f3fc30d8615d421d18fc7537`.
All work uses `D:\anaconda3\envs\torch\python.exe` (Python 3.11,
PyTorch `2.0.1+cu117`, CUDA available); no environment or dependency changes
were made.

The actual checkout had no tracked deletions (`git ls-files -d` was empty), and
no protected historical source was modified. This differs from the
third-pass attachment's stated 36 deletions. Eight pre-existing untracked
Stage 0 artifact directories were left untouched and unstaged:
`stage0_31f7072d4a206d12`, `stage0_341a074daa935971`,
`stage0_4832e2270c72c525`, `stage0_bee149a5010fa818`,
`stage0_ceaf3aa9e3aaed6a`, `stage0_e457ca9e96fed287`,
`stage0_thirdpass_20261029`, and `stage0_thirdpass_20261029_final`.

### Current gate matrix

| Gate | Status | Evidence / remaining issue |
|---|---|---|
| G0 branch and protected files | `PASS` | Correct branch; no tracked deletions or protected-file changes. |
| G1 synthetic protocol | `PASS` | Q-A-TS/Q-B determinism, label blindness, quotas and group limits. |
| G1 Real-IAD metadata | `PASS` | Metadata-only preflight passed for audiojack, regulator, rolled_strip_base and switch. |
| G2 native scoring bridge | `PASS` | Actual train-normal v22/clone/adapter score deltas were all 0.0. |
| G3 Greedy parity | `PASS` | Synthetic native-source matrix: APPEND 8, REPLACE 10, SKIP 22; actual CUDA smoke additionally passed the append path. |
| G4 causal firewall | `PASS` | Prediction precedes disclosure; only queried verified normals write. |
| G5 evidence and LSO | `PASS` | Synthetic protection/current-version tests and native replay verified all 8 queried-source rows excluded. |
| G6 method families | `BLOCKED_FIDELITY` | B5a/B5b paper-equation implementations are reference-tested; B6 Sun-CS remains blocked; B7 is provisional. |
| G7 external reference math | `BLOCKED` | SPOT/dSPOT references pass; no faithful Sun-CS implementation/reference vectors. |
| G8 reproducibility | `PASS` | 63/63 tests; actual native checkpoint at event 17 resumed exactly against uninterrupted trace/state. |
| G9 native core | `PASS` | 34-event native DINO/v22 replay, fixed one-query tape, shared memory, current C64 refresh, LSO and synthetic-label AUROC invariance. |
| G9 full controller matrix | `BLOCKED_FIDELITY` | B6 is unavailable; full G9 remains `BLOCKED`. |

### Validation and audit outputs

- Compilation: `& 'D:\anaconda3\envs\torch\python.exe' -m compileall -q experiments/calibration tests/calibration` — passed.
- Full Stage 0 suite: 63 run, 63 passed, 0 failed, 0 errors, 0 skipped.
- Runner: `outputs/calibration/stage0/stage0_fourthpass_run1_20261029/` — `BLOCKED_WIP`; all artifact checksums and source/config fingerprints verify.
- Compact public package: [stage0_fourthpass_v1](audit/stage0_fourthpass_v1/); checksum verification, source-artifact link and privacy scan all pass.
- SPOT/dSPOT are executable and independently reference-tested against paper equations and a seeded SciPy GPD fit. No compiled libspot C trace parity is claimed. Sun-CS has no controller implementation and remains `BLOCKED_FIDELITY`.

### Native evidence scope

The Greedy matrix uses synthetic feature fixtures with the original `PatchVMB`,
`MemoryArm.update`, `greedy_slot`, native normalization, and shared writer. It
audits prototype bytes, source rows after repeated replacement of row 0,
source-aware LSO, static-support immutability, content-version semantics and
checkpoint restoration. It is an operator test, not an image scorer.

The CUDA native replay encoded 102 development training images (4 supports,
64 additional protected C64 anchors and 34 stream events) and made 326 native
score calls across uninterrupted and resumed runs. Its fixed Q-A-TS tape
queried one verified train normal; eight candidate rows were appended, memory
advanced from version 0 to 1, all 64 C64 anchors were refreshed, and eight
rows for the queried normal were excluded by LSO. The event-boundary checkpoint
at cursor 17 matched the uninterrupted scores, trace, predictions, controller
states and final memory exactly. The binary labels used to verify shared-score
AUROC equality are explicitly synthetic and are not Real-IAD outcomes.

### Scientific boundary and remaining blocker

No Stage 1 or Stage 2 experiment was run. No Real-IAD test image, mask, test
outcome, Mint/PCB confirmation metadata, or confirmation label was opened. The
native diagnostic is limited to development-class training normals and makes
no performance or superiority claim. The independent research review remains
required. B6's missing source-faithful confidence-set, offline-match,
change-detection/reset and startup implementation is the acceptance blocker;
we did not replace it with a generic confidence interval.

The rest of this file preserves the third-pass report as historical context;
its gate table and blocker statements are superseded by the current status
above.

## Historical third-pass report (superseded)

## Branch, environment, and protected worktree

- Branch: `calibration-stage0-contract-20261029`.
- Synchronized base at start of this continuation: `42aecc32f865b5b5a9cf782c32e14755df305106`.
- Interpreter: `D:\anaconda3\envs\torch\python.exe`; Python 3.11,
  PyTorch `2.0.1+cu117`, CUDA available. No environment or dependency changes.
- The shared worktree already contained **36 tracked deletions** under
  `outputs/risk_safe_sprint_20261027/` and six untracked earlier Stage 0 smoke
  directories. They were not read, modified, restored, or staged. G0 remains
  blocked until the unrelated deletions are resolved by the workspace owner.
- Historical v22, VisionAD, P0/P1, V0.2, risk-safe bridge, and archived
  experiment sources were inspected as read-only dependencies. No historical
  source or prior result file is changed by this implementation.

## Implemented Stage 0 infrastructure

New implementation code is confined to `experiments/calibration/` and tests to
`tests/calibration/`.

- Versioned CPU/CUDA-execution-independent scientific profile; query-tape
  identities are label blind and use the official Real-IAD sample grouping.
- Prequential engine gives all threshold controllers one shared score and
  memory trajectory, predicts before disclosure, and routes only verified
  queried labels to the shared writer. Evaluation labels remain offline.
- Protected C64 anchors are additional to the four deployed supports; they are
  read-only. Current-state normal anchors require all-layer leave-source-out
  provenance. Checkpoints include private anchor features, controller/memory
  state, provenance, and RNG state.
- Shared native Greedy wrapper uses the historical candidate selection,
  full-matrix normalization, fp16 prototype storage, and K=256 per-layer cap.
  The static support bank has an explicit content digest.
- Real-IAD metadata preflight verifies the historical train-only D2 support and
  64-anchor selection, official group separation, Q-A-TS and Q-B feasibility,
  budget fractions, one-view-per-group behavior, and realized query/bin counts
  for the four development classes. It reads no confirmation-category
  metadata, opens no image or mask, and does not pass test labels to tape logic.
- One-command Stage 0 runner writes a source/config-locked manifest, method
  cards, tests, synthetic trace, metadata preflight and optional native smoke
  records. Outputs are checksummed, non-overwriting, resumable only under an
  exact source/config identity, and marked `INCOMPLETE` while required gates
  remain open.

## Native parity smoke

The reusable runner smoke uses the existing cached DINOv2 model and P0/v22
scorer on the four historical audiojack train supports. It selects 64 other
train-normal official groups for protected C64 identities and one further
verified train normal as a disposable update fixture. No test image, mask,
confirmation class, or test outcome is opened. The update fixture uses an
explicit synthetic threshold above its native score only to exercise the
normal-write operator; it is not a calibration result.

The smoke compares a live native bank with an untouched clone, direct v22
scoring with the calibration adapter, and genuine V0.2 Greedy `MemoryArm.update`
with the shared operator after an exact K=256 isolation. It records only score
deltas, opaque identity hashes, and memory parity evidence, never absolute
image scores. This proves a bounded implementation path, not a full stream,
theta0 calibration, cross-controller trajectory, or empirical performance.

## Method fidelity

| Family | Status | Record |
|---|---|---|
| B0 frozen threshold | `IMPLEMENTED` | D2 higher q90 of 64 additional C64 scores at M0; zero-based rank 57. |
| B1 rolling quantile | `IMPLEMENTED` | Stale-at-arrival queried verified-normal scores; fixed window and minimum are in the method card. |
| B2 historical sidecar | `IMPLEMENTED_ADAPTED` | Wraps v22 `GuardedDualAnchorThresholdController`; the unavailable unlabelled-normal path is disabled and adaptation is declared. No full-loop parity claim. |
| B3 state-consistent quantile | `IMPLEMENTED_PROPOSED` | Current-memory protected-C64 higher q90 with fixed movement limit; not a proposed-method validation. |
| B4 dual ACI | `IMPLEMENTED_ADAPTED` | Query-only class-specific feedback and discrete threshold geometry; it inherits no original ACI guarantee. |
| B5a SPOT | `BLOCKED_FIDELITY` | No paper-equation/Grimshaw reference parity trace; no substitute EVT estimator is called SPOT. |
| B5b dSPOT | `BLOCKED_FIDELITY` | Pinned author source has no dSPOT implementation; no verified drift update policy. |
| B6 Sun-CS | `BLOCKED_FIDELITY` | Source-consistent startup, reference controller tests, observation mapping, and abstention policy are unresolved. |
| B7 SCDAC-G | `IMPLEMENTED_PROVISIONAL` | Versioned executable proposal with descriptive CP endpoints. Feasible-set selection, conflict fallback, clipping, and secondary FPR diagnostic are separately specified and tested. Numerical choices remain open for independent review. |

The detailed source and method audit is in
`docs/calibration/BASELINE_FIDELITY_AND_REFERENCES.md`; the executable B7
defaults and objective are in `docs/calibration/SCDAC_G_V0_METHOD_SPEC.md`.
Code, tests, protected historical inputs, checkpoint weights, and proposal
source hashes are tabulated in
`docs/calibration/STAGE0_SOURCE_PROVENANCE.md`.
Neither document claims that a provisional mechanism is scientifically
validated.

## Validation record

Commands use only the existing `torch` environment:

```powershell
& 'D:\anaconda3\envs\torch\python.exe' -m compileall -q experiments/calibration tests/calibration
& 'D:\anaconda3\envs\torch\python.exe' -m unittest discover -s tests/calibration -v
& 'D:\anaconda3\envs\torch\python.exe' -m experiments.calibration.runner --stage 0 --all --output-dir outputs/calibration/stage0/stage0_thirdpass_20261029_final
& 'D:\anaconda3\envs\torch\python.exe' -m experiments.calibration.runner --stage 0 --verify-only --output-dir outputs/calibration/stage0/stage0_thirdpass_20261029_final
```

The suite covers causal disclosure, query tape label-swap invariance,
controller/memory trajectory isolation, evidence versioning and LSO,
checkpoint/resume and RNG equivalence, Greedy capacity logic, B7 confidence
endpoints and conflicts, Real-IAD selection/tapes, runner checksum corruption
and provenance mismatch, and honest optional-smoke skip behavior. The runner
also executes the CUDA native score/update smoke and metadata preflight when
prerequisites exist. The final run artifact is saved locally at
`outputs/calibration/stage0/stage0_thirdpass_20261029_final/`; its checksums,
exact test count/output, source hashes, metadata-only preflight, native parity
record, and gate matrix provide the run-level evidence. The generated runner
directory is not committed; the text implementation and provenance reports
carry the reviewable results.

Final recorded unit suite: **51 passed, 0 failed, 0 errors, 0 skipped**.
Compilation passed. The runner reports `G1=PASS`, all four development
metadata classes `PASS`, `G2=PASS`, `G3=PASS`, `G4=PASS`, `G5=PASS`, and
`G8=PASS`; exact statuses and source/config checks are in the run artifact.

Metadata-only test-stream results are: audiojack 3,815 images / 763 official
groups / 114 Q-A-TS queries; regulator 3,855 / 771 / 116; switch 3,820 / 764 /
115; rolled_strip_base 3,805 / 761 / 114. Q-B produced the registered 23 early
queries per class. All Q-A-TS 16-bin realized counts match their registered
quotas; each query tape is label blind and one-view-per-official-group.

The original manual CUDA smoke on this checkout returned exact live/clone and
adapter score agreement (`0.0` absolute difference), and exact fp16 Greedy
update equality on the one dynamic layer, with 8 appended, 0 replaced, and 0
skipped rows. The reusable runner repeats this smoke and stores its complete
versioned record. This diagnostic append branch does not independently verify
the saturated replacement/skip path on the native bank. Two fresh CUDA smoke
runs with P0's math-only SDPA mode produced identical support/C64/update-group
hashes, fixed-bank digest, checkpoint hash, and zero score deltas; both updates
were bit-exact. The cached DINOv2 Hub directory lacks Git commit metadata, so
the artifact pins its Python source tree by a deterministic SHA-256 instead.
The final runner smoke records one native dynamic layer at K=256, exact live /
clone and adapter score deltas of zero, and exact append of eight candidate
rows. It records the checkpoint and Hub Python-tree SHA-256 values and does not
emit absolute image scores.

## Stage 0 gates

| Gate | Expected status | Evidence / blocker |
|---|---|---|
| G0 branch and protected worktree | `BLOCKED` | 36 unrelated tracked archive deletions pre-existed; left untouched and excluded from commit. |
| G1 synthetic query identities | `PASS` | Q-A-TS/Q-B determinism, 16 bins, label blindness, exact budgets, and one query per official group. |
| G1 Real-IAD metadata | `PASS` | Four development-class JSON indexes passed official-group separation, one-view query tapes, 16-bin Q-A-TS, exact budgets, and Q-B split checks. |
| G2 native scorer bridge | `PASS` bounded | Audiojack training-only native v22, untouched clone, and calibration adapter score differences are all zero. |
| G3 native Greedy update | `PASS` bounded | Native K=256 append update matches shared Greedy exactly on the dynamic layer; fixed support digest is unchanged. Replacement/skip remains unverified natively. |
| G4 causal firewall | `PASS` | Synthetic predict-before-disclosure and queried-only feedback tests. |
| G5 calibration evidence | `PASS` | Synthetic C64 immutability, version invalidation, LSO, and source attribution tests. |
| G6 all controller families | `BLOCKED_FIDELITY` | B5a/B5b/B6 remain incomplete; B7 is explicitly provisional. |
| G7 external reference math | `BLOCKED` | Baseline reference tests do not cover every required external family. |
| G8 execution/reproducibility | `PASS` | Synthetic resume equivalence, RNG restore, source/profile mismatch and artifact corruption checks. |
| G9 shared scientific trajectory | `BLOCKED` | No full native sequence/AUROC invariance across all controllers was run. |

Overall status stays `BLOCKED_WIP`: G0 is `BLOCKED`, G6 is
`BLOCKED_FIDELITY`, G7 and G9 are `BLOCKED`. Independent research review is
still required.

## Explicitly not run

- No Stage 1 or Stage 2 experiments.
- No Real-IAD test stream scoring, threshold tuning, or metric optimization.
- No Mint/PCB confirmation metadata, images, labels, outcomes, or hidden class
  histograms.
- No confirmation of empirical superiority or risk guarantees.

## Remaining blockers and remediation

1. Resolve the 36 pre-existing protected archive deletions before G0 can pass.
2. Verify native saturated K=256 replacement/skip behavior and end-to-end
   source attribution; current GPU smoke covers the exact append branch only.
3. Complete pinned source or independent equation tests for SPOT and dSPOT,
   including Grimshaw, update/exclusion, and initialization semantics.
4. Complete Sun-CS reference math, source-consistent startup/reset behavior,
   observation compatibility, and predeclared abstention accounting.
5. Review B7's versioned provisional minima, bounds, constraints, movement
   objective, and conflict fallback; these choices are not settled by software
   tests.
6. Run full shared native score/state trajectories across all controllers only
   after the methodological gates are accepted. G9 remains blocked.
