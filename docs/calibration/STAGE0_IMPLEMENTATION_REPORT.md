# Stage 0 implementation and third-pass audit report

**Status: `BLOCKED_WIP`.** The infrastructure and permitted parity mechanisms
have executable evidence, but G0, G6, G7, and G9 remain open. This is not
scientific validation and does not authorize Stage 1.

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
