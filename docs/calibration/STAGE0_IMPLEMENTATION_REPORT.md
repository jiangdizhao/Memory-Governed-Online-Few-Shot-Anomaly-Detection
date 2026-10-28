# Stage 0 implementation report

**Status: `BLOCKED_WIP` — not scientifically validated and not ready for Stage 1.**

This report records the Stage 0 infrastructure implementation on
`calibration-stage0-contract-20261029`. The run uses synthetic CPU fixtures only.
No Real-IAD image, feature, mask, or confirmation-class outcome was opened; no
Stage 1 or Stage 2 experiment was run. Passing unit tests establish software
mechanisms only, not empirical or scientific validity.

## Branch and source audit

- Working branch was already `calibration-stage0-contract-20261029`; origin and
  local HEAD were synchronized at source parent
  `0b349ce67ce5ff635f35958eda4d0a3717fdfde8` before implementation.
- The required historical sources were inspected read-only: `vmb_visionad_new.py`,
  `test_phase3_AIF_simplified_v22.py`, `experiments/p0_protocol_audit.py`,
  `experiments/a0_a1_strict4_matched.py`, `experiments/v02_memory.py`,
  `experiments/risk_safe/native_bridge.py`, `preflight.py`, `query_protocol.py`,
  `d2_threshold_calibration.py`, and `p01_falsification_native.py`.
- The native score adapter routes through `native_bridge.native_score` and the
  existing `a0_a1_strict4_matched.score_bank` → v22 `score_sample_multiview`
  path. It provides no fallback or nearest-neighbor scorer. The mutable writer
  reuses `extract_v22_candidates`, full-feature-matrix normalization,
  `select_informative_normal_write`, `GreedyBank.proposed`, and the existing
  `v02_memory.greedy_slot` operator.
- P0 historically builds a dynamic bank at K=1024. The new isolated adapter
  deep-copies the bank, requires an empty dynamic bank, changes only dynamic
  codebook capacities to K=256, and compares fixed-support prototype hashes.
  That production adapter has not been exercised against an actual VisionAD
  bank in this run.
- The repository handoff refers to a separate internal Deep Research proposal
  for SCDAC-G, but that report is not present in the workspace. The committed
  contracts provide only an outline, so the exact B7 bound and fallback rules
  remain unresolved.
- At the start of work, Git reported tracked deletions in the protected
  `outputs/risk_safe_sprint_20261027/` archive tree, with access-denied/missing
  directory warnings. These unrelated entries were not read, restored, staged,
  or modified. This blocks G0 pending independent worktree cleanup/review.

## Implemented package

New source is confined to `experiments/calibration/`; new tests are in
`tests/calibration/`.

- `profiles.py` locks Stage 0 to CPU, four deployed supports plus 64 additional
  protected C64 normals, K=256, native P0 scorer parameters, Q-A-TS/Q-B
  parameters, fixed reference theta policy, and stable config hashes.
- `protocol.py` separates trusted tape records, controller-visible observations,
  queried-only feedback, predictions, and versioned evidence. Controllers do
  not receive sample keys, paths, images, masks, or labels except the explicit
  `Feedback` object after a query.
- `query_tape.py` builds and hashes the new Q-A-TS and Q-B schedules. Q-A-TS
  budgets `floor(.03N + .5)` images across 16 fixed temporal bins using
  deterministic largest-remainder quotas proportional to eligible group
  availability. Q-B assigns `floor(.20B + .5)` queries to the first 400 image
  positions. Both select at most one view per official sample key and fail
  closed when the group budget is infeasible. Priority hashes use only
  position/camera metadata; group keys are used for grouping and audit, not
  query priority. The archived Q-A implementation is not imported.
- `native.py` validates the inherited P0 config, builds the K=256 isolated
  native clone, computes native score-state hashes, provides one common
  reference writer, and rejects protected C64 keys. The writer takes fixed
  `theta_ref`/`sigma_ref`, never a controller threshold. It mirrors append,
  replacement, and skip source attribution and checks its staged rows against
  the native Greedy proposal before installing a changed state.
- `anchor_ledger.py` requires exactly four distinct deployed support identities
  and 64 disjoint additional C64 identities. C64 is read-only. Query-verified
  normal anchors are rescored leave-source-out across all layers; defect
  anchors never enter the normal bank. Stale evidence snapshots are rejected.
- `engine.py` provides one score call per event, pure predict-before-feedback
  checks, common query feedback, a single injected memory writer, versioned
  evidence refresh, and an evaluator boundary. It enforces at most one content
  version change per event and checks digest/version consistency.
- `metrics.py` keeps outcomes in an evaluator-only ledger. It reports exact
  whole-stream and 256-image window confusion counts, query-evidence timing,
  shared AUROC/AUPRC, and error streaks. Missing classes and abstentions remain
  unavailable rather than being mapped to zero risk.
- `plugins.py` defines feature, score, memory, query, correction, and future BEA
  interfaces. Stage 0 activates correction `OFF`; no BEA policy is activated.
- `controllers.py` implements B0, B1, B3, B4, and an adapter over the existing
  v22 B2 sidecar. B1 defaults are frozen at W=256 and n_min=16. B3 uses
  protected C64 higher-q90 only, delta=0.01, and updates once per memory
  version. B4 uses queried-label-only class-conditional ACI updates with
  alpha/beta=.10 and gamma=.005; it is an adaptation and claims no ACI theorem.
- `method_registry.py` contains machine-readable cards for B0–B7, with B5a and
  B5b separately listed. Unresolved methods raise `BLOCKED_FIDELITY`; they do
  not emit substitute thresholds.
- `runner.py` implements `--stage 0 --all`, `--verify-only`, and `--resume`.
  Outputs are stable-keyed, non-overwriting, checksummed, provenance-checked,
  CPU-only synthetic smoke artifacts with an `INCOMPLETE` marker.

## Method fidelity records

| ID | Status | Definition and boundary |
|---|---|---|
| B0 | `IMPLEMENTED` | Frozen `theta0`; D2 higher q90 from 64 C64 normals under M0. The implementation validates 64 finite scores and selects index 57 (zero based). |
| B1 | `IMPLEMENTED` | Rolling q90 of queried, verified normal scores at arrival; W=256, n_min=16; hold at theta0 until eligible count is met. It is the explicitly stale-score variant. |
| B2 | `IMPLEMENTED_ADAPTED` | Calls the unchanged v22 `GuardedDualAnchorThresholdController`; disables its unlabeled-normal fallback because this protocol has no such label feedback. It initializes from the additional C64 scalar scores using the source's NumPy quantile convention, so its initial theta may differ from D2 theta0. Fast and balanced rescue remain at source defaults disabled. This is a threshold-sidecar adapter, not a full native-loop parity claim. |
| B3 | `IMPLEMENTED_PROPOSED` | Protected C64 current-M q90, higher order statistic, minimum 16 normals, one bounded delta=.01 move after an actual memory change. FNR is explicitly unavailable without queried defect anchors. Queried-normal LSO is not added to this primary arm. |
| B4 | `IMPLEMENTED_ADAPTED` | Class-conditional ACI-style updates only on queried labels, then current-M normal/defect quantile projection. `gamma=.005`, alpha/beta=.10 are registered implementation defaults, not paper optima. Sparse selective feedback and dynamic-memory dependence void inherited guarantee claims. |
| B5a | `BLOCKED_FIDELITY` | SPOT threshold equation and paper parameters are recorded, but no source-equivalent Grimshaw implementation plus pinned author-reference trace is in place. No different EVT estimator is passed off as SPOT. |
| B5b | `BLOCKED_FIDELITY` | dSPOT remains separate from SPOT; the original drift/update convention is not implemented and reference-tested here. No rolling-quantile surrogate is used. |
| B6 | `BLOCKED_FIDELITY` | Sun et al. source requires a faithful confidence-sequence reference test and an explicit primary-metric policy for its abstentions. These are not complete; no generic interval substitute is labeled Sun-CS. |
| B7 | `BLOCKED_FIDELITY` | The full internal SCDAC-G proposal is missing. Exact descriptive bound, minimum evidence, conflict fallback, and confidence rules are not invented. |

Primary reference provenance is recorded in `method_registry.py` and
`controller_method_cards.json`: ACI (Gibbs & Candès, NeurIPS 2021), SPOT/dSPOT
(Siffer et al., KDD 2017), Sun-CS (Sun et al., ICML 2024), and the executable
v22 sidecar source. File SHA-256 values for the inspected historical files and
new Python sources are in the runner's `source_hashes.json`.

## Validation run

Environment: `D:\anaconda3\envs\torch\python.exe`, Python 3.11, PyTorch
2.0.1+cu117. CUDA is available in this environment, but every Stage 0
validation command below ran in CPU mode. No package was installed and no
environment/dependency was changed. Pytest is absent; tests use standard-library
`unittest`.

Commands executed:

```powershell
& 'D:\anaconda3\envs\torch\python.exe' -m compileall -q experiments/calibration tests/calibration
& 'D:\anaconda3\envs\torch\python.exe' -m unittest discover -s tests/calibration -v
& 'D:\anaconda3\envs\torch\python.exe' -m experiments.calibration.runner --stage 0 --verify-only
& 'D:\anaconda3\envs\torch\python.exe' -m experiments.calibration.runner --stage 0 --all
& 'D:\anaconda3\envs\torch\python.exe' -m experiments.calibration.runner --stage 0 --all --resume
```

The final unit suite contains 25 tests, including label-blind key-text
selection, Q-A-TS/Q-B budgets, infeasible group rejection, predict-before-label
engine behavior, unqueried-label-swap invariance, B1/B2/B3/B4 update rules,
theta0 rank, support/C64 disjointness, protected C64 write rejection, current
version/LSO checks, P0 config locks, abstention/missing-class metrics, and exact
source Greedy proposal behavior.
The synthetic runner event trace is mechanism evidence only.

## Gate matrix

| Gate | Status | Evidence or blocker |
|---|---|---|
| G0 branch/protected tree | `BLOCKED` | Correct branch; protected tracked archive deletions predate this implementation and remain untouched. |
| G1 query identities | `PASS` synthetic / `SKIP` real preflight | Q-A-TS/Q-B synthetic tape and group constraints passed. Real-IAD path presence was checked without enumerating or opening data; the data preflight is not a pass. |
| G2 native scorer parity | `BLOCKED` | Native adapter uses source bridge, but direct/shadow VisionAD scores were not run on image/features. |
| G3 native Greedy parity | `BLOCKED` | Synthetic Greedy and source-map tests pass; native K=256/fp16 update parity was not run. |
| G4 causal firewall | `PASS` synthetic | Predictions are pure before feedback; no unqueried label is delivered; swapping only unqueried labels leaves controller/memory trajectory invariant. |
| G5 evidence ledger | `PASS` synthetic | C64 separation/protection, version checks, and all-layer leave-source-out source-map behavior tested. Native source attribution parity remains under G3. |
| G6 method families | `BLOCKED_FIDELITY` | B5a/B5b, B6, and B7 unresolved as detailed above. |
| G7 reference math | `BLOCKED` | B0–B4 reference cases pass; unresolved baselines prevent all-family acceptance. |
| G8 execution/resume | `BLOCKED` | Runner outputs are non-overwriting, checksummed, and source/config verified on resume. Full mid-stream engine/controller/anchor/memory checkpoint restore is not implemented. |
| G9 scientific parity | `BLOCKED` | No native shared-score trajectory or AUROC invariance run; only synthetic one-score-per-event behavior was checked. |

The final generated synthetic artifact directory for this source snapshot is
`outputs/calibration/stage0/stage0_8f10119808986684/`. It contains
`protocol_manifest.json`, `dependency_environment.json`,
`gates.json`, `controller_method_cards.json`, `synthetic_event_trace.jsonl`,
`synthetic_summary.json`, `test_results.json`, `source_hashes.json`,
`checksums.json`, and `stage0_summary.md`. Its `INCOMPLETE` marker is deliberate:
it is not Real-IAD evidence and does not close the native or baseline gates.

## Remaining blockers

1. Review/resolve the pre-existing protected archive deletions before G0 can
   pass. They were excluded from this change and from staging.
2. Run permitted non-scientific native CPU parity with real support/C64 features
   when an approved cache/source is available; validate direct/shadow score,
   fixed-bank immutability, exact K=256 Greedy writes, fp16 normalization, and
   all-layer source attribution. This run did not read dataset files.
3. Implement and source-reference-test faithful SPOT and dSPOT, including the
   paper's tail fit and dSPOT update semantics; do not weaken either baseline.
4. Complete Sun-CS source-equation/reference tests and settle its abstention
   metric handling before registering an executable comparator.
5. Supply the authoritative SCDAC-G proposal/version so its bound, minimum
   evidence, fallback, and confidence semantics can be implemented verbatim.
6. Add state-complete engine, memory, anchor and controller checkpoint restore;
   current `--resume` verifies completed synthetic artifacts only.

No empirical superiority claim is made. Stage 0 remains a blocked engineering
milestone for independent research review.
