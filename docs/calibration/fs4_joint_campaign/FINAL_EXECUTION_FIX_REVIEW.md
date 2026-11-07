# FS4 final campaign execution and audit correction

**Task:** `FS4_FINAL_FREEZE_EXECUTION_FIX_V1`
**Branch:** `calibration-stage0-contract-20261029`
**Expected starting HEAD:** `50cd629ecf906adba3921c3386bdc26c08d81115`
**Final source-fix HEAD:** `e722ca5d16cd1253d44d456d0ea3e6aaf48f86bb`
**Earlier source-fix commit:** `a2de4bb05cbf3b2f35340d516eb6ffb2e4c08fd4`
**Frozen campaign specification SHA-256:** `1c96077bfdaa58f9ff93a67a9349a4cc6bc8b63578ae8915ff23762b50aee896`
**Scientific execution authorized:** No
**Disposition:** Ready for independent review; this is not authorization to stage real data or run the campaign.

## Scope and commit record

The starting branch and exact starting HEAD matched the task. Work stayed on the requested branch. The frozen campaign specification, arm registry, scientific estimands, execution population, query policy, and native algorithms were not changed.

The principal source-fix commit is `a2de4bb05cbf3b2f35340d516eb6ffb2e4c08fd4`. Its post-commit regression run exposed a worker class-placement defect: `ResearchWorkerTimeout` had interrupted the `BlindedResearchContext` class body, so `_encode_event` was attached to the exception class. The blinded worker then fell back to the path-based research encoder. A narrow corrective source commit, `e722ca5d16cd1253d44d456d0ea3e6aaf48f86bb`, restored the method to `BlindedResearchContext`; the spawned R4 recovery test and full CPU regression were rerun on that final source commit. The final source commit is the source freeze for this review. A separate report-only commit follows it; no source was changed after `e722ca5`.

The source changes are confined to the following five files:

- `experiments/calibration/fs4_joint_campaign.py`
- `experiments/calibration/fs4_joint_campaign_evaluator.py`
- `experiments/calibration/fs4_r4_worker.py`
- `experiments/calibration/fs4_research_stage_a_verify.py`
- `tests/calibration/test_fs4_joint_campaign.py`

This review is the sole file in the separate report-only commit. No push, reset, clean, stash, or history rewrite was performed.

## Accepted threshold-update accounting

The evaluator now treats the controller's `revision` as evidence revision only. It separately counts an accepted estimator update when the sealed post-event `threshold_tokens` value for G or GC changes to a new non-null token. This counts a successfully accepted update even if the numeric threshold is unchanged. Evidence additions or unchanged tokens do not count as accepted updates.

Per controller and scorer, the report retains evidence revision, accepted G and GC counts, earliest accepted positions, distinct events with any accepted update, hold counts by reason, and total numerical threshold movement. A single event that accepts both scorers contributes two scorer updates and one event update. Repeated audit states, duplicate positions, and checkpoint replay cannot add another update. Per-category, segment, and sliding-window diagnostics use the same token accounting.

The focused tests cover zero, 11, and 12 evidence revisions with insufficient ESS; G-only and GC-only updates; both scorers accepting at one event; token changes with no numeric threshold movement; new evidence with unchanged tokens; repeated audit state; replay from a checkpoint state; hold-reason deltas; and segment/window boundary counts. The evaluator and tests do not alter L/E update mathematics.

## Timing enforcement and worker supervision

The append-only `TIMING_LEDGER.jsonl` is bound to the campaign ID, frozen specification identity, and source-freeze commit. Its hash chain records campaign and trajectory origins, per-process wall and monotonic timestamps, cumulative elapsed totals, process boundaries, active trajectory, deadline state, and resume history. Monotonic deltas account for time while a process runs. A persisted wall-clock gap after a restart is conservatively charged to the campaign and the previously active trajectory; a backward or inconsistent clock, broken chain, altered total, or changed binding fails closed. The frozen seven-day trajectory and 21-day campaign limits remain unchanged.

The executor checks the timing ledger before worker launch and event dispatch, after each event, before checkpoints, and before category or campaign completion. Exceeded limits record `TRAJECTORY_WALL_LIMIT_EXCEEDED` or `CAMPAIGN_WALL_LIMIT_EXCEEDED`; neither can produce a Stage A campaign seal. Worker and broker pipe waits accept the remaining budget, and timeout or bounded-shutdown failure terminates the processes. A timeout while label intent may be active is classified as ambiguous disclosure and is not recoverable by supplying `--resume`.

## Recovery and terminal-state rules

The first `INCOMPLETE.json` marker is preserved. Failure records are appended to `INCOMPLETE_FAILURE_HISTORY.jsonl`; recovery does not overwrite or erase previous evidence. Failures are classified as recoverable execution interruption, fatal integrity failure, ambiguous label disclosure, or deadline exhaustion. Only a recoverable interruption can proceed to verified recovery.

A successful recovered campaign has a separate `RECOVERY_COMPLETION.json` receipt bound to campaign ID, specification SHA, source commit, original incomplete-marker digest, selected checkpoint identity, replayed ledger-prefix identity, verified complete Stage A ledger identities, final S/CL/CE seals, recovery classification, and receipt digest. The receipt and timing history are cross-checked against the terminal seal. Stage A verification and Stage B admission reject missing, forged, mismatched, or inconsistent terminal evidence. Integrity failure, ambiguous disclosure, and expired timing budgets are not made recoverable by a resume request.

## Integrated synthetic campaign

`test_integrated_s_cl_ce_campaign_completion_and_checkpoint_recovery` exercises the actual integrated launcher and verification interfaces with synthetic workers. It runs the frozen four-category population shape and checks the common event/group registration across S, CL, and CE; S's shared score/query/feedback/memory path and its separate L/E/FrozenTheta readouts; independent CL/CE memories, checkpoint and disclosure namespaces; query isolation; four category transitions and twelve seals; rejection of Stage B for incomplete execution; metric-report production; and recovery replay without duplicate query or feedback. It also checks missing/forged receipts, ambiguous disclosure, fatal failure, and deadline-exhaustion behavior. All data are synthetic; no Real-IAD image, label, mask, historical score, or C64 feature was read.

## Verification matrix

| Check | Result |
|---|---|
| `python -m compileall -q experiments/calibration` | PASS |
| `python -m py_compile` for the four modified modules and campaign test | PASS |
| Focused timing/recovery/timeout/integrated tests (`-k 'campaign_timing_ledger or fatal_ambiguous or blinded_worker_timeout or integrated_s_cl_ce'`) | PASS; 8 passed, 15 deselected, 83.03 s on final source commit |
| Spawned R4 worker/broker recovery test after encoder-binding correction | PASS; 1 passed, 133.41 s |
| Combined CPU regression: campaign, R4, research execution, threshold, and R2 hardening files | 90 passed; 1 failed; 4 Pillow deprecation warnings; 564.67 s |
| CUDA hardening (`test_fs4_hardening_cuda.py` and `test_fs4_hardening_r2_cuda.py`) | PASS; 6 passed, 4 Pillow deprecation warnings; 84.78 s |
| `spec_identity()` independent check | PASS; `1c96077bfdaa58f9ff93a67a9349a4cc6bc8b63578ae8915ff23762b50aee896` |
| `git diff --check` from the expected starting HEAD | PASS |
| Actual Real-IAD preparation, Support reconstruction, resource preflight, Stage A, or Stage B | NOT RUN; not authorized |

The single CPU failure is the required, unmodified adversarial-production regression `test_production_rejects_preconstruction_encoder_substitution`: it expected `FS4BoundaryError`, but none was raised. It remains a known high-security failure and is not presented as passed. An earlier full CPU attempt also found the worker encoder-binding regression described above; the corrective source commit fixed it, and the final full CPU run passed the R4 recovery test. The protected encoder-substitution test was not edited or weakened.

CUDA was available in the configured Torch environment, so both synthetic CUDA files were run. The CUDA checks establish numerical hardening behavior only; they are not a campaign resource benchmark.

## Scientific and source protection audit

The normalized spec identity was computed through the repository's `spec_identity()` implementation and matches the frozen SHA above. `CAMPAIGN_SPEC.json` and `ARM_REGISTRY.json` are byte-for-byte unchanged from the starting revision. Comparing the expected starting HEAD with the final source-fix HEAD shows only the five listed calibration implementation/test files changed. No historical v22, vmb, V0.2, P0/P1, R0/R1, D1-D4 sources or archived outputs were modified. The frozen native VisionAD, Greedy, C1, L/E, and stochastic BEA algorithms were not edited.

Before adding this report, the existing untracked-file inventory matched the saved initial inventory exactly: 7,030 entries, with no added or missing unrelated paths. No pre-existing outputs or untracked files were staged or deleted. Tracked status was clean at the final source-fix commit before this report was added.

## Remaining activation gates

Independent review remains necessary, followed by the separately authorized real-data gates already registered in the freeze: source/category/group/view/count validation; R7 Seed-0 order and one-time blinded mapping verification; four-Support native bank, score, theta, and Writer-sigma parity; synthetic native resource benchmarking and intended-host resource preflight; and separate authorization for Stage A and Stage B. No empirical conclusion or scientific campaign result exists.

FS4_FINAL_EXECUTION_FIX_READY_FOR_INDEPENDENT_REVIEW
