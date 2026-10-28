# Stage 0 automated implementation, Windows environment and acceptance gates

**All commands in this document describe target behavior; the Stage 0 Runner does not exist at contract-branch creation.** Do not report command success until actually executed and logged.

## A. Exact local environment (user-confirmed 2026-10-29)

- Host: Windows, PowerShell
- Existing Conda environment NAME: **torch**
- Existing environment LOCATION: **D:\anaconda3\envs\torch**
- Preferred interpreter: **D:\anaconda3\envs\torch\python.exe**
- Dataset root (historical, **confirm existence locally**): **G:\dataset\realiad**
- JSON subdirectory (historical, confirm): **G:\dataset\realiad\jsons\realiad_jsons**
- GPU may be an RTX 3090 Ti; test capability locally; **never assume it is available**.
- Do NOT create another Conda/venv environment; do NOT install/upgrade/downgrade PyTorch, CUDA, flash-attention, torchvision, scipy or other packages without user approval. Stage 0 must normally run CPU-only tests; GPU smoke only when specifically enabled and read-only/non-scientific.

Reliable PowerShell preflight from repo root:
```powershell
$ErrorActionPreference = "Stop"
$PY = "D:\anaconda3\envs\torch\python.exe"
if (-not (Test-Path -LiteralPath $PY)) { throw "torch conda interpreter missing: $PY" }
& $PY -c "import sys, torch; print('exe=',sys.executable); print('torch=',torch.__version__); print('cuda_available=',torch.cuda.is_available()); assert sys.prefix.lower() == r'D:\anaconda3\envs\torch'.lower()"
if ($LASTEXITCODE -ne 0) { throw "Incorrect Python environment" }
git status --short
git branch --show-current
```
Alternative for interactive shells: `conda activate torch`; always independently verify `sys.executable`. A Codex sub-process may not inherit interactive conda activation; **prefer the absolute interpreter path**.

## B. One-command target interface

```powershell
& $PY -m experiments.calibration.runner --stage 0 --all
& $PY -m experiments.calibration.runner --stage 0 --verify-only
```

`--all` performs deterministic preflight, imports, tests, pure synthetic mechanisms, optional **explicitly flagged** native parity checks and report generation. It MUST NOT run Stage 1 or Stage 2. `--verify-only` must verify manifests/checksums without recomputing empirical research results. Support `--resume`, `--output-dir`, and JSON machine-readable audit; set stage defaults only in versioned immutable Config.

Prefer a locally implemented `scripts/run_stage0.ps1` thin wrapper using exact $PY; shell should exit nonzero on critical gates and not continue after a failed dependent step.

### Mandatory reproducibility artifacts (under new isolated outputs)
```
outputs/calibration/stage0/<run_key>/
    protocol_manifest.json          # dataset identity, query version, method registry, hashes
    dependency_environment.json     # Python/torch/CUDA mode and exact executable
    gates.json                      # PASS/FAIL/SKIP/BLOCKED with command/traceback
    controller_method_cards.json    # fidelity for all B0..B7, B5a/B5b
    synthetic_event_trace.jsonl
    test_results.json               # all executed test names, duration and status
    source_hashes.json
    checksums.json
    stage0_summary.md
```
Output paths must never overwrite preexisting complete runs. Interrupted runs are `INCOMPLETE`; verify sha digests and resume with source+config matching. A synthetic audit is not Real-IAD evidence.

## C. Six internal milestones — one autonomous execution, not six human approvals

1. **M0 protocol**: immutable config, unit of query, access permissions, C64/Strict4 distinction, Q-A-TS vs archived Q-A, fixed method registry; all config validated before algorithm calls.
2. **M1 native adapter**: import existing scorer, normalize features faithfully, exact K=256 Greedy source/update; parity tests and no source-file modification. If CPU native parity impossible without local data, mark `SKIP_DEPENDENCY`, never `PASS`.
3. **M2 composable causal engine**: single score trajectory / memory version; label firewall; Q-A-TS/Q-B precommitted tape; evaluator isolated. Future BEA/correction plugin slots but no speculative implementations.
4. **M3 method families**: B0, B1, B2, B3, B4, B5a, B5b, B6, B7. Each must have math, source, availability, hyperparameters, fidelity card and reference tests. A smoke-only stub is `NOT_IMPLEMENTED`.
5. **M4 mechanism tests**: synthetic traces proving state transitions, missing-side, risk conflict, LSO source exclusions, update order, deterministic restart. Failure MUST be conspicuous.
6. **M5 final gates/report**: test matrix, acceptance proof, setup docs; no real Real-IAD ranking, no automatic Stage 1 execution.

## D. Test matrix and required gates

| Gate | Required invariant | Expected status when unverified |
|---|---|---|
| G0 branch/protocol | work only in calibration branch; historical files unchanged; config hash and support/C64 disclosures exist | FAIL |
| G1 dataset identities | partitions and 1 query/view/official sample group, 16-bin Q-A-TS and Q-B, deterministic blind tape | PASS synthetic; real data preflight SKIP if unavailable |
| G2 native scorer | unchanged v22 native source, shadow vs direct numeric parity tolerance recorded; original vs new clone immutable | FAIL/BLOCKED if parity source unavailable |
| G3 Greedy | original append/replace operator, per-layer cap, source attribution, controller-independent write; no source tensor mutation | FAIL |
| G4 causal firewall | predict BEFORE feedback, any not-queried label forbidden, full stream prequential, offline outcome inaccessible | FAIL |
| G5 ledger | version invalidation, protected C64 no writer access, exact LSO or explicit BLOCKED augmented path | FAIL |
| G6 controller | all method families correct API plus faithful/adapted declarations, no unknown method disguised as pass | BLOCKED_FIDELITY |
| G7 reference math | exact comparator formulas, empirical risk edge cases and known synthetic traces, monotonic tests | FAIL |
| G8 execution | checkpoint/resume and fail-closed non-overwrite, deterministic source hashes and logging | FAIL |
| G9 scientific parity | identical native score sequence and AUROC across threshold controllers; BA differences only from theta trajectories | FAIL |

Do not allow `SKIP` to satisfy a required `PASS`. Native parity with actual DINO features may require later local non-scientific smoke: until executed, overall stage readiness is `PROVISIONAL`, not `STAGE0_PASS`. If original Sun-CS source fidelity cannot be fully verified, register `BLOCKED_FIDELITY`, not a fabricated method.

## E. Synthetic reference scenarios

At least these deterministic tests:
1. two methods with strongly different theta trajectories produce identical score vectors, query tape, memory hashes and AUROC/AUPRC;
2. identical stream with one queried normal verifies exactly one legal shared normal update, a queried defect verifies zero normal writes;
3. unqueried defect label hidden in evaluator cannot affect any controller or query tape even if changed to normal in test;
4. same official sample with 5 views receives <=1 query; label-blind tapes invariant to permuting hidden labels; 16-bin budget and early Q-B counts validated;
5. C64 normal source is never written, even when protected normal's score is extreme; all 64 references refreshed on actual M change;
6. version-stale anchor explicitly rejected; LSO excludes all rows attributable to the anchor's official sample across all layers, including after replacement and on resume; no match silent shortcuts;
7. empty normal/defect sets preserve UNAVAILABLE flags, normal-only and defect-only states, BOTH, CONFLICT, exact score ties and step-clipping; confidence interval is never mislabelled as anytime-valid;
8. changed proposed controller threshold cannot influence query selection, native memory update candidate count, feature selection, or scorer;
9. exactly reloaded Controller, memory, ledger, RNG and input cursor from checkpoint produces bitwise identical synthetic outputs;
10. if missing source/invalid configuration/insufficient SPOT tail initialization, code fails with machine-readable BLOCKED reason instead of returning good-looking dummy thresholds;
11. native direct/shadow source score and Greedy parity tests exercising fp16 normalization and capacity K256 vs source default K1024 where torch-native dependencies exist;
12. B4 class-conditional ACI updates only on queried labels; SPOT/dSPOT use unlabeled scores legally and do not require selective labels.

## F. Machine-reviewable completion contract

Final message / report must include:
```json
{
  "stage": 0,
  "branch": "calibration-stage0-contract-20261029",
  "source_commit": "<sha>",
  "overall_status": "PASS | PROVISIONAL | BLOCKED | FAIL",
  "gates": {"G0": "PASS", "G1": "..."},
  "method_statuses": {"B0": "IMPLEMENTED", "B5a": "IMPLEMENTED", "B6": "BLOCKED_FIDELITY"},
  "tests": {"passed": 0, "failed": 0, "skipped": 0},
  "native_parity_verified": false,
  "stage1_run": false,
  "changed_original_files": [],
  "results_path": "outputs/calibration/stage0/...",
  "blockers": []
}
```
Use actual values, never copy the example as fake results.

## G. Branch/commit/push boundaries

Codex may push only commits on `calibration-stage0-contract-20261029`, no force push, no merging to default/old branches, no destructive reset/clean, no tags or release. It must check `git status --porcelain`, never stage unrelated dirty files. If pending independent changes exist, stop rather than discard. Commit source/tests/docs plus tiny synthetic text audit as authorized; do not stage datasets, trained weights, protected probe data, absolute local paths inside private artifact manifests unless essential for provenance, secrets or bulk GPU results.

If everything verified and required gates PASS, commit e.g. `feat(calibration): implement stage0 isolated harness and fidelity tests`, then `git push -u origin calibration-stage0-contract-20261029`. If BLOCKED, it MAY push honest WIP implementation with `wip(calibration): stage0 blocked - <reason>` and blockers manifest; the user needs only to report that it pushed for independent review. If `git push` needs authentication, stop and report; never modify credentials or grant itself new access.

## H. Human-in-the-loop boundary

The executor may autonomously edit, test, repair and rerun **Stage 0 implementation** until specified gates pass, including routine dependency-free unit tests. It cannot change statistical hypotheses, initialize Real-IAD GPU performance runs, inspect confirmation labels, invent numerical guarantees, conduct Stage 1/2 or change original v22. Methodological ambiguity -> precise blocked report and a review request. We seek **one Stage 0 review checkpoint**, not micro-approvals after each file.
