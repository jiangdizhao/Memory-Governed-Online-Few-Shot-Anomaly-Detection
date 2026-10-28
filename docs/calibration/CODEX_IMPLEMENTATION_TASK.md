# Codex implementation order — Stage 0 (self-contained task, 2026-10-29)

## Mission

You are the **engineering executor** of a PhD-level industrial visual-anomaly-detection calibration experiment. Implement **Stage 0 only** from the already committed research specifications, run all allowed local tests, write auditable evidence, **commit and push your work to the same Git branch autonomously**. Do not ask for user confirmation after every substep; use strict automatic failure handling. If a scientific decision cannot be determined from authoritative docs/source code/papers, mark the affected gate BLOCKED and report instead of fabricating a solution.

Repository: https://github.com/jiangdizhao/Memory-Governed-Online-Few-Shot-Anomaly-Detection
Branch: **calibration-stage0-contract-20261029** (already created from p0p1-falsification-20261028 on 2026-10-29).
Never merge / edit other branches.

### Required first action

From local repo root use PowerShell:
\`\`\`powershell
$ErrorActionPreference = "Stop"
git status --short
git fetch origin
git switch calibration-stage0-contract-20261029
git pull --ff-only origin calibration-stage0-contract-20261029
git log -1 --oneline
$PY = "D:\anaconda3\envs\torch\python.exe"
if (-not (Test-Path -LiteralPath $PY)) { throw "Missing EXISTING conda torch Python: $PY" }
& $PY -c "import sys,torch; print(sys.executable); print('torch=',torch.__version__); print('cuda=',torch.cuda.is_available()); assert sys.prefix.lower()==r'D:\anaconda3\envs\torch'.lower()"
if ($LASTEXITCODE -ne 0) { throw "Not running correct torch env" }
\`\`\`
If the working tree has unrelated uncommitted changes, avoid switching or modifying them; stop for safety rather than \`git reset --hard\` or \`git clean\`. Conda env is named **torch** at exactly **D:\anaconda3\envs\torch**. DO NOT create a new env or upgrade PyTorch/CUDA. If shell activation is needed, \`conda activate torch\` is acceptable but verify \`sys.executable\`. Use $PY for all Python tests/Runner commands.

### Mandatory documents to read in full, in this order

1. \`AGENTS.md\`.
2. \`docs/calibration/README.md\`.
3. \`docs/Calibration_Falsification_Research_Handoff_2026-10-28_v1.docx\` (when the tool can read DOCX; otherwise all newer constraints are encoded in docs/calibration).
4. \`docs/calibration/STAGE0_RESEARCH_CONTRACT.md\`.
5. \`docs/calibration/ARCHITECTURE_AND_CAUSAL_PROTOCOL.md\`.
6. \`docs/calibration/BASELINE_FIDELITY_AND_REFERENCES.md\`.
7. \`docs/calibration/AUTOMATION_AND_ACCEPTANCE.md\`.
8. Relevant source references: \`vmb_visionad_new.py\`, \`test_phase3_AIF_simplified_v22.py\`, \`experiments/p0_protocol_audit.py\`, \`experiments/a0_a1_strict4_matched.py\`, \`experiments/v02_memory.py\`, \`experiments/risk_safe/native_bridge.py\`, \`experiments/risk_safe/preflight.py\`, \`experiments/risk_safe/query_protocol.py\`, \`experiments/risk_safe/d2_threshold_calibration.py\`, \`experiments/risk_safe/p01_falsification_native.py\`.
9. Primary literature from \`BASELINE_FIDELITY_AND_REFERENCES.md\`. For external source fidelity, read substantive algorithm/appendix sections, not merely paper abstracts.

## Implement using ONE internally automated workflow

**A. Protocol and composable architecture.** Build an \`experiments/calibration\` package with immutable typed experiment profiles, native scorer/memory wrapper, score state hashes, a versioned anchor ledger, separate offline evaluator, causal prequential engine, and plugin interfaces for frozen/off correction, query tape/BEA, and memory/calibration policies. Implement only \`calibration_isolated\` actual execution; future modes must be extensible but need not run.

**B. Native fidelity.** Reuse source v22/VisionAD code and \`v02_memory.greedy_slot\` through \`risk_safe.native_bridge\`, without touching historical scripts. One shared precommitted reference writer uses queried verified-normal only and is independent of all calibration thresholds. Validate exact append/replacement, per-layer capacity K=256, normalization dtype, immutable fixed bank, state digests and scorer parity. No fresh nearest-neighbor shortcut.

**C. New feedback protocol.** Implement *new named and hashed* \`Q-A-TS\` 16 time-bin label-blind 3%-of-image stream query tape and \`Q-B\` early warmup 20% budget in first 400 observations, with one queried view per official sample group, precommitted before labels. Do NOT silently reuse historical flat Q-A. Implement Q-B and synthetic tests now, do NOT run Stage 1 comparisons. Clearly report B and actual query fraction by images/groups.

**D. Implement all agreed eight families in a single registry.** B0 Frozen; B1 Rolling-q90 verified-normal stale control; B2 native v22 Original Sidecar; B3 SCQ; B4 adapted class-conditional SC-Dual-ACI; B5a SPOT, B5b dSPOT; B6 Sun-CS; B7 SCDAC-G. Give each exact equations, access permissions, initial conditions, defaults or explicitly blocked unresolved choices, method cards and source-fidelity unit tests. Consult the original ACI (NeurIPS 2021), SPOT (KDD 2017), Sun-CS (ICML 2024) and Gibbs/Candès (JMLR 2024) sources listed in docs. Never substitute a homebrew CS and label it Sun. If exact Sun-CS or original Sidecar fidelity cannot be demonstrated, label the method \`BLOCKED_FIDELITY\` with a precise source/math issue, do not fake a pass, and still push the complete remaining honest work as a clearly blocked WIP.

**E. Synthetic tests and anti-leakage.** Full-stream prequential predict-before-label; verify memory/score/AUROC invariance across controller implementations, exact same sample/group/query tape, C64 protected anchors only, source-aware LSO attribution or explicit blocked advanced path, stale score rejection, label-swap invariance of label-blind decisions, no future/current unqueried labels, two-sided evidence state transitions, missing-class no zero risk, normal/defect first activation, ACI queried-only update, SPOT score-only update, numeric boundary ties, K=256 and fp16 parity when dependencies available, resume idempotence. Add tests under \`tests/calibration/\`.

**F. Automation.** Implement \`python -m experiments.calibration.runner --stage 0 --all\` / \`--verify-only\` / \`--resume\`, with dry metadata preflight, deterministic synthetic test runs, JSON reports, checksum validation, accurate PASS/FAIL/BLOCKED/SKIP status and non-overwriting outputs. Detect absent local Real-IAD gracefully; Stage 0 synthetic tests should be able to run without dataset. No Stage 1 or Stage 2 runs and no accuracy-driven tuning. Optional non-scientific native parity smoke is permitted if it does not touch confirmation data and is explicitly labelled as smoke.

**G. Update documentation.** Document exact implementation changes, exact external baseline adaptations, tests, incomplete parity or fidelity, and command examples for the user's existing Windows torch env. Change no historical source or archived result. Add a detailed implementation report to \`docs/calibration/STAGE0_IMPLEMENTATION_REPORT.md\`.

## Automated execution and correctness requirements

- Write all source in \`experiments/calibration/\`, new tests in \`tests/calibration/\`, and new calibration docs only; small new \`scripts/run_stage0.ps1\` is optional.
- Do not modify \`test_phase3_AIF_simplified_v22.py\`, \`vmb_visionad_new.py\`, \`experiments/v02_memory.py\`, \`experiments/risk_safe/*\`, \`experiments/p0_protocol_audit.py\`, original docs/outputs, or existing .github automation. If a dependency bug prevents correct use, document it and fail closed.
- No external dependencies added or environment upgrades without approval. If a source library is absent, write a scoped pure-Python reference when faithful and tested, otherwise report BLOCKED.
- Maximum autonomous repair freedom applies to **implementation bugs**, not changing research methods, gate thresholds, labels or data partitions to obtain passing tests.
- Run \`$PY -m compileall -q experiments/calibration tests/calibration\`, unit tests, and the Stage 0 Runner; include actual stdout/log paths and failure diagnostics. Do not claim a test succeeded if it did not run.
- If an issue requires human scientific choice, continue independent nonblocked modules, then emit precise blockers; only stop the affected part, unless causal protocol/source preservation is threatened.

## Autonomous Git commit and push (explicitly authorized)

When implementation/test pass or an honestly labelled blocked milestone is committed:
\`\`\`powershell
git status --short
git diff --stat
git diff --name-only
# Review staged filenames BEFORE committing. Explicitly stage only new
# experiments/calibration, tests/calibration, docs/calibration and approved
# synthetic metadata. Never include datasets/weights/cache/secrets.
git add -- experiments/calibration tests/calibration docs/calibration
git diff --cached --check
git diff --cached --stat
git commit -m "feat(calibration): implement Stage 0 causal harness and baseline contracts"
# If critical fidelity is blocked, instead use an honest WIP commit message.
git push -u origin calibration-stage0-contract-20261029
git rev-parse HEAD
git status --short
\`\`\`
Use \`git add\` for an optional new PowerShell script only after reviewing its path. Never force-push, never silently discard local changes, never open/merge a PR unless asked, never delete old results. Git push authorization is granted **for this new branch only**. If push fails due credentials/permissions, report it without attempting to change authentication settings.

## Final response to user / reviewer (no extra manual requests)

Once pushed, give concise branch name, commit SHA, GitHub compare/commit link, Stage 0 gate summary and any exact BLOCKED/SKIPPED items. Tell the user simply:

**“Stage 0 已 push 到 calibration-stage0-contract-20261029；请让研究审查对话检查代码和报告。”**

If blocked, append: **“当前为 WIP，尚未通过 Stage 0 验收。”**

It is the scientific reviewer's job to inspect actual GitHub Diff, math fidelity, test evidence, source hashes and protocol before authorizing Stage 1. Never declare Stage 1 ready solely because tests executed.
