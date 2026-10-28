# Agent instructions — Memory-Governed Online FSAD

Scope: applies to this repository. For the calibration workstream, **first read**:
1. `docs/calibration/README.md`
2. `docs/calibration/STAGE0_RESEARCH_CONTRACT.md`
3. `docs/calibration/ARCHITECTURE_AND_CAUSAL_PROTOCOL.md`
4. `docs/calibration/BASELINE_FIDELITY_AND_REFERENCES.md`
5. `docs/calibration/AUTOMATION_AND_ACCEPTANCE.md`
6. `docs/calibration/CODEX_IMPLEMENTATION_TASK.md`

## Scientific invariants (not negotiable)
- All newly developed calibration code lives in `experiments/calibration/` and its own tests/docs. Existing v22, vmb, V0.2, P0/P1, R0/R1, D1–D4 files and archived outputs are **read-only**. Existing source functions may be imported as dependencies. If modifying source is genuinely unavoidable, STOP and request explicit review; do not silently patch old experiments.
- Do not substitute a generic nearest-neighbor scorer for native v22/VisionAD. Use `experiments/risk_safe/native_bridge.py` and native source functions; validate score AND memory-update parity. Preserve all frozen support, augmentation, and layer semantics.
- This is **Stage 0 only**: implement infrastructure, eight methods with fidelity declarations, reference tests, preflight/synthetic CPU verification, optional *non-scientific* parity smoke that does not require full Real-IAD. **Do not run Stage 1/Stage 2, tune on Real-IAD scores, open confirmation-class outcomes or claim empirical superiority.**
- Distinguish four-support + **additional** C64 calibration normals from strict four-shot-total; never call C64 “strict four-shot”.
- Predict before current-label disclosure. Query tape must be label blind. Only queried verified normals can update the shared Greedy bank. Never fabricate missing class error rates or treat missing evidence as zero risk.
- In calibration-isolated mode, all controllers receive the same native score trajectory and verified feedback; no controller's threshold can alter memory write, query decision, or correction. A memory/correction experiment may use distinct score trajectories, but must be a separately named experiment profile.
- No claim that sparse-selective-feedback variants inherit ACI or Sun et al. finite-time/anytime guarantees. Faithful versus adapted comparisons must be explicitly labeled and tested.
- External baseline is not “implemented” until exact equations, information assumptions, adaptation policy, and reference tests are recorded. **Fail closed** on unresolved math instead of inventing a plausible method.
- No future image/label, Probe-A/B, defect mask, hidden class histogram, or confirmation-class label in controller state; masks only for later offline metrics.
- No algorithm/parameter edits triggered by Stage 2 data. Stage 1 methodological revisions require evidence, versioned scientific rationale, bounded revisions and a frozen holdout before confirmation.

## Engineering and Git rules
- Work on the requested calibration branch; do not change historical branches or force-push.
- New outputs must be isolated, manifest-locked, checksummed, resumable, and non-overwriting; mark partial/smoke as `INCOMPLETE`.
- Never commit datasets, weights, credentials, images from Real-IAD, large feature caches, or unapproved generated output. Small synthetic fixtures and text-only audit reports are fine.
- Run compilation and tests; commit/push only if Stage-0 critical gates pass, or clearly mark the commit as `WIP / STAGE0_BLOCKED` with blockers. Do not claim PASS when a mandatory test was skipped.
- For math ambiguity, missing source, fidelity failure, dirty unrelated worktree, or unexpected protected-file changes: stop affected action and report the exact blocker. Do not relax a test to make it pass.
- When finished, push branch only if authorized by the task prompt and report: branch, commit SHA, PASS/FAIL/SKIP matrix, changed files, commands, unverified claims, and blockers.
