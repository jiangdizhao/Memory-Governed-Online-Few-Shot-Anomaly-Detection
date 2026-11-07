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

## Experiment A Scoped Authorization — 2026-11-06

The historical Stage 0 contract remains in force for its original research stage. Experiment A is a separately authorized follow-up study under `docs/calibration/EXPERIMENT_A_AUTHORIZATION_20261106.md` and its frozen protocol. The Stage-0-only execution limit does not block the specifically scoped Experiment A work described there: A1, A2, A3a, and A3b may use the authorized development data and perform their registered empirical analyses.

This exception is limited to `audiojack`, `regulator`, `switch`, and `rolled_strip_base`, with the exact data roles, causal event order, group identity checks, and output protections in the authorization document and Experiment A protocol. It does not authorize historical Stage 1/Stage 2 programs, independent confirmation categories or outcomes, Experiments B/C, other Real-IAD categories, pixel masks, unrestricted experimentation, or any relaxation of the scientific invariants above. Queried labels remain unavailable until the registered query action; unqueried outcomes remain evaluator-only after a valid Stage A seal. Any calibration-coupled learning trajectory must be separately identified from calibration-isolated comparisons.

## Experiment A4 scoped exception — 2026-11-07

The separate `docs/calibration/EXPERIMENT_A4_EB68_AUTHORIZATION.md` authorizes only A4_EB68_v1: P0 on the same four Experiment A development categories, using exact A1 S4+C64 identities, draw-0 C32/R32 split, four registered equal-fixed-storage arms, native VisionAD scoring/Greedy, Q-A-TS-v1, and post-seal Stage B evaluation. This narrow follow-up may use its stated development TRAIN/TEST access and empirical analyses; the historical Stage-0-only limit remains in force for all other work. All other scientific invariants above remain unchanged. Do not run other categories, confirmation outcomes, masks, B/C, historical Stage 1/2, new query policies, or unregistered optimization.

## FS4 selection-aware threshold implementation authorization

`docs/calibration/FS4_THRESHOLD_IMPLEMENTATION_AUTHORIZATION.md` authorizes the separate `FS4_SELECTION_AWARE_THRESHOLD_IMPLEMENTATION_V1` implementation and CPU synthetic verification only. It covers the new C64-free stochastic group-slot query design, the historical-score and source-excluded current-state threshold controllers, and their new provenance adapters. It does not authorize scientific data scoring, Real-IAD TEST labels or masks, model-weight downloads, historical-source modifications, or a future experiment. Scientific execution requires a later explicit authorization following independent review.

## FS4 hardening R1 scoped engineering authorization

`docs/calibration/fs4_threshold/FS4_HARDENING_R1_AUTHORIZATION.md` authorizes `FS4_THRESHOLD_HARDENING_R1`: FS4-only repairs and production infrastructure, synthetic CPU/CUDA verification, and the narrowly defined verified four-Support TRAIN-only engineering parity preflight. No scientific experiment, TEST labels or masks, other categories, historical-source edits, or additional normal-reference inputs are authorized. All other scientific and repository invariants remain in force.

## FS4 hardening R2 scoped engineering authorization

`docs/calibration/fs4_threshold/FS4_HARDENING_R2_AUTHORIZATION.md` authorizes only the R2 trusted execution boundary, Stage B integrity and private-storage engineering repair, plus bounded synthetic CPU/CUDA verification. It does not authorize scientific execution, Real-IAD TEST data or outcomes, masks, additional reference images, historical-source modifications, or a future campaign. The earlier R1 preflight receipts remain immutable historical evidence. All other scientific and repository invariants remain in force.

## FS4 trusted local research engineering authorization

`docs/calibration/fs4_threshold/FS4_TRUSTED_LOCAL_RESEARCH_AUTHORIZATION.md` authorizes only the separately named non-adversarial research execution implementation and synthetic verification. It does not authorize a scientific run, Real-IAD TEST data or outcomes, masks, historical-source changes, or a future campaign. High-security production execution remains blocked. All other scientific and repository invariants remain in force.

## FS4 R4 label-blind engineering authorization

`docs/calibration/fs4_threshold/FS4_R4_ENGINEERING_AUTHORIZATION.md` authorizes the FS4 R4 label-blind adapter and query-only broker implementation with synthetic verification only. It does not authorize real TEST data or outcomes, masks, scientific execution, historical-source changes, or a future campaign. The high-security production context remains blocked. All other scientific and repository invariants remain in force.
