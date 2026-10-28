# Calibration workstream — Stage 0 documentation index

**Status: implemented Stage 0 infrastructure with a `BLOCKED_WIP` acceptance status; not scientifically validated.**

## Why this workstream exists
Pattern Recognition submission PR-D-26-11359 was rejected; the October 2026 rejection recovery and R0/R1/P0/P1 audits exposed weak control/feedback attribution and stale-threshold effects. P0/P1 on a **single** Audiojack development condition suggested Greedy memory changed score ranking favorably but an old operating threshold severely increased false negatives. Recalibration also increased FPR; it is not proof that a two-sided safe method already works. Latest research handoff dated 2026-10-28 **stops the current CDARG/SAEG Safe Admission sprint** and pivots to state-consistent online calibration.

This Stage 0 builds a **reusable, switchable experimental engine**; the present experimental profile isolates calibration and shares one Greedy memory trajectory. Later factorial attribution and full BEA/correction closed loop are separate profiles and separate scientific claims, not preapproved experiments.

## Authoritative reading order
1. [STAGE0_RESEARCH_CONTRACT.md](STAGE0_RESEARCH_CONTRACT.md) — scientific question, exact first-round protocol, algorithm candidates, acceptance limits.
2. [ARCHITECTURE_AND_CAUSAL_PROTOCOL.md](ARCHITECTURE_AND_CAUSAL_PROTOCOL.md) — reusable engine, native v22 interfaces, event order, ledger and information firewall.
3. [BASELINE_FIDELITY_AND_REFERENCES.md](BASELINE_FIDELITY_AND_REFERENCES.md) — B0–B7 mathematics, provenance, primary paper URLs, adaptation boundaries and open fidelity gates.
4. [AUTOMATION_AND_ACCEPTANCE.md](AUTOMATION_AND_ACCEPTANCE.md) — Stage 0 deliverables, gates, test matrix, one-command CLI and audit format.
5. [CODEX_IMPLEMENTATION_TASK.md](CODEX_IMPLEMENTATION_TASK.md) — agent execution requirements and automatic push policy.

## Prior local/Git evidence (not to be rewritten)
- `docs/Calibration_Falsification_Research_Handoff_2026-10-28_v1.docx` — **latest decision authority**; supersedes earlier Safe Admission plans. Read directly where a DOCX reader is available; the constraints in this document set encode the agreed instructions.
- `docs/P01_FALSIFICATION_2026-10-28.md`, `outputs/p01_falsification_20261028/` — final-state Probe-A diagnostic, **not** online prequential confirmation.
- `docs/R0_OPPORTUNITY_AUDIT_2026-10-28.md`, `experiments/risk_safe/p01_falsification_native.py` — candidate extraction, native Greedy / P0 parity and calibrated q90.
- `experiments/v02_memory.py` and `experiments/risk_safe/native_bridge.py` — native Greedy and shadow parity reference.
- `experiments/risk_safe/query_protocol.py` — historical label-blind grouped Q-A tape and replay **but not** the newly required time-stratified Q-A-TS variant.
- `vmb_visionad_new.py`, `test_phase3_AIF_simplified_v22.py` — frozen scoring substrate; do not edit.
- User-provided study `Designing and Prioritizing Memory-Dependent / State-Consistent Calibration for Dynamic-Memory Industrial Anomaly Detection` and VisionAD paper `Search is All You Need for Few-shot Anomaly Detection` are contextual design sources. The precise source implementation prevails for parity.

## Work boundaries
**Stage 0:** specification, code, synthetic mechanism tests, offline manifests, baseline fidelity; no Real-IAD result optimization. **Stage 1:** 4 development classes, matched Q-A-TS and Q-B profiles (separate). **Stage 2:** precommitted order stress and untouched Mint/PCB confirmation only after freeze. Pass/fail and scientific results must never be invented.

A single `python -m experiments.calibration.runner --stage 0 --all` command is implemented for Stage 0 synthetic validation and audit generation.
