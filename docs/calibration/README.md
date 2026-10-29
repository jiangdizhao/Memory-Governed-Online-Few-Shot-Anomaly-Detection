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

## Sixth-pass scientific closure candidates

The post-sixth-pass endpoint review is recorded in the versioned
[Sun-CS conservative endpoint specification](SUN_CS_A3_CONSERVATIVE_ENDPOINTS_v2.md),
[B7 activation-feasibility reporting specification](B7_ACTIVATION_FEASIBILITY_REPORTING_v1.md),
and [Stage 1 protocol candidate v2](STAGE1_EVALUATION_PROTOCOL_FROZEN_CANDIDATE_v2.md).
The associated closure decision and compact synthetic audit are separate from
the archived sixth-pass audit. These documents remain candidates for
independent researcher review; they do not approve the Gate Amendment or
authorize Stage 1/2.

The current final protocol decision record is
[STAGE0_FINAL_SCIENTIFIC_PROTOCOL_REVIEW.md](STAGE0_FINAL_SCIENTIFIC_PROTOCOL_REVIEW.md).
It records the Sun-CS endpoint pathology and applicability boundary, B7
sample-size/query-budget feasibility, method-fidelity limits, and decisions
still requiring researcher approval.


## Stage 0.5 exposure and Stage 0.75 design-only plan

The [Stage 0.5 exploratory statistical report](STAGE05_EXPLORATORY_STATISTICAL_REPORT.md) and [data-exposure ledger](STAGE05_DATA_EXPOSURE_LEDGER.md) record authorized access to labels from four development categories, 200 label-blind query schedules per class/rate, and 20 static native scorer configurations. These outcomes are **exposed development information**, not unseen scientific confirmation. Mint/PCB and masks remain unopened.

The **[Stage 0.75 research agenda](STAGE075_RESEARCH_AGENDA_v0_1.md)** explains why total queried defect count is not the same as timely feedback, why fixed-M0 threshold headroom is distinct from dynamic-memory ranking, and how a future **0.75-A paired native Greedy falsification** should account for five-view same-group memorization and current-state C64 transport. The **[B8 MTEAC hypothesis and method design](STAGE075_B8_MTEAC_RESEARCH_DESIGN_v0_1.md)** records a proposed paired-score memory-transport, normal-only calibration objective, evidence-ladder control states, assumptions, negative controls and required B3/B7 ablations. B8 is **not implemented or frozen**; no new method settings or results are claimed.

**Authorization:** documentation-only in this commit. Stage 0.75-A/B/C experiments, Stage 1 and Stage 2 remain **HOLD** until the researcher issues a separate scoped execution instruction. These design documents do not supersede historical experiment source or authorize changing any frozen controller or query tape.


## Stage 0.75-A results and Stage 0.75-B preparation (B8 v0.2)

The [executed Stage 0.75-A ranking report](STAGE075A_GREEDY_RANKING_FALSIFICATION_REPORT.md), [C64 transport report](STAGE075A_C64_MEMORY_TRANSPORT_REPORT.md) and [research decision candidate](STAGE075A_RESEARCH_DECISION_CANDIDATE.md) record 20/20 exploratory development trajectories; Native Greedy improved paired ranking on these configurations, but the old frozen threshold could worsen FNR. The unclipped read-only C64-q90 diagnostic is **not** the actual B3 with movement clip. Paired C64 median shifts were usually zero despite some q90 movements, weakening the median-translation-only hypothesis.

The **[B8 v0.2 EC-MSC research design](STAGE075_B8_ECMSC_RESEARCH_DESIGN_v0_2.md)** preserves the [historical B8 v0.1 MTEAC draft](STAGE075_B8_MTEAC_RESEARCH_DESIGN_v0_1.md) and advances a **new, unimplemented** normal-reference percentile coordinate + defect-informed B3-relative decision residual + evidence-aware conservative update hypothesis. Its intended empirical performance criterion is positive **class-macro paired BA(B8)−BA(actual B3)**, not uniform dominance in long normal-only prefixes; the stronger B3+defect-offset ablation is mandatory to test novelty. The [Stage 0.75-B synthetic implementation-preparation contract](STAGE075B_SYNTHETIC_PROTOCOL_CANDIDATE_v0_1.md) lists open mathematical decisions, causality tests, extreme feedback regimes and negative controls. These are **design candidates**, not frozen B8 parameters or implementation approvals.

**Execution boundary remains unchanged:** no B8 implementation or Stage 0.75-B/C experiment is authorized by this documentation commit; Stage 1 and Stage 2 remain HOLD, Mint/PCB confirmation stays sealed. The four categories analyzed so far are exposed development data, not independent confirmation.
