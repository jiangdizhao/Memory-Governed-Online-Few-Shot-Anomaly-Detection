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

At that documentation revision, no B8 implementation or Stage 0.75-B/C experiment was authorized; the later synthetic-only Stage 0.75-B work is documented below. Stage 1 and Stage 2 remain HOLD, and Mint/PCB confirmation stays sealed. The four categories analyzed so far are exposed development data, not independent confirmation.

## Stage 0.75-B synthetic-only execution (2026-10-29)

The [candidate v3 implementation contract](STAGE075B_IMPLEMENTATION_CONTRACT_CANDIDATE_v3.md) is the primary frozen specification for the synthetic EC-MSC matrix. The preserved [v1](STAGE075B_IMPLEMENTATION_CONTRACT_CANDIDATE_v1.md) and [v2](STAGE075B_IMPLEMENTATION_CONTRACT_CANDIDATE_v2.md) versions are superseded: v1 restricted search to reachable thresholds, and final v2 review found that B8-Full did not use its registered C64 percentile coordinate in optimization. The [version lineage](STAGE075B_VERSION_LINEAGE.md) records these corrections without rewriting prior evidence.

The [full synthetic results](STAGE075B_SYNTHETIC_RESULTS_REPORT.md), [mathematical falsification report](STAGE075B_B8_MATHEMATICAL_FALSIFICATION_REPORT.md), [ablation report](STAGE075B_B8_ABLATION_REPORT.md), and [research decision candidate](STAGE075B_RESEARCH_DECISION_CANDIDATE.md) describe the v3 matrix. The full calibration suite passed 133 tests and the 48 scenario/seed trajectories replayed deterministically. G1–G15 passed, but G16/Control E is BLOCKED: the B8-M/S/P arms do not independently remove each Full component, and removing the two-class gate would leave the paired gain undefined when either class is missing. Overall Stage 0.75-B acceptance is therefore `STAGE075B_BLOCKED_WIP`. The synthetic observation remains negative for Full: its threshold trajectory matched actual B3 in every row; B3+Offset had higher descriptive BA in 35 of 42 defined pairs.

The compact checksummed [final audit package](audit/stage075b_synthetic_v8/README.md) links to the local blocked runner artifact at `outputs/calibration/stage075b_synthetic_v3_blocked_final_20261029/`. The earlier [audit package v1](audit/stage075b_synthetic_v1/README.md) preserves the superseded candidate-v2 run. No Stage 0.75-C, Stage 1/2, Real-IAD performance evaluation, Mint/PCB outcome access, mask access, or model tuning was performed. This candidate does not authorize a later stage.

## Stage 0.75-A actual B3 and local headroom audit

- [Actual B3 replay report](STAGE075A_ACTUAL_B3_REPLAY_REPORT.md)
- [Local operating-point headroom report](STAGE075A_LOCAL_HEADROOM_REPORT.md)
- [B8 simplification research diagnosis](STAGE075A_B8_SIMPLIFICATION_RESEARCH_DIAGNOSIS.md)
- [B3 headroom decision candidate](STAGE075A_B3_HEADROOM_DECISION_CANDIDATE.md)
- [Compact audited evidence package](audit/stage075a_b3_headroom_v1/)

## Stage 0.75 queried-anchor identifiability audit

The frozen [queried-anchor protocol candidate v1](QUERIED_ANCHOR_IDENTIFIABILITY_PROTOCOL_CANDIDATE_v1.md)
registers a measurement-only comparison of legal Q-A-TS feedback against
current-memory native anchor scores and temporally separated fixed-snapshot
future cohorts. The [native rescoring report](QUERIED_ANCHOR_NATIVE_RESCORING_REPORT.md),
[identifiability results](QUERIED_ANCHOR_IDENTIFIABILITY_RESULTS.md),
[research implications](QUERIED_ANCHOR_B8_RESEARCH_IMPLICATIONS.md), and
[decision candidate](QUERIED_ANCHOR_IDENTIFIABILITY_DECISION_CANDIDATE.md)
must be read together with the compact checksummed
[audit package](audit/queried_anchor_identifiability_v1/). Results are from the
four previously exposed development categories; they are not independent
confirmation or population-risk evidence. B8 v4, Stage 0.75-C, Stage 1/2 and
confirmation-class access remain unauthorized without a separate researcher
decision.

The corrected native measurement scored all 5,095 eligible queried anchors
across 120/120 snapshot states (2,450 normal LSO scores and 2,645 unchanged
memory defect scores). Query feedback yielded 90 identifiable states (75%) and
actions of 33 LOWER, 26 KEEP, 30 RAISE, and 31 ABSTAIN. The decision candidate
is `LOCAL_FEEDBACK_UNRELIABLE`: equal-class mean selected-action future ΔBA was
negative for both registered cohorts. N1's seed-position denominator was
corrected from three to fifteen in a postmeasurement gate recalculation; the
action digest and score/evaluation artifacts did not change. All final gates
pass except G2, which remains `BLOCKED` because the original Actual-B3
`complete.json` checksum and file count disagree with its current checksum
manifest. Overall status is `BLOCKED_WIP`; no Stage 0.75-C, Stage 1/2,
confirmation outcomes, or masks were accessed.
