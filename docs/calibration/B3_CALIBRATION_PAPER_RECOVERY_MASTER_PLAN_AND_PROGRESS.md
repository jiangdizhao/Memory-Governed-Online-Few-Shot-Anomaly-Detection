# B3 Calibration Paper Recovery — Master Plan, Progress & Decision Log

> **Document type:** Living research project plan + execution tracker + scientific audit + publication decision record  
> **Created:** 2026-10-30 (Australia/Sydney)  
> **Status:** PLAN_REGISTERED / NO_NEW_EXPERIMENT_EXECUTION  
> **Repository branch at creation:** `calibration-stage0-contract-20261029`  
> **Baseline HEAD before this document:** `3dc81e4f6a5ce6dfbb8cab744ff30d971d9e4240`  
> **Maintainers:** Researcher (scientific approval and experiment execution decisions), ChatGPT (protocol/code/result audits and document updates), Codex (substantial implementation and runs)  
> **Scope:** B3 mechanism ablation; external strong calibration baselines; eventual independent confirmation and manuscript recovery.  
> **Language/terminology:** Chinese project commentary; English method names and reproducible artifact identities.
>
> **IMPORTANT:** This document records a **proposed** research plan. Creating or editing it does not authorize Codex execution, dataset access, additional query labels, protected confirmation classes, GPU work, parameter search, or changes to frozen historical results. A subsequent explicit researcher instruction is required before each new experimental run.

## 0. Objective and decision criteria

### 0.1 Publication goal

Recover the rejected *Pattern Recognition* research into a defensible, more focused journal submission aimed at venues such as *Neurocomputing*. The **single primary methodological candidate** is **State-Consistent Quantile Calibration (SCQ/B3)** for an industrial anomaly detector with a continuously evolving external normal memory.

Do **not** frame the paper as needing to beat every external method. It must instead demonstrate:

1. A clearly identified **scientific mechanism** rather than a loose aggregation of existing modules.
2. **Genuine competitiveness** against honestly implemented, reasonably tuned and applicable external literature baselines and strong simple controls.
3. Correct and transparent **few-shot and calibration-data access**, causal feedback permissions, non-self-matching, and independently checked evaluation.
4. A falsifiable claim with explicit negative-result / stop criteria, rather than adding new controller complexity until a positive result appears.

The underlying frozen visual scorer and shared Greedy normal-memory trajectory remain system context; dynamic memory is **not** a new, unverified paper-wide claim of its own. Threshold-only controllers sharing identical scores have identical score AUROC/AUPRC.

### 0.2 Preferred scientific claim (conditional)

> When external normal memory changes the anomaly score function `s(x; M_t)`, calibration on stale reference scores mixes different scorer states. Rescoring a fixed, legally available reference set against the **current** memory state may yield a more stable and competitive online decision threshold.

This is a **hypothesis**, not yet an established independent innovation or an online statistical guarantee. Rescoring against a current model does not, by itself, confer conformal validity, certified FPR/FNR bounds, or new theoretical coverage guarantees.

The registered primary B3 is **protected-C64-only**, not a queried-normal-LSO pooled-anchors method. The source-group leave-out rule remains mandatory for any separate queried-normal anchor variant; do not silently attach it to B3.

### 0.3 Decision vocabulary

- `PLANNED`: protocol documented, no experiment launched.
- `READY_FOR_RESEARCHER_APPROVAL`: protocol/inputs audited, researcher has not authorized execution.
- `AUTHORIZED`: researcher explicitly approved named scope, permissions, compute bounds and run.
- `RUNNING`: Codex/run underway.
- `SUBMITTED_FOR_AUDIT`: Codex committed code and provided immutable artifacts.
- `PASS_ENGINEERING`: code/integrity/causal contract checks pass; **not** evidence supporting the hypothesis.
- `PASS_EMPIRICAL`: specified empirical evidence supports the stated narrow claim.
- `NEGATIVE_RESULT`: controlled test fails the hypothesis or stronger method dominates.
- `BLOCKED`: missing inputs, authorization, exact state snapshots or fidelity; no workaround through protected labels.
- `HOLD`: scientific decision deferred until a specified issue is resolved.
- `GO_TO_MANUSCRIPT` / `NO_GO_METHOD_CLAIM`: researcher-level publication decisions, not automatic CI statuses.

**Current overall state (2026-10-30): `A0/A1_PASS_ENGINEERING_REVIEWED; A2_REVIEWED_WITH_PROVENANCE_LIMITS; H1_WEAK_OR_MIXED; PAPER_METHOD_CLAIM_HOLD`.** The researcher explicitly authorized development-only Stage 1 evaluation and Codex completed sealed A2/A3/A4 prediction and offline label analysis, commit `9b1e7d65969be88a2cfe5d045e840dbc1908b9cb`. ChatGPT audited the committed code, reporting/paired artifacts and stage gates but could not reopen the original Windows frozen inputs, full local metrics/diagnostics, or independently reproduce the saved compressed prediction streams. The result is a small BA/FNR gain against lag-1 accompanied by increased FPR. **No independent-confirmation run, extra controller, or publication GO has been authorized.**

## 1. Verified inherited evidence (do not silently modify)

### 1.1 Source-of-truth pointers

- Primary implementation registry: [`experiments/calibration/method_registry.py`](../../experiments/calibration/method_registry.py).
- B3 and B1 code: [`experiments/calibration/controllers.py`](../../experiments/calibration/controllers.py).
- Formal Stage 1-A runner: [`experiments/calibration/stage1a_formal_comparison.py`](../../experiments/calibration/stage1a_formal_comparison.py).
- Native replay and private feedback boundaries: [`experiments/calibration/stage1a_p12_runner.py`](../../experiments/calibration/stage1a_p12_runner.py).
- External-baseline source/fidelity ledger: [`BASELINE_FIDELITY_AND_REFERENCES.md`](BASELINE_FIDELITY_AND_REFERENCES.md).
- SPOT/dSPOT details: [`SPOT_DSPOT_IMPLEMENTATION.md`](SPOT_DSPOT_IMPLEMENTATION.md).
- Stage 0.75-A independent actual-B3 replay: [`STAGE075A_ACTUAL_B3_REPLAY_REPORT.md`](STAGE075A_ACTUAL_B3_REPLAY_REPORT.md).
- Formal comparison audit: [`outputs/calibration/stage1a_formal_controller_comparison_v4_20261030/formal_comparison_audit.json`](../../outputs/calibration/stage1a_formal_controller_comparison_v4_20261030/formal_comparison_audit.json).
- Experimental data access and approved design: [`STAGE1_EVALUATION_PROTOCOL_APPROVED_DESIGN_v3.md`](STAGE1_EVALUATION_PROTOCOL_APPROVED_DESIGN_v3.md) and [`STAGE1A_FOCUSED_PROTOCOL_ADDENDUM_CANDIDATE_v1_2026-10-30.md`](STAGE1A_FOCUSED_PROTOCOL_ADDENDUM_CANDIDATE_v1_2026-10-30.md). The latter was an **approval-required candidate at issue time**; do not retroactively infer formal permission from a completed run.

### 1.2 Frozen experiment context

- 4 development classes: `audiojack`, `regulator`, `switch`, `rolled_strip_base`.
- 5 support seeds per class: `0..4`; **20 correlated class × support-seed trajectories**, not 20 independent datasets.
- Fixed Q-A-TS query tape at **3%**, tape seed **37**.
- Native frozen DINOv2/v22 scorer; shared controller-independent Greedy normal memory, **K=256 per feature layer**, only queried verified-normal writes.
- Four deployed normal supports **plus 64 additional protected training-normal C64 anchors**; not a strict four-shot-total data-access setting.
- Prediction at `t` precedes current queried-label disclosure; any valid memory/evidence refresh applies to `t+1`.
- Original historical G2 source-byte parity has been marked BLOCKED in earlier audits; do not rewrite that provenance or conflate current reconstructed replay parity with exact historical source parity.
- Formal Stage 1-A v4 reports **20 trajectories / 76,475 score events**, no new native/GPU scoring for controller comparison, and no unqueried labels in controllers. Keep original output files immutable.

### 1.3 Inherited Stage 1-A v4 descriptive results

Mean over 20 class × seed trajectories. These are **development** results, not independent confirmation:

| Controller | BA ↑ | FPR ↓ | FNR ↓ | Worst-256 BA ↑ |
|---|---:|---:|---:|---:|
| B0 Frozen-Theta | 0.6434 | 0.0099 | 0.7033 | 0.6041 |
| B1 queried-normal stale Rolling q90 | 0.6812 | 0.0239 | 0.6136 | 0.6285 |
| B2 adapted historical v22 sidecar | 0.6458 | 0.0110 | 0.6974 | 0.6078 |
| B3 protected-C64 SCQ | **0.7362** | 0.1042 | 0.4235 | **0.6530** |
| B4 adapted Dual-ACI | 0.6755 | 0.0261 | 0.6229 | 0.6149 |
| B7 SCDAC-G | 0.6986 | 0.0945 | 0.5083 | 0.6253 |
| Stage075BOffset (distinct candidate) | **0.7573** | 0.1718 | **0.3136** | 0.6317 |

B3 exceeds B1 in BA by about **0.0550**, on **19/20** paired trajectories, but at an increased false-positive rate. The separately named Offset candidate exceeds B3 in mean BA but has worse FPR and Worst-256 BA. B3 is not yet established as superior to external methods.

**Already completed partial mechanism ablation:** the Stage 0.75-A actual-B3 replay records Greedy+Frozen BA **0.64339**, Greedy+read-only current-C64-q90 **0.73712**, and Greedy+actual B3 (step clip `Delta=0.01`) **0.73615**. The step clip does **not** explain most of the BA gain; current-state C64 recalibration appears to be the operative candidate mechanism. This is descriptive evidence, **not** isolation of state consistency from dynamic threshold refresh.

**B5a/B5b caveat:** original SPOT/dSPOT have causal startup intervals with unavailable predictions; their source-native Full-stream BA is undefined for affected trajectories. Do not impute or rank missing outcomes; retain native prediction-availability reports and distinguish named Frozen-Theta fallback hybrids.

**B6 caveat:** the method registry has source-equation Sun-CS implementation/tests, but Stage 1-A v4 uses **applicability-only** treatment. The older fidelity ledger contains historical BLOCKED statuses; report dated implementation versus comparison status separately. Do not claim B6 participated in the formal BA leaderboard or transferred its confidence guarantees.

## 2. EXPERIMENT A — B3 mechanism falsification

**Task ID:** `A-B3-STATE-CONSISTENCY`  
**Priority:** P0; conduct **before** new large external-baseline runs.  
**Current status:** `A0/A1_PASS_ENGINEERING_REVIEWED_WITH_LOCAL_VERIFICATION_LIMITS` (2026-10-30, PR #1 open and unmerged); `A2_PLANNED_NOT_AUTHORIZED`.  
**Research hypothesis H1:** conditional on the **same fixed C64 anchors, Greedy score/memory stream, threshold update boundaries, quantile estimator and movement policy**, using reference scores from the current memory version yields an empirically meaningful improvement over using deliberately lagged scores.

### 2.1 Primary causal isolation

`s_t(x) = s(x; M_t)`; for the same protected C64 anchor identities `C`, define `q_v = higher_q90({s(x_i; M_v): x_i in C})`. The strict reference score comparison must be **current** `q_v` versus **lagged** `q_{max(0,v-k)}` at the **same post-write update event**, with identical `Delta=0.01`, q90 convention, update cadence, prediction/event ordering, initial threshold and C64 eligibility.

Where no actual historical C64 vector or q90 receipt exists for a requested version, report `BLOCKED_INPUT_STATE`. Never synthesize historical rescoring, access future memory, or silently substitute arrival scores from a different anchor set.

### 2.2 Registered arms

| Arm | Exact distinction | Role | Status |
|---|---|---|---|
| A0 | Frozen initial C64 q90 under M0 | Existing B0; orientation | HISTORICAL_RESULT_AVAILABLE |
| A1 | Read-only, current-M C64 q90, no movement clip | Existing Stage 0.75-A result; clip sensitivity | HISTORICAL_RESULT_AVAILABLE |
| A2 | Current-M C64 q90, `Delta=0.01` | Existing B3; primary treatment | HISTORICAL_RESULT_AVAILABLE |
| A3 | Same as A2, but C64 q90 from 1 memory-content-version earlier | Primary matched stale-state control | PLANNED |
| A4 | Same as A2, but C64 q90 from 5 memory-content-versions earlier | Lag stress control | PLANNED |

**Primary estimand**: per-class and per-seed paired `BA(A2)-BA(A3)`, conditional on equal inputs apart from memory-version-of-reference. Secondary: `BA(A2)-BA(A4)`; FPR/FNR and worst-window comparisons. A0/A1 are context, **not** sufficient by themselves to establish H1.

**Lag semantics:** lag is in content-changing memory versions, not time steps or number of image observations. At version `v<k`, use registered `q_0` for `q_{max(0,v-k)}`. Updates occur on the same legally observed memory transitions for A2/A3/A4, even if q90 value is unchanged. No future lookups. Record the exact `v`, reference version, q90 value, applied step, current scores' version, and before/after threshold at each transition.

**Do not change**: B3 source class/semantics, normal C64 identities, Greedy writer, C64 scoring source, query tapes, privileged labels, test-stream order, model weights, seed set, or initial theta. New arms must have distinct IDs and separate output paths.

### 2.3 Measurements and falsifiers

Required outputs:

1. Full-stream and preregistered common-horizon `t >= 1050` BA, FPR, FNR, TP/TN/FP/FN and denominators; worst defined nonoverlapping 256-image BA/FPR/FNR; same finite prediction coverage.
2. Per class × seed paired differences, per-class mean over seeds, four-class macro mean, and variability. Seeds are **not independent classes**; category-level generalization is only four development classes.
3. Signed and absolute `q_v - q_{v-k}`; fraction of updates with material divergence; conditional effects around writes; threshold movement and clipping frequency; any observed memory-lag/performance correlation is descriptive and not itself causal proof.
4. Source/version/permission parity tests, causality tests, replay/resume parity, and independent recurrence oracle.
5. CPU time, maximum memory and rescoring operations. If cached state history suffices, **no additional GPU inference** is warranted for A3/A4.

**H1 passes only provisionally** if A2 exhibits a repeatable **meaningful** advantage over at least A3 across categories without an unacceptable FPR/worst-window deterioration, and audits confirm that reference-score state is the sole intervention. There is no post hoc required strict ordering A2>A3>A4: data can be non-monotone. If A2 roughly matches lagged controls or simpler controls dominate under the preselected operating criteria, record `NEGATIVE_RESULT` / `HOLD`; do not invent further arms to rescue H1.

**Decision caveat:** the 0.01 step cap is primarily an engineering stability choice unless an isolated test shows its benefit; A1's slightly higher historical BA must be disclosed.

### 2.4 Delivery and work allocation

**ChatGPT — small scoped tasks only:** review exact frozen event/receipt schema; add or edit this protocol/tracker; review small test fixtures; inspect Codex diff and numerical audit. No broad code refactor or silent GPU/dataset access.

**Codex — after a separate researcher-approved prompt:** implement a versioned, standalone replay/ablation path (proposed `experiments/calibration/stage1a_b3_mechanism_ablation.py`) and targeted tests (`tests/calibration/test_b3_mechanism_ablation.py`); use existing P1.2 frozen receipts/evidence and seal old artifacts. A separate new dataset/feature run requires new explicit authorization.

**Gate A0 — input feasibility:** verify version-addressed current and prior C64 q90/evidence can be read for **all 20** trajectories with original digests; if not, identify exact missing data and stop rather than guessing.  
**Gate A1 — implementation:** A0/A1/A2 historical predictions/metrics independently reconcile within declared numerical tolerance; A2 and A3/A4 exact same event boundaries and reference C64 identities; tests pass.  
**Gate A2 — evidence:** report full paired table, downside metrics and negative results without selective omission.  
**Gate A3 — decision:** Researcher chooses `GO_TO_EXTERNAL_BASELINE`, `HOLD` or `NO_GO_METHOD_CLAIM` after ChatGPT audits Codex's commit and immutable artifacts.

## 3. EXPERIMENT B — competitive external calibration baselines

**Task ID:** `B-COMPETITIVE-LITERATURE-BASELINES`  
**Priority:** P0 after Experiment A implementation feasibility; code-fidelity review can proceed earlier.  
**Current status:** `PLANNED`.  
**Research hypothesis H2:** B3 is competitive with appropriately adapted literature controllers and strong simple calibration rules under the same evolving-memory scoring path and declared observation permissions, even if it does **not** win every metric or beat every comparator.

### 3.1 Current implementation readiness versus publication suitability

| ID | Literature family / comparator | Repo implementation | Scientific status / needed work |
|---|---|---|---|
| B0 | Frozen initial threshold | Implemented and evaluated | Basic mandatory reference |
| B1 | Queried-verified-normal Rolling q90 (stale) | Implemented and evaluated | Mandatory transparent simple comparator; **different normal reference data from B3** |
| B2 | Original v22 Guarded Dual Anchor sidecar | Adapted and evaluated | Rejection-paper linkage; no native full-loop parity claim |
| B4 | Gibbs–Candès ACI-inspired Dual-ACI | Adapted and evaluated | **Not original ACI theorem or source-equivalent algorithm**; label exactly `Adapted Dual-ACI` |
| B5a | Siffer et al. SPOT | Paper equations + tests + comparison available | Review initialization, GPD estimator, source/native availability, q choice; no author-C numeric parity claim |
| B5b | Siffer et al. dSPOT | Paper equations + tests + comparison available | Review local mean/residual window, startup, alarms; no author-code parity claim |
| B5a/B5b_FROZEN_FALLBACK | Deployment hybrid | Already present | Rank only under their **hybrid names**; report native/fallback fractions |
| B6 | Sun et al. confidence-sequence online thresholding | Paper-equation module/tests | **Applicability-only** in formal Stage 1-A; abstention and anomaly target differ; do not force into main binary leaderboard |
| B7 | Internal SCDAC-G | Implemented prototype / evaluated | Internal exploratory comparator, no certified online risk claim; no further development planned |
| Stage075BOffset | Independent Offset variant | Evaluated | Must retain as a strong, inconvenient alternative to B3 |
| B-EWQ (candidate) | Causal EWMA/weighted-quantile score-history control | Not implemented in frozen main suite | Optional **one** inexpensive strong simple baseline, only with preregistered update equation, lookback/effective-memory settings and fair parameter tuning |

Important distinction: **the existing B1 differs from B3 in both scoring state and calibration anchor set / information supply**. It is a valid system comparison but **not** a mechanistic stale-vs-current matched ablation. Experiment A supplies the missing isolation.

### 3.2 Minimum fair comparison plan

**Stage B0 — audit existing results before implementing anything:** identify which method is already in the Stage 1-A v4 leaderboard, which methods have undefined full-stream performance, and whether existing binary hybrid rows can be compared on the predeclared common horizon without re-running any model. Summarize all coverage and denominators before reading full datasets.

**Stage B1 — baseline fidelity audit:** inspect only relevant code, primary algorithm definitions, tests and existing method cards. For each baseline record: source/paper/year/commit; faithful/adapted/different objective; inputs (C64, causal unlabeled scores, queried labels); initialization; score state; tuning/parameter values; missing predictions; runtime; what guarantees **do not transfer**. Never refer to B4 as simply “faithful ACI” or to B6 as a direct physical-defect binary detection competitor.

**Stage B2 — fill minimal comparison gaps:**
- Correct/re-report source-native SPOT/dSPOT availability: full-stream BA stays **undefined** wherever startup predictions are unavailable; selective metrics and denominators are labeled; default evaluation horizon `t>=1050` remains **fixed**, not cherry-picked.
- Report the already implemented Frozen-Theta fallback hybrids separately with fraction actually decided by source algorithm; no hybrid result is attributed to pure SPOT/dSPOT.
- Audit original and adapted algorithms' fairness; do not deliberately deprive unlabeled methods of legal causal scores or expose unqueried labels to any method.
- If strong simple controls are insufficient, implement **at most one** new lightweight EWMA/weighted-quantile score-history comparator as a separately named row. A causal score-only control might see contamination by defects: report this vulnerability, not a handicap engineered after seeing results.
- No new controller family, expanded risk-theory claim, modified B7, or open-ended hyperparameter fishing.

**Stage B3 — predeclared external comparison:** common score/memory path, same development classes and fixed query tape; differentiated input permissions **disclosed explicitly**. Report full-stream, fixed common horizon, normal/defect coverage, FPR/FNR/BA, worst 256, threshold stability and computational cost. All methods using the same score tape have the same AUROC/AUPRC; do not fabricate controller-dependent AUROC gains.

**Stage B4 — methods selection and fairness review:** distinguish directly deployable binary methods (primary table), original source-native methods with abstention/missing prediction (applicability table), and different-target B6 (separate discussion). If a method cannot be validly mapped into this task, explain why instead of weakening it to make B3 win. Where tuning is appropriate, allocate comparable, modest search effort on development data only; freeze selection rules before independent confirmation.

### 3.3 Proposed paper-facing reporting

**Table 1: Common-score full-binary decision performance.** Rows: B0/B1/B2/B3/B4/Stage075BOffset and clearly named operational SPOT/dSPOT hybrids; B-EWQ only if implemented. Include `full-stream BA/FPR/FNR`, `worst-256 BA`, `common-horizon BA/FPR/FNR`, prediction coverage, runtime. Report macro over classes with per-class supplement; include explicit comparison relative to strongest applicable simple baseline.

**Table 2: External-method fidelity / applicability.** Rows: native SPOT, native dSPOT, Sun-CS, adapted ACI. Record algorithm target, inputs, initialization, abstention/startup policy, published guarantee applicability, total/normal/defect output coverage and defined selective metrics. Never silently turn missing into normal predictions or retrospectively adjust horizon.

**Table 3: Mechanism test.** A0–A4, status-consistency lag, paired BA/FPR/FNR/worst-window, thresholds around content changes and evidence of matched variables.

**Paper fairness table:** `4 deployed supports + additional protected C64=64 train-normal reference samples` versus a truly `strict-four-shot-total` protocol. The former must **not** be described as the latter. A separate tightly scoped strict-four-shot control is needed before claiming strict few-shot competitiveness.

### 3.4 Pass / stop criteria

- `PASS_ENGINEERING`: source fidelity/adaptations explicit; exact denominator and coverage accounting; replay integrity and independent tests; no hidden test labels enter controllers.
- `PASS_EMPIRICAL` (conditional): B3 shows a defensible balance of BA, FPR/FNR, temporal robustness, or efficiency compared with **at least one genuinely competitive external/simple method**, without materially unacceptable risk under the paper's **preselected** operating priorities. It need **not** win each baseline.
- `HOLD`: better mean BA is purchased by unacceptable FPR, or strong method comparisons are not faithful or cannot share a meaningful decision definition.
- `NO_GO_METHOD_CLAIM`: matched mechanism fails, a much simpler reference subsumes B3 on meaningful criteria, or independently held-out findings do not support the claimed contribution.

Current B3 FPR is **0.1042**: do not conceal this behind BA. There is currently **no preapproved universal FPR limit** for this paper, so a particular numerical cap (e.g. 5%) must not be imposed after inspecting the test outcomes.

### 3.5 Delivery and work allocation

**ChatGPT — small scoped tasks:** audit method registry/source references; compare already emitted metrics; review code diffs and update this document; optionally make isolated tests or reporting-only fixes via GitHub.

**Codex — substantial changes after a separate researcher prompt:** independently tested baseline replay/reporting path (proposed `experiments/calibration/stage1a_external_baseline_comparison.py`, matching `tests/calibration/test_external_baseline_comparison.py`), optional lightweight baseline, immutable reports, source-commit pinning and reproduced existing rows.

**Required Codex handoff:** commit SHA; branch; exact commands; source/input hashes; 20-trajectory expected/completed counts; tests, PASS/FAIL; new immutable output folder; per-method coverage and fallback fractions; paired results and complete negative findings; unresolved blockers. No giant raw JSON output pasted into the conversation.

## 4. Experiment execution contract and authorization firewall

1. **Planning != execution authorization.** Codex prompts shall name exactly one task and one scope. The researcher approves any access to protected images, queried labels, masks, confirmations, new GPU computation, and fixed computational budget separately.
2. **No confirmation contamination:** Mint/PCB remain sealed; the four development classes can be used for method research. Only after method/parameters are frozen and independently reviewed may researcher authorize an independent confirmation run.
3. **Prequential order:** predict using current pre-event memory; then execute fixed label-blind query; disclose only truly queried label; write only verified-normal to shared memory; update anchor evidence and threshold for the next prediction. No controller-dependent memory/query tape.
4. **Immutable inherited outputs:** do not overwrite any `outputs/calibration/stage1a_formal_controller_comparison_v4_20261030` files or historical source/manifest hashes. New result sets use versioned namespaces and exact SHA-256 records.
5. **Historical fidelity:** earlier G2 source-provenance limitations remain acknowledged. A fresh PASS for a new replay does not retroactively validate original historical runner bytes.
6. **Statistics:** give per-class results and per-seed variability; four classes are the unit of class generalization, repeated support seeds do not create 20 independent classes; do not assert statistical significance solely from 19/20 wins.
7. **No silent scope drift:** new benchmark categories, data, hyperparameter grids, extra method families, or redefinitions of error costs require a logged researcher decision **before** execution.

## 5. Progress dashboard (update after every Codex milestone)

| Milestone | Owner | Status | Deliverable / next action | Audit and evidence |
|---|---|---|---|---|
| P00: register single master plan | ChatGPT | COMPLETE | This Markdown file; no experiment run | Source branch and Stage 1-A docs inspected 2026-10-30 |
| A0: historical-C64-version evidence inventory | Codex, then ChatGPT | PASS_ENGINEERING_REVIEWED_WITH_LIMITS | 20/20 complete C64 histories, 1,190 writes, 76,475 matched event/parity records | PR [#1](https://github.com/jiangdizhao/Memory-Governed-Online-Few-Shot-Anomaly-Detection/pull/1): committed manifest, 110 hashed artifacts and reported zero mismatches; local raw bytes not independently replayed by ChatGPT |
| A1: independent A3/A4 implementation and unit tests | Codex, then ChatGPT | PASS_ENGINEERING_REVIEWED_WITH_LIMITS | Version-checked A2/A3/A4 recurrence and tests; feature code integrated in original research branch | Integration commit `a0e444215e2cfcef9dfeb796bd95cf85e9d2a0f2`; A1 recorded unittest 14/14; source audit reviewed; original raw local test replay unavailable |
| A2: 20-trajectory mechanism results | Codex, then ChatGPT | REVIEWED_WITH_PROVENANCE_LIMITS / H1_WEAK_OR_MIXED | 20/20 sealed predictions, development-only metrics, version divergence and paired effects; label access after seal | A2 report + paired JSON + audit at `9b1e7d6`; ChatGPT code/consistency review 2026-10-30; local compressed predictions/detailed metrics not independently re-executed |
| A3: mechanism GO/HOLD/NO-GO decision | Researcher with ChatGPT | HOLD_METHOD_CLAIM / RESEARCHER_DECISION_PENDING | Do not assert current-state consistency has unqualified competitive advantage; prioritize external-method fidelity and fair FPR/recall tradeoff | A2-A3 macro BA +0.0037 with FPR +0.0051; A2-A4 BA +0.0200 with FPR +0.0224; final research decision remains the researcher's |
| B0: external implementation/fidelity inventory | ChatGPT / Codex | PLANNED | Method-cards reconciled with source versions and existing coverage | Existing baselines partially reviewed |
| B1: baseline comparison reporting and minimal missing baseline | Codex, then ChatGPT | PLANNED | Correct SPOT/dSPOT, adaptation labels; optional EWMA row | Not submitted |
| B2: paired external baseline results | Codex, then ChatGPT | PLANNED | Paper-facing main and applicability tables | Not submitted |
| B3: external-comparison decision | Researcher with ChatGPT | PLANNED | Competitive/insufficient/blocked | Not decided |
| C0: strict-four-shot protocol and independent-confirmation authorization | Researcher | HOLD | Dataset permission, data-access comparison, model freeze | Not authorized |
| C1: independent confirmation | Codex, then ChatGPT | BLOCKED | Only after approved C0 | Mint/PCB sealed |
| M0: Neurocomputing-level manuscript recovery | Researcher / ChatGPT | HOLD | Focused contribution, reviewer issue matrix, revised experiments | Depends on A/B/C |

**Status edits require evidence.** A Codex claim that tests passed is not enough; ChatGPT checks commit content, test outputs and immutable artifacts before changing `SUBMITTED_FOR_AUDIT` to `PASS_ENGINEERING`. Only a scientific audit supports `PASS_EMPIRICAL`.

## 6. Append-only experiment and research decision log

For each new commit or experiment, append a dated entry; **do not overwrite older judgments**. Correction entries must explicitly link earlier mistakes, new evidence, and reason for revision.

### 2026-10-30 — PLAN_REGISTERED

- **Who:** Researcher + ChatGPT; GitHub commit made solely to register this plan.
- **Requested publication objective:** respond to rejection critiques of diffuse novelty and missing strong external baselines; pursue a focused B3-centered journal paper, with competitive rather than universally best performance.
- **Evidence consulted:** Stage 1-A v4 output audit; method registry and controllers; Stage 0.75-A actual-B3 replay; SPOT/dSPOT reference tests/fidelity notes; Stage 1 approved/candidate protocols; P1.2 replay code.
- **Decision:** prioritize A (same-C64 state-version ablation) followed by B (fair external comparison); no new controller complexity, no paper-go claim.
- **Code change:** this Markdown plan only. **No experiment executed.**
- **Known risks:** four development classes only; B3 FPR 0.1042; additional C64 is not strict four-shot; A3/A4 history availability unverified; B5 startup missingness; B6 different target; old historical G2 block.
- **Next:** researcher will request a Codex prompt for the narrowly scoped Experiment A feasibility gate and implementation.

### 2026-10-30 — A0 CODEX READ-ONLY REPORT / A1 WORKTREE BLOCKER

- **Report source:** Researcher pasted Codex's local terminal findings into ChatGPT; the local probe generated no persistent `feasibility_matrix.json`, manifest, commit or PR. This is **reported evidence**, not a ChatGPT-independent verification of full local JSONL or SHA-256 values.
- **Reported environment:** Windows `E:\Memory-Governed-Online-Few-Shot-Anomaly-Detection`; Python `D:\anaconda3\envs\torch\python.exe` 3.11.14; Torch 2.0.1+cu117; NumPy 1.24.0; dataset `G:\dataset\realiad`, JSON metadata and DINO cache found; original planning branch HEAD `b291b5a399d72d38ed608730008e5b54995720e0`.
- **Claimed A0:** all 20 trajectories have complete versioned protected-C64 q90 histories, **20 initial + 1,190 post-write** vectors, with 1,190 content changes in total. Exact recurrence comparison reportedly found **zero pre/post threshold and prediction mismatches** across **76,475** events. Each trajectory has a source-evidence SHA in the researcher-provided report, but **there is no separately committed manifest here**. Original historical G2 source-byte parity remains blocked.
- **Reported access:** read-only receipts/C64 JSONL and historical prediction rows, **0 new image reads, 0 new label-provider reads, 0 new GPU calls**, no confirmation or masks opened. ChatGPT has not independently re-executed the local probe.
- **A1 outcome:** `BLOCKED_WORKTREE` because the existing checkout has unrelated untracked experimental `outputs/` and `docs/calibration/audit/` directories. Codex made **no changes**, ran no A1 implementation tests and created no task branch/commit/PR. This is a workspace hygiene blocker, **not an evidence-feasibility failure**.
- **Corrective path:** do **not delete/move/clean** existing outputs. Prefer a separate `git worktree add -b <task-branch> <new-directory> <verified-base-commit>` for A1. Historical data may be untracked and physically present only under the original checkout; Codex must explicitly identify required read-only sources and use narrowly scoped references or directory junctions without writing into those targets. Keep all new code, tests and reports in the isolated worktree, not in linked historical directories. If source-path/hashes cannot be preserved, stop and report instead of weakening invariants.
- **Next:** researcher provides an A1 continuation prompt, Codex creates a separate task workspace, repeats A0 hash/parity verification with a small committed manifest and conditionally implements A1 tests. **No 20-trajectory outcome evaluation** before separate authorization.
- **Audit designation:** `A0_REPORTED_PASS_PENDING_INDEPENDENT_AUDIT`; `A1_BLOCKED_WORKTREE`; `H1_NOT_TESTED`; no paper GO decision.

### 2026-10-30 — A1 PR #1 INDEPENDENT GITHUB REVIEW

- **Submitted implementation:** [PR #1](https://github.com/jiangdizhao/Memory-Governed-Online-Few-Shot-Anomaly-Detection/pull/1), dedicated branch `calibration-b3-state-lag-a1-20261030`, reviewed head `a32b9c80fd422dc121963ee9a2a9d6bd097256f1`, based on `cfaf0f925a71963333627687a04d8baac7414f0e`. Open/unmerged as of review.
- **Change scope:** exactly 9 added files: one 1,098-line standalone ablation module, one 255-line test module, seven scoped audit/runbook/result files. **No changed historical controller, scorer, query tape, original B3, memory writer, or frozen results.**
- **Reviewed mechanism:** registered `ARMS={A2:0,A3:1,A4:5}`; q90 higher of fixed C64; fixed `Delta=0.01`; exact M0 theta, post-prediction content change; current-v vs v-1 vs v-5 reference lookups; memory digest and history checks; deterministic checkpoint roundtrip. Existing A2 replay comparisons use stored formal predictions and threshold traces, not outcomes.
- **Reviewed tests/audit:** `unittest_output.txt` records 14/14 OK (including real-data parity when `B3_A0_FROZEN_ROOT` is set), compilation PASS; 20-row matrix sums to 76,475 events/1,190 content writes; 1,210 C64 versioned scores; zero recorded threshold/prediction/event mismatches; 110 input artifact hashes. [Recorded PR review](https://github.com/jiangdizhao/Memory-Governed-Online-Few-Shot-Anomaly-Detection/pull/1) comment ID `5711331107`.
- **Independent verification boundary:** ChatGPT has audited GitHub source code, source/manifest organization, test logic and committed reports, but cannot independently access the researcher's Windows E: drive to reproduce those read-only data checks. GitHub combined CI status has no checks on the reviewed commit. This is **`PASS_ENGINEERING` with explicit local-execution limitations**, not `PASS_EMPIRICAL`.
- **Minor environment discrepancy:** report in chat said NumPy 1.24.0; committed manifest says 1.24.1. Verify local installed version before formal A2; it is not itself evidence of incorrect q90 calculation.
- **Historical limitation:** original Stage 0.75-A/G2 source-byte parity remains `BLOCKED_UNCHANGED`. This PR does not solve it.
- **Go/No-Go:** Source-review **GO for researcher-controlled merge of A1**; **NO new experiment execution yet**. H1 novelty/benefit remains untested. Before A2 formal outcome evaluation, construct sealed prediction-only replay with preflight/hash checks, and separately authorize offline test-label scoring. No Mint/PCB access.
- **Next handoff:** researcher decides on PR #1 merge; then an A2 Codex prompt may be issued for tightly scoped 20-trajectory *development-only* evaluation, with immutable artifacts and full metric/coverage reporting.

### Template — append a new entry for every Codex delivery and audit

```markdown
### YYYY-MM-DD — <TASK-ID> — <STATUS>
- Researcher authorization: <exact scope/approval reference, or NOT_AUTHORIZED>
- Codex branch/commit/PR: <branch> / <full SHA> / <URL>
- Parent source/input SHA-256 and output namespace: <exact IDs>
- Requested implementation vs actual diff: <summary; unexpected modifications>
- Tests and parity: <commands, count, passed/failed, reproducibility>
- Data access audit: <queried-only labels; protected classes; mask access; feature/GPU calls>
- Results: <per-class and 20-trajectory aggregates, null denominators, coverage, risks>
- Comparison/claim: <precise supported statements; negative evidence>
- ChatGPT independent audit: <PASS_ENGINEERING / HOLD / BLOCKED with reasons>
- Researcher scientific decision: <GO / HOLD / NO-GO / pending>
- Unresolved limitations and next task: <one concrete action>
```

## 7. Working agreement: researcher × ChatGPT × Codex

- **Researcher:** chooses objective, authorizes data and execution, defines publication priorities, decides final scientific GO/NO-GO.
- **ChatGPT:** prepares targeted specifications and Codex prompts; may implement **small, bounded** source/tests/document edits using GitHub; reviews submitted Codex commits, test receipts, metrics and claims; **updates this same master document after each reviewed milestone** rather than spawning competing trackers.
- **Codex:** implements substantial changes and executes explicitly authorized tests/experiments; commits scoped changes and machine-readable, provenance-rich reports; does not self-certify scientific conclusions.
- Before implementation: check the **current** target branch HEAD, working tree, exact output namespace, and source-fingerprint policy. Branch/PR workflow should preserve historical artifacts and avoid unreviewed code changes.
- For local use: switch to `calibration-stage0-contract-20261029`, fetch, pull fast-forward; check the new master plan at `docs/calibration/B3_CALIBRATION_PAPER_RECOVERY_MASTER_PLAN_AND_PROGRESS.md`.

## 8. A2 formal B3 state-consistency mechanism ablation

### 2026-10-30 - A2 FORMAL B3 STATE-LAG ABLATION - SUBMITTED_FOR_AUDIT

- **Researcher authorization:** The researcher explicitly authorized A2 Stage 1 after being asked to resolve the repository Stage-0-only restriction. Scope was the 20 sealed development trajectories and offline evaluation on the four development-class label vectors after every prediction stream was sealed. Mint/PCB and all confirmation-class outcomes remained excluded.
- **Branch and commits:** calibration-stage0-contract-20261029; starting remote HEAD 20d6ed3cc3a6b414d2761561cb7822ecfd7da2a1. A1 was merged without rewriting history at a0e444215e2cfcef9dfeb796bd95cf85e9d2a0f2. A2 implementation and compact artifacts are commit 71f00a92a8ab8f8a303b6d6647eb177ef5312240.
- **Implementation:** Added experiments/calibration/stage1a_b3_mechanism_formal.py and tests/calibration/test_b3_mechanism_formal.py, plus the paper-facing report and compact provenance artifacts. The implementation reuses A1's frozen-input verification and matched replay; no scorer or GPU inference was added. The prediction pass verified historical A2 parity, identical score/memory event streams and write positions, and sealed all 20 streams before evaluation.
- **Output namespace:** outputs/calibration/b3_state_lag_formal_v1_20261030/. Prediction seal SHA-256 8e1b4a78cbc4037386ff1bda83341b26695448a1d6f75bea9d8ddbad6953ff92; input identity SHA-256 3ffa3305a7876d4ce352b5db7467cc2463049e6f7ba9f6368a6f39be846ce840. Compact receipts, paired results, audit, label-access ledger, run manifest, and checksum index are committed. Compressed prediction streams and the larger detailed metrics, diagnostics, and preflight JSON remain local in that immutable output namespace and are identified by the checksum index.
- **Tests:** D:\anaconda3\envs\torch\python.exe -m compileall -q experiments/calibration tests/calibration passed. With B3_A0_FROZEN_ROOT set to the workspace, D:\anaconda3\envs\torch\python.exe -m unittest discover -s tests/calibration -p 'test_b3_mechanism_*.py' -v passed 23/23 (9 A2 formal tests and 14 A1/A0 tests). Final output verification returned COMPLETE_CHECKSUMS_PASS.
- **Data access audit:** The prediction seal was verified before the label reader ran. Four distinct development-class label vectors were read in each of two evaluator attempts, for eight provider calls total. The first attempt stopped on an aggregation implementation bug; the second wrote metric artifacts and stopped on a stale report key. The report was then generated from saved metrics without further label reads. These were evaluator/report-only fixes after sealing; prediction recurrence and sealed inputs were unchanged. No raw vectors were persisted; no images, masks, confirmation labels, or GPU calls were used.
- **Results:** Full-stream equal-class macro BA/FPR/FNR: A2 0.7362/0.1042/0.4235, A3 0.7324/0.0991/0.4360, A4 0.7162/0.0818/0.4859. Common t>=1050 BA/FPR/FNR: A2 0.7413/0.1137/0.4038, A3 0.7385/0.1109/0.4122, A4 0.7233/0.0977/0.4557. Full-stream paired A2-A3 BA difference was +0.0037137 (17 positive, 3 negative, 0 tied); FPR +0.0050867 (20 worse); FNR -0.0125142 (18 better, 2 worse); worst-complete-256 BA +0.0038575 (8 positive, 6 negative, 6 tied). A2-A4 BA was +0.0199831 (18 positive, 2 negative); FPR +0.0223631 (20 worse); FNR -0.0623292 (20 better); worst-256 BA +0.0123050 (14 positive, 4 negative, 2 tied). Per-class results and the common-horizon paired results are in the report and machine-readable artifacts.
- **Mechanism evidence:** Mean q90 divergence was -0.00097594 at lag 1 and -0.00489062 at lag 5; binary decisions differed on 784 and 3,544 of 76,475 events, respectively. Post-update theta matched on 815/1,190 lag-1 transitions and 274/1,190 lag-5 transitions. Next-event error counts after transitions were 190 vs 190 for A2/A3 and 190 vs 191 for A2/A4.
- **Interpretation:** The runner records H1_WEAK_OR_MIXED. The small A2-A3 BA gain comes with higher FPR. A2-A4 has a larger BA gain, also with higher FPR; class effects are heterogeneous. These descriptive paired counts are not significance tests. No ACI or Sun et al. sparse-feedback guarantee, population claim, or confirmation-class claim is made. Four deployed supports plus additional C64 normals were used, not strict four-shot-total.
- **Audit status and limits:** SUBMITTED_FOR_AUDIT; this entry does not self-certify PASS_ENGINEERING or PASS_EMPIRICAL. Historical Stage 0.75-A/G2 source-byte parity remains blocked. A3 mechanism decision remains pending independent review and researcher judgment.
## 9. ChatGPT independent A2 review — 2026-10-30

### 2026-10-30 — A2 CODE/REPORT REVIEW; H1_WEAK_OR_MIXED; PAPER CLAIM HOLD

- **Evidence reviewed:** research branch at `9b1e7d65969be88a2cfe5d045e840dbc1908b9cb`; A2 source and targeted tests; sealed prediction receipt and source/input manifest; `formal_ablation_audit.json`, `label_access_ledger.json`, `paired_results.json`, and `FORMAL_B3_MECHANISM_RESULTS.md`. Source branch contains the A1 integration at `a0e444215e2cfcef9dfeb796bd95cf85e9d2a0f2`. The committed reports are mutually consistent on key totals and contrast signs. The complete compressed prediction streams, `per_method_metrics.json` and `mechanism_diagnostics.json` are referenced by checksums but **are not in GitHub** and were not independently replayed by ChatGPT.
- **Protocol and engineering:** reviewed the source-level prediction/evaluation separation: 20 complete streams sealed before official development labels are opened; A2 historical B3 parity enforced; A2/A3/A4 reuse the same frozen scores and Memory transitions with reference lags 0/1/5; 76,475 events, 1,190 actual writes, same 64 protected normals, `Delta=0.01`. Recorded tests 23/23 PASS and 0 GPU inference. No observed source-level violation of the prequential controller input boundary. However, the historical Stage 0.75-A/G2 source-byte-parity blocker remains.
- **Label access and retry limitation:** two post-seal evaluator attempts, eight total provider calls reading the **same four development-class vectors**, were explicitly logged. The first failed on an aggregation `NameError`; a later report-generation key required repair without further label reads. No raw label vectors were committed. This is a disclosed engineering retry, **not** independent confirmation. The auditor cannot verify live Windows file-access logs.
- **Post-seal source-provenance reservation:** the prediction seal records formal-runner source hash `f58e0c4f...`, the evaluator audit records `c49b7e04...`, and the run manifest also records a later report-generation source hash `d27a1c54...`. The changes are described as evaluator/report repairs. Because exact pre-seal executable bytes were not independently preserved as a reviewed Git commit, the auditor cannot prove from final source alone that all post-seal changes were evaluator-only. Sealed prediction bytes and their checksums are locked; a later read-only, prediction-only deterministic regeneration from committed code could close this provenance uncertainty. **Do not re-open labels to address it.**
- **Full-stream macro (equal class):** A2 BA/FPR/FNR `0.7362/0.1042/0.4235`; A3 `0.7324/0.0991/0.4360`; A4 `0.7162/0.0818/0.4859`. A2-A3 BA `+0.0037137`, FPR `+0.0050867`, FNR approximately `-0.0125142`; A2-A4 BA `+0.0199831`, FPR `+0.0223631`, FNR approximately `-0.0623292`. BA gains are consistent with the FNR-versus-FPR balance, not an improvement in score ranking (same score AUROC/AUPRC across arms).
- **Paired direction independently read from committed `paired_results.json`:** full-stream BA A2-A3 **17 positive / 3 negative** trajectories, A2-A4 **18 positive / 2 negative**; A2's FPR was **higher in all 20/20** trajectories against **both** lag arms. On fixed `t>=1050` horizon, BA gains in 16/20 and 19/20 respectively, with full details in the machine-readable paired artifact. Five seeds per class are correlated replications, not 20 independent classes.
- **Class heterogeneity:** full-stream A2-A3 BA differences by class: audiojack `-0.00020`, regulator `+0.00207`, switch `+0.00492`, rolled_strip_base `+0.00807`. A2-A4: `+0.00178/+0.01259/+0.02545/+0.04011` in that class order. A2 is not consistently better in every class against short lag.
- **Mechanism activation:** for A2-A3, quantile difference was negligible on **70.84%** of write events, post-update thresholds identical on **815/1190 (68.49%)** events, and predictions differed on **784/76,475** score events (~1.03%). For A2-A4: post-update thresholds identical on **274/1190 (23.03%)**, predictions differed on **3544/76,475** (~4.63%). Current-state C64 q90 and post-update threshold were, on average, **lower** than lagged counterparts. Lower thresholds increase defect recall but also false alarms; the experiments do **not** establish a matched-FPR improvement or a deployment-optimal risk tradeoff.
- **Near-write diagnostic:** next-image error-count differences after writes were `0` for A2-A3 and `-1` for A2-A4 across 1,190 events, so **no detectable immediate next-event gain** can be used as primary evidence; aggregate differences arise across subsequent events and operating-point trajectories.
- **Threshold of scientific evidence:** the runner's meaningful BA threshold `0.02` is its declared descriptive pre-evaluation criterion and is **not** a statistical significance threshold. A2-A4's observed macro BA difference `0.0199831` lies just below this value, whereas A2-A3 is far smaller. Do not round 0.0199831 to `0.0200` and imply a preregistered threshold was passed. More importantly, **both contrasts increase FPR**. Even a positive BA contrast cannot establish unqualified superiority.
- **Verdict:** `PASS_ENGINEERING_REVIEWED_WITH_LOCAL_PROVENANCE_LIMITS` on committed evidence; scientific interpretation `H1_WEAK_OR_MIXED`. **Hold the claim** that state consistency itself yields a large, risk-safe performance gain, a guaranteed calibration advantage, or reliable superiority over simpler lagged decisions. Retain negative/mixed results as genuine paper evidence; do not alter B3 or tune additional lag variants to rescue H1.
- **Recommended next gate (not authorized by this entry):** conduct a **read-only external-baseline fidelity and existing-results inventory** (B0) before any new method implementation or independent confirmation. Explicitly compare FPR/recall and source-native versus fallback methods, and consider whether a fixed operating constraint must be declared before additional empirical comparisons. The researcher makes the final paper GO/HOLD/NO-GO decision. Keep all work in the original `E:\Memory-Governed-Online-Few-Shot-Anomaly-Detection` workspace; no extra worktree or clone.

