# B8 hypothesis and mathematical research design — MTEAC (v0.1)

**Working ID:** `B8` — **Memory-Transported Evidence-Adaptive Calibration** (`MTEAC`).  
**Status:** `DESIGN_HYPOTHESIS_ONLY` / `NOT_IMPLEMENTED` / `NOT_FROZEN` / `NO_EXPERIMENT_APPROVAL`.  
**Researcher direction:** handle very long normal-only/defect-unavailable feedback intervals while native dynamic Greedy memory may change. Capture the mechanism and falsification plan before Stage 0.75-A; do NOT replace B7, B3, or the frozen Stage 1 method registry.

This specification proposes a new method; it is **not** a description of functionality present in the repository, an established source-paper algorithm, or a result. All tunable choices other than references to frozen B3/B7 defaults are **UNSET** pending Stage 0.75-A evidence and a separately versioned preregistration.

## 1. Problem and non-identifiability

At prequential event \(t\), the frozen VisionAD/DINOv2 scorer produces \(S_t=s(X_t;M_t)\) against one controller-independent native memory \(M_t\). Every controller predicts before the current query label is disclosed. The single writer admits only queried verified normals, using source-faithful Greedy with K=256 per layer. There are four distinct deployed normal support identities and an **additional fixed, protected 64-image C64 normal reference**; the latter is never written into mutable memory. Five support seeds (0–4) are preregistered; C64 identities remain fixed.

The physical-defect event \(Y_t\) is observed by a controller **only if the precommitted label-blind tape queries it**. A complete stream may have plenty of defects yet expose **no** defect labels for more than a thousand observations. The separate offline evaluator can see true labels only after score and prediction freeze.

**Identifiability:** the normal score distribution \(P(S_t\mid Y=0)\) and unlabeled marginal \(P(S_t)\) alone do not generally identify \(P(S_t\mid Y=1)\) or \(P(Y_t=1\mid S_t)\) without further assumptions. Under arbitrary class-prior drift and score overlap, no no-defect-label algorithm can truthfully guarantee FNR, Balanced Accuracy, or two-sided population risks. In particular, C64's empirical alert fraction is **not** guaranteed to equal on-stream FPR after memory adaptation or dataset shift.

B8's goal is narrower: preserve **bounded, explainable, current-memory normal-reference adaptation**, quantify its lack of defect-side information, and condition the transition to two-sided treatment on actual eligible defect evidence. It may improve BA empirically; it must not claim certified FPR/FNR.

## 2. Why B8 is scientifically distinct from B3 and B7 (testable, not presumed)

**B3 State-Consistent Quantile:** after valid memory changes, recalculate `higher q90` of C64 at current \(M\), move with \(\Delta=0.01\). It has a stable normal-side input and no defect-conditional objective. Source: `controllers.py::StateConsistentQuantile`.

**B7 SCDAC-G v0.1:** with eligible class evidence counts at least 16 normal / 8 queried defects, optimize a descriptive one-sided CP risk + movement objective with feasible/CONFLICT branches; otherwise HOLD. At 8 zero-error defect anchors the 95% descriptive FNR endpoint is ~0.312; at least 14 zero-error anchors are required for ≤0.20, but not sufficient for feasibility. B7's CP quantities are **descriptive** under selective feedback, repeated rescoring and memory dependence.

**B8 hypothesis:** a memory update can transform the *score coordinate system*. Quantify changes on the **same identified normal anchors** across consecutive memory states, use these *paired, source-controlled* transformations to generate a transported threshold, assess normal-tail shape and evidence reliability, and later select between bounded one-sided transport/quantile control and a two-sided objective. Such a controller may act while B7 HOLDs, without pretending to have FNR evidence.

**Falsifiable novelty requirement:** if B8's paired transport provides no measurable advantage over B3 with the same information, its additional complexity has not been justified. Calling a B3+B7 state switch "new" is insufficient without evidence of an independent mechanism.

## 3. Precise data and state ownership

Inputs allowed to the hypothetical B8:
- Current immutable native scalar score for an image **after** the shared scorer evaluates against pre-event \(M_t\).
- Native memory content version/digest and source map *after* an actual permitted Greedy write.
- Exact pairs of C64 scores under \(M^-\) and \(M^+\) for the **same 64 C64 image identities**. C64 is protected/read-only, independent of controllers; frozen features may be cached with strong provenance.
- Additional *queried verified-normal* anchors, **only** when legally source-aware LSO and current-state rescoring make their scores valid; keeping C64-only as initial primary isolates the proposed mechanism.
- Queried verified defect anchor scores under the current content version, after lawful rescoring; defect images must never enter normal Greedy memory.
- A precommitted, controller-independent query tape; fixed \(\theta_0\), movement constraints, and evidence state.

Inputs forbidden:
- Any unqueried physical labels, defect mask, future image, future label, per-category hidden prevalence, confirmation outcome, oracle threshold, or feedback inferred from a threshold's own predictions.
- Any threshold-driven change to the shared normal writer, query decisions, memory proposals, feature backbone, or native score equation.
- Any empirical test-label calibration of B8 weights or state switches during the online run.

State schema to specify before implementation: last committed \(\theta\), memory digest/content version, last valid C64 score vector with per-anchor identities, paired-score shift summaries, last reliable transport state, valid normal/defect reference ledgers and provenance, evidence revision counter, active regime, last update reason, and checkpoint fingerprint. A merely re-queried score under the *same* memory is not new independent normal evidence.

## 4. Paired normal-reference score transport (central hypothesis)

Let \(Z=(z_1,\ldots,z_{64})\) be the same protected C64 images. For a true shared memory content update \(M^- \to M^+\), define
\[
c_i^-=s(z_i;M^-),\qquad c_i^+=s(z_i;M^+),\qquad d_i=c_i^+-c_i^- .
\]
Paired scoring is causal at the update boundary: \(c_i^-\) is frozen from the old memory content version (or recomputed read-only), \(c_i^+\) after the permitted update. The updated threshold is **for the following event only**. The normal labels of C64 are already known from its training-normal selection and give no new test-label access.

Define a robust translation estimate and heterogeneity:
\[
m=\operatorname{median}_{i=1}^{64} d_i,\qquad
H=Q_{0.75}(d_1,\dots,d_{64})-Q_{0.25}(d_1,\dots,d_{64}) .
\]
Quantile interpolation (including even-\(n\) median convention, ties and lower/upper boundaries) must be versioned and independently tested; using the `higher` convention here is a candidate, **not yet frozen**.

A *translation-only* threshold candidate is
\[
\theta^T = \theta^-+m .
\]
A *direct current-state normal quantile* candidate, matching B3's nominal target, is
\[
\theta^Q = Q_{0.90}^{\mathrm{higher}}(c_1^+,\dots,c_{64}^+) .
\]
These are conceptually different: \(\theta^T\) attempts to preserve a prior threshold under a coordinate translation; \(\theta^Q\) targets a current normal-tail quantile. They coincide under special conditions but not generally. If \(\theta^-\) already contains defect-informed adjustments, transporting it may preserve those adjustments; this is a possible advantage **or** a source of outdated risk. Track regime and threshold provenance explicitly.

Propose a *diagnostic-only* robust scale
\[
\sigma = \max\{Q_{0.75}(c^+) - Q_{0.25}(c^+),\,\epsilon\},
\qquad D = |\theta^T-\theta^Q|/\sigma .
\]
The arbitrary floor \(\epsilon>0\) and any meaning assigned to \(H/\sigma\) or \(D\) must be registered before testing; zero-IQR and ties require fail-safe handling.

**Examples (hypotheses, not measurements):**
- Pure shift \(c_i^+=c_i^-+\delta\): \(d_i=\delta\), \(H=0\), both candidates translate by \(\delta\) if \(\theta^-=Q_{.90}(c^-)\).
- Unequal tail changes: \(m\) small but \(\theta^Q\) moves; translating by \(m\) can miss a changed normal tail.
- Memory unchanged: \(c^+=c^-\) and no new trusted evidence; do not update the threshold based on imaginary information.

A simple affine/quantile mapping could be a later, explicitly preregistered alternative *only if* the scalar translation is falsified; do not add complexity without evidence.

## 5. Normal-only candidate objective (not yet frozen)

For defect-unavailable periods, the controller is allowed to optimize a **normal-reference** objective, not two-sided FNR. For candidate threshold \(\theta\), let
\[
\widehat R_{N}(\theta;M^+)
=\frac{1}{64}\sum_{i=1}^{64}\mathbf1[c_i^+\ge\theta] .
\]
At \(\alpha_N=0.10\) (the existing B3 q90 *reference*), a proposed diagnostic optimization is
\[
\begin{aligned}
\theta^*=\arg\min_{\theta\in\Theta}\bigg[
&\lambda_N\big(\widehat R_N(\theta;M^+)-\alpha_N\big)^2\\
+&\lambda_T\big((\theta-\theta^T)/\sigma\big)^2\\
+&\lambda_M\big((\theta-\theta^-)/\sigma\big)^2
\bigg].
\end{aligned}
\]
**Unspecified choices:** \(\lambda_N,\lambda_T,\lambda_M\), candidate set \(\Theta\), quantile convention for median/IQR, scale \(\epsilon\), disagreement gating, tie breaking and confidence/uncertainty policy. They are *research degrees of freedom*, **not instantiated B8 defaults**. Avoid using these degrees of freedom to fit the exposed four development labels without a new decision/provenance ledger.

The indicator means the normal reference objective is discontinuous in \(\theta\); test candidate enumeration exactly, rather than assuming a smooth optimizer. With \(\mathrm{predict\_defect}=\mathbf 1[S_t\ge\theta]\), equal scores at the threshold count as **normal false alarms**, and q90 is not necessarily exactly 10% false-alarm rate because of ties and finite \(|C64|=64\).

Bound any executed motion, e.g. candidate \(|\theta^+-\theta^-|\le\Delta\) with the existing B3/B7 \(\Delta=0.01\) as a **comparative starting point only**, not a frozen new B8 value. Record if clipping takes an originally feasible candidate outside a proposed normal-side set, and distinguish proposed vs executed feasibility.

Potential alternative hypothesis: adapt the weight on \(\theta^T\) versus \(\theta^Q\) based on observed paired translation heterogeneity/disagreement and *reference* stability; any learned gate must have a label-free declared update and be tested for self-reinforcement. A gate cannot infer true defect-side risk from these normal shifts.

## 6. Evidence ladder and finite-state behavior

Candidate regimes to be specified precisely before implementation:

| State | Legal current evidence | Proposed behavior | Non-claim |
|---|---|---|---|
| `NORMAL_ONLY` | 64 current C64 normal reference scores; zero/insufficient defects; real memory content revisions may occur | bounded transport/normal-tail candidate or HOLD based on normal-reference reliability | no defect FNR estimate or two-sided certificate |
| `DEFECT_SCARCE` | current C64 and 1–7 eligible queried defect anchors | keep bounded normal-side control; monitor defect score ordering and evidence age | cannot treat 1–7 observed defects as certified class risk |
| `BOTH` | ≥16 valid normals and ≥8 *current-memory* eligible defect anchors | optionally hand off to B7-like two-sided descriptive optimization, subject to preregistered constraints and explicit threshold provenance | CP bound remains descriptive; 8 not equivalent to guaranteed ≤0.20 FNR |
| `CONFLICT` | sufficient counts but no threshold satisfying descriptive bilateral requirements | explicit bounded, audited risk compromise or conservative HOLD; compare both *as separate variants* if chosen | no claim of meeting violated constraints |
| `STALE_EVIDENCE` | source-map version/anchor rescoring mismatch; missing valid C64 pairs or invalid defect refresh | fail closed; update evidence first or maintain current threshold; never silently reuse stale labels' old scores | stale anchors must not be treated as current |

A normal-only reference is generally available from protected C64 even when *no new queried normal* is received; however if the **memory content does not change**, rescore data are identical and there may be no legitimate reason to move \(\theta\). This matters for a long defect-only/no-normal admission stretch.

Old queried defect labels may remain *known* and eligible after long gaps; count them separately from truly absent labels, and distinguish their age/freshness from **current-memory** rescoring status. A memory-state refresh fixes score-coordinate staleness, not automatically real-world concept staleness.

Proposed transitions are triggered only by pre-event memory/evidence revisions and legal post-query feedback. Never update current event's past prediction. Keep transition metadata, counts, sign of threshold motion and last evidence revision in the audit.

## 7. Information firewalls, grouped identity and confounding

- Common score and memory trajectory: all calibrators receive the same \(S_t,M_t\), anchor refresh and query tape; a B8 threshold cannot alter Greedy admissions, controller-independent theta_ref or future labels. The sole writer updates only from **queried verified normal** events.
- Five-view same-group reuse: when a normal queried view was admitted, later views of the **same** official group can receive artificially improved scores. Report and/or exclude such overlap in independent ranking diagnostics. For queried-normal anchors contributing patches, use the current all-layer source-aware leave-source-out semantics; protected C64 remains disjoint from supports and writes.
- C64 is a fixed train-normal population and does not necessarily represent current test normals. Normal-tail control cannot by itself assert real-time production FPR.
- Preserve exact feature source, score-map aggregation, K=256/layer bank, fixed support and augmentation settings; no proposal uses gradients or modifies the frozen backbone.
- \(\Delta\) movements must be applied once per genuine evidence revision with deterministic checkpoint/resume, not once every observation on unchanged data.
- A deficit in defect feedback is not filled by pseudo-labeling high scores or self-consistency: this would create the same self-reinforcing loop B8 aims to avoid.

## 8. Required ablation and falsification

| Named method/ablation | Memory and information permissions | Scientific question |
|---|---|---|
| B0 Frozen | common score/memory, fixed \(\theta_0\) | Is no adaptation already sufficient? |
| B3 SCQ | same normal C64, simple current-state q90 with step clip | Can a standard normal-only controller explain the entire gain? |
| B7 SCDAC-G v0.1 | same legally queried labels and shared memory; HOLD on missing defect side | What does fully conditional bilateral control add? |
| B8-T | C64 paired median translation, no defect input | Does score-coordinate transport add value beyond direct q90? |
| B8-N | paired transport + explicit normal-side objective, no defect input | Does constrained shape-aware normal calibration reduce drift errors? |
| B8-Full | B8-N + evidence-ladder two-sided transitions with legal queried defects | Does the evidence-dependent controller provide incremental benefit? |

No ablation may receive a different memory trajectory or extra C64/reference images. Score AUROC/AUPRC must be **identical across threshold-only arms** sharing \(S_t\). Recompute and report BA/FPR/FNR where both labels exist, and **coverage of evidence states**, not only headline BA.

Primary desired outcomes are **incremental gains above B3** at equal information/cost, reduced worst-window failure and delayed-feedback recovery without unacceptable FNR harm. The required negative controls:
1. **No memory change, no new trusted evidence:** B8 must not oscillate.
2. **Pure score translation:** transport should track by construction; check no future leakage.
3. **Heterogeneous normal-tail shape change:** a pure median translation may be falsified.
4. **Arbitrarily shifted defect distribution with unchanged normals:** B8 must report the impossibility of knowing FNR; no fictitious guarantee.
5. **Long normal-only prefix with changing memory:** B8 should remain operationally defined, and compare against B7 HOLD and B3 q90.
6. **Long defect-only/no-normal admission:** no memory-induced shift should be invented.
7. **Defect label returns after long delay:** transition and old-anchor rescore must be causal, limited-motion and version-consistent.
8. **CONFLICT/insufficient precision:** distinguish statistical uncertainty, score-overlap infeasibility, and evidence staleness.

**Early falsification rule:** if after controlled paired tests B8-T and B8-N do not offer a repeatable advantage over B3, do not claim independent B8 novelty from their mere combination with B7; reject/simplify or pursue a different source of information.

## 9. Evaluation design and statistical safeguards (future authorization required)

Stage 0.75-A (Greedy ranking + C64 paired transport *without B8 controller*) precedes B8 hyperparameter choice. Details: [Stage 0.75 agenda](STAGE075_RESEARCH_AGENDA_v0_1.md).

Any eventual 0.75-B implementation must pass synthetic controller trace parity, source-shift/no-shift tests, predictable output contracts, unit tests for median/q90/ties, state transition and audit/replay determinism, no label leakage, source/group exclusion and executable step-clip checks. Any 0.75-C dev comparison must freeze parameter choices first, preserve the matched Q-A-TS 3% tape and report all five support seeds/class outcomes, not select seeds using observed performance.

Classes Audiojack/Regulator/Switch/Rolled Strip Base are **already exposed exploratory development sets**, so hypotheses motivated by their label timing, AUROC and headroom are **not blind**. No performance tuning or validation on Mint/PCB until a future explicit authorization after complete method freeze. No error bars treating images/views as independent units.

**Success is not synonymous with SOTA BA.** A valid outcome might be: B8 offers a measurable early no-defect-label benefit but fails under later defect burst; or B3 dominates and B8 is rejected. The paper must distinguish verified mechanisms, bounded/diagnostic risks, empirical deployment behavior, and missing theoretical guarantees.

## 10. Theoretical support and gaps

Foundational ideas, **not validation/novelty proof for this particular design**:
- PatchCore (Roth et al., CVPR 2022): https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html (normal patch memory; not our paired transport).
- Adaptive Conformal Inference (Gibbs & Candès, NeurIPS 2021): https://proceedings.neurips.cc/paper_files/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html (label-feedback online calibration; source guarantee cannot be transferred).
- Concept drift survey (Gama et al., ACM CS 2014): https://doi.org/10.1145/2523813 (unlabeled versus conditional drift distinction).
- Failing Loudly (Rabanser et al., NeurIPS 2019): https://proceedings.neurips.cc/paper/2019/hash/846c260d715e5b854ffad5f70a516c88-Abstract.html (distribution shift can be detected without identifying unseen physical defect labels).
- Sun-CS (Sun et al., ICML 2024): https://proceedings.mlr.press/v235/sun24h.html (different quantile-anomaly target and assumptions).
- Internal implementation: B3/B7 controllers and `native.py`, `engine.py`; Stage 0.5 stats and historical P0/P1 report.

**Open theory questions:** stability/robustness of paired median transport under nonuniform normal shift; finite-64 empirical quantile/tie behavior; repeated adaptive reuse of C64 and conditional calibration; nonexchangeability of rescores; selective-label bias in CP; identifiability limits for defect risk; computational cost of current-M rescoring; controller state switching and regret/BA attribution. Any formal guarantee requires its own assumptions and proof. No free-energy/Active Inference optimality or conformal coverage theorem is claimed for B8 by analogy.

## 11. Decisions intentionally deferred

1. Is paired normal-reference transport stable and predictive of score-coordinate shift under actual native Greedy updates? Await 0.75-A.
2. Are C64 shifts nearly uniform, heterogeneous or source-group dependent? Await 0.75-A.
3. How should \(\lambda\) weights, scale, gating, conflict response, update frequency, tie behavior, movement budget and initial regimes be fixed? **Not set.**
4. Does the full evidence-ladder add measurable value beyond B3 under fair equal-memory tests? Await 0.75-C.
5. Should B8 be a standalone method, a restricted B7 variant, or dropped? Outcome-dependent research decision after preregistered evidence.

**Stop here**. This document authorizes no B8 implementation, no 0.75-A/GPU run and no Stage 1 or Stage 2 execution.
