# Stage 0.75 research agenda — delayed feedback and native Greedy ranking falsification (v0.1)

**Status: RESEARCH DESIGN / DOCUMENTATION ONLY — NO RUN AUTHORIZATION.**  
**Recorded:** 2026-10-29. **Parent evidence:** Stage 0.5 exploratory commit `fe692a855f283c37fc6919d3e739058eb3248757`.  
**Researcher decision:** investigate B8 and native Greedy ranking *before* comprehensive Stage 1. Preserve Stage 1/Stage 2 HOLD and untouched Mint/PCB. This record is an explicit design hypothesis informed by exposed development outcomes, **not** an independent preregistration.

## 1. Research questions and decision boundary

RQ-A: Can a causal threshold controller continue *limited, interpretable adaptation* using protected normal references and changed memory state when queried defect labels are absent for hundreds or thousands of observations, without pretending to know class-conditional FNR?

RQ-B: Does verified-normal-only native Greedy memory change **normal-versus-defect ranking** rather than merely shift the scores? Are apparent gains robust to chronology, support seed, source-group contamination, and consistent threshold recalibration?

RQ-C: Is an explicit **paired-reference memory transport** mechanism measurably useful beyond B3's current-memory C64 q90 and B7's two-sided conditional controller? Or is B3 already the simplest sufficient solution?

Stage 0.75 consists of (A) ranking/transport diagnostics, (B) B8 mathematical specification and synthetic falsification, and only then (C) authorized development-stream comparison. **Only this written agenda is authorized now**. The researcher will separately request the 0.75-A implementation/execution prompt. No B8 code, native Greedy test, Stage 1/2 task, or data read is authorized by this file.

## 2. Existing facts; boundaries of inference

### 2.1 Stage 0.5 — exposed Real-IAD development results (observations, not B8 evidence)

Sources: [Stage 0.5 report](STAGE05_EXPLORATORY_STATISTICAL_REPORT.md), [exposure ledger](STAGE05_DATA_EXPOSURE_LEDGER.md), and [aggregate audit](audit/stage05_exploratory_v1/README.md). Four **exposed development** classes; images are evaluated in official JSON record order, not demonstrated physical production chronology. Each official sample group has five views. Neither groups nor views may be treated as independent physical parts without evidence.

| Development class | 3% Q-A-TS seed-37 queried normal / defect | First queried defect | First 8 defects | First 14 defects | Static M0 AUROC, five-support-seed mean ± SD | Static M0 Frozen BA | Retrospective best *single fixed* BA |
|---|---:|---:|---:|---:|---:|---:|---:|
| Audiojack | 74 / 40 | 1305 | 1732 | 2023 | 0.6226 ± 0.0470 | 0.5488 | 0.5902 |
| Regulator | 81 / 35 | 116 | 1007 | 1504 | 0.6255 ± 0.0568 | 0.5349 | 0.6134 |
| Switch | 54 / 61 | 55 | 375 | 631 | 0.7353 ± 0.0566 | 0.6474 | 0.6701 |
| Rolled Strip Base | 33 / 81 | 35 | 297 | 501 | 0.9109 ± 0.0127 | 0.8233 | 0.8294 |

- All **200 fixed label-blind query schedules per class/rate** at 3%, 5%, and 7% obtained at least 8 and 14 queried defects by the end of their streams, and at least 16 queried normals. This is only an observed finite-schedule frequency, *not* a general probability or risk feasibility result.
- Audiojack's **first 400 events contained zero queried defects in all 200 sampled schedules** at 3%. In seed 37 the first queried defect was at t=1305; 3% B7 could not reach its nominal 8-defect gate until t=1732. Across the 200 schedules the median Audiojack first-14 event was t=2186 (P05≈1925, P95≈2470).
- The 3% budget is 459 images across four classes. The 5%/7% *nested exploratory* tapes are separate diagnostics; they are **not Stage 1 arms** without later approval.
- Near-ceiling screening (Frozen BA ≥0.95 and fixed-oracle headroom ≤0.01) was 0/20 configurations; this rejects the narrow fear that M0 was near-perfect *in these development configurations*. It does **not** show that dynamic-memory calibration must succeed.
- Three classes' static, full-stream, **single-threshold** Oracle BA was <0.80 across all five support seeds. Under *these frozen scores and a single global threshold*, simultaneous empirical FPR≤0.20 and FNR≤0.20 is impossible, since BA=1−(FPR+FNR)/2. This does **not** establish infeasibility for a time-varying threshold, adaptive memory, or B7's selective current-state anchor objective.
- Long one-class windows and 6–16-image false-negative runs occur in the static M0 audit. Window BA is undefined when the window contains only one true class.
- The Stage 0.5 summary did not activate dynamic Greedy, execute threshold controllers, inspect defect masks, or prove any online gain. Tests and exact scope are in the pinned audit.

### 2.2 Historical P0/P1 (narrow mechanism evidence; not direct Stage 1 comparison)

From [P0/P1 falsification protocol](../../docs/P01_FALSIFICATION_2026-10-28.md) and historical `outputs/p01_falsification_20261028/.../p01_summary.json`:
- One Audiojack seed, a 240-image **Probe-A** retrospective final-memory comparison, historical query budget 128 (different from Q-A-TS 3%); fixed static bank AUROC 0.6898 and BA 0.6225.
- The historical Greedy-all final memory produced AUROC 0.7777; its **stale fixed threshold** had BA 0.5566, while separately C64-q90-recalibrated BA was 0.6972.
- This motivates testing ranking versus operating-point calibration but does **not** imply multi-class, multi-seed, causal prequential superiority. Other historical selection policies were not all deployable online and should not be conflated with this Stage 0.75 writer.

### 2.3 Frozen existing implementation, not to be rewritten

- B3 current-M C64 higher-q90 with movement clip Δ=0.01: `experiments/calibration/controllers.py::StateConsistentQuantile`. Uses existing C64 current-memory scores; no requirement to collect defect labels. FNR unavailable if no defect anchor.
- B7 `experiments/calibration/scdac_g.py::SCDACGController` requires at least 16 eligible normal references and 8 queried defect anchors before its BOTH/CONFLICT objective. A 95% one-sided descriptive Clopper–Pearson FNR upper bound with zero errors needs ≥14 defects for ≤0.20; this is *not* an online risk certificate under selective labels and changing memory. On missing defect-side evidence B7 holds.
- Native/Greedy integration: `experiments/calibration/native.py`, `engine.py`, `native_event_replay.py`. Four normal supports plus 64 **additional protected** C64 normals; native v22/DINOv2 scoring; Greedy K=256 **per layer**; initially empty dynamic bank. The common writer is label-blind until queried feedback, writes only verified queried normals, and does not use any controller's threshold.
- Support seeds (0,1,2,3,4) with the **same** 64 C64 identities/views, Q-A-TS 3% seed37; `docs/calibration/STAGE1_EVALUATION_PROTOCOL_APPROVED_DESIGN_v3.md` remains the existing design baseline. Snapshot/replay experiments must keep their separate profiles and never modify this baseline.

## 3. Why these are separate problems

**Feedback count**: all tested streams eventually provided enough labels, so total count is not the main general obstacle *for these four classes*. **Feedback arrival**: before the first queried defect, current defect conditional risk cannot be identified even if memory changes. **Score separability**: if class-conditional scores overlap, no scalar static threshold can completely fix ranking. **Memory shift**: dynamic normal updates can improve rankings but move the operating score scale so that a stale threshold misses defects.

The JSON order is an accepted deterministic adversarial/deployment *test sequence*, whether or not it is acquisition chronology. Robustness across orderings is a research goal. Do not attribute JSON-order label blocks to physical production drift without independent timestamp evidence. An experiment may later include predeclared order stress, but the construction of label-aware stress sequences must be disclosed, with query tape generated causally using metadata of the resulting order and never choosing labels online.

## 4. Stage 0.75-A — paired native Greedy ranking falsification (NOT YET AUTHORIZED TO EXECUTE)

### A1. Central estimands

For each allowed class/support-seed/query-profile, compare at the same observation `x_t`:

`S_t^0=s(x_t;M_0)` (static frozen initial native bank) versus  
`S_t^G=s(x_t;M_t)` (dynamic verified-normal Greedy state *before* current feedback).

Primary paired full-stream diagnostic: `AUROC(S^G,Y) − AUROC(S^0,Y)`; include AUPRC, normal/defect score quantiles and shifts, full-stream operating confusion at frozen theta0, and current-M C64 q90 as **read-only calibration diagnostic**. This evaluates time-varying scores under a predetermined stream, not an unconditional static representation theorem.

Since an image's own queried normal may later enter memory, predictions must always be *before* that event's memory write. Record actual native append/replace/skip, layer prototype usage, memory content digest/version, source-map identities, and event-specific score-state digest. Any difference in score ordering must trace to a changed memory state, not changing the backbone, support bank, query tape, or current threshold.

**Critically, no B8 or B7 controller may control the writer**. The existing independent writer uses precommitted theta_ref/sigma_ref; record these and preserve exact native Greedy parity.

### A2. Stronger equal-cohort snapshot design

Full-stream AUROC compares scores from changing \(M_t\) and can confound score ranking with temporal label composition. At predeclared checkpoints (proposal: t=0,400,800,1600,2400,3200), freeze the Greedy bank after events strictly before t. Compare `M_0` and `M_t` on **the same future, non-admitted group-disjoint evaluation cohort**. No future observation or label may affect the snapshot.

Requirements:
- Preregister future evaluation horizons (e.g. fixed-length future window and, separately, available remainder) *before* inspecting this experiment's dynamic outcomes. For each snapshot, use an identical eligible set for static and dynamic scores. Counts, class composition and undefined metrics must be reported; no window is dropped silently because of unfavorable BA.
- Exclude any evaluation official sample group whose member contributed to the memory bank, for the clean primary snapshot analysis; also report fraction and identities in a protected private ledger. For source-overlapping diagnostics, source-aware leave-source-out is mandatory. Prefer group-disjoint snapshot scoring as primary evidence.
- Do not confuse future-only evaluation of an earlier frozen snapshot with online prospective performance: later labels enter the evaluator only, after scored arrays are frozen.
- Snapshot BA oracle is an in-sample static-threshold diagnostic on a *fixed scorer and fixed cohort*, not a deployable online control or upper bound for time-varying scoring.
- Include matched supports, group identity/exclusion, same score convention, AUROC/AUPRC, post-update score shifts and anchor drift. Record any class-missing window as AUROC/BA undefined.

### A3. Essential five-view source-group leakage control

Real-IAD has 5 camera views per official sample group; a **3% image query can cover ≈15% of groups**. A verified normal camera view that enters memory can make a later same-group view artificially easy. Merely requiring "predict before *current image* feedback" does NOT exclude cross-view same-group reuse.

For A1 and A2 report paired dynamic-vs-static metrics:
1. **As-operated causal prequential** (all observations, existing single writer and protocol).
2. **Group-clean diagnostic**: restrict evaluation to observations whose official sample group has not supplied any earlier admitted memory prototypes at the scoring event. Preserve the same masks/denominators for both static and dynamic scores; do not use this filter to decide online writes.
3. **Explicit contaminated-stratum diagnostic**: separately report later same-group views after prior admission. Do not claim that this stratum's gain transfers to unseen products.
4. Optionally, a separately named **group-deferred write sensitivity** may be proposed later, but it changes the writer and is NOT part of the source-parity primary experiment without approval.

Official sample keys are verified *grouping* units, not proven unique physical manufactured parts; further serial-level correlation sensitivity may be warranted, but do not invent physical independence.

### A4. Phased run, reproducibility and failure outcomes

Suggested planned sequence, subject to a future Codex authorization:
- Run a bounded native parity/feature-cache test and **one predeclared Seed-0 smoke** per development class to expose engineering faults, not cherry-pick class/seed results.
- Execute a precommitted **all five seeds × four development classes**, Q-A-TS 3% seed 37 in separate isolated runs, if approved and feasible; cache immutable frozen feature encoding only with verified config/source fingerprints. 5%/7% are not automatically authorized dynamic-memory experimental arms.
- Snapshot diagnostics and C64 paired differences must re-use the same shared memory replay, never train a new memory policy per proposed calibrator.
- Report paired category × seed point estimates, aggregate by category first (not 5-view pseudo-replicates), time of memory events, normal/defect means/quantiles and AUROC/AUPRC, retrospective static-oracle BA per snapshot as secondary, and runtime. Retain all negative results.
- **Falsification**: negligible/mixed \(\Delta\mathrm{AUROC}\) across classes and group-clean snapshots weakens the thesis that this locked Greedy writer improves ranking; apparent gains confined to same-group overlap indicate possible contamination; improvement in AUROC combined with stale-threshold FNR deterioration strengthens the *need* to study state-consistent calibration. Do not set empirical success thresholds post hoc.

### A5. Paired C64 score-transport audit (no B8 controller yet)

At **every actual memory content change**, score exactly the same 64 protected C64 anchors against \(M^-\) and \(M^+\), caching frozen backbone features where valid; record pairs and content versions. Let \(d_{i,t}=s(z_i;M^+)−s(z_i;M^-)\). Report:
- \(m_t=\mathrm{median}_i d_{i,t}\), \(H_t=Q_{.75}(d_t)-Q_{.25}(d_t)\); using `higher` or another **declared** quantile convention for IQR.
- \(\theta_t^{T}=\theta_{t-1}+m_t\) and \(\theta_t^Q=Q_{.90}(\{s(z_i;M^+)\})\), **diagnostic targets only**, not enacted decisions.
- normalized discrepancy \(|\theta_t^T-\theta_t^Q|/\sigma_t\), C64 quantile gap, rank reordering, and conditional effects by append/replace/size/source.
- test whether nearly uniform shifts differ in behavior from heterogeneous remappings, whether the simple median transport is a better threshold predictor than B3 q90, and whether memory operations materially alter the normal reference tails.

Important: large or small C64 shift **does not prove** how defect scores shifted; it is evidence about the protected *normal* reference coordinate system only.

## 5. Stage 0.75-B — B8 mathematical design and synthetic falsification (NOT YET AUTHORIZED)

See [B8 method hypothesis/specification](STAGE075_B8_MTEAC_RESEARCH_DESIGN_v0_1.md). Candidate working name **B8 — Memory-Transported Evidence-Adaptive Calibration (MTEAC)**. It is not a source-paper algorithm and not an already accepted method. Design stages:
1. Compare B3 q90 and pure score transport under repeatable synthetic score transformations with stationary normal-only feedback.
2. Introduce a bounded normal-reference objective and evidence-state machine; verify behavior through artificially delayed/missing defect labels, stale anchors, abrupt changes and recovery.
3. Once verified, introduce two-sided branch with B7-inspired descriptive risk only when valid evidence exists; no claim of class-conditional risk without defect feedback.
4. Regime/transport/step-penalty parameters must be justified and preregistered separately *before* empirical B8 comparison. Preserve original B3/B7 and write B8 in calibration-owned namespace only.

## 6. Stage 0.75-C — later *separate* development stream comparison (NOT YET AUTHORIZED)

If 0.75-A shows credible Greedy ranking improvement *or* a useful dynamic-memory stale-threshold failure, and B8 passes theoretical/synthetic checks, conduct a separately approved comparison on exactly the **same** shared score/memory/query trajectory: B0, B3, B7, B8-Transport, B8-Normal, B8-Full. Additional B1–B6 external comparators remain part of the Stage 1 architecture but are not silently dropped from eventual reporting.

Pre-register:
- paired full-stream BA/FPR/FNR (defined denominators), AUROC/AUPRC score invariance, class-wise and group-clean metrics, worst defined 256-image windows, first-defect/first-eight-defect phase metrics, post-feedback recovery and threshold trajectory;
- explicit NORMAL_ONLY/BOTH/CONFLICT/STALE states and count/time spent, cost in queries and rescoring operations, distinct evidence/actual-write versions, CPU/GPU latency and memory;
- phase-stratified metrics with missing class labels marked unavailable rather than fabricating zeros;
- threshold-only controller identity: **AUROC must be identical across controllers sharing one score tape**;
- B8-Full versus B3, B7 and two mechanism ablations; failure to outperform B3 is a valid falsification of claimed incremental novelty.

Stage 0.75-C will be clearly **development-data exploratory**. Stage 0.5 already exposed test labels for these categories. Only untouched confirmation data may later support independent out-of-sample claims after full version freeze.

## 7. Extreme-arrival cases and no-information limits

Proposed later stress profiles (must be versioned separately):
- **S1: no defect feedback for a long prefix** while verified normals may update Greedy. No FNR identifiability: normal-only transport/calibration may help or hurt defect recall.
- **S2: defect-only or no verified normal admissions**: no Greedy state change may occur, so paired C64 values are unchanged; B8 cannot invent adaptation signal or guarantee improved defect classification. Unlabeled drift monitoring is only an alarm, not an inferred physical defect label.
- **S3: normal block → defect burst → normal recovery**, emphasizing delayed evidence, contamination/stale-state prevention, threshold oscillation and recovery.
- **S4: memory-shift controls**, synthetic pure translation, dispersion shift, heterogeneous/local rank reordering, and stable memory with label-conditional shift invisible to C64.

Default JSON order is not assumed physical chronology. Counterfactual order stress is generated offline and may depend on *exposed dev labels*; document it as label-aware construction and keep controllers/query decisions label-blind at runtime. No method can guarantee FNR under arbitrary shifts of \(P(Y\mid S)\) without class labels or additional assumptions. This is an identifiability limit, not a controller bug.

## 8. Research decision matrix / stage gates

| Future evidence | What it would support | Next research action (requires fresh approval) |
|---|---|---|
| Greedy improves group-clean future-cohort AUROC but frozen threshold worsens FNR | Native dynamic memory helps ranking yet breaks operating-point calibration | prioritize B3/B8 state-consistent mechanism |
| Greedy improves only after same-group previous-view admission | leakage or narrow memorization rather than unseen-group transfer | address evaluation/source exclusion; do not claim general ranking gain |
| Greedy ranking remains weak in Audiojack/Regulator/Switch | low ceiling for *pure threshold* proposals given these scores | review memory retrieval/representation separately; no unjustified tuning of frozen scorer |
| Memory changes approximately translate C64 score coordinates | transport hypothesis plausible but not verified for defects | test B8-T vs B3 with equal information and costs |
| C64 shifts are highly heterogeneous | scalar transport insufficient | consider shape-aware normal quantile mapping; preregister before evaluating |
| B8 Normal-only improves early windows but worsens FNR after defect burst | failure of asymmetric adaptation | test explicit movement/uncertainty restraint; retain negative result |
| B8-Full adds no gain above B3 | hierarchical complexity not justified | reject or simplify B8; preserve baseline |
| B8 adaptation emerges even without memory or new trusted evidence | likely artifact/leakage or unsupported self-training | reject state transition and audit input contract |

**Hard boundary:** no Stage 1 / Stage 2 GO, no Mint/PCB outcomes, no defect masks, no new query budget, no controller or protected-source edits, no Stage 0.75 experiment from this documentation commit. Any new 0.75-A prompt requires explicit research authorization and a separate audit namespace.

## 9. Related-work anchors and attribution limits

These are **literature motivation**, not proofs of B8 correctness, performance, or novelty:
- Roth et al., *Towards Total Recall in Industrial Anomaly Detection* (PatchCore), CVPR 2022. https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html — normal patch memory and representative coreset.
- Gibbs & Candès, *Adaptive Conformal Inference Under Distribution Shift*, NeurIPS 2021. https://proceedings.neurips.cc/paper_files/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html — online calibration under different feedback/assumptions; cannot transfer label-aware conformal guarantees to our no-defect-feedback regime.
- Rabanser et al., *Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift*, NeurIPS 2019. https://proceedings.neurips.cc/paper/2019/hash/846c260d715e5b854ffad5f70a516c88-Abstract.html — detecting distribution differences is not equivalent to recovering missing physical labels.
- Gama et al., *A Survey on Concept Drift Adaptation*, ACM Computing Surveys 2014. https://doi.org/10.1145/2523813 — concept-drift framing.
- Sun et al., *Online Adaptive Anomaly Thresholding with Confidence Sequences*, ICML 2024. https://proceedings.mlr.press/v235/sun24h.html — unsupervised quantile thresholding; original task/assumptions differ from Real-IAD physical-defect risk.
- Internal sources: Stage 0/0.5 contracts, native scorer/Greedy source, B3/B7 code, P0/P1 final-memory audit.

No claim that any cited paper already proposes B8's paired-C64 transport + evidence ladder. An eventual novelty claim requires a dedicated prior-art review.

## 10. Record integrity and next authorization

This agenda is design-only, authored after exposure to the four development streams; it does not change the previously reviewed Stage 1 design or frozen test history. The next action, if the researcher requests it, is a **separate, scoped 0.75-A Codex prompt** for paired Greedy ranking/transport diagnostics. Stage 1 remains HOLD until explicitly authorized after further review; Stage 2 confirmation remains sealed.
