# D1-C CDARG: design fidelity, scope and causal safeguards

**Date:** 2026-10-28 (Australia/Sydney). **Status:** design/implementation guardrails only; not a revision of the frozen D1-B v1 protocol and not a claim of CDARG implementation or experimental success.

## Authority and source hierarchy

The primary technical basis for **Counterfactual Dual-Anchor Risk Gate (CDARG)** is the project document **"Risk-Safe Memory Admission for v22: A Deployment-Oriented Research Strategy"**, especially "Priority A — Counterfactual Dual-Anchor Risk Gate", "Integration with v22 and practical computation", and "Validation protocol against ContCore, FOADS, and HITL". Cross-check with the project's "Risk-Safe Memory Admission-7DAY-PLANS.docx" and the currently frozen D1-B protocol. Do not replace the proposal's mechanism with a different novelty-only or class-prior-adjustment algorithm.

External references (for motivation/comparisons, not plug-in formulas or established guarantees):
- HITL Memory Correction, https://arxiv.org/abs/2608.17775 — training-free verified-label bank editing and novelty gate; note its review-pool calibration is not our strict prospective sparse-feedback setting.
- ContCore, https://arxiv.org/abs/2608.15277 and https://github.com/jungyg/ContCore — fixed-capacity greedy representation and replacement, not a verified-defect FNR-risk gate.
- FOADS, https://arxiv.org/abs/2403.18201 — neural-gas few-shot adaptation using mixed streaming observations; different label-access assumptions.
- Conservative Contextual Linear Bandits, https://papers.nips.cc/paper_files/paper/2017/hash/bdc4626aa1d1df8e14d80d345b2a442d-Abstract.html — baseline-relative safety as a *conceptual* motivation; its theorems do not imply finite-sample safety here.
- Learn-then-Test, https://arxiv.org/abs/2110.01052 — offline calibration reference; no automatic guarantee under adaptive feedback/querying.

## 1. The primary research question is safety of verified-normal memory admission

**Main question:** Can the system admit useful normal patch features while preventing degradation of *true defect detection* caused by memory-induced decreases of anomaly scores? The study is **not primarily** a model of changing time-varying normal/defect prevalence or a class-prior correction controller.

Main benchmark = **Q-A shared random query tape, sample-random phase order, identical eligible normal candidates, frozen scorer and frozen threshold theta0, with dynamic Sidecar and Signed Correction off in the mechanism-isolation arms**. Comparing methods within a class/seed/B therefore holds constant the exact verified-normal/defect label set and camera views. Query seed changes realized composition, but all admission arms in its paired stratum share that composition; report it and aggregate within strata first.

Normal-first / Defect-first / Bursty / acquisition-time-blocked schedules remain **separate later stress/diagnostic experiments**, not requirements to implement propensity weighting or prior-shift compensation before the core admission test. This is an **execution-priority clarification**, not a change of the frozen D1-B schedule membership or query policy.

For Q-A, phase/sample order changes **when** anchors become available, **not the eventual count of normal versus defect labels**, since the same queried sample and camera image are held fixed across orders. Do not confuse information-arrival sensitivity with a query-composition confound.

## 2. CDARG counterfactual scores (faithful to the project proposal)

At pre-query time t, predict using pre-event memory M_t, fixed pre-event theta_t and correction state C_t. Query decision precedes disclosure. If a queried image is *individually verified normal*, obtain candidate features C_t^{cand} from the existing v22 8/16/24-patch selection rule. Form a **noncommitted** shadow update

    M_t^{shadow} = GreedyApply(M_t, C_t^{cand})

using the **exact existing append OR full-capacity replacement semantics** and identical static support, normalization, kNN aggregation and scorer. Compare *the same historical anchor features* under live and shadow memory, at the **same** fixed theta_t and correction state.

Let s_t(z) = raw normal-memory anomaly score under M_t, and s_t^{shadow}(z) under M_t^{shadow}. The proposal's normal utility is the reduction in the normal false-positive hinge:

    U_N = (1/|A_N|) SUM_{n in A_N}
          ( [s_t(n)-theta_t]_+ - [s_t^{shadow}(n)-theta_t]_+ ).

The proposal's defect score-erosion risk is a positive downward score shift on previously verified defect anchors:

    R_D = (1/|A_D|) SUM_{d in A_D}
          w_d [s_t(d)-s_t^{shadow}(d)]_+ / max(sigma_N, sigma_0).

Here w_d is the proposal's anchor importance weight and sigma_N, sigma_0 are normal-score scale/floor terms. Precise default values and whether the denominator is per-anchor or global MUST be documented before a production implementation; do not silently invent them. A defensible minimal unweighted/unnormalized variant can be an explicit ablation, not be mislabeled as identical to the PDF method.

Hard guard, as described in the proposal:

    H_D = 1{exists prior defect anchor d with
            s_t(d) >= theta_t + m_D and
            s_t^{shadow}(d) < theta_t + m_D} .

Decision:

    Admit iff H_D = 0 AND R_D <= epsilon_D
                  AND U_N - lambda_D R_D >= tau_U .

All epsilon_D, lambda_D, tau_U, m_D and the risk-aggregation choice are **unfrozen engineering/calibration parameters**. The source document suggests anchors in the tens and rough lambda/margin ranges; treat these as candidates, not established hyperparameters or empirically safe constants. Freeze using development calibration only, without inspecting confirmatory probes.

**Crucial boundary:** If A_D is empty, risk is **UNAVAILABLE**, not zero, and no formal "safe" decision may be claimed. The first implementation must expose an explicit pre-registered conservative handling mode (e.g., reject/defer candidate), log its lost normal utility and compare to always-reject. Never quietly allow update by substituting R_D=0.

Raw-memory score erosion is primary. Operational-score effects with signed correction are a separate follow-up; never let the current event create a correction and then use that correction to declare its own memory update safe.

## 3. Supervision and anchor semantics

- A verified **image-level** defect label may authorize defect-anchor extraction, but not a claim that every patch or every other camera view is abnormal. Choose predeclared top-scoring anomalous patch features without oracle pixel masks; track coverage uncertainty.
- Verified-normal image labels authorize normal candidate patches; they do not authorize propagation to sister views.
- Anchor ledger stores feature/image/camera/group IDs in a **trusted** layer, evidence acquisition step, effective time, source query event ID, selection score/uncertainty and optional query propensity.
- A candidate can only be tested against anchors known **strictly before** the candidate admission evaluation. The same event cannot provide both the new normal candidate and the historical anchors used to approve it.
- Separate **query disclosure time** from **memory update effective time**; first mechanism experiment uses prequential per-image immediate update. Delayed-to-end-of-five-view-group can be later sensitivity, not a requirement for D1-C.
- Never expose label-bearing Real-IAD filenames or group keys to pre-query controllers. Trusted loaders and offline evaluation may retain them.

## 4. Primary scientific comparisons and failure criteria

Within identical class/support seed/query seed/budget/stream:
- Static-only; verified-normal Append; fixed-capacity Greedy; matched-capacity Reservoir; causal HITL-style novelty (past-normal-only calibrated); proposed SAEG and CDARG.
- Anchor-based **counterfactual risk prediction**: for each proposed normal update log pre-admission predicted R_D and true counterfactual erosion on an **independent held-out defect probe evaluated offline only**; report correlation, dangerous-update ranking AUROC/calibration, and whether risk is unavailable.
- Per-step image-level FNR erosion against the **same fixed theta0** and probe; report FNR-degradation AUC, defect-score-erosion AUC, FPR improvement AUC, AUROC/AUPRC and final score/decision performance.
- Report candidates examined, verified normals queried, admission count/rate, actual bank size, replacements, latency/VRAM. A reject-all gate can reduce FNR erosion trivially while preventing all normal adaptation: therefore compare at **matched admission rate and matched FPR improvement** where feasible and plot FPR-gain / FNR-erosion Pareto curves.
- Use class-aggregated/clustered uncertainty, not individual camera frames as independent replicated datasets. No scientific claim of protection against unseen defect modes or selective-feedback representativeness without evidence.

D1-C requires CPU synthetic counterexample tests:
1. A verified-normal candidate improves normal coverage but dangerously lowers defect-anchor scores: CDARG must reject it.
2. Benign normal candidate improves U_N with no anchor harm: accept if thresholds permit.
3. Zero defect anchors stays UNAVAILABLE; conservative fallback and acceptance loss logged.
4. A shadow rejection leaves production bank bytes, capacity, version and retrieval indices unchanged.
5. Exact append and exact full-capacity **replacement** are faithfully simulated in shadow; no silent expansion.
6. Candidate event/label cannot be used as its own past anchor; probe ground truth inaccessible.
7. Signed correction and theta frozen throughout hypothetical evaluation.
8. Repeated evaluation is deterministic; patch/anchor scoring is equivalent to the real scorer under synthetic oracle examples.
9. Fix candidate labels, query tapes and camera images across all compared methods.

**D1-C is initially CPU-only.** No big GPU experiment, no performance claim and no modification to archived v22, V0.2 or D1-A/B manifests.
