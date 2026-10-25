# Post-Rejection Research Roadmap: Routes A and B
**Project:** Memory-Governed Online Few-Shot Anomaly Detection  
**Status:** Research plan ONLY; no experiments or algorithms are authorized by this document.  
**Created:** 2026-10-26  
**Working branch at authoring:** e02-r6-extreme-balanced-validation  
**Purpose:** Repair evaluation validity and demonstrate a defensible independent memory-learning contribution in response to Pattern Recognition rejection (PR-D-26-11359).

> **中文版摘要：** 先执行路线 A（公平评测、严格 four-shot、无自匹配阈值初始化、统一监督预算/记忆容量、强简单记忆基线及必要的外部方法比较），用实验判断现有复杂记忆治理是否有独立价值。只有在确定了剩余缺陷与明确的科学问题后，才决定是否进入路线 B（提出并验证新的风险感知/证据治理式记忆更新规则）。R6 及原投稿结果均应保留，不可重写。**本次任务仅创建计划文档；下一阶段实验尚未启动。**

## 0. Authority, scope, and scientific integrity

### 0.1 Existing evidence and records

- Rejection/recovery handoff: historical record of the three reviews and original evaluation concerns; the submitted manuscript and supplementary files are **historical evidence, not corrected-baseline evidence**.
- Previous studies E02 through R6: frozen, versioned experiments. Especially:
  - [R6 sidecar protocol](E02_R6_SIDECAR_STABILITY_PROTOCOL.md)
  - [R6 cross-category minimal validation protocol](E02_R6_CROSS_CLASS_MINIMAL_VALIDATION.md)
  - [R6 12-condition aggregated results](../outputs/e02_r6_extreme_balanced_validation_summary.json)
  - [Original matched-feedback factorial protocol](E02_MATCHED_FEEDBACK_PROTOCOL.md)
- R6 cross-category study: 12 validated conditions (rolled_strip_base and switch; two support seeds; native, normal_first, defect_first). This shows temporal sensitivity and separates shared-score threshold variants, **not** a complete attribution of online learning or a formal guarantee.
- The original submitted comparisons are not reclassified as strict-four-shot merely because only four exemplars were deployed. The original support-selection pool and self-match-calibration concerns must remain visible.

### 0.2 Research questions

**RQ1 / attribution:** When initial representation, support data, label feedback, mutable memory capacity and thresholding are fairly controlled, does the proposed *memory update/governance rule* provide an independently measurable advantage over simpler methods?

**RQ2 / robustness:** Under sparse, selectively requested labels and nonstationary arrival orders, can memory updates preserve anomaly discrimination without unacceptable memory-induced defect-score erosion or hidden decision risk?

**RQ3 / publication:** Are the resulting method, evidence, comparison protocol and reproducibility strong enough to substantiate a contribution rather than a complicated integration of known modules?

### 0.3 Non-negotiable rules

1. Do **not** edit archived E02–R6 outputs, the historical v22 script, the rejected manuscript, or prior experiments to improve a story.
2. Never use current or future test labels to choose supports, thresholds, query actions, memory writes, hyperparameters or stopping decisions. Any label-informed stress permutation is constructed offline and disclosed as a diagnostic intervention, not a deployment policy.
3. Different information-access protocols must be labeled separately. An author-method native-protocol result is not a protocol-matched result.
4. Compare image-score *ranking* (AUROC/PR), classification *operating points* (BA, FPR, FNR), and streaming *reliability* (time windows, delay, recovery, memory) separately.
5. Neither an AUROC increase nor a reduced FPR alone establishes improved overall deployment safety; always report the accompanying FNR and operating-point trade-off.
6. A statistically undefined class-conditional risk (window with no relevant ground-truth class) is **null**, not zero. Queried-label risks and true-population evaluation risks must not be conflated.
7. Every GPU campaign should be launched by **one reproducible runner** handling ordered dependencies, checkpoints/resume, validation, failure logging and machine-readable aggregation. No dozens of manual per-condition commands.
8. The user uploads results by their own Git workflow. Researchers/assistants read results directly from GitHub. No need to provide git add/push instructions.
9. Preserve support-seed identities, query trajectories, sample IDs, per-step scores, bank fingerprints, actual label-access counts, and CPU/GPU/resource provenance.
10. Planned claims are **hypotheses**, not established results. No new results or expected gains are to be fabricated.

## 1. What the rejection requires us to repair

| Reviewer/editor concern | Why serious | Required evidence |
|---|---|---|
| Limited independent algorithmic novelty | STM/LTM, weighted retention, query decisions and sidecars may be a composition of existing ideas | A clear memory-update rule with demonstrable benefit over FIFO, reservoir and online coreset at matched conditions |
| Memory vs threshold vs query attribution | Adaptive components change both observations and the selected query trajectory | A matched-feedback factorial and then a separate closed-loop policy evaluation |
| Weak or non-aligned external comparisons | No credible estimate of competitiveness | Strong implementable protocol-matched baselines plus clearly annotated native-paper external comparisons |
| Four-shot definition | Four deployed exemplars can have been chosen with access to a larger normal pool | Strict random 4-image access as primary regime; separate fully disclosed pool-assisted 4-image deployment regime |
| Initial support-score self-match | The support images and augmentations scored against their own support bank may bias theta0 | Genuine leave-one-support-image-out (LOSO) and/or independently budgeted calibration data; no leaked defect labels |
| Signed correction localization | Normalized radial kernel can cancel expected absolute decay and create long-distance influence | Distance-response audit and no-correction / gated / unnormalized comparisons; demote if ineffective |
| Industrial and continual-learning scope | A single mean AUROC and limited same-class restarts are insufficient | Fixed-probe tracking, worst-window risk, forgetting, restart cases, pixel metrics where supported, latency and memory accounting |

**Priority:** First fix data access and calibration. An apparent gain over an unfair or miscalibrated static baseline is not sufficient evidence.

## 2. Route A — Repair and fairly validate the existing system

**Aim:** Preserve the useful engineering architecture, but only claim capabilities that survive scientifically fair baselines and clean attribution. This is the lower-algorithmic-risk, faster publication route; acceptance timing is never guaranteed.

### A0. Freeze and audit the evaluation contract — MUST PASS before large experiments

- Use a fixed frozen DINOv2/VisionAD scoring substrate, layer/view fusion and image size across all protocol-matched arms; disclose configuration.
- **Strict four-shot:** for each class/seed, draw exactly 4 normal training images from the eligible training pool **without using any other normal image to select, rank or tune them**. Only permitted transformations of those four images enter the initial bank. Exclude test stream access. Pre-register seeds and report classwise variance.
- **Distinct second regime:** pool-assisted selection of four deployed supports; report pool availability/count and extra access. Never call it strict-four-shot.
- **LOSO calibration:** when estimating support-normal score distributions, hold out the entire support image, including **all of its augmented views**, from the scoring bank. Verify this with exact image IDs/hash fingerprints and test assertions. LOSO with n=4 has substantial sampling uncertainty; disclose it.
- If adding independent calibration normals, charge them as additional data access to **all arms**, and separate that setting from strict 4-image-only.
- Estimate theta0 using normal-only allowed calibration and a **pre-registered rule**; no test labels or retrospectively tuned defect-operating point.
- Verify that a fair static comparator and online variants use the same permitted calibration procedure and equivalent detector settings.
- Record eligibility filters, whether any oracle masks/defect labels influence training, support paths, seed, immutable bank hash, theta0, exact capacities and score calculations.
- Audit signed correction separately; do not silently change it in this fairness phase.

**A0 artifacts:** protocol manifest and lineage hashes, calibration self-match/LOSO tests, per-seed static baselines, pre-specified acceptance checks, discrepancy report against original submitted initialization.

**A0 go/no-go:** if the prior headline improvement collapses after correcting the support-selection or theta0 protocol, mark it as non-transferable, stop citing it as a fair gain, and reframe later claims. Do not tune until it returns.

### A1. Matched-feedback memory comparison — PRIMARY NEXT EXPERIMENT, NOT YET IMPLEMENTED

**Core question:** At equal observed labels, equal mutable memory and equal scoring/threshold setup, does the existing governed memory beat strong simple memory algorithms?

Minimum matched arms:

1. Static VisionAD / immutable reference bank, no mutable memory.
2. Verified-normal online FIFO / recent prototypes.
3. Verified-normal reservoir sampling.
4. Verified-normal online Greedy Coreset / coverage-based admission and replacement.
5. Existing governed memory rule (STM, and LTM/maturation where applicable), with named ablations of each extra stage.
6. Proposed new rule from Route B **only if and when B is actually developed**.

**Fairness controls:**

- First use **one frozen query tape** per stream/seed, common to all arms (e.g., 3% verified labels, label-blind fixed randomized tape). Observed labels are disclosed **after** prediction at each queried time step.
- All arms receive **the identical label identities, order and number**. Only verified-normal cases are admitted to normal memory; report resulting counts and the defect-label consumption used for calibration separately.
- Match **mutable retrieval capacity by layer/view**, initial fixed-bank data, shared feature extractor, augmentation settings, score aggregation and allowed compute; count any extra anchor/correction/calibration buffers toward the relevant resource budgets.
- Report distinct budgets where storage is in patches vs images/prototypes. A system with hidden extra buffers is not capacity-matched.
- Do not allow per-arm threshold policies to change the query tape.
- **Primary memory-only endpoint:** compare frozen calibrated theta0 or a single reference calibration rule across arms without per-arm adaptive calibration. Include AUROC plus fixed-probe and FNR@fixed-FPR diagnostic curves as separate evaluation-only measures.
- **Secondary decision endpoint:** apply the same pre-specified adaptive threshold-controller algorithm to every arm (trained only on each arm's permitted queried scores/labels). It is the same rule, not necessarily identical numerical theta trajectories because score scales differ.
- Four-arm 2x2 contribution audit: immutable vs mutable memory, frozen vs adaptive threshold. Measure main effects, interaction and ranking-vs-calibration effects **within each locked tape**, acknowledging that fixed-tape effects are conditional and not native-policy causal effects.
- Keep a fixed, **held-out evaluation probe** with normal and defect cases; use its labels only for offline evaluation. Never add probe images to memory, query tape, anchor calibration or threshold optimization. Distinguish pooled prequential AUROC from snapshot probe AUROC.
- Report FPR, FNR, BA, image AUROC, PR curves, FNR@FPR target, worst defined temporal-window loss, score-drift, memory changes, throughput/latency, peak RAM/VRAM and wall-time.
- Predeclare effect-size targets and replication criteria **before** seeing new performance. Use enough independent support/query seeds and classes for uncertainty intervals; two support seeds from R6 are exploratory only.

**A1 expected output:** one honest, same-budget primary table of memory methods and their contribution to score discrimination. **No claim of novel memory governance** merely because a complicated controller obtains higher BA.

**A1 decision gate:**

- If existing governed memory consistently improves ranking, fixed-probe robustness or safety-constrained error over strong online coreset under matched budgets, retain it as a potentially defensible independent mechanism and identify *which* operation causes the gain.
- If it ties or loses to reservoir/coreset, do not keep claiming sophisticated memory retention as a demonstrated novelty. Simplify Route A or move to Route B's focused algorithmic hypothesis.
- If the static baseline remains strong after fair calibration and none of the adaptive memories adds useful value, reconsider the core research claim before doing broad sweeps.

### A2. Separate native closed-loop query-policy study

Only after A1:

- Let the allowed online policies propose queries using their own observable history, subject to the same strict total budget and matched initialization.
- Include frozen predetermined tape, a label-blind random baseline, original AIF/BEA, and (if developed) audit-mixed query coverage.
- Report exact realized queries, queried label composition, query-free gaps, delayed feedback, and memory/write trajectories. **Do not call this a matched-feedback causal attribution** when query positions differ.
- Separate changes in the query policy from changes in memory retention and threshold update whenever possible; factorial/counterfactual or replays must state identification assumptions.
- Validate native, normal-first, defect-first and interleaved/mixed streams. Label-defined extreme order is **evaluation-only**, never observable to the policy.

### A3. Comparator literature and external baseline evidence

**Protocol-matched, same substrate (highest priority):** implement transparent FIFO, reservoir, online greedy coreset, plus static and current governed memory; report exact differences and any noncanonical adaptations.

**Published methods to evaluate for applicability, implementation and budget agreement:** FOADS; Online-InReaCh; RareCLIP; ContCore. These use differing online, zero-shot, supervision or compute assumptions; verify original paper/code/version and data access before committing to a particular external reproduction.

Maintain two **separate** comparison tables:

- **Native published setting**: faithful original implementation/architecture/dataset protocol, with supervision, label access, model size, sample resolution, temporal order, memory and compute carefully annotated. Not an equal-supervision ranking.
- **Common-substrate method-inspired adaptation**: if feasible, implement the comparable memory/update idea under the shared DINOv2/VisionAD and 3%-label protocol; explicitly name it an adaptation rather than a reported result from the original paper.

Never substitute incompatible published AUROC numbers for a fair head-to-head evaluation. If an external implementation cannot be made protocol-compatible, state the limitation with a defensible reason.

### A4. Corrective signed-correction and broader deployment checks

- Quantify original normalized Gaussian residual at large anchor distance (including epsilon normalization). Evaluate no correction vs original vs explicit spatial/feature gating vs unnormalized effect, under locked query/memory tapes. Compare actual influence-distance curves and far-anchor contamination.
- If correction benefit is negligible or unstable, remove/demote it from the main method, rather than adding complexity without evidence.
- Confirm correction anchors and masks never receive access beyond declared labeling protocol.
- Verify any image-level vs pixel-level metric with actual mask availability and no mask-based update contamination.
- Add visual algorithm/pipeline, memory state transitions, false-negative case studies, storage curves and restart evidence with claims constrained to tested scope.
- Expand from exploratory two classes/two seeds toward the declared Real-IAD 30-class protocol; include multiple support seeds and other datasets only when legitimate and computationally affordable.

### Route A publication gate

Stay with Route A if, after correcting A0, the existing method has **reproducible, bounded-resource, well-attributed** practical gains and fair comparisons. Claim a deployment/governance systems contribution only if evidence justifies it. Choose an application-/systems-oriented journal when methodological novelty is limited. Do not resubmit the rejected manuscript unchanged or present unsupported prior performance as newly validated.

## 3. Route B — A new independent memory-governance method (conditional R&D)

**Aim:** If Route A reveals that existing STM/LTM retention is little better than online coreset or reservoir, develop a distinct, testable memory-admission/retention principle. This is higher novelty potential, but also higher development and publication risk. Route B is **not yet approved for implementation**.

### B0. Problem and novelty hypothesis

Rather than merely maximize normal-patch coverage, ask:

> At a fixed memory/label budget, can a verified-normal memory update improve normal coverage **without eroding established defect separation or causing unsafe recalibration**?

Candidate research label: **Counterfactual Risk-Governed Memory Consolidation** (working name; not a paper claim).

For current memory M, a proposed verified-normal update z induces two candidate memories:

- M0 = M (reject update).
- M1 = Update(M, z) (admit/replace/consolidate).

Assess both candidate states on explicitly separated, **verified historical anchors** and optional held-out *evaluation-only* probes. Actual admission must **never** use held-out evaluation probe labels.

Potential research signals:

- Normal coverage improvement: G_N = mean over admissible normal anchors of [score(x; M0) - score(x; M1)].
- Known-defect erosion: E_A = mean over admissible defect anchors of positive-part [score(x; M0) - score(x; M1)].
- Replacement loss: C_replace, reflecting evicted normal-mode support/coverage and maintenance burden.
- Example hypothesis: U(z) = G_N - lambda_A E_A - lambda_R C_replace, subject to memory capacity and evidence sufficiency constraints.

**Not established novelty:** this is a starting hypothesis; risk-aware replay, prototype selection, sample influence and online memory are active research areas. A literature novelty check and comparable strong baseline are mandatory.

### B1. Statistical and implementation obstacles that must be solved

1. **Defect evidence absent:** before any verified defect, E_A is not observed, not zero. Maintain unknown/insufficient status and avoid false guarantees.
2. **Active-selection bias:** queried labels are selection-biased. Compare optional random audit/probabilistic exploration with guaranteed recorded probabilities before proposing inverse-probability weighting; deterministic queries cannot be given fabricated propensities.
3. **Memory self-match:** normal calibration samples entered into STM may become artificially easy. Use held-out-from-update anchors, leave-one-update-out re-scoring or equivalent isolation; record all versions.
4. **Memory-state score drift:** historical scores are score(x; M_i), not score(x; M_t). Re-score a small bounded calibration anchor set against the current candidate memory when justified; account for update cost and memory capacity.
5. **Nonstationary class arrivals:** normal-first, defect-first and mixed streams may leave one class temporarily unobserved. Must distinguish identifiable and unidentifiable risks and report the evidence gaps.
6. **Interacting controller:** memory admission changes scores and threshold risk; do not claim independent memory benefit based only on a Sidecar-specific BA.
7. **Efficiency and theory:** analyze update/anchor score cost, finite capacity, replacement stability and any actual assumptions behind claimed bounds. Do not invoke formal Bayesian or conformal guarantees without corresponding assumptions and proof.

### B2. Minimum discriminating experiments

- Replay the **same verified-label tape** and compare proposed governance against FIFO, reservoir, strong online coreset and existing v22 memory retention.
- Match all mutable memory, calibration anchors, correction storage and compute as explicitly as possible.
- **First outcome:** memory-only ranking and frozen-probe discrimination/retention under one common threshold setup.
- **Second outcome:** threshold-calibrated BA/FNR/FPR using the exact same calibration algorithm for all methods.
- Challenge with normal-first, defect-first, mixed/alternating, realistic block drift and support changes; report worst-window performance, recovery delay and seed spread.
- Track each proposed memory write's counterfactual delta scores on **admissible anchors**, final selection, protected-defect score erosion, normal coverage, replacements and cost. Positive results on the optimization anchors alone are not enough: require independent evaluation probes and unseen streams.
- Ablate normal gain, defect erosion, replacement cost, anchor refreshing and missing-evidence safeguard one at a time.
- Compare conceptually and empirically against relevant memory boundedness, robust coreset, prototype retention and continual AD papers; avoid an unsupported 'first-ever' claim.

### Route B go/no-go

- **GO:** a clearly defined new admission/retention rule gives repeatable nontrivial improvements over strong coreset under fair budgets, and survives independent probe, classes and seeds without unacceptable runtime.
- **NO-GO:** improvements vanish under strict four-shot/fair calibration, rely on oracle defect labels, come only from Sidecar tuning, or collapse on different seeds/orders. Return to a simpler application-oriented Route A paper or revisit the hypothesis.
- Do not grow additional STM/LTM/correction/controller modules merely to add nominal complexity.

## 4. Work sequence and decision graph

1. **Now (planning only):** freeze this roadmap, record acceptance checks and do not implement new algorithms.
2. **Next phase, only after explicit user approval:** A0 plus A1 minimal screen — *strict four-shot + fair calibration + uniform threshold/controller + four memory comparison families* (static, FIFO, reservoir, online greedy coreset, current governed memory); develop one automated runner with manifest/validation/summary.
3. Inspect A0/A1 outcomes **before** broad 30-class sweeps. If the system under fair protocol wins simple retention, isolate why; if not, simplify or enter Route B exploratory gate.
4. A2 locked-vs-native query policy analysis and A3 strong external methods, with separate publication tables and source-verified configs.
5. A4 corrected signed correction, figures, operational robustness, and complete class/seed evaluation as justified.
6. B0–B2 only after an explicit scientific go decision. A and B may share test infrastructure but should not be blended into unidentifiable new modules.

### Milestone success criteria / deliverables

| Milestone | Minimum deliverable | Gate |
|---|---|---|
| A0 | Fair strict-four-shot manifest, LOSO control, baseline calibration comparison | No self-match or undeclared normal-pool access |
| A1 | Matched-tape static/FIFO/reservoir/coreset/governed memory matrix and 2×2 attribution | Clear effect sizes; no confounded component claim |
| A2 | Budget-matched native query policy and drift/order runs | Full resource and query accounting |
| A3 | Verified protocol-matched simple baselines and native-publication comparison table | No mislabeled paper results |
| A4 | Correction influence plot, FPR–FNR, algorithm figure, restart/latency/pixel checks | Submission claims match evidence |
| B0 | Novelty map, mathematical update rule and assumptions | Nontrivial difference from coreset/replay/risk-aware memory |
| B1/B2 | Robust counterfactual memory test against strongest Route A control | Independent held-out gains and credible cost |

## 5. Engineering interface for future execution (specification, not code)

When experiments are authorized, prefer a single command in the repository root, for example:

    python experiments/run_post_rejection_validation.py --phase A0_A1 --all --device cuda

**This command is illustrative only. No such script is created by this plan.**

The future runner must:

- Resolve dependencies automatically (sanity and calibration tests before GPU loops, native before stress references).
- Support deterministic seeds, class subsets for screening, final pre-registered full matrix, restart/resume and existing-run validation with no silent overwrites.
- Fail closed for missing reference data, invalid four-shot accesses, probe leakage, excess label/query/memory budgets, wrong scorer or mismatched initialization.
- Produce independently inspectable row-level predictions/scores/thetas/query tapes plus per-condition JSON and one aggregate JSON/CSV/plots.
- Preserve failure logs and enable incremental upload/remote inspection; do not require the user to hand-enter a command per class/seed/order.
- Prefer reproducible cached feature reuse where feature provenance and update semantics remain provably identical; quantify VRAM/RAM/disk/time.

## 6. Claim discipline, writing strategy and decision ownership

### If Route A wins

Working contribution: a verified-memory, budgeted-online industrial AD system with empirical benefits from **specific proven governance operations**, accompanied by strict data-access fairness, matched-budget comparative evidence and auditable deployment behavior. Do not assert new online coreset theory without one.

### If Route A's complex memory does not win

Do not promote memory capacity, UI/sidecars or combined system complexity as mathematical novelty. Either:
- Publish a narrower but honest deployable systems/negative-results finding in an appropriate venue if the contribution remains substantial; **or**
- With explicit approval, initiate Route B to build and validate an independently beneficial memory update algorithm.

### If Route B wins

Lead with the memory-update principle, its optimization/evidence-sufficiency logic, isolated gains over strong matched online coreset, and robustness to observed distribution order. Query controller and Sidecar are supporting components whose independent contributions remain reported.

### Paper/report requirements

- Main paper: problem definition, data access and calibration, exact memory mechanism, comparable baselines, strongest ablation, outcome tradeoffs, clear algorithm/state transition figure and limitations.
- Supplement: full seed/class results, numerical protocols, proof/assumption details where applicable, per-step audits, reconstruction scripts and runtime logs.
- Report failures, negative results and hypothesis changes; don't choose the best stream ordering after seeing outcomes.
- Respect original reviewer concerns and quote them accurately only from the actual decision records.
- Venue selection comes **after** valid claims are known; journal timelines/APCs do not determine scientific reporting standards.

---

**Status at creation:** Documentation committed only. No A0/A1 runner, memory algorithm change, calibration change, benchmark re-evaluation, or new GPU experiment has been performed as a consequence of this roadmap. The user will explicitly authorize the next phase.
