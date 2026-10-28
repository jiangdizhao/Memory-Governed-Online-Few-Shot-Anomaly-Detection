# Calibration algorithms, external-baseline fidelity and primary references (v1)

**Research honesty rule:** method labels indicate algorithm families, not a guarantee that a faithful external method is already implemented. **No baseline is allowed to become silently weaker** through missing data, arbitrary hyperparameters, inappropriate label permissions, or substituting a related but different algorithm. The executor must inspect the cited primary paper and/or official author code, then record concrete equation numbers, source-commit, parameter table, tests, and any divergences. If a full faithful algorithm is incompatible with our input protocol, mark `ADAPTED` with exact differences; if its mathematics cannot be reproduced, mark `BLOCKED_FIDELITY`, do NOT silently invent an implementation or advertise a complete eight-method comparison.

## Primary sources and links (verified 2026-10-29)

1. **VisionAD:** *Search is All You Need for Few-shot Anomaly Detection* (2025), arXiv:2504.11895, https://arxiv.org/abs/2504.11895 ; author code https://github.com/Qiqigeww/VisionAD . Source-code versions must be pinned before claiming parity. This is the visual **foundation**, not a threshold-calibration external baseline.
2. **ACI:** Isaac Gibbs and Emmanuel Candès, *Adaptive Conformal Inference Under Distribution Shift*, NeurIPS 34, 2021, https://papers.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html ; PDF at https://papers.nips.cc/paper/2021/file/0d441de75945e5acbc865406fc9a2559-Paper.pdf . Standard coverage-controller concept; **not** a direct class-conditional threshold algorithm for sparsely labeled defect streams.
3. **Adaptive step-size ACI extension:** Isaac Gibbs and Emmanuel Candès, *Conformal Inference for Online Prediction with Arbitrary Distribution Shifts*, Journal of Machine Learning Research 25 (2024), paper 22-1218, https://www.jmlr.org/papers/v25/22-1218.html ; https://www.jmlr.org/papers/volume25/22-1218/22-1218.pdf . Adaptive step-size variant is a **secondary extension**, not to be relabeled as original ACI.
4. **SPOT/dSPOT:** Alban Siffer, Pierre-Alain Fouque, Alexandre Termier, Christine Largouet, *Anomaly Detection in Streams with Extreme Value Theory*, KDD 2017, pp. 1067-1075, DOI https://doi.org/10.1145/3097983.3098144; primary paper https://www.eecs.yorku.ca/course_archive/2018-19/F/6412/reading/kdd17p1067.pdf. The author C implementation was inspected at `asiffer/libspot` commit `d57e921a08930b960d07adcbec3a839bd228fe94`; `src/spot.c` blob SHA-1 `fdebc8c216fffeea93afcc51efab83a2d6013351` identifies itself as version 3.0.0. It provides SPOT code but no dSPOT implementation in that pinned source. Neither method is enabled until paper/source estimator conventions have reference tests.
5. **Sun-CS:** Sophia Huiwen Sun, Abishek Sankararaman, Balakrishnan Murali Narayanaswamy, *Online Adaptive Anomaly Thresholding with Confidence Sequences*, ICML 2024 / PMLR 235:47105–47132, https://proceedings.mlr.press/v235/sun24h.html ; full paper https://raw.githubusercontent.com/mlresearch/v235/main/assets/sun24h/sun24h.pdf . The **original paper's data model, observation protocol, objective, initialization, and assumptions** control what is faithful; do not conflate its unsupervised online setup with our sparse queried-label setup.
6. **Historical source:** `test_phase3_AIF_simplified_v22.py` for Guarded Dual Anchor / Balanced Rescue Sidecar; use repo source as authoritative; rejected paper `pattern_recognition_main_elsarticle_v_10(3).pdf` and supplement are contextual method descriptions, not replacements for executable formula audit.
7. **Internal method proposal:** research report `Designing and Prioritizing Memory-Dependent / State-Consistent Calibration for Dynamic-Memory Industrial Anomaly Detection` (provided as project PDF) and `docs/Calibration_Falsification_Research_Handoff_2026-10-28_v1.docx`. These are proposed designs, NOT peer-reviewed statistical guarantees.

## Common scoring, prediction and empirical operating-point quantities

Let `s>=theta` predict defect, `s<theta` predict normal. For current-memory rescored labelled anchors:
```text
N_t = {n_i = S_native(feature(anchor_i), M_t, source_exclusion_i) : known normal}
D_t = {d_j = S_native(feature(anchor_j), M_t, source_exclusion_j) : queried known defect}
FPR_hat(theta) = #{n_i>=theta}/|N_t|          [undefined if |N_t|=0]
FNR_hat(theta) = #{d_j<theta}/|D_t|           [undefined if |D_t|=0]
BA_hat(theta)  = 1 - (FPR_hat(theta)+FNR_hat(theta))/2 [undefined if missing]
theta_{t+1}   = clip(theta_star, theta_t-Delta, theta_t+Delta)
```
`C64` has known normal labels but not defect labels. An anchor's score is valid only under its declared memory state; if a queried-normal source contributed own normal prototypes, source-aware leave-source-out must exclude the relevant dynamic rows before re-evaluation. Strictly preserve `higher` q90 semantics. Endpoints include ties/midpoints and `theta_t`; use deterministic threshold sorting and tie-breaking.

**Confidence warning:** exact binomial intervals (e.g. Clopper–Pearson) may bound error on independent data under their *own assumptions*; a changing memory, adaptively sampled labels, temporally dependent camera observations and selection do NOT justify automatic anytime CS or safety certification. Name any naive interval `descriptive_bound`, not a valid risk certificate.

## B0 — Frozen Threshold (reference control)

```text
theta_t = theta0 = higher_quantile(source_C64_normals_scored_against_M0, .90)
```
No dynamic threshold state. Requires equality to D2 normal-source registry with n=64 and rank index 57 for the source configuration. Never recalibrate from hidden test labels.

## B1 — Rolling Quantile (transparent simple control)

Default primary variant `RollingQ90QueriedNormal`: append a normal score only if the image was queried and verified normal; maintain a **fixed, prespecified** window of the most recent W eligible observations; threshold becomes `higher_quantile(history, .90)` when count>=n_min, otherwise initial theta0/HOLD as explicitly declared. Record whether this variant uses original observed (stale) scores or rescored current-state normal scores. **Primary simple control should be stale rolling**; SCQ is current-state comparison. C64 may initialize prior but may not appear as extra online feedback without explicit labeling. A second `UnlabeledRollingQ90` is permitted only as a separately named additional row since unlabeled defect contamination is possible. Choose W, n_min, step clipping and refresh rule in frozen protocol, not after performance results.

## B2 — Native v22 Original Guarded Dual Anchor / Balanced Rescue Sidecar

Reproduce actual source methods and configurations from v22; extract exact formula, anchor-eligibility, `sigmaN`, cooldown, step-clipping, balanced rescue, anti-self-training and historical vs source-score behavior. Implement source wrapper when feasible. Apply same Query Tape and shared memory scores; allow only queried true labels. If original method's control decisions also affect memory/query in native v22, isolate *only threshold output* and document the intervention, avoiding phantom native parity claims. Reference-test against original source with a deterministic synthetic stream; log known intentional differences (e.g. access to score-only stream, C64) before `PASS`.

## B3 — State-Consistent Quantile (SCQ, crucial falsifier)

Proposed simple mechanism:
```text
on actual memory-content version change:
    N = protected C64 normal anchor scores rescored against current M_t
    if too few valid normals: HOLD, emit NORMAL_EVIDENCE_INSUFFICIENT
    else: raw = higher_quantile(N, 0.90)
          theta_next = clip(raw, theta_t-Delta, theta_t+Delta)
on no memory change: choose frozen policy for optional update/cooldown; no fake rescore
FNR evidence status = FNR_UNAVAILABLE if no queried defect anchors
```
Protected C64 ensures no mutable source self-match. An optional `SCQ+QueriedNormalLSO` arm can include additional source-excluded anchors only after exact attribution/LSO tests pass; it is not the primary SCQ. Step `Delta`, cooldown and minimum eligible evidence are **proposed hyperparameters** needing preregistration. Report source score staleness `|s_i(M_t)-s_i(M_at_arrival)|` and rescore count.

## B4 — State-Consistent Dual-ACI (adaptation, not faithful original ACI)

Original ACI conceptual update for coverage/miscoverage target a:
```text
a_{t+1} = a_t + gamma*(a_target - err_t)
```
Proposed binary class-conditional adaptation:
```text
aN_(t+1) = clip(aN_t + gamma_N*(alpha_target - 1[pred=defect]))
             ONLY when current queried true y is normal
aD_(t+1) = clip(aD_t + gamma_D*(beta_target  - 1[pred=normal]))
             ONLY when current queried true y is defect
other side has NO update if not disclosed
on current M_t use exact discrete error-count boundaries for predict=1[score>=theta]:
  kN=floor(aN*|N|); lower=nextafter(N_(|N|-kN), +infinity) when kN<|N|
  when kN=|N|, lower=nextafter(min(N), -infinity)
  kD=min(floor(aD*|D|), |D|-1); upper=D_(kD+1) in ascending order
project prior theta into [lower, upper] if consistent; otherwise CONFLICT/HOLD
with only normals, raise theta to lower; with only defects, lower theta to upper
an unavailable class side remains unavailable; it is never assigned zero risk
```
The projection orientation follows the score threshold: FPR constraint typically requires theta >= normal upper quantile; FNR constraint typically requires theta <= defect lower quantile. Verify quantile conventions and equality ties with unit tests. Initial gamma=0.005 was suggested by internal report as an illustrative first choice; **do not attribute this as a mandated general source optimum**. Fix gamma grid and handling of missing sides pre-evaluation. Original ACI typically sees realized error every step; our queried-only variant cannot inherit its long-run coverage guarantee. May compare a **faithful generic ACI reference unit test** separately but the reported B4 is `ADAPTED_DUAL_ACI`. The Gibbs–Candès JMLR 2024 extension is an optional additional arm if independently implemented, not secretly folded into B4.

## B5a/B5b — SPOT and dSPOT (external, score-only)

Implement the original EVT peaks-over-threshold idea as specified in the KDD 2017 paper / pinned official implementation:
- fixed label-blind initialization policy, initial level u (threshold empirical quantile), upper-tail exceedances, GPD shape xi and scale sigma fitted by a disclosed method, target tail-risk q, risk threshold from the source equation, alarm handling and update exclusion. Avoid mixing high anomalies into tail reference contrary to the source algorithm.
- For standard POT tail threshold, check the conventional formula, on source conventions:
```text
z_q = u + (sigma_hat/xi_hat)*((q*n/N_u)^(-xi_hat)-1)   [xi_hat != 0]
z_q = u + sigma_hat*log(N_u/(q*n))                    [xi_hat -> 0]
```
where n is source-defined total reference count and N_u source-defined exceedance count; **verify orientation, convention and online n definition against cited paper**, rather than blindly copying an equation. Handle xi=0 numerically and invalid small-tail samples safely.
- dSPOT additionally performs the original specified drift adjustment / local level tracking, not an arbitrary rolling-q90 relabeled as dSPOT.
- SPOT does **not** need queried labels: it may use the full legal unlabeled score stream and should not be artificially starved. On a memory-dependent score, source stationarity assumptions may be violated; report descriptive adaptation, not theoretical SPOT guarantees.
- C64 only has 64 training-normal views and may be statistically insufficient for some tail-fit hyperparameters. If so, use a **frozen causal score-only warm-up** with explicitly reported delayed predictions and identical predeclared evaluation handling; do not borrow future scores, probe data or normal labels from stream. If faithful warm-up cannot be done within common protocol, `BLOCKED_FIDELITY`.
- Report SPOT and dSPOT as distinct rows, not one cherry-picked baseline.

## B6 — Sun et al. 2024 Confidence Sequence online anomaly thresholding

**High-priority fidelity gate:**
1. Read the **entire** ICML 2024 paper and appendix, note formal observation model, assumptions, Algorithm numbers, online update schedule, warm-start handling, confidence sequence construction, risk targets and exact evaluation meaning. If author code exists, pin URL and commit.
2. Reproduce primary algorithm on a source-consistent **synthetic reference sequence** independently of our vision/memory and verify against source equations/examples. Provide the reference code and its tests; do not replace the method with a generic Hoeffding/Clopper–Pearson interval plus threshold optimization.
3. Declare whether original algorithm is allowed to observe exactly our available unlabeled s_t plus C64; distinguish unsupervised source method from variants that use queried labels. If original requires different observations/feedback, create `SunCS_AdaptedSelective` as a separate *honestly labeled* comparator, and state precisely which theorem hypotheses fail.
4. On dynamic memory, anchor score distribution changes and observed scores can be dependent. Neither confidence-sequence coverage nor FPR/FNR control automatically transports to our system. Use no phrase `certified risk` absent a new proof.
5. If reference fidelity cannot be established, register `BLOCKED_FIDELITY` and stop Stage-0 **all-baselines** gate; do not fake a pass or claim the eight-arm comparison is ready.

## B7 - SCDAC-G (executable provisional v0; not validated)

The internal proposal leaves confidence-bound, evidence-minimum, support-risk,
movement-penalty, score-normalization, cooldown, and conflict choices open.
The executable choices are recorded in
`docs/calibration/SCDAC_G_V0_METHOD_SPEC.md`, which carries the proposal PDF
SHA-256. Its status is `IMPLEMENTED_PROVISIONAL`, not a faithful unique
translation and not a statistically certified controller. Version 0 uses
current-state protected C64 and eligible queried anchors, one-sided fixed-time
Clopper-Pearson upper endpoints labelled descriptive, a BA-oriented two-sided
objective with normalized movement penalty, and a bounded minimum-descriptive-
risk fallback when the two-sided support set is empty. `NORMAL_ONLY`,
`DEFECT_ONLY`, and `NONE` hold. Stale two-sided intervals are diagnostic only.
All provisional numerical values remain open for independent review. No claim
is made that selective feedback, serial dependence, repeated memory rescoring,
or post-selection preserves a binomial guarantee.

## Preregistration and strong-baseline requirements

For every method ship `method_card.json` or `method_card.md` with:
```text
paper_title, authors, venue, year, paper_url, author_code_url, pinned_source_sha
method_label, reference_alg_or_equation, faithful_or_adapted, exact_differences
input_permissions (scores/labels/C64/anchors), initialization and warm-up
parameter_name/value/units and fixed dev grid, random seed, independent unit-test reference
missing_class and insufficient_data behavior, numerical edge cases, complexity
claimed_guarantee (if any), invalidated_assumptions, source_and_test_hash
```

**No intentional handicap:** tune external methods on the same development categories and comparable search effort to proposed methods in Stage 1; never optimize on Mint/PCB confirmation outcomes. Count all feature/anchor buffers, startup delay and computation. Every adaptation must have its own name; do not silently call SC-Dual-ACI “ACI theorem validated”, or an arbitrary CS baseline “Sun et al. reproduction”.

**B5 and B6 fidelity may be legitimately difficult.** The correct scientific response is a BLOCKED gate with a precise remediation request, not a fabricated method.

## Third-pass source audit (2026-10-29)

This audit records what the primary papers establish and what remains
unimplemented. The distinctions below are fidelity blockers, not a relaxation
of the required baseline set.

### B5a/B5b — POT, SPOT, and dSPOT

The full Siffer et al. KDD 2017 paper was read. Its SPOT update predicts against
the current risk threshold, excludes alarms from the reference tail, adds
non-alarming observations to the total count, and refits the GPD after peaks
above the initial level. The paper's risk-threshold convention uses its own
total-observation and exceedance-count definitions; those source meanings must
be transcribed and independently tested before implementation. The author
repository was pinned to commit
`d57e921a08930b960d07adcbec3a839bd228fe94` (`src/spot.c` blob
`fdebc8c216fffeea93afcc51efab83a2d6013351`). That source supplies SPOT code,
but no dSPOT implementation. No exact Grimshaw/reference trace is currently
registered, so neither B5a nor B5b is enabled. A third-party drift implementation
without a pinned version and source-equation parity does not resolve this gate.

Remediation: pin a faithful SPOT source or provide an independent Grimshaw
reference against the paper, then test initialization, `xi -> 0`, nonfinite or
insufficient tail samples, equality at the alarm threshold, alarm exclusion,
normal/peak updates, and the online `n`/`N_u` counters. Implement dSPOT as a
separate method only after its local-mean initialization and update schedule are
verified against the paper or a pinned source. Until then, both method cards
remain `BLOCKED_FIDELITY`.

### B6 — Sun et al. confidence-sequence thresholding

The complete ICML 2024 paper and appendix were read. Its setting uses an
independent score stream with piecewise-stationary i.i.d. segments, targets a
specified upper-quantile anomaly rule, and returns a three-way anomaly,
benign, or abstain decision from a quantile confidence sequence. The paper's
algorithm family includes an explicit change-detection/reset path; it is not a
generic binomial confidence interval followed by threshold optimization. The
reference quantile definition averages adjacent order statistics at
non-integer ranks, and the confidence-sequence radius is time-dependent.

No source-consistent empty-history/startup decision has been established for
our evaluation protocol, and an original-source reference controller with
independent test vectors has not been implemented. C64-only normal calibration
and a changing, memory-dependent score path do not satisfy the paper's complete
observation assumptions by inspection; sparse queried feedback is not a
substitute for its unlabeled score stream. The abstention treatment in primary
metrics also needs to be frozen before any comparator run. Therefore B6 remains
`BLOCKED_FIDELITY`; no “Sun-CS” or adapted controller is reported as implemented.

Remediation: add a paper-equation reference module and tests for quantile ranks,
confidence-sequence radii, tri-state decisions, change detection, reset timing,
and startup. Separately declare a source-faithful observation mapping or name
and test any adapted selective-feedback method, including its delayed-prediction
and abstention accounting policy. Do not transfer the paper's guarantee to a
dynamic-memory score stream without a separate proof.

### B7 — provisional SCDAC-G specification correction

The third-pass executable and method specification now distinguish the
primary balanced upper-risk objective from the secondary FPR-constrained
diagnostic objective. Primary selection minimizes the balanced objective only
over descriptive-feasible candidates when that set exists; otherwise it uses
the documented minimum-risk compromise over the full grid and explicitly enters
`CONFLICT`. The bounded threshold is clipped afterward and feasibility is
recomputed at the executed threshold. The secondary FPR diagnostic minimizes
FNR upper endpoint plus movement cost and cannot alter the primary trajectory.
These are deterministic provisional software choices, not choices uniquely
dictated by the proposal and not certified guarantees.
