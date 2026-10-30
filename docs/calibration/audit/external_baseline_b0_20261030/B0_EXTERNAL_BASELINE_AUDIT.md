# B0 External Baseline Fidelity & Existing Results Audit

**Audit date:** 2026-10-30

**Repository:** E:\Memory-Governed-Online-Few-Shot-Anomaly-Detection

**Branch:** calibration-stage0-contract-20261029

**Audited source HEAD:** 896ee9b426c27fd8332e34773094e41d4442a11c
**Scope:** read-only source, literature, manifest, checksum, and aggregate-metric audit. No experiment, image inference, test, or algorithm change was made.

## 1. Executive findings

Three published algorithm families are represented in the repository:

- B4 is an **Adapted Dual-ACI** inspired by Gibbs and Candès. It is not their original ACI update law or target and does not inherit ACI coverage guarantees.
- B5a SPOT and B5b dSPOT are Python implementations of Siffer et al.’s paper equations, with independent equation/state tests and no compiled author-code trace parity. Their application to a changing-memory visual score stream is an adaptation whose stationarity assumptions may fail.
- B6 implements Sun et al.’s confidence-sequence score-quantile procedure with declared endpoint conventions and reference-trace tests. Its output is a score-quantile label with abstention, not physical image-level defect truth. The formal Stage 1-A audit correctly treated it as applicability-only and omitted it from the binary leaderboard.

B0 Frozen-Theta, B1 RollingQ90QueriedNormal, B2 historical v22 Sidecar, B3 StateConsistentQuantile, B7 SCDAC-G, and Stage075BOffset are project controls, adaptations, or internal proposals rather than additional external published algorithms. Stage075BOffset is the strongest internal alternative by mean BA in the existing run and must stay visible.

The existing Stage 1-A run supports a **development-only operating-point comparison**, not independent category generalization. B3 has mean BA 0.7362, FPR 0.1042, and FNR 0.4235. B1 has lower BA at 0.6812 but much lower FPR at 0.0239. B3’s paired BA is higher than B1 on 19/20 trajectories, while B3’s FPR is higher on 20/20 and FNR lower on 20/20. Stage075BOffset has higher mean BA than B3 (0.7573) with higher FPR (0.1718) and lower mean worst-window BA (0.6317 versus 0.6530). No universal FPR ceiling was pre-registered, so these results do not establish a risk-safe winner.

Native SPOT and dSPOT are unavailable during startup: 1,000 and 1,050 events per trajectory, respectively. Their full-stream BA and FPR are undefined and were not imputed. Their fixed common-horizon results are available with 100% coverage. The Frozen-Theta startup wrappers are separate deployment hybrids; they use fallback on 26.15% of B5a-Frozen decisions and 27.46% of B5b-Frozen decisions over the full stream, and no fallback after the fixed t >= 1050 horizon begins.

The Stage 1-A rows all share the same native score and Greedy-memory trajectory, Q-A-TS-v1 tape, and evaluator boundary. The regime is **four deployed supports plus 64 additional protected C64 training-normal calibration images**, not strict four-shot-total. B1 and B3 differ in both their calibration-reference source and memory-state treatment; their direct contrast is an end-to-end comparison, not an isolated test of state consistency. The matched A2/A3/A4 state-lag test is the appropriate mechanism intervention, and it returned H1_WEAK_OR_MIXED.

**Useful paper-facing claims now:** report the C64 operating regime exactly; show BA, FPR, FNR, coverage, and worst-window metrics together; name B4 as adapted; distinguish paper-equation SPOT/dSPOT from author-code parity and their deployment hybrids; keep B6 out of the physical-defect binary ranking; and retain Stage075BOffset as an inconvenient but strong internal comparator.

## 2. Literature and algorithm fidelity

The machine-readable inventory in **baseline_fidelity_matrix.json** records each method’s source paths, equations/state, initialization, input permissions, fidelity label, caveats, and test evidence. Test files were inspected but **not run for B0**.

### 2.1 B4 — Adapted Dual-ACI

**Original reference:** Isaac Gibbs and Emmanuel Candès, “Adaptive Conformal Inference Under Distribution Shift,” NeurIPS 2021 ([official paper](https://papers.neurips.cc/paper_files/paper/2021/file/0d441de75945e5acbc865406fc9a2559-Paper.pdf), [NeurIPS page](https://papers.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html)).

Original ACI updates one miscoverage state after each realized outcome, commonly written alpha_(t+1) = alpha_t + gamma(alpha - err_t), where err_t indicates whether the observed outcome missed the prediction set. Its objective is sequential prediction-set miscoverage/coverage. The repository instead maintains separate normal- and defect-side states, updates only when a query discloses the corresponding label, and maps those states to empirical anchor bounds for binary physical-defect decisions.

The current defaults are alpha=beta=0.10 and gamma_N=gamma_D=0.005. Missing evidence on one side remains unavailable; it is not treated as zero risk. This is a sparse-feedback, class-conditional adaptation. It does not implement original ACI’s every-step outcome update, one-alpha recursion, or conformal prediction-set target. Its FPR/FNR results are legitimate results for the named adaptation, but neither original ACI’s finite-time/anytime statements nor its long-run coverage interpretation transfers to B4.

Relevant implementation and evidence: **experiments/calibration/controllers.py**, **experiments/calibration/method_registry.py**, and **tests/calibration/test_controllers.py**.

### 2.2 B5a SPOT and B5b dSPOT

**Original reference:** Alban Siffer, Pierre-Alain Fouque, Alexandre Termier, and Christine Largouet, “Anomaly Detection in Streams with Extreme Value Theory,” KDD 2017, pp. 1067–1075, DOI [10.1145/3097983.3098144](https://doi.org/10.1145/3097983.3098144); [paper PDF](https://www.eecs.yorku.ca/course_archive/2018-19/F/6412/reading/kdd17p1067.pdf).

SPOT starts from a high empirical threshold u, fits a generalized Pareto distribution to strict exceedances using Grimshaw maximum likelihood, and derives a rarer-tail alarm threshold from risk q. Here n_init=1000, initial level=0.98, q=0.001, ten Grimshaw starts, and root tolerance 1e-6. An alarm is score > threshold and is excluded from the running observation and peak updates. Source predictions are unavailable through initialization.

dSPOT first subtracts the mean of the last d accepted normal raw scores, then applies SPOT to residuals and restores the local mean to its threshold. Here d=50; the first 50 raw-score warmup events plus 1,000 residual initialization events leave 1,050 predictions unavailable. Alarms update neither the POT state nor the moving mean window.

The repository has independent tests for the GPD fit against SciPy, POT threshold equation and zero-shape limit, alarm exclusion, accepted-peak update, startup, dSPOT residual/window behavior, and checkpoint restore. That supports the label **paper-equation implementation with declared numerical conventions**, not author-code parity. The pinned libspot C source at commit d57e921a08930b960d07adcbec3a839bd228fe94 was not compiled for numerical trace comparison; that C source does not contain dSPOT. The implementation uses NumPy linear interpolation for the initial empirical level, adds deterministic sign-bracket recovery for Grimshaw roots, and uses a strict greater-than alarm. These details are recorded rather than hidden.

The external paper concerns univariate streams and POT tail behavior. The repository applies the equations to scores produced by a changing memory and shared online adaptation, which can make the score stream dependent and nonstationary. Therefore report these as adapted paper-equation comparators on this stream; do not claim their statistical assumptions or guarantees were verified here.

Relevant files: **experiments/calibration/spot_dspot.py**, **experiments/calibration/method_registry.py**, **docs/calibration/SPOT_DSPOT_IMPLEMENTATION.md**, and **tests/calibration/test_spot_dspot.py**.

### 2.3 B6 — Sun-CS

**Original reference:** Sophia Huiwen Sun, Abishek Sankararaman, and Balakrishnan Murali Narayanaswamy, “Online Adaptive Anomaly Thresholding with Confidence Sequences,” ICML 2024, PMLR 235, pp. 47105–47132 ([PMLR paper](https://proceedings.mlr.press/v235/sun24h.html)).

The implementation follows the paper’s quantile-confidence-set, forward/backward intersection change detector, and reset/abstention structure. At a decision, scores above the upper confidence endpoint are anomaly-like score-quantile labels, scores below the lower endpoint are benign-like, and scores inside the interval abstain. The empty prefix abstains. The primary profile uses p=0.99, delta=0.01, clipped endpoints q=0 -> negative infinity and q=1 -> positive infinity; an empirical-extrema endpoint policy is a separately named sensitivity variant.

The source specification was initially marked BLOCKED_FIDELITY before implementation. Current **sun_cs.py** and the later endpoint document implement the paper equations with explicit conventions, independent reference/optimized trace checks, and no author-code parity claim. At p=.99 and delta=.01, the conservative upper endpoint remains at the q=1 boundary until n=353,017; at streams around 3,815 events, this policy cannot label a score as anomalous. Its score-quantile target and abstentions do not equal physical image-level defect truth. The paper’s segment assumptions (including piecewise-stationary score behavior, continuity/no atoms, and change-point spacing) are also not established for the dynamic-memory score stream.

B6 is consequently an **applicability study**, not a binary competitor that B3 can beat. The formal comparison marks B6 applicability-only and supplies no B6 BA/FPR/FNR row. Algorithm 3 has O(T^2 log T) worst-case cumulative work without resets and O(T log T) peak auxiliary space in the current implementation; Stage 1-A has no per-method runtime or peak-memory benchmark.

Relevant files: **experiments/calibration/sun_cs.py**, **experiments/calibration/method_registry.py**, **docs/calibration/SUN_CS_SOURCE_FIDELITY_SPEC.md**, **docs/calibration/SUN_CS_A3_CONSERVATIVE_ENDPOINTS_v2.md**, **tests/calibration/test_sun_cs_reference.py**, and **tests/calibration/test_sun_cs_diagnostics.py**.

### 2.4 Internal and historical comparators

- **B0 Frozen-Theta:** D2 higher q90 over 64 C64 normals under M0, order index 57. It is a basic reference, not a published baseline.
- **B1 RollingQ90QueriedNormal:** W=256, q=0.90, n_min=16; holds theta0 until enough queried verified-normal arrival scores are available, then uses their higher q90. It does not consume the full unlabeled stream and is a strong, transparent simple comparator.
- **B2 OriginalGuardedDualAnchorSidecar:** wraps historical v22 source with its unqueried-normal fallback disabled and source NumPy quantile initialization from C64 M0 scores. It has no native full-loop parity claim and is a historical linkage/adaptation, not an external published algorithm.
- **B3 StateConsistentQuantile:** on a memory-content change, rescales protected C64 normals at the current memory state, refreshes higher q90, and clips movement to 0.01. It is the internal proposed method, not an external baseline.
- **B7 SCDAC-G:** provisional internal selector using descriptive one-sided Clopper–Pearson upper endpoints, queried normal/defect anchors, C64, and movement cost. Defaults include minimum counts 16/8, per-side delta=.05, support risks .20/.20, lambda=.05, and max step .01. Selective labels, dependence, repeated rescoring, and candidate selection invalidate a transferred population or online risk guarantee.
- **Stage075BOffset:** a separate internal raw-median anchor midpoint candidate with max step .01 and a missing-side hold. It is not B8-Full and not a published method.
- **B8:** historical synthetic negative control only; no Stage 1-A Real-IAD row. B8-Full’s synthetic match to B3 cannot support a physical defect comparison.
- **B-EWQ:** an unimplemented causal EWMA/weighted-quantile idea with no frozen equation. It was not authorized for implementation in this audit.

## 3. Existing performance

### 3.1 Evaluation contract and provenance

The existing output is **outputs/calibration/stage1a_formal_controller_comparison_v4_20261030/**. It records 4 development classes (audiojack, regulator, switch, rolled_strip_base), support seeds 0–4 (20 correlated class-by-seed trajectories), 76,475 events, Q-A-TS-v1 at 3% with seed 37, and a fixed zero-based common horizon t >= 1050. Metrics use equal-class macro aggregation through the five trajectory means per class. They are not 20 independent categories. The native score and shared Greedy normal-memory trajectory are common across controller rows; controller-specific AUROC/AUPRC gains are not available in this threshold-only comparison.

The run manifest is COMPLETE. Its actual source HEAD is 5546f037834b780eb71019a70655396499bb7800; its requested frozen source HEAD is 89e2cf4992408cbc5e86facb621749aa7210642b. The requested source is an ancestor; the preflight records only tests/calibration/test_stage1a_final_readiness.py among the intervening source changes. The run’s runner SHA-256 is a9391ab94e9a994198484cd3fb5efc7f534b3515c870723c141c3417ea717c84 and matches the current runner file. All 27 output checksum entries matched in the read-only integrity check. The output directory is local and was preserved; it is not bundled into this documentation commit.

The **four deployed supports plus 64 protected C64 normal references** are part of this comparison. C64 is additional training-normal calibration data and is never called strict four-shot-total. No confirmation-class outcomes, masks, images, raw label vectors, or queried-label provider were accessed for this B0 audit. The existing run audit records predictions sealed before offline evaluation and no unqueried test labels in controller inputs.

### 3.2 Full-stream results

Means are over 20 class/support-seed trajectories. “Worst 256” is the mean, across trajectories, of each trajectory’s worst defined non-overlapping complete 256-image window; BA uses windows with both classes, while FPR/FNR are defined only where their own denominator exists. Missing startup predictions remain unavailable.

| Method | Mean BA | Mean FPR | Mean FNR | Coverage | Mean per-trajectory worst-256 BA / FPR / FNR |
|---|---:|---:|---:|---:|---:|
| B0 Frozen-Theta | 0.6434 | 0.0099 | 0.7033 | 1.0000 | 0.6041 / 0.0690 / 0.7759 |
| B1 RollingQ90QueriedNormal | 0.6812 | 0.0239 | 0.6136 | 1.0000 | 0.6285 / 0.0852 / 0.7196 |
| B2 v22 sidecar adaptation | 0.6458 | 0.0110 | 0.6974 | 1.0000 | 0.6078 / 0.0718 / 0.7656 |
| B3 StateConsistentQuantile | 0.7362 | 0.1042 | 0.4235 | 1.0000 | 0.6530 / 0.1868 / 0.5911 |
| B4 Adapted Dual-ACI | 0.6755 | 0.0261 | 0.6229 | 1.0000 | 0.6149 / 0.1041 / 0.7440 |
| B5a native SPOT | undefined | undefined | 0.9529 (5/20 defined) | 0.7385 | 0.5014 / 0.0008 / 0.9972 |
| B5a-Frozen hybrid | 0.5574 | 0.0047 | 0.8804 | 1.0000 | 0.5014 / 0.0649 / 0.9972 |
| B5b native dSPOT | undefined | undefined | 0.9441 (5/20 defined) | 0.7254 | 0.5035 / 0.0015 / 0.9954 |
| B5b-Frozen hybrid | 0.5609 | 0.0049 | 0.8732 | 1.0000 | 0.5021 / 0.0663 / 0.9954 |
| B7 SCDAC-G | 0.6986 | 0.0945 | 0.5083 | 1.0000 | 0.6253 / 0.2149 / 0.7096 |
| Stage075BOffset | 0.7573 | 0.1718 | 0.3136 | 1.0000 | 0.6317 / 0.3515 / 0.6140 |

For native SPOT/dSPOT, full-stream BA and FPR are undefined on all 20 trajectories because missing startup decisions prevent the full denominators from being evaluated; the table does not impute those predictions. Native full-stream FNR is defined on only 5/20 trajectories. Selective BA among answered events is 0.5078 for SPOT and 0.5095 for dSPOT; it is not full-stream BA. SPOT’s mean normal/defect coverage is 0.7797/0.7349, and dSPOT’s is 0.7683/0.7222. Native source decision fractions are 0.7385 and 0.7254. Their separately named Frozen-Theta hybrids fall back on 26.15% and 27.46% of full-stream events, respectively.

### 3.3 Fixed common horizon, t >= 1050

Every row has 100% normal, defect, and total coverage on this horizon. For native SPOT and dSPOT this is after their startup interval; the fallback hybrids make no fallback decisions in this horizon and therefore exactly match their source-native results.

| Method | Mean BA | Mean FPR | Mean FNR | Coverage | Mean per-trajectory worst-256 BA / FPR / FNR |
|---|---:|---:|---:|---:|---:|
| B0 Frozen-Theta | 0.6375 | 0.0062 | 0.7187 | 1.0000 | 0.6086 / 0.0342 / 0.7662 |
| B1 RollingQ90QueriedNormal | 0.6828 | 0.0252 | 0.6092 | 1.0000 | 0.6452 / 0.0644 / 0.6808 |
| B2 v22 sidecar adaptation | 0.6405 | 0.0073 | 0.7118 | 1.0000 | 0.6095 / 0.0362 / 0.7619 |
| B3 StateConsistentQuantile | 0.7413 | 0.1137 | 0.4038 | 1.0000 | 0.6956 / 0.1741 / 0.4887 |
| B4 Adapted Dual-ACI | 0.6699 | 0.0264 | 0.6337 | 1.0000 | 0.6251 / 0.0684 / 0.7247 |
| B5a native SPOT | 0.5078 | 0.0001 | 0.9842 | 1.0000 | 0.5023 / 0.0008 / 0.9953 |
| B5a-Frozen hybrid | 0.5078 | 0.0001 | 0.9842 | 1.0000 | 0.5023 / 0.0008 / 0.9953 |
| B5b native dSPOT | 0.5095 | 0.0003 | 0.9808 | 1.0000 | 0.5037 / 0.0016 / 0.9922 |
| B5b-Frozen hybrid | 0.5095 | 0.0003 | 0.9808 | 1.0000 | 0.5037 / 0.0016 / 0.9922 |
| B7 SCDAC-G | 0.7160 | 0.1120 | 0.4560 | 1.0000 | 0.6686 / 0.2053 / 0.5854 |
| Stage075BOffset | 0.7564 | 0.1954 | 0.2918 | 1.0000 | 0.6771 / 0.3183 / 0.4701 |

The near-zero SPOT/dSPOT FPR comes with FNR near 0.98 on this operating point. It is evidence about these registered settings on this score stream, not evidence that the method is inferior under every operating objective. B3’s higher BA comes with substantially more false alarms than B0/B1/B2/B4 and the SPOT/dSPOT operating points.

### 3.4 Per-class, paired, and computational detail

Per-class means, all 20 support-seed records per evaluated controller, confusion counts and denominators, separate full/common-horizon metrics, window extrema, defined counts, and descriptive paired effects are preserved in **existing_results_matrix.json**. No raw labels or label vectors are included.

The full-stream paired B3-minus-B1 differences are BA +0.05493, FPR +0.08023, and FNR -0.19009. BA is higher for B3 on 19/20 trajectories; FPR is higher on 20/20 and FNR lower on 20/20. Against Stage075BOffset, B3’s mean differences are BA -0.02112, FPR -0.06765, and FNR +0.10988; the offset method has higher BA on 14/20 trajectories. These are descriptive paired directions, not significance tests.

All threshold-only methods share the same score ranking in this profile, so AUROC/AUPRC cannot distinguish their thresholds. The run records 2,263.34 seconds for the prediction phase across all 20 trajectories; this is not per-method time. No comparable per-method runtime or peak-memory figure is present. Some receipts contain operation counts, but they do not substitute for runtime/memory benchmarks.

## 4. Information and comparison fairness

The Stage 1-A controller input boundary is common: all evaluated methods receive the same current native scalar score, shared memory trajectory, and label-blind query schedule. No threshold controls whether a query occurs or whether a verified normal updates shared Greedy memory. Predictions precede disclosure of the current queried label. The fixed support images are inputs to the common native scorer; the threshold controller does not receive those images directly.

| Method | Four supports through shared scorer | C64 reference access | Queried verified normals | Queried defect feedback | Full causal unlabeled score history | Future scores / unqueried labels / masks | Offline evaluator labels |
|---|---|---|---|---|---|---|---|
| B0 | yes | M0 theta0 only | no | no | no | no / no / no | after prediction seal |
| B1 | yes | M0 theta0 only | yes; stale arrival scores | no | no; only selected queried-normal scores | no / no / no | after prediction seal |
| B2 | yes | M0 initialization | yes | yes when queried | no full-stream buffer | no / no / no | after prediction seal |
| B3 | yes | protected C64 rescored at current M | no | no | no | no / no / no | after prediction seal |
| B4 | yes | current-memory normal/defect anchors | yes | yes when queried | no full-stream buffer | no / no / no | after prediction seal |
| B5a / B5b native | yes | none | no | no | yes; causal score stream | no / no / no | after prediction seal |
| B5a-Frozen / B5b-Frozen | yes | M0 theta0 only for fallback | no | no | yes; causal score stream | no / no / no | after prediction seal |
| B6 | not in Stage 1-A | none in method | no | no | yes; causal score stream | no / no / no | no binary evaluation |
| B7 | yes | protected C64 rescored at current M | yes | yes | no full-stream buffer | no / no / no | after prediction seal |
| Stage075BOffset | yes | current-memory C64 and queried anchors | yes | yes | no full-stream buffer | no / no / no | after prediction seal |

The evaluator uses outcome labels only after prediction sealing for the existing Stage 1-A metrics. That evaluator access is separate from online controller inputs; B0 itself did not open a raw label source. The full per-method access flags and scopes are described in the two JSON inventories.

Three differences matter to interpretation:

1. **B1 versus B3:** B1’s adaptive buffer uses queried-normal arrival scores, while B3 rescales protected C64 normals under current memory. B1 and B3 therefore differ in both data source and memory-state treatment. Their direct comparison is useful as a system-level contrast, not a state-consistency causal estimate.
2. **A2/A3/A4:** these arms hold the frozen scores, memory transitions, C64 identities, cadence, and movement rule fixed while changing the reference-memory lag. This isolates the intended state-version intervention more directly, but the observed effects are weak/mixed and have increased FPR.
3. **Operating point and warmup:** B1, B3, SPOT, and dSPOT use distinct threshold objectives and inputs. B5’s missing startup is part of native behavior and is reported as such. The common horizon is fixed at t >= 1050, not selected after inspecting results. A shared operating constraint was not pre-registered.

A future matched-FPR or false-alarm-budget comparison must choose the operating criterion before looking at confirmation outcomes. Any per-method threshold selection must use a separate, predesignated calibration split (for example normal-only calibration data) and freeze the complete policy before the held-out evaluation. Selecting a threshold from the current development labels and reporting performance on those same classes would leak evaluation outcomes and create optimistic comparisons. If no suitable disjoint calibration normals exist, obtaining them is a data requirement, not something to infer from the test metrics.

## 5. Contradictions and provenance

| Historical statement | Later evidence and date/provenance | Current supported reading | Effect on results |
|---|---|---|---|
| The third-pass baseline ledger says B5a/B5b are blocked and disabled. | The same ledger’s later fourth-pass section and **SPOT_DSPOT_IMPLEMENTATION.md** record paper-equation implementations and tests; current registry says reference-tested; current code is evaluated in Stage 1-A. | B5a/b exist as paper-equation implementations with explicit numerical conventions, but no compiled author-code parity. | The run’s rows are real; claims must state implementation level and startup missingness. |
| The historical Sun source-fidelity spec says BLOCKED_FIDELITY and no implementation. | Commit ced0682 (2026-10-29) records pre-implementation blocked status; sixth-pass code commit ae5bce9 and endpoint correction/document commit cf79855 (2026-10-29) add current B6 source and conventions. | B6 is currently implemented and reference-tested, still without verified author-code parity, and still not a direct physical-defect binary comparator. | No B6 leaderboard number exists; no B3-versus-B6 win claim is valid. |
| The 2026-10-30 method matrix says execution_authorized=false and calls Stage075BOffset synthetic-only. | Matrix commit fd29a31 is a readiness snapshot. The later Stage 1-A run manifest records COMPLETE at source HEAD 5546f037 (2026-10-30); the output contains Stage075BOffset. | Readiness snapshot and later local completed run are different points in the timeline. Do not silently edit or relabel the old snapshot. | Stage075BOffset’s existing numbers are included here with later-run provenance; its absence from method_registry.py is a registry/documentation gap. |
| method_registry.py contains B6 and a B6 fallback; the formal row list omits them. | Current formal audit explicitly marks B6 applicability-only; neither B6 nor B6-Frozen has a Stage 1-A physical-defect numeric row. | Registry presence means implementation availability, not inclusion in the binary leaderboard. | Report as “not scored/not ranked,” not “lost to B3.” |

The Stage 1-A run requested frozen source HEAD 89e2cf4…, used actual HEAD 5546f037…, and its source hash manifest agrees with current code hashes for the recorded runner and comparison components. The aggregate output was checksum-verified, but the local output directory remains untracked and is not included in this audit commit. The audit does not independently replay the predictions.

The A2 review reports that predictions were sealed before offline label evaluation and documents evaluator/report-source repairs after sealing. It also records that the full local prediction/metric artifacts were not independently reproduced from a reviewed pre-seal executable commit. The saved A2 report, manifest, and hashes are retained as evidence; this audit did not re-open outcome vectors. That provenance reservation limits reproduction claims and is not cured by the B0 inventory.

## 6. Scientific implications

B3 is competitive in BA on these four development classes when compared with several registered thresholds, but it is not dominant across error costs. Relative to B1 it exchanges about 8.02 percentage points more FPR for 19.01 points lower FNR and 5.49 points higher BA. Relative to Stage075BOffset it has lower BA and FNR recall, but lower FPR and better mean worst-window BA. These are materially different operational choices. No cost ratio or maximum tolerable FPR was fixed in this audit, so “risk-safe superiority” is unsupported.

The mechanism result narrows the claim further. Full-stream A2 has BA/FPR/FNR 0.73615/0.10417/0.42353; A3 (lag one) has 0.73244/0.09909/0.43604; A4 (lag five) has 0.71617/0.08181/0.48586. A2-A3 is approximately +0.00371 BA, +0.00509 FPR, and -0.01251 FNR; BA is positive on 17/20 trajectories, but FPR is higher on 20/20. A2-A4 is approximately +0.01998 BA, +0.02236 FPR, and -0.06233 FNR; BA is positive on 18/20, FPR higher on 20/20. The latter BA contrast is just below the runner’s descriptive 0.02 meaningful-difference marker; it is not a significance threshold. The independent review records H1_WEAK_OR_MIXED and advises holding a claim of a large, risk-safe state-consistency gain.

For a journal paper, the safe interpretation is a **high-recall, higher-false-alarm operating point on development classes**, with a small/mixed matched state-version effect. B4 can be shown only as Adapted Dual-ACI. SPOT/dSPOT can be shown as paper-equation implementations on the fixed common horizon and as explicitly named startup-fallback hybrids on the full stream. B6 belongs in a separate applicability discussion. The current data do not establish category-level generalization, author-code parity for SPOT/dSPOT/Sun-CS, or superiority under a preselected operational risk constraint.

## 7. Minimum additional experiments for a defensible submission

No experiment is authorized or run by B0. The smallest credible next package is conditional on the paper claim:

1. **Preselect the operating criterion, then do one independent holdout comparison.** Freeze an FPR budget or explicitly state that comparison is at each method’s registered default operating point. If a matched-FPR claim is necessary, use a disjoint normal-only calibration split and lock all thresholds/configurations before the held-out categories are evaluated. Reuse the current sealed-replay runner and unchanged registered methods. This addresses category generalization and operating-point criticism. It cannot be answered by the four development classes. Compute is low-to-medium if frozen native scores/memory receipts exist, otherwise high due to new native inference; evaluator labels are opened only after prediction sealing.
2. **Run strict-four-only sensitivity only if the manuscript claims strict four-shot competitiveness.** Current results use four deployed supports plus 64 additional C64 normals. Reuse the scorer and replay pipeline, but remove C64 access and define each baseline’s legal initialization before the holdout. This addresses a protocol-label criticism. Existing C64 results cannot answer it. Cost depends on whether score/memory receipts suffice; no B0 rerun was made.
3. **Treat author-code parity as optional, claim-dependent work.** If the paper says “libspot-compatible” or “source-faithful,” add a small synthetic trace comparison against pinned author SPOT C code. No Real-IAD data or labels are required; CPU burden is low. Current equation tests already support the narrower statement “paper-equation implementation,” so parity is not needed for that narrower claim. The pinned C source does not implement dSPOT; keep dSPOT tied to paper-equation tests unless an independent implementation is found.

Do not add an open-ended threshold search, tune against the same development labels, expand B7, or implement B-EWQ without a specific reviewer-facing gap. Keep all reported negative and mixed outcomes. The final researcher decision remains pending independent review.

## Audit artifacts and hash references

The machine-readable matrices contain current source hashes, exact Stage 1-A and A2 artifact hashes, per-class/support-seed metrics, denominators, coverage, and selected paired effects. Principal Stage 1-A artifact hashes are:

- per_method_metrics.json: fdd69b8ec38d34c3f95fbdc3fe6c8ad0950d27948eae5ca75e31b3c0d11703e2
- paired_results.json: f9177ad0d68bd9fade9bc38cf79e0cda06c9f20e8ad320bfa254c3a95117174f
- run_manifest.json: 763a17f756f9e1ffcf25790346bda74d23995a166c753384f8f3317597ed957c
- checksums.json: 5e635545d414275f5b8e8339d0ef90110e342f55cd70ef53890c434015f73754
- query-tape binding: dbe72a2f989ee314becc07e0c8b8c7a8353ffacabbfc93863d5666cff9d28183
- frozen label-blind metadata: 48b80202e4b5e02a85b8a950431b386199ee59046d220956062eef25804d0166

External references were checked on 2026-10-30: Gibbs and Candès via the official NeurIPS paper, Siffer et al. via the KDD paper/DOI, and Sun et al. via the official PMLR page. No author-code parity claim beyond the repository evidence above is made.
