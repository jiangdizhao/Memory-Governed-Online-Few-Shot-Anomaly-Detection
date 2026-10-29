# B8 v0.2 research design — Evidence-Conditioned Memory-State Calibration (EC-MSC)

**Status: RESEARCH HYPOTHESIS / DESIGN CANDIDATE; NOT IMPLEMENTED, NOT PERFORMANCE-VALIDATED, NOT FROZEN.**  
**Recorded:** 2026-10-29; follows the 20-trajectory Stage 0.75-A exploratory audit (commit `5e855c5be402f3dfd70a0294f894a5846b4f235b`).  
**Research objective:** develop a **new, independently justified threshold controller** whose **class-macro Balanced Accuracy (BA) over matched class × support-seed trajectories exceeds the actual B3 SCQ controller**. It is acceptable for B8 to match or slightly underperform B3 in an early normal-only feedback regime **if** other regimes produce a credible net macro improvement and adverse class-wise FNR/memory costs are transparently reported. **This is a falsifiable research target, not a guaranteed outcome.**

Version history: [B8 v0.1 MTEAC design](STAGE075_B8_MTEAC_RESEARCH_DESIGN_v0_1.md) remains unchanged as the historical paired-median-transport proposal. v0.2 **changes the preferred mechanism** to normal-reference percentile coordinates, retained defect-informed decision information, and baseline-aware conservative action. The old median-transport path remains a named ablation, not the main proposed solution. Read together with the [Stage 0.75 agenda](STAGE075_RESEARCH_AGENDA_v0_1.md), [0.75-A ranking audit](STAGE075A_GREEDY_RANKING_FALSIFICATION_REPORT.md), [0.75-A C64 transport audit](STAGE075A_C64_MEMORY_TRANSPORT_REPORT.md), and [Stage 0.75-B synthetic protocol candidate](STAGE075B_SYNTHETIC_PROTOCOL_CANDIDATE_v0_1.md).

## 1. Scientific question, prior-art boundary and intended contribution

**Problem:** a normal-only verified Greedy memory writer changes the anomaly scoring function over time; the operating threshold may become misaligned. Sparse, selected defect feedback may be temporally absent or unrepresentative. How can a controller retain usable defect-informed decision information **in the current memory-dependent score coordinate system**, without treating reference-normal C64 scores as calibrated production risk?

**Do not claim the following is new:** memory improves AUROC while an old threshold worsens FNR; adaptive threshold control changes the FPR/FNR operating trade-off. Both were already described in the rejected Pattern Recognition manuscript's staged attribution and guarded sidecar (historical Table 4, M1–M4). The present work must distinguish itself through **a specifically defined, falsifiable state-transfer and conservative decision algorithm** with matched-score/controller ablations, not by rediscovering that phenomenon.

Hypothesized B8 contribution (must be experimentally established):
1. **Current-memory normal-reference coordinate:** represent decision position by a reference percentile rather than a raw, possibly obsolete score.
2. **Defect-evidence residual memory:** retain/refresh legal defect-informed deviations from the contemporaneous B3 baseline, rather than blindly resetting to current C64 q90 after each memory update.
3. **Evidence-conditioned baseline deviation:** estimate *incremental* BA/Youden gain against an **actual run of B3**, penalize unreliable evidence and movement, and cautiously shrink/fallback when support is inadequate.
4. **Causal, auditable memory/version coupling:** no controller-dependent memory writes or query choices; source-group and leave-source-out (LSO) constraints remain enforced.

Avoid claiming generic "conformal validity", distribution-free FPR/FNR, Bayesian posterior, active-inference/free-energy optimality, guaranteed B3 improvement, or theoretical novelty merely by citing adjacent literature.

## 2. Evidence that motivates, and limits, this design

### 2.1 Stage 0.5, exposed development data
- Four development classes: `audiojack`, `regulator`, `switch`, `rolled_strip_base`; 5 preregistered deployed support seeds `(0,1,2,3,4)`; four training-normal supports **plus 64 additional protected C64** normal references. Exactly the frozen Q-A-TS-v1 3% seed37 label-blind tape. Mint/PCB confirmation classes remain sealed.
- The 200 sampled legal query schedules at 3%/5%/7% eventually provided ≥8 and ≥14 defects in all four categories; **feedback arrival is still delayed**. Audiojack seed37 first queried defect at event 1305 and eighth at 1732. These are empirical finite-seed, JSON-order **development** observations, not natural production arrival probabilities.
- Zero of the 20 static M0 configurations met the exploratory near-ceiling BA rule. Static-score ranking and *single-threshold* BA headroom cannot be extrapolated directly to dynamic Greedy.

### 2.2 Stage 0.75-A; fixed Greedy, no controller
- All 20 class × support-seed **causal prequential** dynamic Greedy trajectories improved full-stream AUROC and AUPRC against paired M0, as-operated; mean `ΔAUROC`: Audiojack +0.0693, Regulator +0.1537, Switch +0.1281, Rolled Strip Base +0.0575. Conservative group-clean differences remained similar; fixed-snapshot future-cohort results mostly supported improvement but contain negative/undefined late windows. These are exposed development results, not population generalization.
- Four-class equally weighted mean BA (five-seed means): M0 + frozen theta0 ≈0.6386, Greedy + frozen theta0 ≈0.6434, Greedy + **read-only, unclipped** current-C64 q90 ≈0.7371. **The latter is NOT B3**, because actual B3 applies `Δ=0.01` and event/update conditions. It is not a proven B8 achievable benchmark.
- Four-class mean FNR: M0+frozen theta0 ≈0.6152; dynamic Greedy+stale theta0 ≈0.7033; dynamic+unclipped current C64 q90 ≈0.4204. These comparisons support the *need to study calibration*, **not** a new method's superiority.
- Among 1,190 actual Greedy content changes, paired C64 median score shift had median zero and mean ≈−0.0000039; q90 changed 347 times (295 down, 52 up) and was unchanged 843 times. Normal anchor distributions can change in the tails with zero median shift. Median translation is therefore not the assumed default; finite C64/ties/rank changes matter. C64 evidence says **nothing by itself about defect-score transport**.
- Group-clean controls excluded future same-group views after earlier admission. Five camera views per official group must not be treated as five independent manufactured objects.

### 2.3 Existing baselines (freeze their real implementations)
- **B3 SCQ:** current-memory higher C64 q90, its own state and `Δ=0.01` update clip; `experiments/calibration/controllers.py::StateConsistentQuantile`. Instantiate the **actual** controller as a shadow baseline; do not silently substitute a direct q90 diagnostic.
- **B7 SCDAC-G:** descriptive CP-informed two-sided candidate and bounded motion with 16/8 evidence gates and CONFLICT behavior; `experiments/calibration/scdac_g.py`. B7's CP endpoints are **descriptive** under selective labels/adaptive memory, not certified population risks.
- **B3+raw-offset** must become a stronger *new* ablation: a simple defect-informed offset on the exact same B3 reference using identical legal queried labels. The additional defect labels alone must not be mistaken for an original B8 innovation.
- B0, B1/B2, ACI/SPOT/dSPOT/Sun-CS, their adapted/fallback variants and scientific provenance remain intact for later full Stage 1 reporting. Do not remove competitors to favor B8.

## 3. Shared causal state / permitted inputs

At event \(t\), the frozen backbone and native v22 scores use current shared memory \(M_t\), with prediction **before** any present-event feedback or Greedy write. The writer admits **only queried verified normals**, is controller-independent, uses original `theta_ref`, `sigma_ref`, selection and native Greedy K=256 **per layer**. The frozen query tape is likewise controller-independent.

B8 may read: \(s(x_t;M_t)\), current C64 image identities and their rescored values, current memory content digest and true writes, valid source-aware LSO queried-normal anchors, valid queried defect labels *with their current-memory rescored score vectors*, their immutable source/timestamp/group provenance and original prediction/feedback order, and the live internal state of a separately replayed **real B3** controller. All controllers share one scorer/memory/query trajectory.

B8 may NOT read: unqueried test labels, labels of future events, full-stream defect prevalence, offline oracle threshold, masks, Mint/PCB confirmation outcomes, controller-derived pseudo-labels treated as truth, or an extra reader/writer path. Re-scoring an old queried defect in a newer memory gives a fresh **score**, not another independently acquired defect **label**.

C64 normal anchors may be rescored; they are **read-only train normal**, not 64 new online queries and not guaranteed representative of production normals. Source-identity/group exclusion is required for admitted queried normals. Keep genuine content revision separate from bookkeeping and repeated zero-change rescoring.

## 4. Memory-state normal-reference coordinates

Let \(Z=\{z_1,\dots,z_{64}\}\) be the fixed C64 anchor identities. Under \(M_t\), define C64 normal scores and a reference empirical CDF:
\[
c_{i,t}=s(z_i;M_t),\qquad
\widehat F^{C}_{0,t}(a)=\frac1{64}\sum_{i=1}^{64}\mathbf 1[c_{i,t}\le a].
\]
For score \(s_t(x)\), define \(\displaystyle u_t^C(x)=\widehat F^{C}_{0,t}(s_t(x))\). This is a **C64-relative rank coordinate**, not \(P(Y=1\mid x)\), calibrated production probability, or guaranteed \(F_{0,\mathrm{stream}}\).

The empirical CDF has discrete steps of \(1/64\), ties and saturation beyond the anchor range. A future implementer must explicitly choose **one** reproducible convention for ties, empirical CDF and inverse-quantile score mapping, including off-grid B3 thresholds. Candidate decisions must ultimately be executed in the **original score domain** with \(\widehat Y_t=\mathbf1[s_t(x)\ge\theta_t]\) (retain the native controller convention), not by silently changing the equality case when converting to \(u\).

**Useful but narrow invariance:** if every anchor and evaluation score is subjected to the same *strictly increasing* scalar transform \(g_t\), then the finite rank relation \(\widehat F^C_{0,t+1}(g_t(a))=\widehat F^C_{0,t}(a)\) holds with consistent tie handling. Native Greedy is **not guaranteed** to induce a common monotone transform; conditional defect distribution can change independently. This result does not prove preserved BA.

The current normal reference q90 is \(q_t=Q^{\mathrm{higher}}_{.90}(c_{\cdot,t})\), and a diagnostic robust scale is \(\sigma_t=\max\{Q_{.75}(c_{\cdot,t})-Q_{.25}(c_{\cdot,t}),\epsilon\}\). Use \(\epsilon>0\) only as a declared numerical stabilizer; its value and quantile/interpolation policy remain to be frozen. A *standardized residual* \(k_t=(\theta_t-q_t)/\sigma_t\) is a useful **ablation state**, not proven superior to percentile rank. The primary proposed B8 state is a baseline-relative decision coordinate \(u_t\) and a legally updated **defect-informed deviation from live B3**.

## 5. Online BA/Youden objective and the B3-relative decision

For a *fixed* score distribution with real class labels and higher score = defect:
\[
\mathrm{BA}(\theta)=\frac12(\mathrm{TPR}+\mathrm{TNR})
=1-\frac12(\mathrm{FNR}+\mathrm{FPR}),\quad
J(\theta)=\mathrm{TPR}-\mathrm{FPR}=2\mathrm{BA}-1.
\]
Maximizing BA equals maximizing Youden \(J\). **Evaluation** is class-macro BA, not pooled-image Accuracy, FPR-only, or in-sample oracle optimum.

On legal, deduplicated, current-memory rescored queried anchors, provisional class-conditional error proxies for any candidate raw-score threshold \(\theta\) are:
\[
\widehat{\mathrm{FNR}}^Q_t(\theta)
={\sum_{j\in D^Q_t}w_{j,t}\mathbf1[s(x_j;M_t)<\theta]\over\sum_{j\in D^Q_t}w_{j,t}},
\]
\[
\widehat{\mathrm{FPR}}^{Q,N}_t(\theta)
={\sum_{i\in N^Q_t}v_{i,t}\mathbf1[s(x_i;M_t)\ge\theta]\over\sum_{i\in N^Q_t}v_{i,t}}.
\]
Here \(D^Q_t\) are *distinct source-aware queried verified defects*, \(N^Q_t\) legal *queried verified normals* with LSO where required; weights/temporal decay and effective count are **UNSET design choices**, never invented probabilities or independent replicated samples. If a denominator is zero, that term is **unavailable**, not zero. C64 may supply a separate normal-reference proxy \(\widehat{\mathrm{FPR}}^C_t(\theta)\), but it cannot be silently substituted for stream FPR or pooled with queried normals as if IID. Query selection and changing memory can bias all empirical proxies.

Let \(\theta_t^{(3)}\) be the threshold of the **actual B3 shadow controller** given identical legal update events, memory, and C64, including its \(\Delta=0.01\). Define a provisional paired incremental gain (only where both class proxies exist):
\[
\widehat{\Delta J}_t(\theta;\theta_t^{(3)})
=\left[\widehat{\mathrm{FPR}}_t(\theta_t^{(3)})
-\widehat{\mathrm{FPR}}_t(\theta)\right]
+\left[\widehat{\mathrm{FNR}}_t(\theta_t^{(3)})
-\widehat{\mathrm{FNR}}_t(\theta)\right].
\]
Its half is an *estimated* \(\Delta\mathrm{BA}\) on the **selected, adaptive evidence**, not true online population improvement. Reference-only normal estimates, any unmodeled propensity/selection bias, and lack of evidence must be made explicit in the audit.

The *candidate research decision* maximizes a penalized, baseline-relative objective on a predeclared finite threshold/percentile candidate set:
\[
\theta_t^\star\in\arg\max_{\theta\in\Theta_t}
\Big[
\widehat{\Delta J}_t(\theta;\theta_t^{(3)})
-\mathcal U_t(\theta)
-\lambda_{\rm base}\,d(\theta,\theta_t^{(3)})
-\lambda_{\rm move}\,d(\theta,\theta^{B8}_{t^-})
\Big].
\]
\(\mathcal U_t\) denotes a **provisional uncertainty penalty** for few/selected/dependent/stale labels and candidate search; there is no verified confidence sequence or lower confidence bound. Candidate set, penalty, tie breaks, error weighting, propensity assumptions and \(\lambda\)'s are **NOT YET FROZEN**.

An *evidence gate* chooses \(\theta_t^{B8}\) **near B3** when expected incremental benefit is unconvincing, and allows a bounded deviation only when benefit exceeds uncertainty and movement costs. A *graded shrinkage* candidate is \(u^{B8}=u^{(3)}+w_t(u^\star-u^{(3)})\), \(0\le w_t\le1\), but it must be inverted and motion-bounded in raw score space. \(w_t\) must depend on legally available independent-source label counts, temporal coverage, current-memory rescore validity and benefit uncertainty, not on unqueried outcomes. The original B7 rigid 8-defect gate is **not assumed for B8**; a hard minimum or soft gate must be justified later.

**Critical fallback subtlety:** if B8 previously departed from B3, a request to return to \(\theta_t^{(3)}\) can exceed B8's per-event \(\Delta\) motion limit. A correct controller must either (i) project the B3 target into B8's feasible next-step set and **record delayed rejoining**, or (ii) state a separately justified escape rule. Do not pretend that fallback is immediately exact after a prior large deviation. If no prior deviation and the same update rules apply, an exact B3 fallback should be testable. Baseline proximity does NOT imply formal no-worse BA.

## 6. Evidence validity, asymmetric arrival, and state management

Provisional regimes (not fixed algorithmic transitions): `NORMAL_REFERENCE_ONLY`, `DEFECT_SCARCE`, `BOTH_VALID`, `EVIDENCE_STALE`, `CONFLICT`, `BASELINE_REJOIN`.

- **Long normal-only prefix:** if verified normals change \(M_t\), the B3 reference and normal coordinates update; B8 is allowed to track/reference B3 even without defects, not invent defect risk. Early-window parity or small inferiority is acceptable **only as part of an eventual net-macro improvement**.
- **Delayed defects arrive:** retain the source identity and immutable label, rescore in the *current* memory, measure newly available paired relative benefit and uncertainty; never pretend the eighth sample guarantees 0.20 FNR.
- **Long interval without new defects but memory changes:** refresh legally known defect anchors at current memory; score staleness and **concept-age/representativeness staleness** are distinct. A fresh score does not restore unknown current defect distribution.
- **Defect-only interval / no admitted normals:** Memory and C64 may be constant. B8 cannot generate memory adaptation signal simply from time passage; label updates may still change its defect-evidence state.
- **Normal-reference mismatch:** C64 may differ from current-stream queried normals; monitor provenance-aware disagreement and shrink or HOLD when reference mismatch is large. No claim C64 q90 = production FPR 10%.
- **Selective label bias and correlated views:** one official sample group can have five camera views. Query selection must be tape-fixed; distinct images and rescores cannot be treated as independent physical products. Sources contributing to memory require true group-aware LSO for reused normal anchors.
- **High score overlap:** no threshold controller can repair rank inseparability for a fixed score vector. `CONFLICT` should distinguish lack of class evidence, broad uncertainty, actual empirical overlap, invalid evidence, and incompatible movement constraints.
- **No gain evidence:** B8 should avoid oscillating or changing thresholds solely because the same reference is rescored under unchanged memory.

## 7. Baselines, mechanism ablations and falsification

All arms must have **identical** native Greedy score tape \(S_t\), current C64, support seed, query tape, independent writer and allowed labels. Threshold-only arms therefore **must have identical AUROC/AUPRC**; any divergence is an implementation error.

| Arm | Scientific reason |
|---|---|
| **B3 actual shadow SCQ** | Hard, strong primary comparator; real update timing, \(\Delta=0.01\), exact equality convention. |
| B7 SCDAC-G v0.1 | Current bilateral descriptive risk/CONFLICT and HOLD baseline. |
| **B3 + simple raw defect offset** | Tests whether simply adding queried defect labels explains B8 gains. |
| B8-M (v0.1 paired median shift) | Explicitly falsify a translation-only mechanism. |
| B8-S (q90 + normalized IQR residual) | Tests whether scale-normalized residual alone is sufficient. |
| B8-P (percentile transport, simple rule) | Tests percentile-coordinate invariance without full evidence gating. |
| **B8-Full** | Current normal percentile state + defect evidence + conservative B3-relative optimization/validity. |
| B0 and other frozen Stage 1 comparators | Preserved for eventual separate full evaluation; do not retroactively change protocol. |

Primary future *development* measure:
\[
\Delta_{\mathrm{macro}}^{BA}
=\frac1{4}\sum_{c\in\mathcal C}\frac1{5}\sum_{s=0}^{4}
(\mathrm{BA}_{B8,c,s}-\mathrm{BA}_{B3,c,s}).
\]
The researcher's desired Go/No-Go performance direction is \(\Delta_{\mathrm{macro}}^{BA}>0\), under a **precommitted** reporting/hyperparameter-selection plan. Do not assert achieved or significance-tested improvement before running the full matched comparison. Category-level FNR damage, minimum per-class \(\Delta\), worst 256-image defined window, absolute threshold movement, runtime and update frequency are mandatory diagnostics. Do not demand uniform dominance in every time segment, but do not conceal severe harm behind macro gain.

**Novelty falsification:** even if B8-Full beats B3, a gain that vanishes against **B3+simple-offset** or B8-S means percentile/evidence machinery may be unnecessary. B8's contribution is *incremental mechanism*, not extra queried labels alone. The four development classes were exposed during Stage 0.5 and cannot yield independent holdout confirmation. Mint/PCB remain untouched until final method/version/hyperparameters freeze and explicit authorization.

## 8. Synthetic tests needed before any Real-IAD B8 evaluation

S0 exact B3 fallback with no B8 deviation; S1 translation, positive affine and strictly monotone nonlinear common transforms with tie handling; S2 sparse C64 tail-only modification with median shift zero; S3 C64/defect transforms that differ; S4 delayed/no defect feedback then legal returns; S5 no normal admission and unchanged memory; S6 conflicting errors, few defect anchors, heavily imbalanced/correlated query groups; S7 score-state versus concept-stale labels; S8 C64-vs-current-normal shift; S9 motion/step-limit, B3 rejoin after deviation, repeated same memory revision; S10 extreme percentile saturation/ties/zero IQR; S11 identity/source-aware LSO and repeated anchor rescoring; S12 frozen-score AUROC identity for threshold-only controllers; S13 checkpoint/replay determinism. For each test specify desired behavior, counterexample and unavailable risk claims. Full details in [Stage 0.75-B synthetic protocol](STAGE075B_SYNTHETIC_PROTOCOL_CANDIDATE_v0_1.md).

## 9. Theory/literature anchors and open issues

Theoretical inspiration, not source-code parity or direct guarantee transfer:
- **Youden J / balanced classification:** maximize sensitivity + specificity − 1; classical operating-point criterion. Under genuinely representative class-conditional samples, \(J=2BA-1\).
- **Gibbs & Candès, Adaptive Conformal Inference Under Distribution Shift**, NeurIPS 2021: https://proceedings.neurips.cc/paper_files/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html — online calibration, but their feedback/coverage assumptions do not follow here.
- **Tibshirani et al., Conformal Prediction Under Covariate Shift**, NeurIPS 2019: https://papers.nips.cc/paper/8528-conformal-prediction-under-covariate-shift — distribution-weighting ideas under explicit assumptions; do not assert they apply to arbitrary class-conditional drift.
- **Conformal Risk Control**, ICLR 2024: https://openreview.net/forum?id=33XGfHLtZg — explicit loss/risk framing, not an automatic B8 coverage certificate.
- **Sun et al., ICML 2024**, https://proceedings.mlr.press/v235/sun24h.html — reference for quantile anomaly control under a different event definition and assumptions; not physical-defect FNR control.
- **B3/B7 implementation**, `experiments/calibration/controllers.py`, `experiments/calibration/scdac_g.py`, `experiments/calibration/native_event_replay.py` and the linked 0.75-A audit — source of the exact local comparison contract.
- An eventual related-work/novelty claim must separately verify source papers and explicit overlap with reference quantile mapping, selective-label debiasing, conservative policy improvement, and state transfer. These are not solved just by citing the above list.

Open theory questions: finite-C64 rank resolution, tie-aware inverse mapping, class-conditional selection bias, adaptive candidate search optimism, dependence/LOO, memory-induced representation change not expressible as monotonic transform, uncertainty estimation with sparse defective samples, baseline regret versus future policy, and end-to-end runtime.

## 10. Authorization and decisions still open

This document **records an agreed research direction, not an accepted implementation or a final mathematical specification**. The following remain unset: empirical CDF inverse/tie convention, candidate set, label weights/decay, reliability gate/uncertainty penalty, minimum defect count, \(\lambda\) values, update frequency and state transition details, movement/rejoin rule, effective-sample-size estimation, class-risk proxies, and any claim of formal statistical validity.

Next action is a **separately authorized Stage 0.75-B synthetic implementation and mechanism audit**, not unapproved Real-IAD tuning. New experiment results must be recorded without deleting failures. Stage 0.75-C, Stage 1/2 and confirmation access remain HOLD. **No code, GPU run, method registry or old audit is changed by documenting this design.**
