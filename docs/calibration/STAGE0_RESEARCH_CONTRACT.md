# Stage 0 scientific research contract (v1 — 2026-10-29)

**Status:** protocol-only; no new Stage 0 code or GPU evidence exists as of this commit. Superseding decision: `docs/Calibration_Falsification_Research_Handoff_2026-10-28_v1.docx`. The stopped Safe Admission / CDARG / SAEG sprint must NOT be resumed implicitly.

## A. Scientific question

Under identical frozen DINOv2 / VisionAD-style scoring configuration, strict-four-support fixed bank, one verified-normal Greedy memory trajectory, identical stream order and identical precommitted label-blind human-query tape, does **memory-state-consistent online threshold calibration** improve full-stream prequential balanced accuracy and error-rate stability over strong online calibration/threshold baselines?

Let y=0 normal and y=1 defect, and let M_t be the **shared** mutable normal memory at time t:

```text
s_t = S_native(x_t, FixedSupport, M_t)
prediction_t^k = 1[s_t >= theta_t^k], for each calibrator k in {B0,...,B7}
BA = 1 - (FPR + FNR)/2
FPR = FP/(FP+TN), FNR = FN/(FN+TP), with unavailable denominators marked NULL
```

Score-ranking metrics AUROC/AUPRC **must be identical** across all calibration-only controllers on the same stream (same sequence and scoring of every observation). Threshold methods may improve BA/FPR/FNR, not AUROC in this profile. Observed difference is a parity/protocol failure.

### Falsifiable research hypotheses
- H1: stale calibration scores under changing M_t explain substantial operating-point error. Test **SCQ** versus B0/B1 at the same memory trajectory; initial P0/P1 evidence is single-class, final-state, exploratory only.
- H2: valid queried defect evidence and explicit missing-side guards add **incremental** benefit beyond current-state normal q90. Test SCDAC-G against SCQ and original Sidecar.
- H3: online external alternatives (ACI-adaptation, SPOT/dSPOT, Sun-CS) may be stronger; use them to diagnose algorithmic defects rather than weakening baselines.
- H4: any gain survives Q-B, temporal class imbalance and independent classes/seeds; no success is guaranteed.

## B. Non-negotiable substrate and information budget

**Frozen main profile (C64):**
- Encoder: frozen DINOv2 feature extractor from historical `vmb_visionad_new.py` / v22 sources. Use `experiments/p0_protocol_audit.py` native settings and audit config values below, not unconstrained new defaults.
- Four distinct training-normal **deployed supports**, original consistent support augmentation/reference setup. `fixed_capacity=25000`, `fixed_per_image=5000`, `augmentations=4`, `include_identity=False`, `multiplier=2.5`, `feature_fuse=concat`, `knn=5`, `image_aggregation=max`, `image_topk=64`, `image_quantile=0.95`, `topk_patches=32`; record exact full `p0.Config` including preprocessing, view settings, dtype/seed. These are the D2/P0 reference settings, **not** proof all historical scripts used these defaults. A check against actual inherited `Config` is required.
- Mutable memory: verified-normal-only original `experiments/v02_memory.py` Greedy implementation, initially empty, `capacity=256` as the Stage 1 primary candidate; dynamic patch capacity semantics must be reported per-layer and total. Do not silently use v22 default K=1024. Memory is identical for every calibrator. No defect bank, Signed Correction, STM/LTM/maturation, gating policy, self-training, BEA, model fine-tuning or mixed scorer.
- **C64 = four supports + 64 additional disjoint training-normal calibration image identities (one camera view per official sample group). NOT strict four-shot-total.** C64 is protected: can be rescored but is never admitted into mutable memory. Separately design strict-four-support-only (no C64) sensitivity later; do not infer strict-four-only performance from C64.
- `theta0`: historical D2 `higher` q90 using the C64 source normal scores under M_0. With n=64 this is the 58th order statistic (one-based), array index 57 (zero-based). Normal tail alpha=0.10. Every controller's initial threshold and its own initial information source must be recorded. No probe labels used for theta initialization.
- Source file/reference integrity: fixed memory SHA, dynamic bank version/digest, support/C64/group identities, original output manifest, exact scorer config hash.
- For algorithms using scores, unlabelled **whole-stream** s_t is available after scoring; current image label is available only after its precommitted query is delivered. Protected C64 normals (and their known normal labels) are available under the C64 access regime to every eligible method. No defect labels before query. Metadata with offline labels is held by evaluator, not controller.

### Historical comparability is NOT protocol identity
Historical `experiments/risk_safe/query_protocol.py` builds label-blind grouped Q-A with nested priority budgets. The **new** principal calibration Q-A-TS is ~3% of *image observations* with 16 predefined time bins and at most one queried camera image per `official_sample_key`; Q-B is a precommitted mild early warm-up. Neither may masquerade as the archived Q-A. Implement as a **new query protocol version and digest**; preserve historical file and seed 37 as provenance, not evidence of equivalence.

For each full stream with N image events, B=round(0.03*N) with a single registered rounding rule (Python round/bankers vs half-up must be declared; recommended `floor(0.03*N+0.5)`), unique eligible groups. Predeclare 16 temporal bin edges and within-bin allocation using only label-blind group availability. If requested B is infeasible under the group limit, fail closed. Queries are drawn using fixed hash/seed before seeing labels; actual early counts must be audited, not selected via class outcomes.

Q-B: allocate approximately 20% of B to earliest 400 observations (if eligible), remaining 80% to the rest, no label awareness. Example for N=3815, B=114 under half-up, early target 23, later 91; subject to actual stream length/eligibility and explicitly recorded rounding. Q-C 40% early is optional **stress**, not main. No query schedule should be tuned based on anomaly prevalence.

**Critical distinction:** official sample key is verified camera-grouping metadata, not a proven distinct physical manufactured item. Query at most one view per key, and count both queried **images** and queried **groups**; report both image and group denominators.

## C. Exact causal event contract

At event t, before any update:
1. Read stream image x_t, group/view/time metadata without label. Compute ONE original-native scorer s_t using M_t; preserve raw native score and score-map provenance.
2. Every calibrator predicts with its own pre-update theta_t. No calibrated prediction influences the shared memory, label tape or score.
3. Deliver **precommitted**, label-blind Query Tape decision; only if requested, reveal label to permitted feedback handlers and ALL controllers simultaneously.
4. For queried **verified normal** image only, use common reference, pre-registered threshold-independent candidate proposal and original Greedy append/replacement. If rejected/ineligible, record. Query defects NEVER enter normal memory.
5. Increment memory_version only on actual content change. Recompute eligible protected and queried historical anchor scores against the new M_version, using source-aware LSO for anchors whose originating event contributed mutable prototypes; never leak labels of unqueried data. No full backbone rerun for cached frozen anchor features.
6. Let every controller observe the current unlabeled score and any newly revealed feedback under its registered information privileges, refresh current-state references if needed, update **theta_{t+1}** for next event. For SPOT/dSPOT the score-only observation may update despite no query; for feedback algorithms it cannot use a missing label.
7. Log `t, sample_key, camera, query_event, label_disclosed, native_score, theta_pre[k], pred_pre[k], theta_post[k], evidence_state[k], memory_version_before/after, bank hashes, anchor rescore versions, computational costs`. Evaluator stores true y_t in a separate private outcome ledger, never passed to methods.

**Timing ambiguity rule:** a controller may use current score s_t without its label to prepare theta_{t+1}, not to retrospectively alter prediction at t. The single exception would be a separately registered transductive score model and must be excluded from the primary comparison.

### Locked reference Greedy writer
The source `v02_memory.MemoryArm.update` calls `impl.select_informative_normal_write(score,theta,sigma,...)`. Passing each controller's theta would change keep=8/16/24, future M and future scores (invalidates calibration isolation). Implement ONE shared frozen reference writer with a precommitted reference scoring stream and theta_ref/sigma_ref policy, **never a controller theta**. The exact writer source/config, pre-event reference score timing and feature normalization must be disclosed and matched to native behavior in parity tests. Source Greedy `greedy_slot` and normalisation are reused. The reference decision is permitted to derive from shared static/native score and precommitted theta0, but cannot consult future labels or another controller's updated theta.

## D. Eight-arm primary matrix; all must run on ONE shared score tape

| ID | Method | First-round operational definition | Status |
|----|--------|------------------------------------|--------|
| B0 | Frozen-Theta | theta_t=theta0 | simple control |
| B1 | Rolling-q90 | rolling eligible label-verified normal buffer, fixed window/quantile and missing-data HOLD; must distinguish optional unlabeled rolling variant | simple control |
| B2 | Original Guarded Dual Anchor / Balanced Rescue Sidecar | source-faithful v22 rule; same queried labels, explicit adaptations for native v22 interface | historical external-to-proposed comparator |
| B3 | SCQ | current M-version, C64 normal-only q90, source exclusion if using queried normals, precommitted step clip | simple **state-consistent falsifier** |
| B4 | State-consistent Dual-ACI | class-conditional error feedback on queried labels, two online ACI states and current-state anchor-score quantiles | adapted external |
| B5 | SPOT and dSPOT | original EVT score-only stream tail thresholding; report SPOT and dSPOT separately as B5a/B5b if both implemented | strong external; no selective-label handicap |
| B6 | Sun-CS online anomaly thresholding | faithful source algorithm if assumptions can be met; otherwise explicitly adapted under sparse disclosure with lost guarantees | strong external, fidelity gate |
| B7 | SCDAC-G | current-state normal+queried-defect score ledger, two-sided empirical risk/bounds, statuses BOTH/NORMAL_ONLY/DEFECT_ONLY/NONE/CONFLICT, bounded update and explicit missing-side handling | proposed candidate, not validated |

**Important:** B5a and B5b are distinct reported rows; "eight family slots" does NOT imply silently averaging two algorithms. Methods may legitimately have different **legal** score-only/feedback use. Compare on common labels and evaluate common query cost; distinguish identical availability from identical consumption.

No hyperparameter value without a citation or an explicit **proposed** preregistered design; strong baselines are not intentionally under-tuned. Mathematical details and unresolved fidelity gates live in `BASELINE_FIDELITY_AND_REFERENCES.md`.

## E. Evaluation separated from algorithm state

Primary: **whole-stream prequential BA**, FPR and FNR with exact confusion counts. Report unweighted class macro and support-seed uncertainty separately; BA undefined if either class absent over the evaluated horizon. Diagnostic: worst **defined** 256-image window FPR, FNR, BA; windows lacking a class have unavailable class-specific risk, never zeros. Other metrics: time-to-first queried defect, time to both class evidence, theta drift/TV/lag, long consecutive errors, memory updates/replacements, anchor refresh and leave-source-out costs, CPU/GPU latency and allocated state bytes, query fraction by images and groups, AUROC/AUPRC shared-score invariants. Offline ground truth evaluation only.

Historical P0/P1 is *final-state* Probe-A diagnostic, not full-stream prequential and not independent confirmation. Do not reuse its final-state numbers as Stage 1 predictions.

## F. Stage 1 / Stage 2 preregistration, not permission to run now

- Stage 1A: four dev classes `audiojack`, `regulator`, `switch`, `rolled_strip_base`; support seed 0; native stream; Q-A-TS first. Four shared trajectories, multiple calibration decisions per trajectory. Check C64 availability separately by class and fail infeasible combinations.
- Stage 1B: same development classes/config with separate Q-B mild-warmup tape.
- Stage 2: Normal-first/Defect-first on `switch` and `rolled_strip_base` (labels used to create stress order only; controller blinded); then untouched `mint`, `pcb` two support seeds only AFTER method freeze. Stage 2 confirmation outcomes must not be explored for tuning.
- Engineering R&D GO *targets* from 2026-10-28 handoff (not significance tests): meaningful dual-side activation >=3/4 dev classes; macro BA +0.02 over strongest admissible baseline in >=3/4 classes, no class BA loss >0.03, worst-window FNR loss <=0.05, independent confirmation improvements across seeds; if not met, report failures honestly. No forced passing.
- Up to two explicit, mechanism-hypothesis-driven algorithm revisions on development data before frozen confirmation. All versions and failures remain archived.

## G. Future full-system integration without contamination

All modules are pluggable with explicit state ownership:
- scorer: immutable DINOv2 encoder, bank/version-aware native anomaly score,
- memory: Greedy, future independently evaluated admission/hierarchy,
- correction: OFF now, later signed local correction with raw and corrected scores separately logged,
- query: Q-A-TS/Q-B now, later BEA closed-loop,
- calibration: independent controller registry.

Three profiles: `calibration_isolated` (shared M_t, same s_t, query tape); `factorial_isolated` (controlled factor contrasts, recompute score for divergent memory/correction); `full_closed_loop` (system-specific memory/query/score trajectories and feedback). The common event engine must **not** assume identical scores across profiles. Full closed-loop results do not substitute for conditional calibration-only attribution.

## H. Stage 0 scope and acceptance

Stage 0 delivers protocol manifests, native bridge/source parity, prequential engine with evaluator label firewall, new Q-A-TS/Q-B tape builders, component API, versioned anchor ledger, eight baseline/controller families with honest fidelity statuses, synthetic mechanism activation and tests, one-command resumable execution/verification, and generated report. No claims from Real-IAD tuning; no Stage 1/2 GPU tasks until explicit human authorization.

A proposed method is not implemented merely by having a Python class; functional equations, source fidelity, tests and declared missing-evidence behavior must pass. Critical failures stop automatically. Do not modify original v22/vmb/source experiments/history or original outputs.
