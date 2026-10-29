# Stage 1 evaluation protocol — frozen candidate v2

**Status: `RESEARCHER_APPROVAL_REQUIRED`. Stage 1 is not authorized by this
document.** This candidate freezes an evaluation plan for independent review
before any Stage 1 experiment. It does not certify a method, a risk guarantee,
or expected Real-IAD performance.

## 1. Research question and scope

The proposed comparison asks whether memory-state-consistent online threshold
calibration improves prequential physical-defect decisions under sparse
queried feedback when every calibration-only method receives the same native
score and shared controller-independent memory trajectory.

The experiment uses the frozen DINOv2/VisionAD feature path and native v22
scorer, exactly four deployed training-normal support images, 64 *additional*
protected training-normal C64 calibration anchors, and one verified-normal-only
Greedy bank with capacity K=256 per feature layer. C64 is rescored when memory
changes and is never written to the mutable bank. This is not strict
four-shot-total.

The outcome target for evaluation is the Real-IAD physical-defect label
`Y_defect,t`. Sun-CS's source target is the score-quantile event
`Y_quantile,t = 1[S_t > Q_Ft(p)]`. Those are different targets.

## 2. Method families and comparison roles

### Group I — core sparse-feedback calibration comparisons

| ID | Registered method | Information and role |
|---|---|---|
| B0 | Frozen-Theta | `theta0`; shared score and no feedback adaptation. Control. |
| B1 | Rolling-q90 queried normal | Queried verified-normal labels and stale arrival scores; fixed W=256 and n_min=16. |
| B2 | Historical v22 Guarded Dual Anchor sidecar | Native sidecar wrapped to receive C64 scalar scores and queried labels; source normal-only fallback disabled because unqueried labels are unavailable. |
| B3 | State-Consistent Quantile (SCQ) | Protected C64 current-memory q90; one-sided normal evidence; Delta=0.01. |
| B4 | Adapted Dual-ACI | Queried-label class-conditional error updates and current-memory anchor bounds; explicitly adapted, with no ACI guarantee claim. |
| B7 | SCDAC-G v0.1 | Protected C64, queried verified-normal LSO anchors, and queried defects; provisional descriptive two-sided constraints and bounded movement. |

### Group II — unsupervised online score-threshold references

| ID | Registered method | Information and source target |
|---|---|---|
| B5a | SPOT | Every causal unlabeled score; upper-tail EVT alarms. It ignores physical labels and queries. |
| B5b | dSPOT | Every causal unlabeled score; detrended upper-tail EVT alarms. It ignores physical labels and queries. |
| B6 | Sun-CS Algorithm 3, conservative extended-real primary | Every causal unlabeled scalar score; unsupervised score-quantile anomaly target and native abstention. It ignores physical labels, queried feedback, and C64 offline matching. Empirical-extrema is a named sensitivity version, not the primary. |

Group II methods retain their original score-only update rules. The evaluator
may score their predictions against Real-IAD physical-defect labels after the
run, but those labels are not method inputs.

### Group III — separate deployment adaptations

| ID | Operational variant | Decision source |
|---|---|---|
| B5a_FROZEN_FALLBACK | SPOT + Frozen-Theta fallback | `theta0` while SPOT is unavailable; SPOT after initialization. |
| B5b_FROZEN_FALLBACK | dSPOT + Frozen-Theta fallback | `theta0` while dSPOT is unavailable; dSPOT after initialization. |
| B6_FROZEN_FALLBACK | Sun-CS + Frozen-Theta fallback | `theta0` for cold start, change-detection abstention, or confidence-set abstention; Sun-CS otherwise. |

Each event records `ORIGINAL_METHOD`, `STARTUP_FALLBACK`, or
`ABSTENTION_FALLBACK`. A fallback row is an operational adaptation, not the
unmodified source algorithm. It adds no labels, support images, C64 anchors, or
feature memory.

## 3. Causal event order and information permissions

At each stream position, the engine shall:

1. Score the current event with the frozen native scorer and pre-event shared
   memory state `M_t`.
2. Ask every controller for a prediction using its state before current-label
   disclosure.
3. Apply the precommitted label-blind tape. Only a queried event discloses its
   verified label to permitted feedback handlers and controllers.
4. Permit the single shared writer to consider only queried verified normals.
   The writer uses one locked reference policy independent of every
   controller's threshold. Queried defects never enter the normal bank.
5. On a memory content or anchor-evidence revision, rescore all eligible
   anchors under current memory; use all-layer source-aware LSO for queried
   normal anchors that contributed prototypes.
6. Update controllers for the next event. A score-only controller may consume
   the current unlabeled score after its prediction, never before.
7. Store true labels only in the private evaluator ledger. Images, masks,
   future labels, unqueried labels, hidden class histograms, and confirmation
   labels are not controller state.

Calibration-isolated controllers share one score trajectory, one query tape,
and one Greedy writer. A memory or correction intervention with distinct
scores requires a separately named profile and is outside this protocol.

## 4. Locked dataset, support, memory, and query configuration

| Item | Frozen candidate setting |
|---|---|
| Development classes | `audiojack`, `regulator`, `switch`, `rolled_strip_base` only. |
| Confirmation isolation | `mint` and `pcb` remain unopened and unavailable for tuning; Stage 2 requires a later explicit authorization. |
| Deployed support | Four distinct training-normal identities per class; support seed 0. |
| Additional C64 | 64 disjoint training-normal identities, one view per official sample group; historical D2 history seed 190021 and support-view offset 891. Protected/read-only. |
| theta0 | Historical D2 `higher` q90 from C64 scored under M0; 58th order statistic (zero-based index 57). No probe labels. |
| Native scorer | `dinov2_vitl14_reg`; 392 image input, 448 resize; fp16 feature path; 4 augmentations, no identity augmentation, multiplier 2.5; concatenated features; 32 top patches; image max aggregation, top 64, q=.95; kNN=5; fixed capacity 25,000 and 5,000 per image; locked v22 mode/layer/view fusion settings in `NativeScorerConfig`. |
| Dynamic memory | Shared verified-normal-only native Greedy bank; K=256 per layer; initially empty; no controller threshold may alter writes. |
| Main query tape | Q-A-TS-v1, 3% of image observations, half-up `floor(.03*N+.5)`, 16 temporal bins, seed 37, at most one queried view per official sample key. Reject infeasible schedules. |
| Secondary query tape | Q-B via Q-A-TS-v1 allocation; about 20% of total queries in first 400 observations when eligible; exact realized counts reported. |
| Query identities | Count queried images and official-sample groups separately; official sample keys are verified grouping metadata, not proof of distinct physical products. |
| Stream order | Frozen category JSON test-record order; tape built only from public identity, timestamp, camera, and position metadata before labels are accessed. |
| Checkpoint | Event-boundary only; stream/tape/outcome digest, method config, RNG, memory, source map, and anchor ledger must match on restore. |

The native scorer's complete configuration hash and each class's support/C64,
stream, and query tape hashes must be saved in the per-run manifest. No
generated output may overwrite an earlier run.

## 5. Method configurations and startup rules

| Method | Frozen candidate configuration |
|---|---|
| B0 | `theta_t=theta0`; predict defect iff `score >= theta_t`. |
| B1 | Last W=256 queried verified-normal stale scores; `higher` q90 after 16 eligible normals; otherwise hold theta0. |
| B2 | Historical v22 sidecar source defaults; normal-only unlabeled fallback disabled; C64 scalar initialization; queried labels only. Record its native initialization convention separately from D2 theta0. |
| B3 | Current-M protected C64 `higher` q90; Delta=0.01 movement clip; min normal count 16; queried-normal LSO excluded in primary. |
| B4 | alpha/beta targets 0.10; gamma-normal/gamma-defect 0.005; class-side state updates only on the matching queried label; exact registered empirical bound convention. |
| B5a SPOT | n_init=1000 scores, high empirical level 0.98, tail risk q=0.001, ten Grimshaw starts, tolerance 1e-6. Source-native predictions unavailable before initialization. |
| B5b dSPOT | d=50 local window, then 1000 residual warm-up scores, level .98, risk q=.001, ten Grimshaw starts, tolerance 1e-6. Both stages are source-native unavailable predictions. |
| B6 Sun-CS | Version `B6-SUN-CS-A3-CONSERVATIVE-EXTENDED-REAL-v2.0.0`; p=.99, delta=.01; primary `conservative_extended_real` endpoint policy; raw Equation (2) radius, no clamp; empty past -> `ABSTAIN_COLD_START`; decision-before-append; change-event abstention and reset. `SUN_CS_A3_EMPIRICAL_EXTREMA` is a separate sensitivity version. Checkpoints use schema `calibration.sun_cs_checkpoint.v2` and reject v1 identity. |
| B7 SCDAC-G | v0.1 provisional defaults: min normals 16, min queried defects 8, delta=.05, descriptive support FPR/FNR .20/.20, diagnostic FPR targets .05/.10/.20, movement penalty .05, epsilon 1e-12, max threshold step .01, no cooldown. These choices remain subject to independent method review. |

The B6 source-paper target is `1[S_t > Q_Ft(.99)]`, an anomaly relative to
the score-stream quantile. Evaluation may additionally compare the resulting
predictions with the Real-IAD physical-defect target, but those targets are not
interchangeable. The conservative endpoint candidate can have limited
source-native binary coverage at finite stream lengths; this is reported as
applicability evidence, not repaired by using defect labels. The empirical
endpoint sensitivity is reported separately. There is no verified author-code
parity, and neither endpoint convention inherits a theorem for scores changed
by adaptive memory or selective physical-label feedback.

No parameter is tuned on Real-IAD outcomes in Stage 0. Any post-approval
methodological change requires a versioned rationale and a pre-frozen holdout
before confirmation access.

## 6. Evaluation metrics and denominators

### Full-stream operational metrics

For an all-event binary method with complete predictions, report confusion
counts and `BA = 1-(FPR+FNR)/2`, with `FPR=FP/(FP+TN)` and
`FNR=FN/(FN+TP)`. A missing class denominator is `NULL`; it is never zero.
Report defect recall, class counts, query counts/fractions, runtime, peak
memory, and the worst defined nonoverlapping 256-image window FPR/FNR/BA.

Do not assign Full-stream BA to source-native methods with startup-unavailable
events or native abstentions. Report their full-stream coverage and mark binary
BA undefined when any event lacks a binary answer. The explicit fallback
variants may receive Full-stream BA when they issue a binary decision at every
event.

### Common-horizon metrics

The locked common post-initialization horizon begins at zero-based `t=1050`.
Evaluate every method on precisely the same positions. SPOT/dSPOT predictions
are source-native available by this horizon under the specified warm-up; any
later unavailability must remain missing. Methods with abstentions report
selective performance and coverage. The horizon is not moved after results are
seen.

### Selective and diagnostic metrics

For each method report overall, normal, and defect coverage; abstention rate;
startup-unavailable rate; queried-label count; selective FPR/FNR and selective
BA where denominators exist; and startup/abstention fallback counts and rates.
Startup-unavailable and deliberate abstention are separate categories.
Neither is an automatic human query.

Report shared-score AUROC/AUPRC, memory update counts, threshold movement,
HOLD/conflict frequency, evidence counts/revisions, C64 and queried-anchor
rescore counts, LSO-excluded rows, candidate/bound evaluations, runtime, and
peak RAM. Threshold-only controllers must have identical AUROC/AUPRC because
they share the exact same native score vector. Any difference is a protocol
failure. Report image-level and official-sample-group-level query fractions.

For B7, label its one-sided Clopper-Pearson values as descriptive diagnostics.
They do not certify online risk control under selective labels, serial
dependence, memory rescoring, and post-selection.

### B7 activation and descriptive-feasibility reporting

For B7, report queried normal and defect image counts separately from the
eligible current-memory normal-anchor and queried-defect-anchor counts. Record
the first event reaching each evidence state (`NORMAL_ONLY`, `BOTH`,
`CONFLICT`), each state's event fraction, HOLD frequency, actual threshold
update count, first event with two-sided evidence, first event with a
descriptive feasible set, and first nonzero threshold change. Also report
candidate and risk-bound evaluations, descriptive feasibility rate (both
conditional on two-sided evidence availability and over the full stream),
conflict reasons (`INSUFFICIENT_PRECISION`,
`RISK_CONSTRAINT_INCOMPATIBILITY`, and
`UNIDENTIFIABLE_SCORE_DISTRIBUTIONS`), memory changes, and anchor-evidence
revision counts.

Use separate fields for `TWO_SIDED_EVIDENCE_AVAILABLE` and
`DESCRIPTIVE_TWO_SIDED_CONSTRAINT_FEASIBLE`. The first means the current
minimum eligible counts are met; the second means at least one candidate
threshold satisfies both registered descriptive upper-endpoint limits. An
available two-sided sample is not automatically feasible.

With minimum queried defects 8, confidence delta .05, and descriptive FNR
target .20, zero errors among 8 defects gives an upper endpoint of about
.31234. The minimum sample size whose zero-error endpoint is at most .20 is 14:
`U(0,n)=1-.05^(1/n)`, with `U(0,13)>.20` and `U(0,14)<=.20`. At least 14 zero-error
queried defect anchors are required for this specific descriptive
+precision condition. Thus eight
defects can activate nominal two-sided processing but cannot by themselves
support the target, even under zero observed errors. This is a fixed-time
descriptive sample-size calculation, not an online risk guarantee.

### B6 endpoint identifiability and applicability reporting

The primary endpoint policy represents a clipped lower q=0 endpoint as
negative infinity and a clipped upper q=1 endpoint as positive infinity.
For p=.99 and delta=.01, the upper requested level is 1 at n=64, 256, 1,050,
and 3,815. At those stream sizes the primary controller cannot issue a finite
score anomaly decision (`score > +infinity` is false); it can issue benign
decisions below the lower endpoint or abstain. Report this as an endpoint
identifiability limitation, not a software failure or Real-IAD performance
estimate. The empirical-extrema sensitivity remains fully reported to expose
its numerical behavior.

### Preregistered source-native coverage decision

For each unsupervised source-native method, report full-stream, common-horizon,
normal-class, and defect-class coverage from its actual output modes. A
source-native method is eligible for ordinary Full-stream binary BA only when
it emits a binary prediction for every event in that denominator. If any
startup-unavailable or abstention event remains, report BA as undefined and
report coverage and selective metrics instead. This is a denominator rule,
not a post-result coverage cutoff; no comparator is removed because its
coverage is low. Report the registered common horizon at zero-based `t=1050`
on the identical event positions for all methods.

If a source-native method has little or no usable binary coverage, retain it
as an unsupervised applicability reference and report the result as
supplementary source-task evidence. Do not describe it as a full-coverage
binary detector. Its separate Frozen-Theta row may receive full-stream binary
metrics only if it actually emits a binary decision for every event; always
report the fractions decided by the original method, startup fallback, and
abstention fallback. A hybrid's result is not attributed to the unmodified
source method.

### Locked query-count and B7 feasibility context

Metadata-only Stage 0 counts give 3% Q-A-TS budgets of 114 Audiojack, 116
Regulator, 115 Switch, and 114 Rolled Strip Base queried images (459 total
across 15,295 images). These counts use half-up rounding and the label-blind
one-view-per-official-group tape; they do not disclose defect prevalence.
The separate [B7 activation feasibility specification](B7_ACTIVATION_FEASIBILITY_REPORTING_v1.md)
contains hypothetical independent-Bernoulli calculations at queried defect
fractions .05, .10, .20, and .30. Those assumptions are sensitivity scenarios,
not Real-IAD prevalence estimates and not query-policy changes.

If Stage 1 B7 fails to reach two-sided evidence or never obtains a descriptive
feasible threshold, report the state occupancy, HOLD frequency, counts,
feasibility denominator, and conflict reasons as the result. Do not lower the
minimum defect count, change delta, raise the query budget, use unqueried
defects, or substitute another method while retaining the B7 label.

## 7. Cross-method assumptions and valid claims

| Method | Original task / data assumptions | Feedback and stationarity/drift | Memory compatibility, startup, abstention | Source-faithful and adapted components | Guarantees that do not transfer | Valid empirical conclusion if approved |
|---|---|---|---|---|---|---|
| B0 | Frozen operating threshold control. | No learning or feedback. | Uses shared memory-dependent scores; no startup or abstention. | Registered control. | No statistical guarantee. | Physical-defect metrics at fixed theta0 on the evaluated classes/seeds. |
| B1 | Rolling empirical normal-score quantile control. | Queried verified normals only; stale scores; nonstationarity is not corrected by rescoring. | Shared score path; immediate binary decision; HOLD until 16 eligible normals. | Proposed fixed W/q/min-count implementation. | No finite-sample defect-risk guarantee. | Effect of sparse queried-normal stale-score updates under the frozen tape. |
| B2 | Historical v22 Guarded Dual Anchor sidecar. | Queried feedback; source unlabeled normal fallback disabled. | Shared memory; starts from additional C64 scalar scores; binary decisions. | Historical v22 source wrapped; explicit C64 and feedback-access adaptations. | No historical full-loop or risk guarantee in this engine. | Performance of the named adapted sidecar wrapper, not all v22 behavior. |
| B3 | State-consistent normal quantile falsifier. | Protected known-normal C64; no defect feedback; reacts to current-M rescoring. | Directly accepts changed memory-state scores; immediate binary decision. | Proposed SCQ rule, source-exclusion and Delta frozen above. | No calibrated coverage or finite-time risk guarantee. | Whether current-memory normal calibration differs empirically from stale B1/B0. |
| B4 | Adaptive conformal inference for sequential prediction under its source feedback assumptions. | The source uses prediction errors; this variant updates only on queried class-matched physical labels. | Current-memory anchor scores and selective queries differ from source assumptions; abstention is not native here. | Explicit class-conditional sparse-feedback and memory adaptation. | ACI coverage guarantees do not transfer to selective feedback, dependence, or dynamic memory. | Empirical performance of this adapted B4 under Q-A-TS/Q-B. |
| B5a SPOT | EVT/POT upper-tail anomaly detection in streams; GPD tail approximation and adequate excesses. | Whole unlabeled score stream; source adapts accepted nonalarms; stationarity/tail stability required. | Memory-induced score shifts can invalidate fitted tail; 1000-score warm-up; binary after ready. | Paper-equation implementation with disclosed optimizer/interpolation additions; pinned libspot is provenance, not compiled parity. | No physical-defect class-conditional risk guarantee; no dynamic-memory guarantee. | Empirical physical-defect operating metrics after source-native warm-up, separate from startup fallback. |
| B5b dSPOT | EVT/POT after local-mean detrending. | Whole unlabeled score stream; accepted nonalarms advance d-window; intended drift handling is local-mean only. | Dynamic memory may alter more than local mean; d=50 plus 1000 residual warm-up; binary after ready. | Independent paper-equation implementation; pinned C source contains no dSPOT implementation. | No transferred EVT guarantee under dependent memory-driven scores. | Empirical physical-defect metrics for this exact dSPOT implementation. |
| B6 Sun-CS | Unsupervised score-quantile anomaly labels; IID score assumptions for stated coverage; piecewise-stationary results need their change-separation assumptions. | Whole unlabeled score stream; no labels or selective feedback in source method; Algorithm 3 resets after detected change. | Dynamic-memory dependence may violate assumptions; cold-start/change abstentions are native. | Equations and Algorithms 1-3 independently reference-tested with explicit endpoints/reset conventions; no author-code parity. | No physical-defect FPR/FNR or dynamic-memory theorem transfer. | Physical-defect empirical utility/coverage of source-target predictions on common scores, with no theorem claim. |
| B7 SCDAC-G | Proposed descriptive two-sided risk constrained calibration. | Current evidence consists of protected normals and queried physical labels; sparse feedback and drift explicit. | Rescores current memory and uses queried-normal LSO; holds if a side is missing; has BOTH/CONFLICT branches. | Internal v0.1 proposal with provisional numerical choices. | CP bounds do not certify online risk under selective labels, dependence, repeated rescoring, or selection. | Mechanism and physical-defect results for the named provisional version only. |

### Claim boundary

**Empirical comparability does not imply theoretical equivalence.** SPOT,
dSPOT, Sun-CS, ACI, and SCDAC-G solve different source tasks and use different
information and assumptions. Physical-defect FPR/FNR on Real-IAD are empirical
measurements for all rows, not transferred source-paper guarantees. Sun-CS
targets a score-quantile anomaly event, not physical defects. No theorem
transfer is claimed for sparse selective feedback or dynamic Greedy memory.

## 8. Reporting, exclusions, and approval boundary

Report every method row, including unavailable and abstaining methods; do not
remove a weak or underpowered comparator. Show method-version and source
provenance, all denominators, query budget, startup, fallback fraction, and
source differences next to its metrics. Aggregate across the four development
classes using the preregistered unweighted class macro and show per-class and
support-seed uncertainty; do not treat image rows as independent classes.

This candidate is submitted for independent researcher approval. It neither
authorizes execution nor approves the proposed gate amendment. No Stage 1 or
Stage 2 experiment, Real-IAD performance tuning, Mint/PCB outcome inspection,
or confirmation-label access is included in Stage 0.
