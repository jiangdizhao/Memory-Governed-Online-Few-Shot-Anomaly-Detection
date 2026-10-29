# Stage 0.75-B synthetic implementation contract (candidate v1)

**Status:** superseded before the primary synthetic matrix; preserved as the original candidate for audit. It does not authorize Stage 0.75-C, Real-IAD evaluation, Stage 1/2, or confirmation access. Its open choices were provisionally frozen for a development smoke, then the rejoin test identified a local-search trap. See v2 for the revised primary synthetic contract.

## 1. Scope and estimand

The target is the incremental, selected-evidence estimate of Youden gain relative to the *actual* `StateConsistentQuantile` controller. It is not population BA, a calibrated probability, or a risk guarantee. The synthetic scorer/memory adapter tests state mechanics only and makes no claim about native VisionAD or Greedy parity.

All threshold methods use `int(score >= theta)`. Controllers receive only `Observation`, queried `Feedback`, and version-matched `EvidenceSnapshot`. Stream labels remain in `PrivateOutcomeLedger`; stream payloads contain only score-generation inputs. Synthetic source-group IDs stay in the trusted evidence authority and are exposed to controllers only as opaque anchor IDs/counts.

## 2. Numerical conventions

All score and threshold calculations use finite IEEE-754 binary64 values. Nonfinite input is rejected. Prediction ties are exact: a score equal to the threshold is defect. No epsilon is used in prediction, empirical error counts, or CDF comparisons. `math.nextafter` is used to construct adjacent representable threshold candidates. Equality of rank coordinates never substitutes for raw-threshold equality.

For sorted C64 values `c_(1) <= ... <= c_(64)`, define

`F_C(a) = count(c_i <= a) / 64`.

Ties are included on the right. `F_C(a)=0` below the minimum and `1` at or above the maximum, so the representation saturates. The C64 higher-q90 remains the source B3 value at zero-based index 57. Linear interpolated quartiles use `h=(n-1)p`, linear interpolation of adjacent order statistics, matching NumPy's `method="linear"`; `IQR=max(Q3-Q1, 1e-12)`. The even-sample median is the arithmetic mean of the two central values.

The diagnostic generalized inverse is `Q_C(u)=c_(max(0,min(63,ceil(64u)-1)))` for `u in [0,1]`; thus `u=0` maps to the minimum. Because this inverse cannot represent every raw decision boundary, B8 actions are always selected and executed in raw score space. Candidate construction includes current B3 and B8 thresholds; finite `nextafter` neighbors of each distinct current evidence score; each distinct score itself; midpoints of adjacent distinct evidence scores; C64 order statistics; and the two movement-boundary values. Candidates are finite, sorted, deduplicated, and restricted to the reachable movement interval. This grid includes equality and open-interval decisions. Percentile-coordinate ties are resolved by minimum absolute raw distance to the requested reference threshold, then lower raw threshold. Every coordinate-to-raw conversion records the requested coordinate, selected raw threshold, selected coordinate, saturation, and whether the raw prediction boundary changed.

## 3. Evidence and synchronization

Evidence is unweighted and has no temporal decay. One queried verified label per immutable official sample group is eligible; a duplicate group label is rejected. Rescoring the same anchor after a memory change changes its score version, not its source count. `n_eff` is the number of distinct eligible source groups, not the number of views, rescoring operations, or rows. Queried-normal anchors are eligible only after the synthetic evidence authority applies all-layer leave-source-out. C64 is read-only and supplies percentile/scale coordinates only; it is not pooled into queried-normal FPR evidence.

Every anchor score must match the snapshot's actual memory version. Evidence scores are recomputed on every genuine memory content change. Defect label age and score-state age are separately logged; no age decay is used. The controller does not claim that fresh rescoring cures concept drift.

The B8 shadow is an instance of the unchanged `StateConsistentQuantile(theta0, delta=0.01, min_normals=16)`. It receives the same snapshots at the same event boundary as external B3 and finalizes once after the event's evidence update. Tests compare full serialized state, threshold and prediction at every event. B3 ignores evidence-only revisions at unchanged memory version exactly as its source does.

## 4. Queried-evidence empirical objective

For current-state queried-normal LSO scores `N` and unique queried-defect scores `D`:

`FPRhat(theta)=sum(n >= theta)/|N|`, `FNRhat(theta)=sum(d < theta)/|D|`, `Jhat(theta)=1-FPRhat-FNRhat`.

The B3-relative improvement is `DeltaJhat(theta)=Jhat(theta)-Jhat(theta_B3)`. Either missing denominator makes the objective undefined; B8 then requests the actual B3 threshold. C64 does not fill a missing queried-normal denominator. There is no assumption that query-conditioned anchors represent deployment classes.

For `K=max(1, number of reachable raw candidates)` and `delta_heuristic=0.05`, define the dimensionless reliability penalty

`U = sqrt(log(2K/delta_heuristic)/(2 n_N)) + sqrt(log(2K/delta_heuristic)/(2 n_D))`.

This is a **heuristic penalty inspired by finite-sample concentration**, not a confidence bound: adaptive/selective labels, temporal dependence, memory rescoring, candidate selection, and source-group dependence invalidate an automatic coverage interpretation. If it exceeds all observed B3-relative gain, the controller stays at B3. No minimum count beyond at least one distinct eligible source per class is imposed; the penalty naturally discourages action with scarce evidence. Counts 1/2/3/7/8/14 are explicit diagnostics, not certification cutoffs.

The displacement scale is the current C64 IQR floor above. With `lambda_base=lambda_move=0.05`, maximize

`DeltaJhat(theta) - U - lambda_base*abs(theta-theta_B3)/IQR - lambda_move*abs(theta-theta_B8_prev)/IQR`.

Candidates are ranked by descending objective; exact ties choose minimum distance to B3, then minimum distance to prior B8, then lower raw threshold. The candidate set includes movement-boundary values and is restricted to the reachable movement interval. If the maximum is not strictly positive, the desired target is B3. The executed threshold is projected to `[theta_prev-0.01, theta_prev+0.01]`. `0.01` is a provisional raw-score move limit aligned with B3's existing step size. No second shrinkage is applied.

If B8 has never departed and the desired target is B3, it copies the shadow threshold exactly (the shared move bound makes this reachable). After any prior departure, it moves toward B3 by at most `0.01` per update and records desired target, reachable threshold, residual, and rejoin update count. It never calls a partial move an exact fallback.

## 5. Registered arm definitions

* **B3:** unchanged actual SCQ.
* **B7:** unchanged `SCDACGController` and its registered 16-normal/8-defect minima, descriptive CP endpoints, conflict and movement behavior. If the tested input adapter cannot supply valid evidence, record unavailable; do not substitute.
* **B3+Offset:** with both evidence classes present, target the raw midpoint of queried-normal and queried-defect sample medians; otherwise target B3. Project using the same 0.01 raw move limit and same evidence-update opportunities. This is a simple equal-information baseline, not an optimal-threshold claim.
* **B8-M:** transport the prior B8 raw displacement by the change in the C64 median since the prior memory version; set desired threshold to current B3 plus transported displacement. It tests paired median translation and has no defect-driven objective.
* **B8-S:** preserve the prior standardized residual `kappa=(theta_B8-q90_C64)/IQR_C64` across memory versions; on a new version target `q90_new+kappa*IQR_new`. When both queried classes are present, choose the candidate with maximal empirical `DeltaJ` minus the same reliability penalty, with distance regularization only in standardized-residual units (`0.05` each); raw execution remains movement-limited. Missing either class returns to B3.
* **B8-P:** take the queried-defect median as a defect-informed raw target, convert it to `F_C`, then convert back through the frozen generalized inverse/candidate rule. It intentionally has no two-class reliability gate; no defect evidence means B3. This is an ablation and its risks are reported.
* **B8-Full:** use Sections 3–4, including the actual B3 shadow, current-memory percentile state, two-class queried-evidence gate, reliability penalty and movement-constrained rejoin.

Each new arm is threshold-only and consumes the same legal query evidence and opportunities. B8-M is a prespecified mechanism ablation that only uses C64. No arm writes memory or controls query positions.

## 6. Triggers, state, and replay

An update occurs only when the versioned evidence digest changes. Genuine memory changes refresh C64 and all eligible anchors and advance memory version; a queried defect may advance evidence revision without changing memory. An unchanged digest is a no-op. The engine event order is score, predict, reveal queried label, shared verified-normal write, advance memory version only on content change, rescore legal evidence, update controllers for the next event.

Every stateful arm serializes an immutable method/version ID, configuration, theta, B3 shadow where applicable, current/previous C64 statistics, evidence IDs/scores/version/revision/digest, update digest, candidate and movement diagnostics, counters and rejoin state. Restore rejects another method version/configuration and nonfinite or version-inconsistent values. The engine checkpoint also binds stream/tape/evaluator identity and RNG state. Checkpoint files remain in the local output directory and are not included in the compact audit package.

## 7. Scenario protocol and interpretation

All S0–S14 and controls A–E in the implementation task run on fixed, label-blind synthetic tapes and declared seeds. Main matrix seeds are `37, 101, 2029`; focused edge/control traces use the declared deterministic fixture seed `20261029`. Scenario equations and sizes are stored in the run manifest before execution. The implementation runs the full matrix twice and requires identical canonical result digests. Synthetic memory is an explicit score-state perturbation adapter, not native Greedy. Full-stream BA/FPR/FNR, confusion counts, common-score AUROC/AUPRC, paired differences, threshold movement/TV, updates, evidence counts, rescore counts, rejoin delay, runtime and controller state size are descriptive. Undefined one-class BA is JSON `null`. No temporal-image confidence interval or real-world guarantee is produced.

## 8. Frozen acceptance mapping

G1 source/branch integrity; G2 this contract; G3 exact source B3; G4 engine causal firewall and missing-label paired-world test; G5 common score/ranking invariance; G6 snapshot/memory version and LSO tests; G7 numeric edge tests; G8 group deduplication/effective counts; G9 every S0–S14; G10 same-label/query/move-budget comparisons and A–E; G11 repeated-run digest plus checkpoint/resume; G12 reports state limits; G13 compilation and calibration tests; G14 manifest/checksum/audit linkage; G15 no Real-IAD, Stage 0.75-C, Stage 1/2, masks, or confirmation access. A failed mandatory gate is not converted to PASS.
