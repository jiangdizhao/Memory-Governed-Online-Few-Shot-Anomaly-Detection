# B0 Recommended Next Actions

The B0 audit is complete and read-only. These are recommendations for a later researcher decision; none is authorized or executed by this document.

## P0 — necessary before publication

### 1. Freeze the operating claim and run one independent holdout comparison

- **Scientific purpose:** establish whether the B3 operating point generalizes to categories not used in the current four-class development analysis, while showing its false-alarm/defect-recall tradeoff.
- **Missing evidence:** independent categories/seeds and a paper-level operating priority. Current full-stream results are development-only; no FPR ceiling or cost ratio was fixed.
- **Reusable code:** the Stage 1-A sealed prediction runner, shared score/memory path, query tape, metrics writer, and existing B0/B1/B3/B4/B5a/B5b/Offset controllers. Keep native SPOT/dSPOT and their Frozen-Theta hybrids as separate rows; retain the hybrid fractions.
- **Required new implementation:** no new method is necessary for a default-operating-point confirmation. Before prediction, register either the existing method defaults or an operational false-alarm budget. If a matched-FPR claim is required, predefine a calibration procedure and threshold selection on a separate normal-only calibration split, then freeze it before confirmation.
- **Compute burden:** low-to-medium if frozen score and memory receipts cover the holdout; high if new categories require native image scoring. Do not estimate per-method runtime from the aggregate Stage 1-A elapsed time.
- **Data access:** holdout images/scores under the same support/query protocol; use the precommitted label-blind query tape. Open outcome labels only in the offline evaluator after predictions are sealed. For matched FPR, use disjoint normal-only calibration data; do not select thresholds from confirmation labels.
- **Reviewer criticism addressed:** four development classes are too few to establish category generalization; default operating points trade false alarms against missed defects; selecting thresholds on reported outcomes is optimistic.
- **Can existing artifacts answer it?** No. Existing artifacts establish the development comparison and its paired tradeoffs, but not independent category generalization or a previously unregistered risk constraint.

At minimum retain B0, B1, B3, and Stage075BOffset so the simple control, proposed method, and strongest internal challenger remain visible. Include B4 and B5a/b in the independent comparison if the manuscript continues to claim external-literature comparisons. B2 and B7 can remain clearly labeled historical/internal development context if they are not central claims.

### 2. Use “four supports plus C64” precisely; run a strict-four sensitivity only for a strict-four claim

- **Scientific purpose:** distinguish the current C64 calibration regime from a true four-shot-total setting.
- **Missing evidence:** no B0/B1/B3 comparison without the 64 additional protected training-normal references.
- **Reusable code:** the existing native scorer, support manifest, memory replay, and Stage 1-A controller runner.
- **Required new implementation:** a separately named strict-four profile that removes C64 access and defines legal initialization/fallback behavior for every retained method before evaluation.
- **Compute burden:** a CPU replay may suffice if score and memory receipts are reusable; otherwise new native scoring is required. Feasibility must be checked without substituting a different scorer.
- **Data access:** four deployed supports only for calibration; any held-out labels remain sealed until prediction completion.
- **Reviewer criticism addressed:** the phrase “few-shot” could otherwise hide extra training-normal calibration data.
- **Can existing artifacts answer it?** No. They all use four deployed supports plus 64 additional C64 normals. If the manuscript makes no strict-four-total claim and states the C64 regime accurately, this experiment can be deferred.

## P1 — valuable but optional

### 3. Add a small SPOT author-code parity fixture only if stronger fidelity wording is needed

- **Scientific purpose:** test numerical trace agreement with the pinned libspot SPOT implementation, beyond equation-level and SciPy reference tests.
- **Missing evidence:** no compiled author-C parity trace. The current implementation already documents its equation-level status and root/interpolation conventions.
- **Reusable code:** current Grimshaw/GPD and POT equation tests in tests/calibration/test_spot_dspot.py; pinned libspot commit and source-blob identity in the method registry.
- **Required new implementation:** a tiny deterministic synthetic fixture that runs both implementations on identical scalar sequences and compares initialization, thresholds, and alarms. The pinned C source does not implement dSPOT, so this cannot establish dSPOT author-code parity.
- **Compute burden:** low; CPU only, no image data.
- **Data access:** no Real-IAD images, labels, masks, or confirmation data.
- **Reviewer criticism addressed:** whether the named SPOT implementation matches the available author implementation numerically.
- **Can existing artifacts answer it?** No for author-code parity; yes for the narrower “paper-equation implementation” claim. Do this only if the manuscript uses stronger source-faithful/libspot-compatible wording.

## P2 — defer or exclude

### 4. Keep Sun-CS outside the physical-defect binary leaderboard

- **Scientific purpose:** preserve an accurate applicability comparison for a method whose target is score-quantile labeling with abstention.
- **Missing evidence:** no validated mapping from Sun-CS quantile labels to physical defect labels, and the conservative endpoint yields no upper-tail anomaly labels at current stream lengths.
- **Reusable code:** current paper-equation implementation, endpoint-policy documentation, and independent reference tests.
- **Required new implementation:** none for the current paper claim. A separate research question would be needed to define and validate a different physical-defect target.
- **Compute burden:** none for an applicability discussion; potentially high for a new target study.
- **Data access:** none for the current applicability explanation.
- **Reviewer criticism addressed:** prevents an invalid “B3 beat Sun-CS” claim.
- **Can existing artifacts answer it?** Yes. The implementation, endpoint analysis, and Stage 1-A applicability flag already support exclusion from the binary ranking.

### 5. Defer B-EWQ, new B7 variants, and post-hoc threshold fishing

- **Scientific purpose:** avoid expanding the baseline suite without a specific missing comparison.
- **Missing evidence:** B-EWQ has no frozen equation; B7 is provisional and already has development results; no pre-registered matched-risk selection rule exists.
- **Reusable code:** current B1, SPOT/dSPOT, and threshold replay framework already cover queried-normal adaptation and causal score-history alternatives.
- **Required new implementation:** none unless a reviewer identifies a concrete gap and a versioned protocol is approved before outcomes are inspected.
- **Compute burden:** currently none. A new controller or search would add implementation and evaluation cost.
- **Data access:** do not use the current development outcomes to choose a new equation, parameter, or risk cap.
- **Reviewer criticism addressed:** avoids an open-ended comparator search and outcome-driven method selection.
- **Can existing artifacts answer it?** Existing results already show the low-FPR B1 point, source-native score-history SPOT/dSPOT, B3, and Stage075BOffset. They do not validate an unspecified B-EWQ.

The final paper GO/HOLD/NO-GO decision remains with the researcher after independent review.
