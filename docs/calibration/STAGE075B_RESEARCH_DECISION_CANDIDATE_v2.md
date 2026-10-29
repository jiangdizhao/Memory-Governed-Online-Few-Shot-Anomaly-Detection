# Stage 0.75-B research decision candidate

**Candidate decision:** `SIMPLER_BASELINE_SUFFICIENT`
**Mechanism status:** `SYNTHETIC_MECHANISM_NOT_SUPPORTED`
**Engineering status:** `ENGINEERING_IMPLEMENTATION_COMPLETE`
**Review status:** candidate only; researcher approval required for any next experiment.

## Decision basis

The implementation passed all 15 Stage 0.75-B engineering and audit gates. It used the actual B3 SCQ and B7 SCDAC-G classes, the shared causal engine, a fixed label-blind synthetic query tape, current-version evidence, source-group deduplication, and exact checkpoint replay. All 48 scenario/seed trajectories were repeated with identical canonical digests. The existing calibration suite passed 132 tests, compilation succeeded, and the output/audit checksums verified.

The scientific result is negative for the proposed full B8 mechanism under the frozen candidate v2 contract:

* B8-Full exactly matched B3's threshold trajectory in all 48 rows and had mean paired `BA(B8-Full) − BA(B3) = 0` across the 42 rows where BA was defined.
* The B8 reliability penalty averaged `1.2222` over final snapshots with both queried classes and exceeded the best unpenalized empirical delta-J in every one. B8-Full therefore followed its registered B3 fallback.
* The simple B3+Offset control averaged BA `0.760482` against B3 `0.746193` and B8-Full `0.746193`; B8-Full was lower than the offset in 35 of 42 paired rows. This is synthetic and descriptive, not a claim of general superiority.
* B8-M was weakened by the zero-median/q90-change counterexample. B8-S matched B3. B8-P changed behavior but had mean BA below B3 and was more exposed to finite-rank ambiguity.
* Candidate-selection optimism appeared in the registered small-sample demonstration: train delta-J `+0.5000`, independent future delta-J `−0.2857` using eight queried sources and six candidates.

The result is scientifically useful because it does not convert a successful software run into a positive mechanism claim. It indicates that the additional B8 machinery produced no incremental full-arm action under these frozen defaults, while the same-information simple comparator often had better synthetic BA.

## Recommendation

Do **not** propose Stage 0.75-C on the basis of this candidate. Do not reduce the reliability penalty, tune on these evaluation labels, or reinterpret B8's fallback as a win. A researcher may decide to reject B8, simplify it to a named raw-offset method, or authorize a separately frozen B8 v3 study. Each option requires an explicit decision and new versioned rationale; none is authorized by this report.

## Limits and unverified claims

The memory/scorer is synthetic and does not establish native VisionAD or Greedy parity. The uncertainty term is a heuristic, not a confidence bound. The sampled evidence does not resolve label-selection bias, temporal dependence, concept drift, C64 representativeness, or deployment risk. Three deterministic seeds do not support a generalization or significance statement. No Real-IAD labels or scores were read during this task.

No Stage 0.75-C, Stage 1, Stage 2, Real-IAD performance evaluation, Mint/PCB outcome access, mask access, or model tuning was performed.
