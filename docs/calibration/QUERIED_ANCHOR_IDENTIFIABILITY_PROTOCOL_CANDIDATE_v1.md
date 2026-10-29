# Queried-Anchor Identifiability Protocol Candidate v1

**Run status:** frozen for the Stage 0.75 measurement-only audit. This is a
candidate protocol and requires independent researcher review. It does not
authorize B8 v4, Stage 0.75-C, Stage 1, Stage 2, or confirmation-class access.

## 1. Scope and frozen inputs

This audit measures whether Q-A-TS labels disclosed before a frozen memory
snapshot identify the direction of a local threshold correction for the actual
B3 threshold. It is measurement-only: no online controller, B8, Greedy writer,
training, or score stream is changed.

The only categories are `audiojack`, `regulator`, `switch`, and
`rolled_strip_base`; support seeds are 0–4; snapshots are zero-based positions
`0, 400, 800, 1600, 2400, 3200`. The intended matrix is 120
category/seed/snapshot states. The Stage 0.75-A source run is
`outputs/calibration/stage075a_greedy_ranking_repaired3_20261029`, based on
source commit `4d715a9b27027ee0c52dab4552d7296153ca8633`. Its protocol manifest
SHA-256 is
`948bdb5396269b391a9c8fb51681acabbcda46fe4dee145844344815d0404bf9`, its root
checksum manifest SHA-256 is
`f687d585ad432b043b342e143f141c01754bb670f65d8ae37328bd279bf7fda3`, and its
online completion manifest SHA-256 is
`bc0ad9a02398db43c2167b47926ab3895111f51cd528753d4cc0b595ca1ed629`. The
Stage 0.75-A private snapshot hashes are bound through each trajectory
manifest and are verified before deserialization. The historical run's stale
`INCOMPLETE` marker is preserved and disclosed.

The Actual-B3 diagnostic input is
`outputs/calibration/stage075a_b3_headroom_v1_20261030_verified2`. Its current
`run_manifest.json` SHA-256 is
`c657105b8d632ef3f477afeb06470926078c3fcd38ecbbd93b7bf123b6b2f1d8`, its
`checksums.json` SHA-256 is
`2410074ab82b1f2e108996fc4deb0b0e16bd64a203054932241cf9ec599e3f98`, and its
`complete.json` SHA-256 is
`d9aee6328563a9f9847a3a70af99a85c7eaef2042128b476e3961a6a7a2c7f96`. All four
files listed by its current checksum manifest verify, the listed replay hash
matches the run manifest, and an independent label-free replay from the
verified Stage 0.75-A event traces reproduces all 20 saved B3 threshold
vectors exactly. However, `complete.json` records a different checksum-manifest
SHA-256 and `files_checked=5`, while the current manifest has four entries.
This is preserved as a blocked Actual-B3 package-linkage gate; the original
package is not edited. Measurement uses the independently reproduced
source-faithful B3 thresholds and must continue to report that gate as
`BLOCKED`.

The inherited scorer is the frozen v22/VisionAD path with local DINOv2
`dinov2_vitl14_reg4_pretrain`, four deployed supports plus the separate
protected 64-image C64 reference, `K=256` dynamic capacity per layer, and the
recorded Stage 0.75-A configuration. The DINO checkpoint SHA-256 is
`36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51`; the local
DINO source-tree SHA-256 is
`1de67887ca330bdfaad8ccd45128d2b9313d63aa707bbe8e4f38310e52005b24`; the
scorer-configuration SHA-256 is
`1b8e9d96b121e3f1f6c1bbee13ab9f0063facee8fb08ad696dc5f9cb43dbd452`. The
machine-readable run manifest records current source hashes, exact input
hashes, environment, and code identity before queried labels are requested.
The only permitted accelerator/model source is the already verified local
cache; missing or mismatched source/checkpoint files block inference without a
download attempt.

## 2. Eligibility and information separation

For snapshot position `t`, the eligible set is exactly
`Q_<t = {i : i < t and i is selected by the original Q-A-TS-v1 tape}`. A
position at `t` is not eligible. No query after position 3199 is needed for
this matrix. Query labels are obtained only for those selected positions via a
process-isolated oracle bound to the unchanged official metadata SHA-256.
Only one anchor per official sample group counts; any duplicate group on the
frozen tape is rejected. Camera and view identities are retained as provenance
and never treated as independent products. Any overlap with that trajectory's
four deployed support groups or protected C64 groups is rejected.

The feature/scoring component receives opaque image/group identities, the
frozen scorer, native memory, and features; it does not receive evaluator-only
future labels. The query-evidence component receives only verified labels from
`Q_<t` and scores bearing the exact snapshot version and digest. The future
evaluator is a separate process and is started only after the query estimates,
actions, and their canonical digest have been written and independently
rechecked. Future labels are never passed to the estimator or action selector.

## 3. Native snapshot reconstruction and score policies

Each saved `.pt` snapshot is deserialized only after its resolved path is
confirmed to be inside the original private-checkpoint directory and its
SHA-256 matches the trusted trajectory manifest. Every state must match the
recorded memory content version, memory digest, content digest, layer count,
capacity, source-map digest, source-map row alignment, support identity, and
prior-admitted-group set. State mismatch blocks the affected cell; q90 or a
later snapshot is never used to synthesize missing memory.

The fixed support bank is rebuilt through the unchanged P0 `build_vmb` path,
with the original supports, augmentation, local DINO weights/source, and
configuration. Its fixed-bank digest must match the original score-tape
manifest exactly. The restored dynamic state is installed on a clone through
`native_bridge.install_on_clone`; the native `score_encoded`/v22 scorer is
used for all image scores.

For a queried defect, score against the actual frozen memory `M_t` only after
the group is verified absent from every layer of the source map and from the
prior-admitted history. A contradiction makes the anchor unavailable and
blocks its cell.

For a queried normal group `g`, the primary score is
`s(x; M_t^(-g))`: remove every dynamic prototype row attributed to `g` in every
layer, while preserving the fixed support bank and all other rows. Source-map
misalignment or an unsupported native codebook state makes that anchor
unavailable; no alternate scorer is substituted. Also compute the diagnostic
full-memory score `s(x; M_t)` and record removed rows per layer. This
self-contamination comparison does not replace LSO evidence.

Where the original event trace provides the query's arrival-time dynamic
score, compare it with the current-memory score as a staleness diagnostic. For
normal anchors, the comparison is considered source-policy matched only when
the group was not resident at arrival; otherwise report the policy mismatch
and do not treat the difference as pure memory drift. Arrival scores never
enter the primary estimator.

At every restored snapshot, rescore all 64 protected C64 references and
compare the higher-q90 (zero-based index 57) with the stored snapshot q90, the
NPZ snapshot q90, and the event-boundary q90. The declared parity tolerance is
`abs(delta) <= 1e-6 + 1e-6 * max(abs(a), abs(b))`. Fixed-bank and memory
prototypes/digests require exact equality. Any q90 parity failure blocks that
state. All measurement operations run on private clones and must leave the
checkpoint memory digest and original snapshot score arrays unchanged.
For every state with at least one eligible queried anchor, repeat the first
eligible anchor's primary native score from the same cached features and the
same score policy (all-layer LSO for a normal, unchanged snapshot for a
defect). Record the absolute difference and require the same declared
`1e-6 + 1e-6 relative` tolerance. A state with no eligible query anchor is
marked `NO_ELIGIBLE_ANCHOR` for this repeat check; it is not a parity pass by
substitution.

## 4. Estimator and diagnostic action selector

Let `theta_t` be the pre-event actual B3 threshold from the source-faithful
`StateConsistentQuantile` replay. Predictions use exact `score >= threshold`
with no epsilon. The frozen primary offsets are `h in {-0.01, 0, +0.01}`;
secondary sensitivity offsets are `h in {-0.005, 0, +0.005}`. They are
measurement probes, not controller parameters.

For current-state queried-normal LSO scores `N_t` and queried-defect scores
`D_t`:

```text
FPR_Q(theta) = count(n >= theta) / |N_t|
FNR_Q(theta) = count(d < theta) / |D_t|
J_Q(theta)   = 1 - FPR_Q(theta) - FNR_Q(theta)
DeltaJ_Q(h)  = J_Q(theta_t + h) - J_Q(theta_t)
```

When either class is absent, return `TWO_CLASS_GAIN_UNIDENTIFIABLE`; do not
substitute C64 or zero risk. Reject nonfinite scores, stale or mismatched
memory versions, duplicate groups, and malformed labels before calculation.
The KEEP estimate is exactly zero.

The primary selector evaluates LOWER (`h=-0.01`) and RAISE (`h=+0.01`):

1. Return `ABSTAIN` for missing class evidence, invalid/stale scores, or an
   exact positive-gain tie.
2. Return `LOWER` if its gain is strictly positive and strictly greater than
   RAISE.
3. Return `RAISE` if its gain is strictly positive and strictly greater than
   LOWER.
4. Return `KEEP` when both gains are nonpositive.

No tolerance is used to break gain ties. The secondary grid is reported only;
it never changes primary actions. Report defect evidence bins `0`, `1–2`,
`3–7`, `8–13`, and `14+`, as well as normal source-group count and snapshot
position. These bins do not certify risk or define activation requirements.

## 5. Future evaluation cohorts and outcomes

After the pre-future action file is saved and its hash verified, a separate
evaluator loads the original test labels for metric calculation. It uses only
the original snapshot score vector and finite group-clean mask, rechecked
against the saved snapshot provenance. It evaluates `next256_group_clean`
(`t` through `min(t+256,n)`) and `all_remaining_group_clean` (all eligible
positions `>=t`). It never invokes a scorer or changes any historical score,
mask, memory, or action.

For each registered primary and secondary offset, compute future
`DeltaJ = J(theta_t+h)-J(theta_t)` and `DeltaBA = DeltaJ/2`, plus FPR/FNR
changes, on exactly the same frozen score positions. `KEEP` is the baseline.
Future best among the primary three actions is a retrospective diagnostic;
it is unique only when exactly one action has the strictly greatest future
gain. A one-class cohort has null BA and null DeltaJ. It is neither a success
nor an error for direction agreement.

Report all 120 state rows and both cohorts, then class-level summaries and an
equal-class macro. Overlapping snapshot cohorts are dependent; seeds, views,
snapshots, and cohorts are not treated as independent replications. Report
explicit denominators for identifiable-action coverage, unique future-best
direction agreement, nonzero false activation (selected nonzero action with
negative future DeltaJ), conditional harm, mean query-selected future DeltaBA
relative to KEEP, mean loss versus the best registered action, and abstention
by defect-count bin. A valid-data KEEP counts as identified; missing-evidence
fallback/abstention does not.

## 6. Negative controls

- **N1 Missing defect evidence:** snapshots before the first queried defect
  must be unidentifiable; in particular, early Audiojack states cannot select
  a local direction without defect evidence.
- **N2 Wrong-memory rejection:** an anchor score tagged with another snapshot
  version/digest must be rejected by the estimator.
- **N3 Self-contamination:** compare full-memory and all-layer-LSO normal
  scores and report changed estimates/actions.
- **N4 Label leakage:** after future evaluation begins, alter only future
  evaluator labels and recompute the query-only action digest from frozen
  query inputs; it must be bit-identical to the pre-future digest.
- **N5 Query-label shuffle:** deterministically shuffle queried labels within
  each eligible snapshot set while preserving identities, scores, and counts;
  report action/estimate changes without substituting shuffled results.
- **N6 No-useful-gain fixture:** a small independent two-class fixture whose
  exact best registered choice is KEEP must select KEEP.
- **N7 Same-group duplication:** repeated camera-view rows for one official
  group cannot increase effective evidence counts.
- **N8 Frozen-score invariance:** input checksums, snapshot digests, stored
  score vectors, and masks must be unchanged after measurement/evaluation.

## 7. Reproducibility, artifacts, and scientific limits

Run output is non-overwriting at
`outputs/calibration/queried_anchor_identifiability_v1_20261030/`. Private
features, paths, labels, row-level score traces, source maps, and restored
checkpoints stay local and are excluded from Git. The compact audit package
contains aggregate results, hashed provenance, tests, gates, runtime, report
hashes, and checksums only.

Repeat the deterministic estimator/action pass from the same verified query
scores and require identical canonical digests. Recompute a fixed sample of
native anchor scores from the same features and require the declared numeric
tolerance. Run all calibration tests and the focused N1–N8 tests. Protected
historical source hashes and all pre-existing artifact checksums must remain
unchanged.

Queried anchors are sparse, selectively sampled, temporally and group
dependent, and rescored under evolving memory. These estimates are descriptive
for the observed queried anchors. They are not unbiased population balanced
accuracy estimates, IID confidence claims, deployment guarantees, or
independent confirmation results. The four development categories have prior
exposure. This protocol answers a measurement question only and can validly
conclude that useful local direction is unidentifiable or unreliable.

## 8. Frozen acceptance and scope gates

The run records G1–G20 individually as PASS, FAIL, BLOCKED, or SKIPPED. In
particular: verify the branch and protected source tree; verify all original
Stage 0.75-A/Actual-B3 artifacts; reconcile 20 trajectories and 120 states;
validate every memory/source map; verify native support and C64 parity; enforce
strictly pre-snapshot queried evidence and process separation; use all-layer
normal LSO and unchanged-memory defects; freeze actions before future labels;
preserve future scores/masks; run N1–N8; handle undefined metrics honestly;
repeat deterministically; run tests and checksum linkage; and attest that no
writer, B8, Stage 0.75-C, Stage 1/2, confirmation data, or masks were accessed.

**Researcher approval:** required before any subsequent method, protocol, or
Stage 0.75-C/Stage 1 decision. This candidate does not grant that approval.
