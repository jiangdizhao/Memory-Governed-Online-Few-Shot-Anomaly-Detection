# Stage 0.75-A — Native Greedy Ranking and C64 Transport Protocol v1

**Status:** Frozen for the authorized exploratory development audit. This is not
independent scientific approval of a future stage.

## Scope and disclosure

This audit asks whether causal, queried-verified-normal-only Native Greedy
memory changes normal/defect score ranking relative to the frozen four-support
M0 scorer, and whether the resulting normal-side C64 score movement resembles
a scalar translation. It runs only on `audiojack`, `regulator`, `switch`, and
`rolled_strip_base`; those four development categories were already exposed in
Stage 0.5. The result is exploratory. No Mint/PCB confirmation data, masks,
Stage 0.75-B/C, Stage 1, or Stage 2 are accessed.

## Frozen inputs

- Support seeds: `0, 1, 2, 3, 4` for each category (20 trajectories total).
- Each trajectory uses exactly four distinct training-normal deployed support
  groups and the same additional protected C64 set of 64 train-normal
  references selected under the historical seed `190021` and historical
  support-view rule. This is four supports **plus** C64, not strict four-shot
  total.
- The C64 set and each support set are checked for disjointness and against the
  test stream. The native frozen backbone, augmentation, fixed-bank build and
  scoring configuration are inherited unchanged from the verified Stage 0.5
  artifacts.
- Dynamic Greedy memory starts empty with capacity `K=256` in each native
  layer. C64 references remain protected and are never writable.
- The query tape is the frozen `Q-A-TS-v1` tape at image fraction `0.03`, seed
  `37`, with 16 temporal bins and at most one selected view per official
  sample group. Stage 0.5 tape positions and digests are reused without
  changing the schedule.
- Stage 0.5 score tapes are the immutable M0 reference. Seed 0 is reconstructed
  against the same native scorer for every stream image and must match exactly;
  initial C64 M0 score vectors are likewise checked against Stage 0.5.

At `freeze`, the runner records the active branch and parent SHA, code and
native-source hashes, checkpoint and local DINOv2 source-tree hashes, support
and C64 identity digests, static-score checksums, query-tape digests, scorer
configuration, environment, and stream lengths. The machine-readable frozen
record is `protocol_manifest.json` in the local run output.

## Causal event order

At each image-stream position `t`, the runner:

1. Encodes the current image once and shares those immutable features across all
   five support-seed trajectories for that category.
2. Reads the frozen `S_static[t]` M0 score and calculates
   `S_dynamic[t]` using the current live Greedy memory before current feedback.
3. Records both scores, the pre-event memory version/digest, frozen-theta0
   diagnostic predictions, current-memory C64 higher-q90 diagnostic threshold,
   and strictly pre-event group/source status.
4. Uses only the frozen Q-A-TS tape to decide whether this position is queried.
   For queried positions, it computes the native static M0 map before label
   disclosure; an isolated label-oracle process returns only the requested
   queried position's label. No label is requested for an unqueried image.
5. Calls the unchanged `SharedNativeGreedyWriter` with the frozen M0 score and
   M0 score map, the source group, and the existing feature bundle. Per seed,
   `theta_ref` is that seed's initial M0 C64 higher-q90 threshold and
   `sigma_ref=0.02`; `novelty=0` and controller thresholds are not involved.
6. Records append, replace, skip, source-map, capacity, and content-version
   results. Only a real prototype-content change advances calibration memory
   version or triggers paired C64 rescoring.

The existing writer and native Greedy implementation are dependencies; this
audit does not patch them or replace them with Greedy-all. A queried normal
does not imply a content change. If no content changes occur, that is reported.

## Ranking estimands

For each class and seed, report static M0 versus prequential dynamic AUROC,
AUPRC, deltas, cross-class ties, score distributions and normal/defect
conditional score shifts. Report full-stream metrics as **as-operated**; the
dynamic vector contains scores from varying memory states and is not a
single-fixed-model ranking estimate.

The conservative group-clean mask excludes event `t` if that official group
had supplied any prototype before `t`. A separate resident-source mask flags
groups currently represented in the pre-event memory; previously admitted
same-group observations have their own stratum. Each mask is determined from
prior writer/source state and affects evaluation only. Static and dynamic
scores use identical positions. Single-class strata report AUROC and balanced
accuracy as undefined. Five camera views are not treated as five independent
manufactured specimens.

Fixed memory snapshots are frozen at `t=(0,400,800,1600,2400,3200)` before
processing event `t`, so they contain only writes from positions `<t`. For each
snapshot, compare M0 and the frozen snapshot on (1) the next 256 positions and
(2) all remaining positions, using the same eligibility rule: the group was
not represented in the snapshot and had not previously supplied an admitted
prototype by `t`. Cohorts depend only on position and prior source identities,
not labels, scores or later updates. Report cohort class counts, ranking,
threshold metrics and retrospective single-threshold BA oracle. The oracle is
offline only and is never used by the writer.

## Paired C64 memory transport

At every actual content change, pair the previous 64 M0/current-memory C64
score vector with C64 scores under the new memory, using the same frozen image
features. The prior vector is keyed to its exact memory digest; each new vector
is recalculated by the native scorer. Record anchor-level shifts locally and
publish only aggregate diagnostics and hashes.

Use `m=median(d)`, `H=Q75(d)-Q25(d)`,
`theta_transport=theta_previous+m`, direct `theta_SCQ=higher_Q90(c_after)`,
and normalized disagreement `abs(theta_transport-theta_SCQ)/max(IQR(c_after),
1e-12)`. The q90 is the higher order statistic at index
`ceil(0.90*(64-1))=57`; quartile IQR uses linear interpolation; the median is
the standard even-sample median. Exact machine-uniform shifts and descriptive
heterogeneity/IQR bands of 1%, 5%, and 10% are reported. Those bands are
sensitivity summaries, not B8 parameters. Rank reordering is tie-aware over
the 2,016 anchor pairs.

Threshold values are read-only diagnostics. Neither candidate threshold
changes predictions or memory writes. Translation and direct q90 coincide
under a pure score translation if the previous threshold equals the higher q90
of the before scores. Normal-side C64 transport cannot establish that defect
scores underwent the same transformation. B8 is neither implemented nor
validated by this experiment.

## Execution and artifacts

The fixed order is: input/source preflight; synthetic engineering tests; one
predeclared first-512-position seed-0 smoke for each category; all 20 complete
trajectories; offline snapshot and label evaluation; integrity audit. A smoke
is not a performance subset. It cannot select or remove a full trajectory.
The runner uses bounded event-boundary checkpoints and rejects mismatched
input, source, protocol or query-tape fingerprints on resume.

Raw score traces, per-anchor C64 scores, dynamic prototype checkpoints and
source maps remain in the local Stage 0.75-A output directory and are not
published in the compact audit package. The audit package contains provenance,
aggregated metrics, writer/memory summaries, tests, status gates, linkage hashes
and SHA-256 checksums. Incomplete work remains marked `INCOMPLETE`/`BLOCKED`.

No empirical result from this protocol independently validates Stage 1, B8,
population risk, or generalization to confirmation categories.
