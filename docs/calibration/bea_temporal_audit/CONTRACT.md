# BEA-T1 temporal acquisition audit — implementation contract

Status: **R1 SYNTHETIC IMPLEMENTATION FOR INDEPENDENT REVIEW; REAL EXECUTION BLOCKED.** This package is a
CPU-only, exploratory post-hoc audit. It grants no access to outcomes, masks,
private feedback, model weights, or scientific execution. No real-data run was
performed in this implementation task.

## Boundaries and interfaces

* `integrity.load_verified` reuses the frozen `verify_attempt_receipt`, hashes
  the persisted category-seal bytes, and requires the same digest in the
  attempt receipt, the returned-seal category report and the matching Stage B
  scientific record. It checks source, category, condition, event and group
  counts, row/group digests and every journal envelope, reconstructs the group
  receipt digests, and compares rows with the source-bound stream manifest.
  It never reads the private feedback file. No digest is inserted into the
  historical seal payload.
* `campaign.VerifiedCampaign` requires the 90 registered category-condition
  pairs, the 36/54 source split, registered per-category lengths and release
  identities, total events and registered query/suppression totals. It is the
  prerequisite for `projection.project_after_all_seals`.
* Projection requires both the existing Stage B category grant and a separate
  externally issued `bea-t1.audit-authorization.v1` record. The latter binds the
  independently reviewed audit commit, all 90 distinct seal hashes, both source
  plan hashes, the release and identity hashes, the official Stage B metric-file
  hash, configuration-file hash, aggregate-only output and no masks. A further
  `bea-t1.label-projection-authorization.v1` grant binds each externally
  produced Stage B identity-projection file to the audit grant, Stage B grant,
  annotation, historical source, category, condition, campaign and seal. No
  authorization or event-label projection is minted here. After checking all
  90 grants and artifacts, the audit-only adapter rechecks the relevant files,
  loads the authorized event-label mapping, and calls the frozen
  `evaluation.evaluate_sealed_category`. Every full-stream metric field must
  match the hash-bound Stage B reference, including nested window metrics.
  Queried-label composition and first queried-defect position must also match
  exactly, including nulls and value types.
  The Stage B summary contains aggregates; it is never treated as an event-label
  source.
* `core.analyze_trajectory` checks five-view source groups, exact query/cap
  order, and independently replays each BEA price from the 256 most recent
  **executed** flags. All five proposals use the price before the group; replay
  advances only after its five decisions. Fixed price 0.8 is a read-only
  diagnostic and never feeds a state update.
* Deciles use `floor(10 * group_index / number_of_groups)`, capped at 9;
  early/middle/late are deciles 1–3, 4–7 and 8–10. Rates include class counts;
  a missing class yields undefined conditional rates. N-shadow receives AUROC
  and AP only, with no invented threshold. B/N/D terms are NOT_IDENTIFIABLE:
  the frozen row writer records the objective and price, not those terms.
* `aggregate.aggregate_campaign` averages P0/P1/P2 within each category, then
  averages categories. It uses 10,000 category-cluster resamples, seed
  20261114, and descriptive percentile intervals. A missing condition or an
  undefined category/condition is never silently dropped. Per-trajectory cap
  positions remain in memory; only normalized aggregate positions are output.
* `orchestrate.run_verified` performs projection and parity before any output.
  `output.write_outputs` uses an allowlisted aggregate schema and exclusive
  creation; no event identities, paths, raw labels, receipts or per-image rows
  are allowed in public files. Real output requires an already existing,
  Git-ignored `outputs/calibration/bea_temporal_audit/<new-run-id>` directory
  namespace; no ignore rule or authorization is created by this package.

## Existing-artifact integration and execution CLI (R2)

`--preflight --config <absolute-path>` inspects source and artifact metadata only.
It skips the values of scientific metrics and outcome fields, and reports only
readiness categories. It never opens the annotation files. `--run --config`
requires a committed reviewed source revision and a separate researcher-issued
audit authorization before reading any image-level outcomes. Neither mode is an
authorization to execute scientific analysis by itself.

The private configuration schema `bea-t1.execution-config.v2` contains
`audit_source_commit`, `audit_authorization_path`, `stage_b_authorization_path`,
`official_stage_b_summary_path`, `output_directory`, and `campaign_plans`: two
entries with `source_commit`, absolute `path` and `private_binding_root`. Each
root contains the existing `<category>_<condition>_private_bundle.json`. The
stream manifest is checked in its registered category bundle, not copied.
The output must be a new direct child of the existing Git-ignored BEA-T1 output
root. The audit authorization binds the exact configuration byte hash.

The original cross-campaign Stage B authorization is checked against its file
hash in the official summary, the frozen evaluator and metric source hashes,
both historical source commits and plans, and all 90 category-condition
entries including annotation, stream, source, journal and seal hashes. Only a
validated in-memory category view is passed to the frozen evaluator; this is
provenance, not authority for this new audit. The separate audit grant must bind
the reviewed source commit, both historical commits and plan hashes, all 90
seals, original authorization hash, release and identity hashes, official
summary and configuration hashes, `POST_SEAL_IMAGE_LEVEL_OUTCOMES`, no masks,
and aggregate-only output. No grant is issued by this implementation.

After all 90 trajectories are verified, the projector checks annotation bytes
against the registered SHA-256 before decoding only `normal_class`, image keys
and image-level `anomaly_class`. The original TEST positions, camera, group,
image digest and sealed event digest must match. Mask values are traversed as
opaque JSON values; mask files are never opened. The event-label map exists
only in memory. The frozen evaluator runs on sealed rows, and every registered
metric, query count, queried-label composition and first-defect position must
match before aggregate output is created. No learning or query is replayed.

Formal execution remains blocked until independent review and a committed audit
source, a separately issued scoped BEA-T1 grant, a suitable Git-ignored output
root and resolution of the unrelated dirty `final_benchmark/__init__.py` source
identity. No real outcome projection or audit is authorized by this task.

Observed associations can be **OBSERVED**, **COMPATIBLE_WITH_HYPOTHESIS** or
**INCONCLUSIVE**. The unobserved counterfactual value of an unpurchased query
is **NOT_IDENTIFIABLE**; authorization or integrity failure is **BLOCKED**.
These exploratory diagnostics do not establish causality, independent
generalization, or confirmatory p-values.
