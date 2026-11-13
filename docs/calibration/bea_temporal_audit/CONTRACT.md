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

## Stage B reference and execution CLI

The registered summary is `realiad.stage-b-final-metrics.v1` with status
`STAGE_B_ALL_90_CPU_EVALUATED` and exactly 90 `scientific_results` records.
Each record supplies `category`, `condition`, `campaign_run_id`,
`source_commit`, `stage_a_category_seal_sha256`, `metrics`, and `query_count`.
All 30 categories and P0/P1/P2 are required; the first 36 original trajectories
and the remaining 54 continuation trajectories retain their respective source
commits. The reference file's byte hash is bound by the audit authorization.

`python -m experiments.calibration.bea_temporal_audit --run --config <absolute-private-json>`
is the real-execution entrypoint; without `--run`, `--config`, or the required
authorizations it fails closed. The private configuration schema is
`bea-t1.execution-config.v1` and contains:

* `audit_source_commit`, `audit_authorization_path`,
  `official_stage_b_summary_path`, and `output_directory`;
* two `campaign_plans` entries containing `source_commit` and absolute `path`;
* 90 `trajectories` entries, each containing `category`, `condition`,
  `attempt_status_path`, `expected_attempt_identity`, `stream_manifest_path`,
  `stage_b_authorization_path`, `stage_b_authorization_file_sha256`,
  `projection_authorization_path`, and `projection_file_path`.

The audit grant binds the configuration's actual file SHA-256; it is issued
after the configuration is finalized. The per-trajectory projection grant is
issued afterward and binds the audit grant's canonical digest and Stage B
authorization's canonical digest, avoiding a circular file-hash dependency.
The external projection file uses schema
`realiad.stage-b-event-label-projection.v1` and contains its source/category/
condition/campaign/seal/annotation bindings and `labels_by_event_id`. Its byte
hash is bound by the separately issued projection grant. This package does not
produce that input or infer event labels from the metric summary.

## Remaining real-execution blockers

1. Independent source review, a committed audit-source revision and the
   audit-specific and per-trajectory label-projection grants are required.
   Availability and provenance of the 90 separately authorized identity
   projection files have not been established; no such files were accessed.
2. `outputs/calibration/bea_temporal_audit/` is not currently Git-ignored. A
   separately authorized ignore configuration and existing output root are
   required. The existing unrelated `final_benchmark/__init__.py` modification
   is substantive and the reviewed-source clean check also blocks real run
   until its source identity is separately resolved.
3. The project-provided outer Stage B schema is implemented and tested with
   synthetic data. The actual files, their metric field coverage, campaign plan
   paths and authorization paths were not inspected during this task. Exact
   metric-schema mismatch remains fail-closed.

The frozen `FinalBenchmarkEntrypoint.evaluate_stage_b` wrapper remains
incompatible with the historical seal payload: it expects a self-digest field
that the sink does not write. The audit-only adapter deliberately verifies the
file/report/receipt/Stage B cross-bindings first and then calls the existing
`evaluate_sealed_category` directly under both independently issued grants.
It neither rewrites the seal nor weakens the evaluator's authorization checks.
The synthetic suite uses the original sink and attempt-status writer only in
temporary directories inside the authorized test tree, which are cleaned up.

Observed associations can be **OBSERVED**, **COMPATIBLE_WITH_HYPOTHESIS** or
**INCONCLUSIVE**. The unobserved counterfactual value of an unpurchased query
is **NOT_IDENTIFIABLE**; authorization or integrity failure is **BLOCKED**.
These exploratory diagnostics do not establish causality, independent
generalization, or confirmatory p-values.
