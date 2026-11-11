# Final Benchmark Execution Commands

These commands verify the frozen schedule and describe later authorized launches. No Stage A, full FOADS stream, Stage B, TEST outcome, or mask operation was performed for this integration task.

## Validate and list the plan

The checked-in template intentionally has no final source commit, campaign run ID, or execution authorization. This schedule-only validation is safe now:

```powershell
$Python = 'D:\anaconda3\envs\torch\python.exe'
$Plan = 'docs/calibration/final_benchmark_launch/FINAL_RUN_CONFIGURATION_TEMPLATE.json'
& $Python -m experiments.calibration.final_benchmark.launch validate --config $Plan --allow-unbound-template
& $Python -m experiments.calibration.final_benchmark.launch list --config $Plan
```

The verified pinned K-NG checkout used by source-dependent CPU tests and a future FOADS launch is outside the repository at `D:\Whishing-K-NG-63dd8e7c9ade5619771d4b5544f032206818bb22-lf`. Its detached HEAD is `63dd8e7c9ade5619771d4b5544f032206818bb22`; `verify_upstream_source()` confirmed the three registered SHA-256 values. The explicit setup below does not rely on an inherited environment variable:

```powershell
$KngSourceRoot = 'D:\Whishing-K-NG-63dd8e7c9ade5619771d4b5544f032206818bb22-lf'
$env:KNG_SOURCE_ROOT = $KngSourceRoot
& $Python -m pytest tests/calibration/final_benchmark/test_foads_kng.py -q -ra
```

After independent review and commit, create a new source-bound copy of the plan, set `source_identity.final_source_commit` to the actual 40-character HEAD and `campaign_run_id` to the researcher-authorized immutable run ID, then recompute `schedule_sha256` over the complete JSON object with that one field removed. Do not edit the historical template to claim a source commit that has not been created.

Validate the bound copy and exact checkout before any later launch:

```powershell
$ExpectedSourceCommit = '<ACTUAL_REVIEWED_40_CHARACTER_SHA>'
$ActualSourceCommit = (git rev-parse HEAD).Trim()
if ($ActualSourceCommit -cne $ExpectedSourceCommit) { throw "Source commit mismatch: $ActualSourceCommit" }
& $Python -m experiments.calibration.final_benchmark.launch validate --config $Plan
```

`validate` and `list` do not load DINO, inspect image bytes, or access TEST outcomes. `validate` fails for a source-bound plan when the config SHA, schedule, current Git HEAD, or immutable run identity differs.

## Check resource availability and output collisions

```powershell
$RunId = '<RESEARCHER_AUTHORIZED_CAMPAIGN_RUN_ID>'
$CampaignRoot = Join-Path 'outputs/calibration/final_benchmark' $RunId
& $Python -m experiments.calibration.final_benchmark.launch resources --config $Plan --output-path $CampaignRoot
```

The `resources` command reports current free disk and any existing output-root collision; it does not initialize the GPU. The launch boundary additionally snapshots host and GPU availability before an attempt. It reserves ten times the bounded-preflight bytes-per-event observation plus a 10 MiB per-attempt receipt allowance and a 10 GiB free-space reserve. It also stops below 1.25 times the measured host-RSS peak, below the measured GPU peak plus a 1 GiB free-VRAM margin before execution, or above 1.25 times the measured GPU reservation during execution. These are fail-closed operational stop limits, not measured full-category output or runtime guarantees. Final read-only assessment on 2026-11-11: 381,233,139,712 bytes disk free; 41,316,900,864 bytes host RAM available; RTX 3090 Ti with 23,894,949,888 bytes VRAM free. The extrapolated all-trajectory output guard is 23,221,070,420 bytes, plus the 10 GiB disk reserve. No full-category output growth measurement exists.

The launcher creates the attempt directory only after configuration, source, registration, authorization, and initial resource checks pass. Do not pre-create attempt leaves. Existing leaves are never reused; select a new `attempt_<id>` for retries. There is no automatic resume.

## Private input bundle

The `--bundle` argument names a private JSON file with path references to the certified Gate 1 inputs. Keep this file and authorizations outside public research documents and Git:

```json
{
  "schema": "final-benchmark.private-category-bundle.v1",
  "category_bundle_path": "<absolute path to Gate 1 private/realiad/<category>/<condition>.json>",
  "identity_manifest_path": "<absolute path to certified identity manifest>",
  "private_locator_manifest_path": "<absolute path to private locator manifest>"
}
```

The launcher verifies the identity, role, stream, release, locator, annotation-hash, and source-commit bindings before it creates an attempt. Principal Stage A authorization uses schema `final-benchmark.execution-authorization.v1`, scope `stage_a_category`, `label_access: QUERY_ONLY`, `stage_b_authorized: false`, and exact fields for the config-file SHA, source commit, campaign run ID, category-bundle file SHA, private-locator-manifest file SHA, identity-manifest SHA, role-manifest SHA, stream-manifest SHA, dataset-release SHA, annotation SHA, category/condition, and event count. The researcher must issue this authorization separately after source review and commit.

## One principal Stage A trajectory

This is the implemented command shape; it was **not run**:

```powershell
$Bundle = '<PRIVATE_CATEGORY_BUNDLE_JSON>'
$Authorization = '<PRIVATE_STAGE_A_CATEGORY_AUTHORIZATION_JSON>'
& $Python -m experiments.calibration.final_benchmark.launch run-one `
  --config $Plan `
  --bundle $Bundle `
  --authorization $Authorization `
  --category audiojack `
  --condition P0 `
  --attempt-id attempt_01
```

The command invokes the existing `FinalBenchmarkEntrypoint.run_stage_a_category()` only. It writes `attempt_status.json`, uses the existing fsynced group journal/private feedback and Stage A seal, records `INCOMPLETE` with phase/event count/source/config identity on an exception, and records `SEALED_COMPLETE` only after the existing full-stream seal and report succeed. The run stops on a failed category. It does not resume partial state.

## Explicit sequential principal schedule

`run-schedule` requires a separate researcher-issued `final-benchmark.sequential-launch-authorization.v1` bound to a checksummed launch schedule that contains all 90 registered trajectories exactly once, each category bundle, each category authorization, and each fresh attempt ID. `failure_policy` must be `STOP_ON_FIRST_FAILURE`. Only invoking this command begins that explicit sequential schedule:

```powershell
$LaunchSchedule = '<PRIVATE_CHECKSUMMED_90_TRAJECTORY_LAUNCH_SCHEDULE_JSON>'
$CampaignAuthorization = '<PRIVATE_SEQUENTIAL_CAMPAIGN_AUTHORIZATION_JSON>'
& $Python -m experiments.calibration.final_benchmark.launch run-schedule `
  --config $Plan `
  --launch-schedule $LaunchSchedule `
  --authorization $CampaignAuthorization
```

It stops at the first failed trajectory and releases category-scoped Python/GPU allocations before continuing to the next successful category.

## One separate FOADS P0 category

FOADS uses the same private bundle plus the absolute private-locator manifest path, an independently issued FOADS authorization, and a local checkout of the exact pinned K-NG source tree. The authorization schema is `final-benchmark.foads-p0-execution-authorization.v1`, scope `foads_kng_p0_category`; it binds source/config/run IDs, category P0 identity/role/stream/release, FOADS projection, category/locator file hashes, pinned upstream commit and source-file hashes, the registered ResNet18 checkpoint SHA, expected event count, zero paid labels, no TEST outcome/mask access, and no Stage B authorization.

```powershell
$FoadsAuthorization = '<PRIVATE_FOADS_P0_CATEGORY_AUTHORIZATION_JSON>'
$LocatorManifest = '<ABSOLUTE_PRIVATE_LOCATOR_MANIFEST_JSON>'
$KngSourceRoot = 'D:\Whishing-K-NG-63dd8e7c9ade5619771d4b5544f032206818bb22-lf'
& $Python -m experiments.calibration.final_benchmark.launch foads-run-one `
  --config $Plan `
  --bundle $Bundle `
  --locator-manifest $LocatorManifest `
  --authorization $FoadsAuthorization `
  --category audiojack `
  --attempt-id attempt_01 `
  --kng-source-root $KngSourceRoot
```

The output namespace is `foads_kng/<category>/P0/<attempt-id>/`, separate from principal Stage A. It records one prequential score per registered TEST image, model-state digests, native ten-image update receipts, durable `INCOMPLETE` state, and a checksummed category seal. It performs no outcome projection or metric calculation. The command passes `--kng-source-root` explicitly; the source environment variable is used only by source-dependent tests.

## Stage B receipt gate

Stage B remains separately authorized and must not begin unless the matching durable Stage A attempt receipt and seal verify. The implemented checker does not read labels:

```powershell
$AttemptStatus = '<ABSOLUTE_STAGE_A_ATTEMPT_STATUS_JSON>'
$ExpectedIdentity = '<PRIVATE_EXPECTED_ATTEMPT_IDENTITY_JSON>'
& $Python -m experiments.calibration.final_benchmark.launch verify-attempt `
  --kind principal_stage_a --status $AttemptStatus --expected-identity $ExpectedIdentity
```

FOADS post-seal outcome analysis has its own separate authorization and seal check:

```powershell
$FoadsAttemptStatus = '<ABSOLUTE_FOADS_ATTEMPT_STATUS_JSON>'
$FoadsExpectedIdentity = '<PRIVATE_FOADS_EXPECTED_ATTEMPT_IDENTITY_JSON>'
& $Python -m experiments.calibration.final_benchmark.launch verify-attempt `
  --kind foads_kng_p0 --status $FoadsAttemptStatus --expected-identity $FoadsExpectedIdentity
```

Neither command grants Stage B access or reads TEST outcomes. No authorization JSON, launch schedule, attempt directory, or completion receipt was generated by this integration task.
