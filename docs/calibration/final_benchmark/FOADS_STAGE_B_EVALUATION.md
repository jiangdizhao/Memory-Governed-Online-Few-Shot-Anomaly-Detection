# FOADS-KNG Stage B Evaluation

This adapter evaluates image-level AUROC and Average Precision only after a
FOADS P0 Stage A attempt is complete. It uses the registered stream's
`original_position` to join predictions to the official Real-IAD annotation.
It checks the attempt status, canonical seal, journal file digests, registered
category/role/stream commitments, five-view group order, event identities, and
the category-bound post-seal authorization before reading TEST outcome fields.
It never opens pixel masks and never reports threshold metrics, because FOADS
has no registered deployment threshold.

The historical Stage A source commit is validated against the attempt's own
status and seal. It is not compared with the current Git HEAD, so a valid
sealed result remains evaluable after later source commits.

## Evaluate a completed category

The researcher issues a local authorization only after the category has a
valid Stage A seal. The authorization is bound to the category bundle, role
and stream hashes, annotation SHA-256, campaign, Stage A source commit, seal,
and prediction/update journals. It uses schema
`final-benchmark.foads-stage-b-label-projection-authorization.v1`, status
`AUTHORIZED_POST_STAGE_A_LABEL_PROJECTION`, and scope
`stage_b_category_evaluation`. Its `authorization_basis` records the actual
researcher authorization; it must not imply an independent review that did
not occur. Set `test_outcomes_after_valid_seal` to true,
`test_masks_authorized` to false, and `online_outcome_access` to false.

Then run the CPU-only evaluator:

```powershell
& $Python -m experiments.calibration.final_benchmark.launch foads-evaluate-one `
  --attempt-root $AttemptRoot `
  --category-bundle $CategoryBundle `
  --annotation $OfficialAnnotation `
  --authorization $StageBAuthorization `
  --output-dir $EvaluationOutput
```

The evaluator repeats seal and journal verification before it opens the
annotation. Output is non-overwriting and must be outside the Stage A attempt.
It writes `evaluation_summary.json`, `EVALUATION_REPORT.md`, and
`evaluation_receipt.json`. If Stage B fails, preserve Stage A and retry Stage
B to a fresh output directory after correcting the input/grant issue.

## Run a future trajectory and evaluate it after sealing

Keep Stage A and Stage B in separate processes. The Stage A child receives no
annotation or post-seal authorization path. Stop the sequence if the Stage A
command returns nonzero. Afterward, issue the category/attempt-specific
post-seal grant and run the evaluator above; it will refuse to read outcomes
unless the complete seal and grant bindings pass.

```powershell
& $Python -m experiments.calibration.final_benchmark.launch foads-run-one `
  --config $Plan --bundle $PrivateBundle --locator-manifest $PrivateLocators `
  --authorization $StageAAuthorization --category $Category `
  --attempt-id $AttemptId --kng-source-root $KNGSourceRoot
if ($LASTEXITCODE -ne 0) { throw "FOADS Stage A failed; Stage B was not run." }

# After checking the sealed attempt and issuing its post-seal grant, run
# foads-evaluate-one with that attempt, registered category bundle, annotation,
# authorization, and a fresh output directory as shown above.
```

This sequence does not automatically restart a failed Stage A run. A failed
Stage B evaluation does not rerun or alter the sealed Stage A trajectory.

## Aggregate Real-IAD P0

The aggregator validates each summary against its evaluation receipt and the
registered P0 trajectory. It reports the unweighted category-equal AUROC and
AP only when all 30 unique registered categories have valid results; otherwise
it records an incomplete status and missing categories.

```powershell
$Arguments = @(
  "-m", "experiments.calibration.final_benchmark.launch", "foads-aggregate-stage-b",
  "--config", $Plan, "--output-dir", $MacroOutput
)
foreach ($Summary in $EvaluationSummaries) {
  $Arguments += @("--evaluation-summary", $Summary)
}
& $Python @Arguments
```

Do not include event-level predictions, projected TEST labels, private path
locators, authorization JSON, or masks in version control.
