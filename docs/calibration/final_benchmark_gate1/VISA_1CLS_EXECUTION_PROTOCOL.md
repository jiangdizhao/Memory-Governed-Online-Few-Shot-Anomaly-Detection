# VisA 1cls P0 Native Execution Adapter

Status: `RESEARCHER_AUTHORIZED_SINGLE_CATEGORY_EXECUTION_IMPLEMENTED`

This adapter adds VisA-specific registration, path binding, query-label access,
and post-seal evaluation around the existing `NativeInputEncoder`,
`NativeFourArmBackend`, `GroupAtomicCategoryRunner`, `FsyncGroupSink`, and
`evaluate_sealed_category`. It does not replace or alter the detector, BEA,
Margin, Gaussian C1, Signed kNN, B3, Greedy, or N-shadow algorithms.

## Identity and allocation

The registration rechecks the existing identity-only VisA 1cls census against
the local snapshot SHA-256
`15535268a41c5c163b9906874fa148478f9da0bf9394659f448d320de778a7f2` and the
split-table/license artifact-set SHA-256
`5612ff8ec0b9ca5c1545b9f08aaba71b89f3b36bcaf8dc2e9fee82589949d029`. The
release identity is explicitly `RESEARCHER_APPROVED_LOCAL_SNAPSHOT`; it is not
publisher authenticated. Four Supports and 64 additional TRAIN-normal C64
images are deterministically selected by the existing Gate 0 hash domains.
Static-36+32 uses the registered 36-reference/32-C32 partition and the
existing A3b CUDA K-Center implementation. No alternative coreset or
threshold is introduced.

The registered image-level unit digest is an opaque query-unit ID, not a
physical-object ID. Exact-byte TEST duplicates share a unit. The current
registered snapshot expects no duplicates; any change fails closed during
registration. The fixed P0 stream is outcome blind and category-specific.

## Information boundary

Identity census, registration, path binding, and Stage A broker setup use only
VisA identity/split fields and verified image bytes. TEST label cells and mask
cells are skipped by the projected CSV parser; no mask image is opened. For a
query, the broker requires an executed query receipt in the fsynced atomic-unit
journal and decodes only that row's image-level label. Stage B verifies the
sealed attempt, complete journal, event order, score identities, query receipts,
candidate and authorization bindings, and current split-table hash before it
projects the complete TEST label vector. Stage B never opens pixel masks.

The researcher task authorizes cashew/P0 Stage A and its separate post-seal
Stage B only. It does not assert independent source review, publisher
authentication, or physical-object independence. Other categories have
registrations but no execution authorization in this task. The `visa-plan`
command may prepare their source-bound metadata plan later; it deliberately
does not create a Stage A authorization for them.

## Commands

Run from the repository root in the existing Torch environment. The first
registration command is non-overwriting; choose a new registration leaf if
the specified one already exists.

```powershell
$PY = "D:\anaconda3\envs\torch\python.exe"
$DATA = "G:\dataset\VisA"
$IDENTITY = "outputs\calibration\final_benchmark_gate1\attempt_01\visa_1cls_identity_manifest.private.json"
$REG = "outputs\calibration\final_benchmark\visa_1cls\registration_01"
$DECISION_SHA = "86146542d860d3784a0ca656ea94606729f59bff0f3b9dc0c82bf09bc9100d8b"

& $PY -m experiments.calibration.final_benchmark.launch visa-register `
  --dataset-root $DATA --identity-manifest $IDENTITY --output-root $REG `
  --researcher-decision-sha256 $DECISION_SHA
& $PY -m experiments.calibration.final_benchmark.launch visa-plan --registration-root $REG --category cashew
& $PY -m experiments.calibration.final_benchmark.launch visa-run-one `
  --registration-root $REG --dataset-root $DATA --category cashew `
  --attempt-id cashew_P0_attempt_01
& $PY -m experiments.calibration.final_benchmark.launch visa-evaluate-one `
  --registration-root $REG --dataset-root $DATA --category cashew `
  --attempt-id cashew_P0_attempt_01 --evaluation-id stage_b_attempt_01
```

After separate category-specific researcher authorization exists for each
remaining category, the following loop can run and evaluate those categories
sequentially. It stops if any category lacks its private authorization or if
an attempt fails; there is no fallback or restart. Do not run it until that
separate scope is authorized.

```powershell
$CATEGORIES = @("candle", "capsules", "chewinggum", "fryum", "macaroni1", "macaroni2",
  "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum")
foreach ($CATEGORY in $CATEGORIES) {
  $CATEGORY_DECISION_SHA = "<SHA-256 of the separate researcher authorization for this category>"
  & $PY -m experiments.calibration.final_benchmark.launch visa-plan `
    --registration-root $REG --category $CATEGORY `
    --researcher-decision-sha256 $CATEGORY_DECISION_SHA
  $AUTH = Join-Path $REG "private\$CATEGORY\execution_authorization.json"
  if (-not (Test-Path -LiteralPath $AUTH)) { throw "Missing separate researcher authorization for $CATEGORY" }
  & $PY -m experiments.calibration.final_benchmark.launch visa-run-one `
    --registration-root $REG --dataset-root $DATA --category $CATEGORY `
    --authorization $AUTH --attempt-id "${CATEGORY}_P0_attempt_01"
  if ($LASTEXITCODE -ne 0) { throw "Stage A failed for $CATEGORY" }
  & $PY -m experiments.calibration.final_benchmark.launch visa-evaluate-one `
    --registration-root $REG --dataset-root $DATA --category $CATEGORY `
    --attempt-id "${CATEGORY}_P0_attempt_01" --evaluation-id stage_b_attempt_01
  if ($LASTEXITCODE -ne 0) { throw "Stage B failed for $CATEGORY" }
}
```

After twelve category reports exist, aggregate them with explicit paths:

```powershell
$RESULTS = @((Join-Path $REG "..\cashew\P0\cashew_P0_attempt_01_stage_b_attempt_01\stage_b_metrics.json"))
foreach ($CATEGORY in $CATEGORIES) {
  $RESULTS += (Join-Path $REG "..\$CATEGORY\P0\${CATEGORY}_P0_attempt_01_stage_b_attempt_01\stage_b_metrics.json")
}
$ARGS = @("-m", "experiments.calibration.final_benchmark.launch", "visa-aggregate-stage-b",
  "--registration-root", $REG, "--output-directory",
  "outputs\calibration\final_benchmark\visa_1cls\stage_b_macro_01")
foreach ($RESULT in $RESULTS) {
  $ARGS += @("--result", $RESULT)
}
& $PY @ARGS
```

The macro command requires all twelve valid category results from the same
source commit. It reports category-equal arithmetic means; one-category
cashew results are not a cross-dataset superiority claim.
