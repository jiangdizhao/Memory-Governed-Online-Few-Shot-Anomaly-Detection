param(
    [Parameter(Mandatory = $true)][string]$ExpectedSourceCommit,
    [Parameter(Mandatory = $true)][string]$RunId,
    [Parameter(Mandatory = $true)][string]$AuthorizationId,
    [string]$PythonPath = 'D:\anaconda3\envs\torch\python.exe',
    [string]$DatasetRoot = 'G:\dataset\realiad'
)

$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$ModuleName = 'experiments.calibration.r1_s3_native_map_parity'
$ProfileId = 'r1_s3_native_map_parity_v1'
$IdentityContract = 'r1-s3.source-bound.v1'
$TimeoutSeconds = 600
$OutputLimitBytes = 128 * 1024 * 1024
$OutputPrefix = 'r1_s3_native_map_switch_s0_16_20261031_'

function Write-JsonAtomic([string]$Path, [object]$Value) {
    $temporary = "$Path.tmp"
    $json = $Value | ConvertTo-Json -Depth 24
    [System.IO.File]::WriteAllText($temporary, $json + [Environment]::NewLine,
        [System.Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $temporary -Destination $Path -Force
}

function Get-OutputBytes([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return [int64]0 }
    $files = @(Get-ChildItem -LiteralPath $Path -Recurse -File -Force)
    if ($files.Count -eq 0) { return [int64]0 }
    return [int64](($files | Measure-Object -Property Length -Sum).Sum)
}

function Get-PeakWorkingSet([int]$ProcessId) {
    try {
        $process = Get-Process -Id $ProcessId -ErrorAction Stop
        return [int64]$process.PeakWorkingSet64
    }
    catch { return $null }
}

function Set-RunFields([string]$Path, [hashtable]$Fields) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return }
    try {
        $manifest = Get-Content -Raw -LiteralPath $Path | ConvertFrom-Json
        foreach ($key in $Fields.Keys) { $manifest | Add-Member -NotePropertyName $key -NotePropertyValue $Fields[$key] -Force }
        $manifest.output_bytes = Get-OutputBytes (Split-Path -Parent $Path)
        Write-JsonAtomic $Path $manifest
    }
    catch {
        Write-Output "Could not update partial run manifest: $($_.Exception.Message)"
    }
}

function Set-LauncherFields([string]$Path, [hashtable]$Fields) {
    $manifest = Get-Content -Raw -LiteralPath $Path | ConvertFrom-Json
    foreach ($key in $Fields.Keys) { $manifest | Add-Member -NotePropertyName $key -NotePropertyValue $Fields[$key] -Force }
    Write-JsonAtomic $Path $manifest
}

function ConvertTo-ArgumentLine([string[]]$Values) {
    return (($Values | ForEach-Object {
        '"' + ([string]$_ -replace '"', '\"') + '"'
    }) -join ' ')
}

if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "Authorized Python interpreter is unavailable: $PythonPath"
}
if ($RunId -notmatch '^[A-Za-z0-9][A-Za-z0-9_-]{4,63}$') {
    throw 'Run ID has an invalid or unsafe format.'
}
if ($ExpectedSourceCommit -notmatch '^[0-9a-fA-F]{40}$') {
    throw 'Expected source commit must be a full Git SHA.'
}

$branch = (& git -C $RepoRoot branch --show-current).Trim()
$head = (& git -C $RepoRoot rev-parse HEAD).Trim()
$trackedStatus = (& git -C $RepoRoot status --porcelain --untracked-files=no | Out-String).Trim()
if ($branch -ne 'calibration-stage0-contract-20261029' -or
    $head.ToLowerInvariant() -ne $ExpectedSourceCommit.ToLowerInvariant() -or
    $trackedStatus) {
    throw "Source-bound S3 launch rejected: branch=$branch HEAD=$head tracked_changes=$trackedStatus"
}

$outputDir = Join-Path $RepoRoot (Join-Path 'outputs\calibration' ($OutputPrefix + $RunId))
if (Test-Path -LiteralPath $outputDir) {
    throw "Refusing to reuse existing S3 output namespace: $outputDir"
}

# This pass reads only source identities, frozen JSON metadata and receipt hashes.
# It does not decode images, load Torch/CUDA, or instantiate a model.
$preflightArgs = @(
    '-m', $ModuleName, '--mode', 'preflight',
    '--expected-source-commit', $ExpectedSourceCommit,
    '--profile-id', $ProfileId,
    '--run-id', $RunId,
    '--authorization-id', $AuthorizationId,
    '--dataset-root', $DatasetRoot
)
$preflightOutput = & $PythonPath @preflightArgs
if ($LASTEXITCODE -ne 0) {
    throw "S3 CPU preflight failed before output creation: $($preflightOutput -join ' ')"
}
try { $preflight = ($preflightOutput -join "`n") | ConvertFrom-Json }
catch { throw "S3 CPU preflight returned invalid JSON: $($preflightOutput -join ' ')" }
if ($preflight.profile_id -ne $ProfileId -or
    $preflight.source_commit -ne $ExpectedSourceCommit -or
    $preflight.identity_contract -ne $IdentityContract -or
    $preflight.event_count -ne 16 -or $preflight.support_count -ne 4) {
    throw 'S3 CPU preflight returned an incompatible profile/source receipt.'
}

New-Item -ItemType Directory -Path $outputDir | Out-Null
New-Item -ItemType Directory -Path (Join-Path $outputDir 'events') | Out-Null
$startedAt = [DateTime]::UtcNow
$runManifestPath = Join-Path $outputDir 'run_manifest.json'
$launcherManifestPath = Join-Path $outputDir 'launcher_manifest.json'
Write-JsonAtomic (Join-Path $outputDir 'preflight.json') $preflight
$initialRunManifest = @{
    schema = 'calibration.r1_s3_native_map_evidence_tape.v1'
    identity_contract = $IdentityContract
    branch = $branch
    source_commit = $head
    source_commit_used_for_gpu = $head
    profile_id = $ProfileId
    authorization_id = $AuthorizationId
    run_id = $RunId
    status = 'INCOMPLETE'
    events_processed = 0
    event_plan = $preflight.event_plan
    output_bytes = 0
    supervisor_started_at_utc = $startedAt.ToString('o')
    supervisor_output_limit_bytes = $OutputLimitBytes
}
Write-JsonAtomic $runManifestPath $initialRunManifest
$launcherManifest = @{
    schema = 'calibration.r1_s3_bounded_launcher.v1'
    identity_contract = $IdentityContract
    status = 'INCOMPLETE'
    branch = $branch
    source_commit = $head
    profile_id = $ProfileId
    run_id = $RunId
    authorization_id = $AuthorizationId
    gpu_attempt_count = 1
    timeout_seconds = $TimeoutSeconds
    output_limit_bytes = $OutputLimitBytes
    cpu_preflight_completed = $true
    gpu_child_started_at_utc = $null
    gpu_child_finished_at_utc = $null
    gpu_child_wall_seconds = $null
    gpu_child_exit_code = $null
    peak_cpu_working_set_bytes = $null
    output_bytes_observed = 0
    supervisor_stop_reason = $null
}
Write-JsonAtomic $launcherManifestPath $launcherManifest

$stdoutPath = Join-Path $outputDir 'stdout.log'
$stderrPath = Join-Path $outputDir 'stderr.log'
$runArgs = @(
    '-m', $ModuleName, '--mode', 'run',
    '--expected-source-commit', $ExpectedSourceCommit,
    '--profile-id', $ProfileId,
    '--run-id', $RunId,
    '--authorization-id', $AuthorizationId,
    '--dataset-root', $DatasetRoot,
    '--output-dir', $outputDir
)
$argumentLine = ConvertTo-ArgumentLine $runArgs
$childStartUtc = [DateTime]::UtcNow
$stopwatch = [Diagnostics.Stopwatch]::StartNew()
$process = Start-Process -FilePath $PythonPath -ArgumentList $argumentLine `
    -WorkingDirectory $RepoRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
Set-LauncherFields $launcherManifestPath @{
    gpu_child_pid = [int]$process.Id
    gpu_child_started_at_utc = $childStartUtc.ToString('o')
}
Set-RunFields $runManifestPath @{ supervisor_pid = [int]$process.Id }

$timedOut = $false
$outputLimitReached = $false
$stopReason = $null
$peakWorkingSet = $null
$lastNotice = 0
while (-not $process.HasExited) {
    Start-Sleep -Milliseconds 100
    $process.Refresh()
    $bytes = Get-OutputBytes $outputDir
    $workingSet = Get-PeakWorkingSet $process.Id
    if ($null -ne $workingSet -and
        ($null -eq $peakWorkingSet -or $workingSet -gt $peakWorkingSet)) {
        $peakWorkingSet = $workingSet
    }
    if ($bytes -gt $OutputLimitBytes) {
        $outputLimitReached = $true
        $stopReason = '128 MiB output storage ceiling exceeded'
        break
    }
    if ($stopwatch.Elapsed.TotalSeconds -ge $TimeoutSeconds) {
        $timedOut = $true
        $stopReason = '600 second GPU subprocess wall-time ceiling reached'
        break
    }
    if ($stopwatch.Elapsed.TotalSeconds -ge ($lastNotice + 30)) {
        $lastNotice = [int]$stopwatch.Elapsed.TotalSeconds
        Write-Output "R1-S3 GPU child active: elapsed=$lastNotice seconds; output_bytes=$bytes"
    }
}

if ($timedOut -or $outputLimitReached) {
    # The PID was created above by this supervisor; taskkill is bounded to its process tree.
    & taskkill.exe /PID $process.Id /T /F | Out-Null
    $process.WaitForExit(10000) | Out-Null
}
else { $process.WaitForExit() }
$stopwatch.Stop()
$process.Refresh()
$childExitCode = $null
if ($process.HasExited) { $childExitCode = [int]$process.ExitCode }
$childFinishedUtc = [DateTime]::UtcNow
$bytesAfterGpu = Get-OutputBytes $outputDir
if ($timedOut -or $outputLimitReached) {
    $gpuStatus = 'RESOURCE_LIMIT_REACHED'
    Set-RunFields $runManifestPath @{
        status = $gpuStatus
        gpu_execution_status = $gpuStatus
        failure_message = $stopReason
    }
}
else {
    try { $gpuManifest = Get-Content -Raw -LiteralPath $runManifestPath | ConvertFrom-Json }
    catch { $gpuManifest = $null }
    if ($null -ne $gpuManifest -and $gpuManifest.status) {
        $gpuStatus = [string]$gpuManifest.status
    }
    else { $gpuStatus = 'INCOMPLETE' }
}

$readback = $null
$readbackExitCode = $null
$readbackStart = $null
if (Test-Path -LiteralPath $runManifestPath -PathType Leaf) {
    $verifyArgs = @('-m', $ModuleName, '--mode', 'verify', '--output-dir', $outputDir)
    $verifyLine = ConvertTo-ArgumentLine $verifyArgs
    $readbackStart = [DateTime]::UtcNow
    $verifyProcess = Start-Process -FilePath $PythonPath -ArgumentList $verifyLine `
        -WorkingDirectory $RepoRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $outputDir 'tape_readback_stdout.log') `
        -RedirectStandardError (Join-Path $outputDir 'tape_readback_stderr.log')
    $verifyProcess.WaitForExit()
    $readbackExitCode = [int]$verifyProcess.ExitCode
    try {
        $readbackText = Get-Content -Raw -LiteralPath (Join-Path $outputDir 'tape_readback_stdout.log')
        if ($readbackText.Trim()) { $readback = $readbackText | ConvertFrom-Json }
    }
    catch { $readback = $null }
    if ($null -eq $readback) {
        $readback = @{
            readback_integrity_pass = $false
            error = 'Readback process did not return a valid JSON receipt.'
            stderr = (Get-Content -Raw -LiteralPath (Join-Path $outputDir 'tape_readback_stderr.log') -ErrorAction SilentlyContinue)
        }
    }
    Write-JsonAtomic (Join-Path $outputDir 'tape_readback.json') $readback
}

$finalStatus = $gpuStatus
$runNow = $null
if (Test-Path -LiteralPath $runManifestPath -PathType Leaf) {
    try { $runNow = Get-Content -Raw -LiteralPath $runManifestPath | ConvertFrom-Json }
    catch { $runNow = $null }
}
$tapeIntegrityPass = ($readbackExitCode -eq 0 -and $null -ne $readback -and
                      $readback.readback_integrity_pass -eq $true)
$tapeIntegrityFailure = ($null -ne $readback -and
                          $readback.readback_integrity_pass -eq $false -and
                          $readback.readback_integrity_failure -eq $true)
$tapeIntegrityStatus = 'NOT_VERIFIED'
if ($tapeIntegrityPass) { $tapeIntegrityStatus = 'PASS' }
elseif ($tapeIntegrityFailure) { $tapeIntegrityStatus = 'FAIL' }
$eventCountComplete = ($null -ne $readback -and $null -ne $runNow -and
                       $readback.events_present -eq 16 -and
                       $runNow.events_processed -eq 16)
$readbackParityPass = ($tapeIntegrityPass -and $eventCountComplete -and
                       $readback.dense_map_exact_readback_count -eq 16 -and
                       $readback.top32_exact_readback_count -eq 16 -and
                       $readback.scalar_max_readback_count -eq 16 -and
                       $readback.scalar_tolerance_readback_count -eq 16 -and
                       $readback.zero_correction_readback_count -eq 16 -and
                       $readback.dynamic_memory_readback_count -eq 16 -and
                       $readback.source_feature_immutability_readback_count -eq 16)
$replayPass = ($null -ne $runNow -and $runNow.events_processed -eq 16 -and
               $runNow.gpu_replay_parity_status -eq 'ALL_16_PARITY_CHECKS_PASS_PENDING_TAPE_READBACK' -and
               $runNow.dense_map_exact_event_count -eq 16 -and
               $runNow.top32_exact_event_count -eq 16 -and
               $runNow.scalar_tolerance_event_count -eq 16 -and
               $runNow.fixed_memory_unchanged -eq $true -and
               $runNow.dynamic_memory_unchanged_and_empty -eq $true -and
               $runNow.correction_defect_state_unchanged -eq $true -and
               $runNow.runtime_profile_id -eq 'stage05_set_seed_0_math_sdp_v1')
$rootExceptionType = $null
if ($null -ne $runNow -and $runNow.root_exception_type) {
    $rootExceptionType = [string]$runNow.root_exception_type
}
elseif ($tapeIntegrityFailure -and $readback.root_exception_type) {
    $rootExceptionType = [string]$readback.root_exception_type
}
if ($timedOut -or $outputLimitReached) {
    $finalStatus = 'RESOURCE_LIMIT_REACHED'
}
elseif ($gpuStatus -eq 'INCOMPLETE' -and $childExitCode -eq 0) {
    if ($tapeIntegrityFailure) { $finalStatus = 'TAPE_INTEGRITY_FAILED' }
    elseif ($readbackParityPass -and $replayPass) { $finalStatus = 'S3_NATIVE_MAP_PARITY_PASS' }
    else { $finalStatus = 'INCOMPLETE' }
}
elseif ($tapeIntegrityFailure -and $gpuStatus -eq 'INCOMPLETE') {
    $finalStatus = 'TAPE_INTEGRITY_FAILED'
}

$finishedAt = [DateTime]::UtcNow
$finalBytes = Get-OutputBytes $outputDir
$launcherStatus = $finalStatus
Set-LauncherFields $launcherManifestPath @{
    status = $launcherStatus
    overall_execution_status = $finalStatus
    tape_integrity_status = $tapeIntegrityStatus
    event_count_complete = $eventCountComplete
    root_exception_type = $rootExceptionType
    gpu_child_finished_at_utc = $childFinishedUtc.ToString('o')
    gpu_child_wall_seconds = [Math]::Round($stopwatch.Elapsed.TotalSeconds, 6)
    gpu_child_exit_code = $childExitCode
    gpu_child_pid = [int]$process.Id
    peak_cpu_working_set_bytes = $peakWorkingSet
    output_bytes_observed = $finalBytes
    supervisor_stop_reason = $stopReason
    tape_readback_exit_code = $readbackExitCode
    tape_readback_started_at_utc = if ($readbackStart) { $readbackStart.ToString('o') } else { $null }
    tape_readback_result = $readback
    finished_at_utc = $finishedAt.ToString('o')
}
Set-RunFields $runManifestPath @{
    status = $finalStatus
    overall_execution_status = $finalStatus
    tape_integrity_status = $tapeIntegrityStatus
    event_count_complete = $eventCountComplete
    root_exception_type = $rootExceptionType
    supervisor_status = $finalStatus
    supervisor_finished_at_utc = $finishedAt.ToString('o')
    total_gpu_subprocess_wall_seconds = [Math]::Round($stopwatch.Elapsed.TotalSeconds, 6)
    output_bytes = $finalBytes
    tape_readback_integrity_pass = if ($null -ne $readback) { $readback.readback_integrity_pass } else { $false }
    tape_readback = $readback
    resource_metrics = @{
        peak_cpu_working_set_bytes = $peakWorkingSet
        peak_cuda_allocated_bytes = if ($gpuManifest) { $gpuManifest.resource_metrics.peak_cuda_allocated_bytes } else { $null }
        peak_cuda_reserved_bytes = if ($gpuManifest) { $gpuManifest.resource_metrics.peak_cuda_reserved_bytes } else { $null }
    }
}
$finalBytes = Get-OutputBytes $outputDir
if ($finalBytes -gt $OutputLimitBytes) {
    $finalStatus = 'RESOURCE_LIMIT_REACHED'
    Set-LauncherFields $launcherManifestPath @{ status = $finalStatus; output_bytes_observed = $finalBytes; supervisor_stop_reason = '128 MiB output storage ceiling exceeded' }
    Set-RunFields $runManifestPath @{ status = $finalStatus; supervisor_status = $finalStatus; failure_message = '128 MiB output storage ceiling exceeded' }
}

Write-Output "R1-S3 bounded run finished: status=$finalStatus exit_code=$childExitCode wall_seconds=$($stopwatch.Elapsed.TotalSeconds) output_bytes=$finalBytes readback_exit=$readbackExitCode"
if ($finalStatus -eq 'S3_NATIVE_MAP_PARITY_PASS') { exit 0 }
if ($finalStatus -eq 'RESOURCE_LIMIT_REACHED') { exit 5 }
exit 3
