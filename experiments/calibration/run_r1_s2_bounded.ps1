param(
    [Parameter(Mandatory = $true)][string]$ExpectedSourceCommit,
    [Parameter(Mandatory = $true)][string]$RunId,
    [Parameter(Mandatory = $true)][string]$AuthorizationId,
    [string]$PythonPath = 'D:\anaconda3\envs\torch\python.exe',
    [string]$DatasetRoot = 'G:\dataset\realiad'
)

$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$ModuleName = 'experiments.calibration.r1_s2_c64_score_parity'
$ProfileId = 'r1_s2_c64_score_parity_v1'
$TimeoutSeconds = 1200
$OutputLimitBytes = 100 * 1024 * 1024
$OutputPrefix = 'r1_s2_c64_parity_switch_s0_20261031_'
$OutputDir = Join-Path $RepoRoot (Join-Path 'outputs\calibration' ($OutputPrefix + $RunId))

function Write-JsonAtomic([string]$Path, [object]$Value) {
    $temporary = "$Path.tmp"
    $json = $Value | ConvertTo-Json -Depth 16
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

function Get-TreePeakRss([int]$ProcessId) {
    try {
        $process = Get-Process -Id $ProcessId -ErrorAction Stop
        return [int64]$process.PeakWorkingSet64
    }
    catch { return $null }
}

function Set-RunStatus([string]$Status, [string]$Reason) {
    $runManifestPath = Join-Path $OutputDir 'run_manifest.json'
    if (Test-Path -LiteralPath $runManifestPath) {
        try {
            $runManifest = Get-Content -Raw -LiteralPath $runManifestPath | ConvertFrom-Json
            $runManifest.status = $Status
            $runManifest.supervisor_stop_reason = $Reason
            $runManifest.supervisor_finished_at_utc = [DateTime]::UtcNow.ToString('o')
            Write-JsonAtomic $runManifestPath $runManifest
        }
        catch {
            # Keep the original partial manifest intact if it cannot be parsed.
        }
    }
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
    throw "Source-bound launch rejected: branch=$branch HEAD=$head tracked_changes=$trackedStatus"
}
$untrackedPython = @(& git -C $RepoRoot ls-files --others --exclude-standard) |
    Where-Object { $_ -match '\.py$' -and
        ($_ -match '^experiments/calibration/' -or $_ -match '^experiments/risk_safe/' -or
         $_ -in @('test_phase3_AIF_simplified_v22.py', 'vmb_visionad_new.py',
                  'experiments/p0_protocol_audit.py', 'experiments/v02_memory.py')) }
if ($untrackedPython.Count -gt 0) {
    throw "Source-bound launch rejected due to untracked Python sources: $($untrackedPython -join ', ')"
}
if (Test-Path -LiteralPath $OutputDir) {
    throw "Refusing to reuse existing output namespace: $OutputDir"
}

# Phase A is repeated by the external launcher immediately before it starts the
# single bounded GPU child. This subprocess imports no DINO model and performs
# no CUDA query or image decode.
$preflightArgs = @(
    '-m', $ModuleName, '--mode', 'preflight',
    '--expected-source-commit', $ExpectedSourceCommit,
    '--profile-id', $ProfileId,
    '--run-id', $RunId,
    '--authorization-id', $AuthorizationId,
    '--dataset-root', $DatasetRoot
)
$preflightOutput = & $PythonPath @preflightArgs 2>&1
$preflightExitCode = $LASTEXITCODE
if ($preflightExitCode -ne 0) {
    $preflightOutput | ForEach-Object { Write-Output $_ }
    throw "CPU preflight failed with exit code $preflightExitCode; GPU child was not started."
}
try { $preflight = ($preflightOutput -join "`n") | ConvertFrom-Json }
catch { throw 'CPU preflight did not return valid JSON; GPU child was not started.' }
if ($preflight.status -ne 'PREFLIGHT_PASS') {
    throw "CPU preflight status was $($preflight.status); GPU child was not started."
}

New-Item -ItemType Directory -Path $OutputDir | Out-Null
$stdoutPath = Join-Path $OutputDir 'stdout.log'
$stderrPath = Join-Path $OutputDir 'stderr.log'
$launcherManifestPath = Join-Path $OutputDir 'launcher_manifest.json'
$preflightPath = Join-Path $OutputDir 'preflight.json'
Write-JsonAtomic $preflightPath $preflight
$launcherManifest = [ordered]@{
    schema = 'r1_s2.external_gpu_launcher.v1'
    status = 'INCOMPLETE'
    branch = $branch
    source_commit = $head
    run_id = $RunId
    authorization_id = $AuthorizationId
    profile_id = $ProfileId
    preflight_status = $preflight.status
    gpu_child_started_at_utc = $null
    gpu_child_finished_at_utc = $null
    gpu_child_wall_seconds = $null
    gpu_child_exit_code = $null
    timeout_seconds = $TimeoutSeconds
    output_limit_bytes = $OutputLimitBytes
    output_bytes_observed = [int64](Get-OutputBytes $OutputDir)
    peak_cpu_rss_bytes = $null
    stdout_path = 'stdout.log'
    stderr_path = 'stderr.log'
    supervisor_stop_reason = $null
}
Write-JsonAtomic $launcherManifestPath $launcherManifest

$runArgs = @(
    '-m', $ModuleName, '--mode', 'run',
    '--expected-source-commit', $ExpectedSourceCommit,
    '--profile-id', $ProfileId,
    '--run-id', $RunId,
    '--authorization-id', $AuthorizationId,
    '--dataset-root', $DatasetRoot,
    '--output-dir', $OutputDir
)
$argumentLine = ($runArgs | ForEach-Object {
    '"' + ([string]$_ -replace '"', '\"') + '"'
}) -join ' '
$childStartUtc = [DateTime]::UtcNow
$stopwatch = [Diagnostics.Stopwatch]::StartNew()
$process = Start-Process -FilePath $PythonPath -ArgumentList $argumentLine `
    -WorkingDirectory $RepoRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
$launcherManifest.gpu_child_started_at_utc = $childStartUtc.ToString('o')
$launcherManifest | Out-Null
Write-JsonAtomic $launcherManifestPath $launcherManifest

$stopReason = $null
$timedOut = $false
$outputLimitReached = $false
$peakRss = $null
$lastReportSeconds = 0
while (-not $process.HasExited) {
    Start-Sleep -Milliseconds 100
    $process.Refresh()
    $bytes = Get-OutputBytes $OutputDir
    $rss = Get-TreePeakRss $process.Id
    if ($null -ne $rss -and ($null -eq $peakRss -or $rss -gt $peakRss)) {
        $peakRss = $rss
    }
    if ($bytes -gt $OutputLimitBytes) {
        $outputLimitReached = $true
        $stopReason = '100 MiB output storage ceiling exceeded'
        break
    }
    if ($stopwatch.Elapsed.TotalSeconds -ge $TimeoutSeconds) {
        $timedOut = $true
        $stopReason = '1200 second GPU subprocess wall-time ceiling reached'
        break
    }
    if ($stopwatch.Elapsed.TotalSeconds -ge ($lastReportSeconds + 30)) {
        $lastReportSeconds = [int]$stopwatch.Elapsed.TotalSeconds
        Write-Output "R1-S2 GPU child active: elapsed=$lastReportSeconds seconds; output_bytes=$bytes"
    }
}

if ($timedOut -or $outputLimitReached) {
    # This PID was created by this launcher. /T bounds termination to its child tree.
    & taskkill.exe /PID $process.Id /T /F | Out-Null
    $process.WaitForExit(10000) | Out-Null
}
else {
    $process.WaitForExit()
}
$stopwatch.Stop()
$process.Refresh()
$childExitCode = $null
if ($process.HasExited) { $childExitCode = [int]$process.ExitCode }
$endUtc = [DateTime]::UtcNow
$finalBytes = Get-OutputBytes $OutputDir
$status = 'INCOMPLETE'
if ($timedOut -or $outputLimitReached) {
    $status = 'RESOURCE_LIMIT_REACHED'
    Set-RunStatus $status $stopReason
}
else {
    $runManifestPath = Join-Path $OutputDir 'run_manifest.json'
    if (Test-Path -LiteralPath $runManifestPath) {
        try {
            $runManifest = Get-Content -Raw -LiteralPath $runManifestPath | ConvertFrom-Json
            if ($runManifest.status) { $status = [string]$runManifest.status }
        }
        catch { $status = 'INCOMPLETE' }
    }
}
if ($finalBytes -gt $OutputLimitBytes) {
    $status = 'RESOURCE_LIMIT_REACHED'
    $stopReason = '100 MiB output storage ceiling exceeded'
    Set-RunStatus $status $stopReason
}
$launcherManifest.status = $status
$launcherManifest.gpu_child_finished_at_utc = $endUtc.ToString('o')
$launcherManifest.gpu_child_wall_seconds = [Math]::Round($stopwatch.Elapsed.TotalSeconds, 6)
$launcherManifest.gpu_child_exit_code = $childExitCode
$launcherManifest.output_bytes_observed = $finalBytes
$launcherManifest.peak_cpu_rss_bytes = $peakRss
$launcherManifest.supervisor_stop_reason = $stopReason
$launcherManifest.stdout_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $stdoutPath).Hash.ToLowerInvariant()
$launcherManifest.stderr_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $stderrPath).Hash.ToLowerInvariant()
Write-JsonAtomic $launcherManifestPath $launcherManifest

Write-Output "R1-S2 bounded GPU child finished: status=$status exit_code=$childExitCode wall_seconds=$($launcherManifest.gpu_child_wall_seconds) output_bytes=$finalBytes"
if ($timedOut -or $outputLimitReached) { exit 5 }
if ($status -ne 'C64_PARITY_PASS') { exit 3 }
exit 0
