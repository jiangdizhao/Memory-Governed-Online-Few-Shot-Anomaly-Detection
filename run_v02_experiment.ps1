# V0.2 one-command Real-IAD paired cold/mature architecture experiment.
# Run from the repo root in an initialized Conda PowerShell:
#   .\run_v02_experiment.ps1
param(
    [string]$DatasetRoot = "G:/dataset/realiad"
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
git fetch origin
if ($LASTEXITCODE -ne 0) { throw "Git fetch failed" }
git switch v02-paired-cold-mature-architecture
if ($LASTEXITCODE -ne 0) { throw "Cannot switch V0.2 branch" }
git pull --ff-only origin v02-paired-cold-mature-architecture
if ($LASTEXITCODE -ne 0) { throw "Git fast-forward failed" }
$env:PYTHONIOENCODING = "utf-8"
conda activate torch
if ($LASTEXITCODE -ne 0) { throw "Conda torch environment not available" }
python -m py_compile experiments/v02_architecture_core.py experiments/v02_memory.py experiments/v02_architecture_trial.py experiments/run_v02_architecture.py
if ($LASTEXITCODE -ne 0) { throw "Python compile failed" }
python -m unittest discover -s tests -p "test_v02_architecture.py" -v
if ($LASTEXITCODE -ne 0) { throw "V0.2 offline tests failed" }
python experiments/run_v02_architecture.py --self-test
if ($LASTEXITCODE -ne 0) { throw "V0.2 self test failed" }
# Metadata/physical-object preflight is included automatically in --all.
# All 36 cases are run consecutively and previously validated results resume.
python experiments/run_v02_architecture.py --all --dataset-root "$DatasetRoot" --math-sdpa
if ($LASTEXITCODE -ne 0) {
    throw "V0.2 stopped at failed/invalid condition. Preserve logs; do not delete evidence."
}
Write-Host "[V02] All runs validated. Final report: outputs/v02_paired_cold_mature/v02_final_decision.json"
