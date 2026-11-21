$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

$steps = @(
    'audit_real_inputs.py',
    'audit_spatial_coverage.py',
    'audit_firms_history.py',
    'run_preprocess.py',
    'multiplicative_followup.py',
    'review_followup.py',
    'review_aux.py',
    'review_calibration.py',
    'make_empirical_figures.py'
)
foreach ($step in $steps) {
    Write-Host "Running $step"
    & python -u $step
    if ($LASTEXITCODE -ne 0) { throw "$step failed with exit code $LASTEXITCODE" }
}
