param([switch]$WaitForDownload)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
New-Item -ItemType Directory -Force logs | Out-Null
if ($WaitForDownload) {
    while (-not (Test-Path -LiteralPath 'data/raw/modis_public/MOD13A1_2020_2024_samples.nc')) {
        Start-Sleep -Seconds 10
    }
}
foreach ($step in @('build_standalone_data.py','train_standalone.py','evaluate_standalone.py','export_formulas.py','make_standalone_paper.py')) {
    Write-Host "Starting $step"
    & python -u $step *> (Join-Path 'logs' ($step + '.log'))
    if ($LASTEXITCODE -ne 0) { throw "$step failed; see logs" }
    Write-Host "Completed $step"
}
& python check_tex_static.py final_paper.tex
if ($LASTEXITCODE -ne 0) { throw 'Static paper check failed' }
Write-Host 'All calculations complete. Compile final_paper.tex with Tectonic.'
