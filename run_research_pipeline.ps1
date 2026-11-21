$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
New-Item -ItemType Directory -Force logs | Out-Null
foreach ($step in @('run_research.py','evaluate_research.py','make_research_paper.py')) {
    & python -u $step *> (Join-Path 'logs' ($step + '.log'))
    if ($LASTEXITCODE -ne 0) { throw "$step failed; see logs" }
}
& python check_tex_static.py research_paper.tex
if ($LASTEXITCODE -ne 0) { throw 'Static paper check failed' }
if (Get-Command tectonic -ErrorAction SilentlyContinue) {
    & tectonic research_paper.tex
    if ($LASTEXITCODE -ne 0) { throw 'Paper compilation failed' }
}
