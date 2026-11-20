$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".gpu-venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "GPU Python environment missing" }

Set-Location $ProjectRoot
$env:PYTHONIOENCODING = "utf-8"
$env:TRADEMIND_XGB_DEVICE = "cuda"
$RunName = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$env:TRADEMIND_AI_FILTER_OUTPUT_DIR = Join-Path $ProjectRoot "data\processed\ai_filter_profit_search_gpu_optimized\$RunName"
New-Item -ItemType Directory -Force -Path $env:TRADEMIND_AI_FILTER_OUTPUT_DIR | Out-Null

& $Python backend/services/ml/verify_gpu.py
if ($LASTEXITCODE -ne 0) { throw "CUDA validation failed; research not started" }
Write-Host "Optimized results directory: $env:TRADEMIND_AI_FILTER_OUTPUT_DIR"
& $Python -u backend/services/ml/research_ai_filter_profit_search.py
exit $LASTEXITCODE
