param(
    [switch]$SkipXgb,
    [switch]$SkipTransformer,
    [switch]$SkipRegime
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root

$GpuPython = Join-Path $Root ".gpu-venv\Scripts\python.exe"
if (-not (Test-Path $GpuPython)) {
    $GpuPython = "python"
}

$env:TRADEMIND_XGB_DEVICE = "cuda"

if (-not $SkipXgb) {
    & $GpuPython -u backend\services\ml\tune_v4_xgb_trade_success.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    & $GpuPython -u backend\services\ml\tune_v4_precision_trade_success.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if (-not $SkipTransformer) {
    & python -u backend\services\ml\tune_v4_transformer_trade_success.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if (-not $SkipRegime) {
    & $GpuPython -u backend\services\ml\tune_v4_regime_specialists.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

& python -u backend\services\ml\evaluate_v4_promotion_gate.py
exit $LASTEXITCODE
