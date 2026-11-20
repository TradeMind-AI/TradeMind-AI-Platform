$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

Write-Host "TradeMind raw microstructure collector"
Write-Host "Spot + USDT-M Futures BTCUSDT"
Write-Host "Press Ctrl+C to stop."
Write-Host ""

python backend/services/data_pipeline/collect_live_microstructure.py
