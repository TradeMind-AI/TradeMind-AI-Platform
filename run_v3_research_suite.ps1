param(
    [switch]$RebuildFeatures
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$SystemPython = "python"
$GpuPython = Join-Path $ProjectRoot ".gpu-venv\Scripts\python.exe"

if (-not (Test-Path $GpuPython)) {
    $GpuPython = $SystemPython
}

Set-Location $ProjectRoot

$LogDirectory = Join-Path $ProjectRoot "logs\v3_research"
New-Item -ItemType Directory -Force -Path $LogDirectory | Out-Null

$Timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$LogFile = Join-Path $LogDirectory "v3_research_$Timestamp.log"

function Run-Step {
    param(
        [string]$Label,
        [string]$Python,
        [string]$Script,
        [string[]]$Arguments = @(),
        [hashtable]$Environment = @{}
    )

    Write-Host ""
    Write-Host ("=" * 78)
    Write-Host $Label
    Write-Host ("=" * 78)

    Add-Content -Path $LogFile -Value ""
    Add-Content -Path $LogFile -Value ("=" * 78)
    Add-Content -Path $LogFile -Value $Label
    Add-Content -Path $LogFile -Value ("=" * 78)

    $Previous = @{}

    foreach ($Name in $Environment.Keys) {
        $Previous[$Name] = [Environment]::GetEnvironmentVariable(
            $Name,
            "Process"
        )

        [Environment]::SetEnvironmentVariable(
            $Name,
            $Environment[$Name],
            "Process"
        )
    }

    try {
        & $Python -u $Script @Arguments 2>&1 |
            Tee-Object -FilePath $LogFile -Append

        if ($LASTEXITCODE -ne 0) {
            throw "$Label failed with exit code $LASTEXITCODE"
        }
    }
    finally {
        foreach ($Name in $Environment.Keys) {
            [Environment]::SetEnvironmentVariable(
                $Name,
                $Previous[$Name],
                "Process"
            )
        }
    }
}

"TradeMind V3 research suite" | Out-File $LogFile
"Started: $(Get-Date)" | Out-File $LogFile -Append
"Project: $ProjectRoot" | Out-File $LogFile -Append
"Rebuild features: $RebuildFeatures" | Out-File $LogFile -Append

if ($RebuildFeatures) {
    Run-Step -Label "Rebuild 1m/2m/5m/10m candles" -Python $SystemPython -Script (
        Join-Path $ProjectRoot "backend\services\data_pipeline\resample_timeframes.py"
    ) -Arguments @(
        "--timeframes",
        "1m",
        "2m",
        "5m",
        "10m"
    )

    foreach ($Timeframe in @("1m", "2m", "5m", "10m")) {
        Run-Step -Label ("Rebuild " + $Timeframe + " research features") -Python $SystemPython -Script (
            Join-Path $ProjectRoot "backend\services\data_pipeline\feature_engineering_$Timeframe.py"
        )
    }
}

Run-Step -Label "V3 multi-timeframe profit-first scan" -Python $SystemPython -Script (
    Join-Path $ProjectRoot "backend\services\ml\research_v3_multitimeframe_profit_first.py"
)

Run-Step -Label "V3 path-aware barrier opportunity scan" -Python $GpuPython -Script (
    Join-Path $ProjectRoot "backend\services\ml\research_v3_barrier_opportunity.py"
) -Environment @{
    "TRADEMIND_XGB_DEVICE" = "cuda"
}

Run-Step -Label "V3 Transformer barrier trade-success benchmark" -Python $SystemPython -Script (
    Join-Path $ProjectRoot "backend\services\ml\research_v3_transformer_barrier.py"
)

Run-Step -Label "V3 strict promotion gate" -Python $SystemPython -Script (
    Join-Path $ProjectRoot "backend\services\ml\evaluate_v3_promotion_gate.py"
)

Write-Host ""
Write-Host ("=" * 78)
Write-Host "V3 RESEARCH SUITE COMPLETE"
Write-Host ("=" * 78)
Write-Host "Log: $LogFile"

Add-Content -Path $LogFile -Value ""
Add-Content -Path $LogFile -Value "Finished: $(Get-Date)"
