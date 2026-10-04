$ProjectRoot = "C:\Users\USER\TradeMind-AI"

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$SystemPython = "C:\Users\USER\AppData\Local\Python\pythoncore-3.14-64\python.exe"

if (Test-Path $VenvPython) {
    $Python = $VenvPython
}
else {
    $Python = $SystemPython
}

$TradingScript = Join-Path `
    $ProjectRoot `
    "backend\services\trading\run_paper_cycle.py"

$LogDirectory = Join-Path `
    $ProjectRoot `
    "logs\paper_trading"

New-Item `
    -ItemType Directory `
    -Force `
    -Path $LogDirectory `
    | Out-Null

$Timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"

$LogFile = Join-Path `
    $LogDirectory `
    "paper_cycle_$Timestamp.log"

Set-Location $ProjectRoot

"============================================================" `
    | Out-File $LogFile

"TradeMind automated paper cycle" `
    | Out-File $LogFile -Append

"Started: $(Get-Date)" `
    | Out-File $LogFile -Append

"Python: $Python" `
    | Out-File $LogFile -Append

"============================================================" `
    | Out-File $LogFile -Append

& $Python $TradingScript *>> $LogFile

$ExitCode = $LASTEXITCODE

"`nFinished: $(Get-Date)" `
    | Out-File $LogFile -Append

"Exit code: $ExitCode" `
    | Out-File $LogFile -Append

exit $ExitCode