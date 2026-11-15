$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RuntimeDir = Join-Path $ProjectRoot 'data\runtime'
foreach ($name in @('vlm', 'detector')) {
    $pidFile = Join-Path $RuntimeDir "$name.pid"
    if (-not (Test-Path -LiteralPath $pidFile)) { continue }
    $record = Get-Content -LiteralPath $pidFile -Raw | ConvertFrom-Json
    $proc = Get-CimInstance Win32_Process -Filter "ProcessId = $($record.pid)" -ErrorAction SilentlyContinue
    $owned = $false
    if ($proc -and $name -eq 'vlm') { $owned = $proc.ExecutablePath -eq $record.executable }
    if ($proc -and $name -eq 'detector') {
        $owned = $proc.ExecutablePath -eq $record.executable -and
                 $proc.CommandLine.Contains((Join-Path $ProjectRoot 'tools\detector_server.py'))
    }
    if ($owned) {
        Stop-Process -Id $record.pid -Force
        Write-Host "Stopped owned $name process $($record.pid)"
    } else {
        Write-Host "Skipped $name PID $($record.pid): process identity did not match"
    }
    Remove-Item -LiteralPath $pidFile -Force
}
