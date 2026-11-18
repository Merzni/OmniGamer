param([int]$ReadyTimeoutSeconds = 120)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RuntimeDir = Join-Path $ProjectRoot 'data\runtime'
$LogDir = Join-Path $ProjectRoot 'data\logs'
New-Item -ItemType Directory -Force -Path $RuntimeDir, $LogDir | Out-Null

function Wait-Health([string]$Name, [string]$Url) {
    $deadline = (Get-Date).AddSeconds($ReadyTimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $response = Invoke-WebRequest -Uri $Url -TimeoutSec 2 -UseBasicParsing
            if ($response.StatusCode -eq 200) { Write-Host "$Name ready: $Url"; return }
        } catch { Start-Sleep -Milliseconds 700 }
    }
    throw "$Name did not become ready at $Url within $ReadyTimeoutSeconds seconds"
}
function Test-Health([string]$Url) {
    try {
        $response = Invoke-WebRequest -Uri $Url -TimeoutSec 2 -UseBasicParsing
        return $response.StatusCode -eq 200
    } catch { return $false }
}

$Llama = $env:OMNIGAMER_LLAMA_SERVER
$Model = $env:OMNIGAMER_VLM_MODEL
$Projector = $env:OMNIGAMER_VLM_PROJECTOR
$CudaDllDir = $env:OMNIGAMER_CUDA_DLL_DIR
$LlamaDevice = $env:OMNIGAMER_LLAMA_DEVICE
if ($CudaDllDir) {
    if (-not (Test-Path -LiteralPath $CudaDllDir -PathType Container)) {
        throw "Configured CUDA DLL directory does not exist: $CudaDllDir"
    }
    $env:PATH = "$CudaDllDir;$env:PATH"
}
$VlmUrl = if ($env:OMNIGAMER_VLM_HEALTH_URL) { $env:OMNIGAMER_VLM_HEALTH_URL } else { 'http://127.0.0.1:8081/health' }
$VlmPidFile = Join-Path $RuntimeDir 'vlm.pid'

# A dead or PID-reused record must not prevent recovery of a stopped server.
if (Test-Path -LiteralPath $VlmPidFile) {
    $record = $null
    try { $record = Get-Content -LiteralPath $VlmPidFile -Raw | ConvertFrom-Json }
    catch { Write-Warning "Ignoring invalid VLM PID metadata: $($_.Exception.Message)" }
    $tracked = if ($record -and $record.pid) {
        Get-CimInstance Win32_Process -Filter "ProcessId = $($record.pid)" -ErrorAction SilentlyContinue
    } else { $null }
    if (-not $tracked -or $tracked.ExecutablePath -ne $record.executable) {
        Remove-Item -LiteralPath $VlmPidFile -Force
        Write-Host 'Removed stale VLM PID metadata.'
    }
}

if (-not (Test-Health $VlmUrl)) {
    if (-not $Llama -or -not $Model -or -not $Projector) {
        throw 'Set OMNIGAMER_LLAMA_SERVER, OMNIGAMER_VLM_MODEL, and OMNIGAMER_VLM_PROJECTOR.'
    }
    if (-not (Test-Path -LiteralPath $Llama) -or -not (Test-Path -LiteralPath $Model) -or
        -not (Test-Path -LiteralPath $Projector)) { throw 'Configured llama.cpp executable/model/projector path does not exist.' }
    if ($LlamaDevice) {
        $deviceOutput = @(& $Llama --list-devices 2>&1)
        $deviceList = $deviceOutput -join "`n"
        if ($LASTEXITCODE -ne 0) { throw "llama.cpp --list-devices failed: $deviceList" }
        foreach ($device in ($LlamaDevice -split ',')) {
            $device = $device.Trim()
            if ($device -and $deviceList -notmatch [regex]::Escape($device)) {
                throw "Requested llama.cpp device '$device' is not available. Devices: $deviceList"
            }
        }
    }
    $llamaArgs = @('-m', $Model, '--mmproj', $Projector,
        '--host', '127.0.0.1', '--port', '8081', '-ngl', '99', '-c', '4096',
        '--alias', 'qwen3.5-4b-local', '--reasoning', 'off')
    if ($LlamaDevice) { $llamaArgs += @('--device', $LlamaDevice) }
    $proc = Start-Process -FilePath $Llama -ArgumentList $llamaArgs -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $LogDir 'vlm.stdout.log') `
        -RedirectStandardError (Join-Path $LogDir 'vlm.stderr.log')
    @{ pid = $proc.Id; executable = (Resolve-Path -LiteralPath $Llama).Path } |
        ConvertTo-Json | Set-Content -LiteralPath $VlmPidFile -Encoding UTF8
}
Wait-Health 'VLM' $VlmUrl
if ($LlamaDevice -and $VlmUrl -eq 'http://127.0.0.1:8081/health') {
    $listener = Get-NetTCPConnection -LocalPort 8081 -State Listen -ErrorAction SilentlyContinue |
        Select-Object -First 1
    $running = if ($listener) {
        Get-CimInstance Win32_Process -Filter "ProcessId = $($listener.OwningProcess)" -ErrorAction SilentlyContinue
    } else { $null }
    $expectedExecutable = if ($Llama -and (Test-Path -LiteralPath $Llama)) {
        (Resolve-Path -LiteralPath $Llama).Path
    } else { '' }
    if (-not $running -or $running.ExecutablePath -ne $expectedExecutable -or
        $running.CommandLine -notmatch [regex]::Escape("--device $LlamaDevice")) {
        throw "VLM health endpoint is ready, but its listener does not match the configured executable/device '$LlamaDevice'. Stop the unowned server or correct its launch settings."
    }
}

$DetectorPython = $env:OMNIGAMER_DETECTOR_PYTHON
if (-not $DetectorPython) { $DetectorPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $DetectorPython)) { throw "Detector Python not found: $DetectorPython" }
$DetectorPidFile = Join-Path $RuntimeDir 'detector.pid'
if (-not (Test-Health 'http://127.0.0.1:8082/health') -and -not (Test-Path $DetectorPidFile)) {
    $proc = Start-Process -FilePath $DetectorPython -ArgumentList @((Join-Path $ProjectRoot 'tools\detector_server.py'),
        '--fp16') -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $LogDir 'detector.stdout.log') `
        -RedirectStandardError (Join-Path $LogDir 'detector.stderr.log') -WorkingDirectory $ProjectRoot
    @{ pid = $proc.Id; executable = (Resolve-Path -LiteralPath $DetectorPython).Path } |
        ConvertTo-Json | Set-Content -LiteralPath $DetectorPidFile -Encoding UTF8
}
Wait-Health 'Detector' 'http://127.0.0.1:8082/health'
Write-Host 'OmniGamer services are ready. No game window or input was used.'
