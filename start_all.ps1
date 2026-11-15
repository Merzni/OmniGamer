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
$VlmUrl = if ($env:OMNIGAMER_VLM_HEALTH_URL) { $env:OMNIGAMER_VLM_HEALTH_URL } else { 'http://127.0.0.1:8081/health' }
$VlmPidFile = Join-Path $RuntimeDir 'vlm.pid'
if (-not (Test-Health $VlmUrl) -and -not (Test-Path $VlmPidFile)) {
    if (-not $Llama -or -not $Model -or -not $Projector) {
        throw 'Set OMNIGAMER_LLAMA_SERVER, OMNIGAMER_VLM_MODEL, and OMNIGAMER_VLM_PROJECTOR.'
    }
    if (-not (Test-Path -LiteralPath $Llama) -or -not (Test-Path -LiteralPath $Model) -or
        -not (Test-Path -LiteralPath $Projector)) { throw 'Configured llama.cpp executable/model/projector path does not exist.' }
    $proc = Start-Process -FilePath $Llama -ArgumentList @('-m', $Model, '--mmproj', $Projector,
        '--host', '127.0.0.1', '--port', '8081', '-ngl', '99', '-c', '4096',
        '--alias', 'qwen3.5-4b-local', '--reasoning', 'off') -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $LogDir 'vlm.stdout.log') `
        -RedirectStandardError (Join-Path $LogDir 'vlm.stderr.log')
    @{ pid = $proc.Id; executable = (Resolve-Path -LiteralPath $Llama).Path } |
        ConvertTo-Json | Set-Content -LiteralPath $VlmPidFile -Encoding UTF8
}
Wait-Health 'VLM' $VlmUrl

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
