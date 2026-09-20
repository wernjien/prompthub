# PromptHub acceptance-criteria checks - Windows (see requirements.md
# Section 8, adapted for the Docker-free Functions architecture - see
# README.md "Architecture note"). Run after setup\install_windows.ps1 or
# setup\start.ps1 (both seed Open WebUI automatically, no manual UI steps).
$RepoRoot = Split-Path -Parent $PSScriptRoot
$HomeDir = if ($env:PROMPTHUB_HOME) { $env:PROMPTHUB_HOME } else { "$HOME\PromptHub" }
$VenvPython = "$HomeDir\venv\Scripts\python.exe"
$TextModel = "dolphin3:8b"
$VisionModel = "llava:13b"
$PassCount = 0
$FailCount = 0

function Pass($msg) { Write-Host "  PASS: $msg"; $script:PassCount++ }
function Fail($msg) { Write-Host "  FAIL: $msg"; $script:FailCount++ }

# `ollama list`/`ollama ps` have been observed to transiently omit a row
# under memory pressure (see README "Known risks" on 16GB being tight with
# two large models loaded) - retry the match a few times before treating an
# absence as real.
function Get-RetryMatch($pattern, [scriptblock]$cmd) {
    for ($i = 0; $i -lt 5; $i++) {
        $out = & $cmd | Select-String -Pattern ([regex]::Escape($pattern))
        if ($out) { return $out }
        Start-Sleep -Seconds 1
    }
    return $out
}

Write-Host "== 1. Models present in Ollama =="
if (Get-RetryMatch $TextModel { ollama list }) { Pass "$TextModel pulled" } else { Fail "$TextModel missing - run: ollama pull $TextModel" }
if (Get-RetryMatch $VisionModel { ollama list }) { Pass "$VisionModel pulled" } else { Fail "$VisionModel missing - run: ollama pull $VisionModel" }

Write-Host "== 2. GPU utilization (not CPU fallback) =="
# Generate once and wait for it to finish, rather than racing a background
# request: the model stays resident for ollama's keep-alive window
# afterwards, so 'ollama ps' can be read with no timing window to miss.
# A cold model can take tens of seconds to load into VRAM, which no short
# sleep alongside an in-flight request can reliably cover.
foreach ($model in @($TextModel, $VisionModel)) {
    $body = @{ model = $model; prompt = "hi"; stream = $false } | ConvertTo-Json
    $generated = $true
    Write-Host "  (loading $model - first run can take a minute)"
    try {
        Invoke-RestMethod -Uri "http://localhost:11434/api/generate" -Method Post `
            -Body $body -ContentType "application/json" -TimeoutSec 300 | Out-Null
    } catch {
        $generated = $false
        Fail "$model : generation request to Ollama failed - $($_.Exception.Message)"
    }
    if ($generated) {
        $psLine = Get-RetryMatch $model { ollama ps }
        if (-not $psLine) {
            Fail "$model : generated, but 'ollama ps' does not list it"
        } elseif ($psLine -match "100% CPU") {
            Fail "$model : running on CPU, not GPU - check NVIDIA driver/CUDA setup ($psLine)"
        } else {
            Pass "$model : GPU-accelerated ($psLine)"
        }
    }
}

Write-Host "== 3. ffmpeg =="
if (Get-Command ffmpeg -ErrorAction SilentlyContinue) { Pass "ffmpeg on host" } else { Fail "ffmpeg not found on host" }
if (Get-Command ffprobe -ErrorAction SilentlyContinue) { Pass "ffprobe on host" } else { Fail "ffprobe not found on host (ships with ffmpeg)" }

Write-Host "== 4. Open WebUI reachable =="
try { Invoke-WebRequest -Uri "http://localhost:8080" -UseBasicParsing -TimeoutSec 5 | Out-Null; Pass "Open WebUI reachable at http://localhost:8080" }
catch { Fail "Open WebUI not reachable - run setup\start.ps1" }

Write-Host "== 5. Picker holds exactly the 2 PromptHub Functions =="
if (Test-Path $VenvPython) {
    $checkOutput = & $VenvPython "$RepoRoot\setup\check_seeded.py"
    $checkOutput | ForEach-Object {
        Write-Host $_
        if ($_ -match "PASS:") { $script:PassCount++ }
        if ($_ -match "FAIL:") { $script:FailCount++ }
    }
} else {
    Fail "venv not found at $HomeDir\venv - run setup\install_windows.ps1 first"
}

Write-Host ""
Write-Host "== 6. Manual checks (not automatable) =="
Write-Host "  Both entries accept typed text, an attachment, or both - check a few"
Write-Host "  combinations, especially text + attachment together."
Write-Host "  [ ] Krea2 - a natural-language paragraph, no comma tags, no (word:1.3)"
Write-Host "      weighting syntax. With text + image, the text drives the scene."
Write-Host "  [ ] MiniMax H3 - three fields plus a footer naming the inferred mode."
Write-Host "      Spot-check: nothing attached -> T2VA; one image -> I2VA; two"
Write-Host "      images -> FL2VA; one image + ends on -> L2VA; use L2VA -> L2VA."
Write-Host "  [ ] Keyframe modes report reference-frame path(s) that exist on disk"
Write-Host "      (default: ~\PromptHub\output), or say none was saved."
Write-Host "  [ ] No network activity while any of the above run."
Write-Host ""
Write-Host "== Summary: $PassCount passed, $FailCount failed (automated checks only) =="
if ($FailCount -gt 0) { exit 1 }
