# PromptHub setup - Windows (CUDA/NVIDIA), no Docker.
# Run in an elevated PowerShell prompt.
$ErrorActionPreference = "Stop"

$HomeDir = if ($env:PROMPTHUB_HOME) { $env:PROMPTHUB_HOME } else { "$HOME\PromptHub" }
$OutputDir = "$HomeDir\output"
$VenvDir = "$HomeDir\venv"
$TextModel = "dolphin3:8b"
$VisionModel = "llava:13b"

# winget updates the machine/user PATH in the registry, but this process's
# $env:Path is a snapshot taken at startup - without refreshing it here,
# a tool installed a few lines below stays "not found" for the rest of this
# same script run (and this same terminal session) until a new shell opens.
# Appends rather than replaces, so anything this session added to PATH but
# never wrote to the registry (an activated conda env, say) survives.
function Update-Path {
    $registryPath = @(
        [System.Environment]::GetEnvironmentVariable("Path", "Machine"),
        [System.Environment]::GetEnvironmentVariable("Path", "User")
    ) -join ";"
    $current = @($env:Path -split ";" | Where-Object { $_ })
    foreach ($dir in ($registryPath -split ";" | Where-Object { $_ })) {
        if ($current -notcontains $dir) { $env:Path += ";$dir" }
    }
}

Write-Host "==> Installing Ollama"
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    winget install --id Ollama.Ollama -e
    Update-Path
}

Write-Host "==> Installing ffmpeg"
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    winget install --id Gyan.FFmpeg -e
    Update-Path
}

Write-Host "==> Installing Python 3.11 (Open WebUI needs 3.11 or 3.12, not 3.13)"
$HasPy311 = $false
if (Get-Command py -ErrorAction SilentlyContinue) {
    try {
        $null = & py -3.11 --version 2>$null
        $HasPy311 = ($LASTEXITCODE -eq 0)
    } catch {
        $HasPy311 = $false
    }
}
if (-not $HasPy311) {
    winget install --id Python.Python.3.11 -e
    Update-Path
}

Write-Host "==> Pulling models: $TextModel, $VisionModel"
ollama pull $TextModel
ollama pull $VisionModel

Write-Host "==> Creating Python venv at $VenvDir"
New-Item -ItemType Directory -Force -Path $HomeDir, $OutputDir | Out-Null
py -3.11 -m venv $VenvDir
& "$VenvDir\Scripts\pip.exe" install --upgrade pip | Out-Null
& "$VenvDir\Scripts\pip.exe" install open-webui

Write-Host "==> Starting Open WebUI and seeding PromptHub Functions"
$RepoRoot = Split-Path -Parent $PSScriptRoot
# Reset first: a script that succeeds leaves $LASTEXITCODE at whatever the
# previous command set, so without this a stale non-zero would read as a
# failure that never happened.
$global:LASTEXITCODE = 0
& "$RepoRoot\setup\start.ps1"
# `exit` inside the called script does not stop this one, so without checking
# here a failed start/seed would still be followed by "Done - everything is
# running and configured".
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Setup did NOT complete - Open WebUI failed to start or seed (see above)."
    Write-Host "Nothing else was configured. Fix the error, then re-run .\setup\start.ps1"
    exit 1
}

Write-Host ""
Write-Host "Done - everything is running and configured, nothing to click through."
Write-Host "  - Open http://localhost:8080 and log in (admin credentials were printed"
Write-Host "    above on first run, and are saved to $HomeDir\.admin_credentials.json)."
Write-Host "  - .\setup\verify.ps1 to check the acceptance criteria."
Write-Host ""
Write-Host "I2V reference frames will be written to: $OutputDir"
