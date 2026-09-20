# Starts Open WebUI in the background (Windows) and seeds it with
# PromptHub's Pipe Functions via the API - no manual UI
# steps. Safe to re-run any time (e.g. after editing a function/prompt).
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$HomeDir = if ($env:PROMPTHUB_HOME) { $env:PROMPTHUB_HOME } else { "$HOME\PromptHub" }
$VenvDir = "$HomeDir\venv"
$PidFile = "$HomeDir\openwebui.pid"
$LogFile = "$HomeDir\openwebui.log"
$Exe = "$VenvDir\Scripts\open-webui.exe"
$PythonExe = "$VenvDir\Scripts\python.exe"

if (-not (Test-Path $Exe)) { throw "Run setup\install_windows.ps1 first" }

function Start-OpenWebUI {
    if (Test-Path $PidFile) {
        $existing = Get-Process -Id (Get-Content $PidFile) -ErrorAction SilentlyContinue
        if ($existing) { return $true }
    }

    # Deleting the home directory also deletes the pid file, so an Open WebUI
    # from an earlier run can still hold :8080 while being untracked here.
    # Seeding would then silently target that stale instance and its old
    # database - which looks exactly like seeding "not working" - so refuse
    # to start rather than guess.
    try {
        Invoke-WebRequest -Uri "http://localhost:8080" -UseBasicParsing -TimeoutSec 2 | Out-Null
        Write-Host "Something is already serving http://localhost:8080, but it wasn't started"
        Write-Host "by this script. It is probably an Open WebUI left over from an earlier run,"
        Write-Host "still holding the old database. Close it and re-run this script:"
        Write-Host '  Get-Process | Where-Object { $_.Path -like "*PromptHub*" } | Stop-Process -Force'
        return $false
    } catch {
        # nothing on the port - good, carry on and start our own
    }

    $env:DATA_DIR = "$HomeDir\data"
    # WorkingDirectory = $HomeDir, not the repo checkout: Open WebUI writes
    # a couple of small files (e.g. .webui_secret_key) relative to the
    # current directory, and those must never end up inside the git repo.
    $proc = Start-Process -FilePath $Exe -ArgumentList "serve", "--port", "8080" `
        -WorkingDirectory $HomeDir `
        -RedirectStandardOutput $LogFile -RedirectStandardError "$LogFile.err" `
        -WindowStyle Hidden -PassThru
    $proc.Id | Out-File -FilePath $PidFile

    # 5 minutes, not 2: the first launch after a clean install fetches the
    # embedding model from Hugging Face, and on a slow link that alone can
    # outlast a shorter timeout while the process is perfectly healthy.
    $timeoutSeconds = 300
    Write-Host "Waiting for Open WebUI to come up (first launch downloads an embedding"
    Write-Host "model and can take several minutes; later starts are much faster)..."
    for ($i = 0; $i -lt ($timeoutSeconds / 2); $i++) {
        # Checked before the HTTP probe, not only when it fails: if our own
        # process died (e.g. lost a race for the port) while something else
        # answers on :8080, a successful probe would otherwise look like success.
        if (-not (Get-Process -Id $proc.Id -ErrorAction SilentlyContinue)) {
            Write-Host "Open WebUI process exited unexpectedly - check $LogFile and $LogFile.err"
            return $false
        }
        try {
            Invoke-WebRequest -Uri "http://localhost:8080" -UseBasicParsing -TimeoutSec 2 | Out-Null
            return $true
        } catch {
            if ($i -gt 0 -and $i % 15 -eq 0) {
                Write-Host "  still starting ($($i * 2)s elapsed, process alive)..."
            }
            Start-Sleep -Seconds 2
        }
    }
    Write-Host "Still not responding after $($timeoutSeconds / 60) minutes. The process is running"
    Write-Host "but isn't serving yet. Last lines of $LogFile :"
    Get-Content $LogFile -Tail 20 -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "    $_" }
    Get-Content "$LogFile.err" -Tail 20 -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "    $_" }
    return $false
}

function Stop-OpenWebUI {
    if (Test-Path $PidFile) {
        $existing = Get-Process -Id (Get-Content $PidFile) -ErrorAction SilentlyContinue
        if ($existing) {
            Stop-Process -Id $existing.Id -Force
            Wait-Process -Id $existing.Id -Timeout 10 -ErrorAction SilentlyContinue
            Remove-Item $PidFile
            Start-Sleep -Seconds 1
        }
    }
}

if (-not (Start-OpenWebUI)) { exit 1 }
Write-Host "Open WebUI ready at http://localhost:8080"

# seed.py prints its own recovery instructions to stdout before exiting
# non-zero (e.g. when an admin account already exists but the saved
# credentials are gone). Relaxing ErrorActionPreference around the call stops
# a stderr write from becoming a terminating error that would discard those
# lines unprinted, leaving only a PowerShell stack trace to debug from.
$prevEAP = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$seedOutput = & $PythonExe "$RepoRoot\setup\seed.py" 2>&1
$seedExit = $LASTEXITCODE
$ErrorActionPreference = $prevEAP
$seedOutput | ForEach-Object { Write-Host $_ }
if ($seedExit -ne 0) { exit 1 }

if ($seedOutput -contains "NEEDS_RESTART") {
    Write-Host "New Function(s) were created - restarting once so their pip requirements install..."
    Stop-OpenWebUI
    if (-not (Start-OpenWebUI)) { exit 1 }
    Write-Host "Open WebUI ready at http://localhost:8080 (pid $(Get-Content $PidFile)). Logs: $LogFile"
}
