# Stops the background Open WebUI process started by setup\start.ps1.
$HomeDir = if ($env:PROMPTHUB_HOME) { $env:PROMPTHUB_HOME } else { "$HOME\PromptHub" }
$PidFile = "$HomeDir\openwebui.pid"

if (Test-Path $PidFile) {
    $procId = Get-Content $PidFile
    $existing = Get-Process -Id $procId -ErrorAction SilentlyContinue
    if ($existing) {
        # -Force plus waiting for exit: a plain Stop-Process returns as soon
        # as the signal is sent, not once the process (and any file handles
        # it holds under $HomeDir) are actually gone.
        Stop-Process -Id $procId -Force
        Wait-Process -Id $procId -Timeout 10 -ErrorAction SilentlyContinue
        Remove-Item $PidFile
        Write-Host "Stopped."
        exit 0
    }
}
Write-Host "Open WebUI is not running (or was started outside setup\start.ps1)."
