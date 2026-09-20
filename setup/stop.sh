#!/usr/bin/env bash
# Stops the background Open WebUI process started by setup/start.sh.
set -euo pipefail

HOME_DIR="${PROMPTHUB_HOME:-$HOME/PromptHub}"
PID_FILE="$HOME_DIR/openwebui.pid"

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  pid="$(cat "$PID_FILE")"
  kill "$pid"
  # Wait for it to actually exit: kill only sends the signal, and start.sh
  # refuses to boot while anything is still answering on :8080.
  for _ in $(seq 1 20); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.5
  done
  rm -f "$PID_FILE"
  echo "Stopped."
else
  echo "Open WebUI is not running (or was started outside setup/start.sh)."
fi
