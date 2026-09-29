#!/usr/bin/env bash
# Starts Open WebUI in the background (macOS / Linux) and seeds PromptHub's Functions; safe to re-run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOME_DIR="${PROMPTHUB_HOME:-$HOME/PromptHub}"
VENV_DIR="$HOME_DIR/venv"
PID_FILE="$HOME_DIR/openwebui.pid"
LOG_FILE="$HOME_DIR/openwebui.log"

[ -x "$VENV_DIR/bin/open-webui" ] || { echo "Run setup/install_<os>.sh first"; exit 1; }

running() { [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; }
serving() { curl -sf http://localhost:8080 >/dev/null 2>&1; }

boot() {
  if running; then
    serving && return 0
    echo "Open WebUI is already starting (pid $(cat "$PID_FILE")); waiting for it..."
  else
    # An untracked instance (e.g. pid file deleted with the home dir) would get seeded with its old database.
    if serving; then
      echo "Something is already serving http://localhost:8080, but it wasn't started"
      echo "by this script. It is probably an Open WebUI left over from an earlier run,"
      echo "still holding the old database. Close it and re-run this script:"
      echo "  pkill -f open-webui"
      return 1
    fi

    # Run from $HOME_DIR: Open WebUI writes files like .webui_secret_key into the working directory.
    cd "$HOME_DIR"
    DATA_DIR="$HOME_DIR/data" nohup "$VENV_DIR/bin/open-webui" serve --port 8080 >"$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    disown  # so a later stop() kill isn't reported as "Terminated"
    echo "Waiting for Open WebUI to come up (first launch downloads an embedding"
    echo "model and can take several minutes; later starts are much faster)..."
  fi

  # 5 minutes: the first launch fetches an embedding model, which can be slow on a poor link.
  for i in $(seq 1 150); do
    # Checked first, so another server answering on :8080 can't pass for ours after it died.
    if ! running; then
      echo "Open WebUI process exited unexpectedly — check $LOG_FILE"
      return 1
    fi
    serving && return 0
    if [ $((i % 15)) -eq 0 ]; then
      echo "  still starting ($((i * 2))s elapsed, process alive)..."
    fi
    sleep 2
  done
  echo "Still not responding after 5 minutes. The process is running but isn't"
  echo "serving yet. Last lines of $LOG_FILE :"
  tail -20 "$LOG_FILE" 2>/dev/null | sed 's/^/    /'
  return 1
}

stop() {
  running || return 0
  pid="$(cat "$PID_FILE")"
  kill "$pid"
  for _ in $(seq 1 20); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.5
  done
  # boot() refuses to start while the old process still holds :8080.
  kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null
  rm -f "$PID_FILE"
}

boot || exit 1
echo "Open WebUI ready at http://localhost:8080"

set +e
seed_output="$("$VENV_DIR/bin/python" "$REPO_ROOT/setup/seed.py")"
seed_status=$?
set -e
echo "$seed_output"
[ "$seed_status" -eq 0 ] || exit 1

if echo "$seed_output" | grep -q "^NEEDS_RESTART$"; then
  echo "New Function(s) were created — restarting once so their pip requirements install..."
  stop
  boot || exit 1
  echo "Open WebUI ready at http://localhost:8080 (pid $(cat "$PID_FILE")). Logs: $LOG_FILE"
fi
