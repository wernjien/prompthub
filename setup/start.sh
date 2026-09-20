#!/usr/bin/env bash
# Starts Open WebUI in the background (macOS / Linux) and seeds it with
# PromptHub's Pipe Functions via the API — no manual UI
# steps. Safe to re-run any time (e.g. after editing a function/prompt).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOME_DIR="${PROMPTHUB_HOME:-$HOME/PromptHub}"
VENV_DIR="$HOME_DIR/venv"
PID_FILE="$HOME_DIR/openwebui.pid"
LOG_FILE="$HOME_DIR/openwebui.log"

[ -x "$VENV_DIR/bin/open-webui" ] || { echo "Run setup/install_<os>.sh first"; exit 1; }

boot() {
  if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    return 0
  fi

  # Deleting the home directory also deletes the pid file, so an Open WebUI
  # from an earlier run can still hold :8080 while being untracked here.
  # Seeding would then silently target that stale instance and its old
  # database — which looks exactly like seeding "not working" — so refuse to
  # start rather than guess.
  if curl -sf http://localhost:8080 >/dev/null 2>&1; then
    echo "Something is already serving http://localhost:8080, but it wasn't started"
    echo "by this script. It is probably an Open WebUI left over from an earlier run,"
    echo "still holding the old database. Close it and re-run this script:"
    echo "  pkill -f open-webui"
    return 1
  fi

  # Run from $HOME_DIR, not the repo checkout: Open WebUI writes a couple of
  # small files (e.g. .webui_secret_key) relative to the current directory,
  # and those must never end up inside the git repo.
  cd "$HOME_DIR"
  DATA_DIR="$HOME_DIR/data" nohup "$VENV_DIR/bin/open-webui" serve --port 8080 >"$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"

  # 5 minutes, not 2: the first launch after a clean install fetches the
  # embedding model from Hugging Face, and on a slow link that alone can
  # outlast a shorter timeout while the process is perfectly healthy.
  echo "Waiting for Open WebUI to come up (first launch downloads an embedding"
  echo "model and can take several minutes; later starts are much faster)..."
  for i in $(seq 1 150); do
    # Checked before the HTTP probe, not after: if our own process died (e.g.
    # lost a race for the port) while something else answers on :8080, a
    # successful probe would otherwise look like success.
    if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      echo "Open WebUI process exited unexpectedly — check $LOG_FILE"
      return 1
    fi
    if curl -sf http://localhost:8080 >/dev/null 2>&1; then
      return 0
    fi
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
  if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    pid="$(cat "$PID_FILE")"
    kill "$pid"
    # Wait for it to actually exit rather than assuming a fixed sleep is
    # enough: boot() below refuses to start while :8080 still answers.
    for _ in $(seq 1 20); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.5
    done
    rm -f "$PID_FILE"
  fi
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
