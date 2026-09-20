#!/usr/bin/env bash
# PromptHub setup — macOS (Apple Silicon / Metal), no Docker.
set -euo pipefail

HOME_DIR="${PROMPTHUB_HOME:-$HOME/PromptHub}"
OUTPUT_DIR="$HOME_DIR/output"
VENV_DIR="$HOME_DIR/venv"
TEXT_MODEL="dolphin3:8b"
VISION_MODEL="llava:13b"

command -v brew >/dev/null || { echo "Homebrew is required: https://brew.sh"; exit 1; }

echo "==> Installing Ollama, ffmpeg, Python 3.11 via Homebrew"
brew list ollama >/dev/null 2>&1 || brew install ollama
brew list ffmpeg >/dev/null 2>&1 || brew install ffmpeg
brew list python@3.11 >/dev/null 2>&1 || brew install python@3.11

echo "==> Starting Ollama"
brew services start ollama >/dev/null 2>&1 || true
sleep 2

echo "==> Pulling models: $TEXT_MODEL, $VISION_MODEL"
ollama pull "$TEXT_MODEL"
ollama pull "$VISION_MODEL"

echo "==> Creating Python venv at $VENV_DIR"
mkdir -p "$HOME_DIR" "$OUTPUT_DIR"
PYBIN="$(brew --prefix python@3.11)/bin/python3.11"
"$PYBIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --upgrade pip >/dev/null
"$VENV_DIR/bin/pip" install open-webui

echo "==> Starting Open WebUI and seeding PromptHub Functions"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$REPO_ROOT/setup/start.sh"

cat <<EOF

Done — everything is running and configured, nothing to click through.
  - Open http://localhost:8080 and log in (admin credentials were printed
    above on first run, and are saved to $HOME_DIR/.admin_credentials.json).
  - ./setup/verify.sh to check the acceptance criteria.

I2V reference frames will be written to: $OUTPUT_DIR
EOF
