#!/usr/bin/env bash
# PromptHub setup — Linux (CUDA/NVIDIA or ROCm/AMD), no Docker.
set -euo pipefail

HOME_DIR="${PROMPTHUB_HOME:-$HOME/PromptHub}"
OUTPUT_DIR="$HOME_DIR/output"
VENV_DIR="$HOME_DIR/venv"
TEXT_MODEL="dolphin3:8b"
VISION_MODEL="llava:13b"

echo "==> Installing Ollama"
if ! command -v ollama >/dev/null; then
  curl -fsSL https://ollama.com/install.sh | sh
fi

echo "==> Installing ffmpeg + Python 3.11/3.12"
PYBIN=""
for cand in python3.11 python3.12; do
  command -v "$cand" >/dev/null && { PYBIN="$cand"; break; }
done

if command -v apt >/dev/null; then
  sudo apt update
  command -v ffmpeg >/dev/null || sudo apt install -y ffmpeg
  [ -n "$PYBIN" ] || { sudo apt install -y python3.11 python3.11-venv 2>/dev/null && PYBIN="python3.11"; } || true
elif command -v dnf >/dev/null; then
  command -v ffmpeg >/dev/null || sudo dnf install -y ffmpeg
  [ -n "$PYBIN" ] || { sudo dnf install -y python3.11 2>/dev/null && PYBIN="python3.11"; } || true
elif command -v pacman >/dev/null; then
  command -v ffmpeg >/dev/null || sudo pacman -S --noconfirm ffmpeg
  [ -n "$PYBIN" ] || { sudo pacman -S --noconfirm python 2>/dev/null && PYBIN="python3"; } || true
else
  echo "Unrecognized package manager — install ffmpeg and Python 3.11/3.12 manually"; exit 1
fi

if [ -z "$PYBIN" ]; then
  echo "Could not find or install Python 3.11/3.12 (Open WebUI requires one of these,"
  echo "3.13 is not yet supported). Install one manually (e.g. via pyenv) and re-run."
  exit 1
fi

echo "==> Starting Ollama (systemd if available, else background)"
if command -v systemctl >/dev/null && systemctl list-unit-files | grep -q ollama; then
  sudo systemctl enable --now ollama
else
  nohup ollama serve >/tmp/ollama.log 2>&1 &
  sleep 2
fi

echo "==> Pulling models: $TEXT_MODEL, $VISION_MODEL"
ollama pull "$TEXT_MODEL"
ollama pull "$VISION_MODEL"

echo "==> Creating Python venv at $VENV_DIR (using $PYBIN)"
mkdir -p "$HOME_DIR" "$OUTPUT_DIR"
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

NOTE: if you have an NVIDIA GPU, confirm the driver is installed
(nvidia-smi should work) — Ollama uses it directly and needs no container
toolkit now that everything runs natively.
EOF
