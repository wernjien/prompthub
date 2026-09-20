#!/usr/bin/env bash
# PromptHub acceptance-criteria checks — macOS / Linux (see requirements.md
# Section 8, adapted for the Docker-free Functions architecture — see
# README.md "Architecture note"). Run after setup/install_<os>.sh or
# setup/start.sh (both seed Open WebUI automatically, no manual UI steps).
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOME_DIR="${PROMPTHUB_HOME:-$HOME/PromptHub}"
VENV_DIR="$HOME_DIR/venv"

TEXT_MODEL="dolphin3:8b"
VISION_MODEL="llava:13b"
PASS=0
FAIL=0

pass() { echo "  PASS: $1"; PASS=$((PASS + 1)); }
fail() { echo "  FAIL: $1"; FAIL=$((FAIL + 1)); }

# `ollama list`/`ollama ps` have been observed to transiently omit a row
# under memory pressure (see README "Known risks" on 16GB being tight with
# two large models loaded) — retry the match a few times before treating an
# absence as real.
retry_grep() {
  local pattern="$1"; shift
  local out=""
  for _ in 1 2 3 4 5; do
    out="$("$@" 2>/dev/null | grep -F "$pattern" || true)"
    if [ -n "$out" ]; then
      printf '%s' "$out"
      return 0
    fi
    sleep 1
  done
  printf '%s' "$out"
  return 1
}

echo "== 1. Models present in Ollama =="
if retry_grep "$TEXT_MODEL" ollama list >/dev/null; then pass "$TEXT_MODEL pulled"; else fail "$TEXT_MODEL missing — run: ollama pull $TEXT_MODEL"; fi
if retry_grep "$VISION_MODEL" ollama list >/dev/null; then pass "$VISION_MODEL pulled"; else fail "$VISION_MODEL missing — run: ollama pull $VISION_MODEL"; fi

echo "== 2. GPU utilization (not CPU fallback) =="
# Generate once and wait for it to finish, rather than racing a background
# request: the model stays resident for ollama's keep-alive window
# afterwards, so 'ollama ps' can be read with no timing window to miss. A
# cold model can take tens of seconds to load, which no short sleep covers.
for model in "$TEXT_MODEL" "$VISION_MODEL"; do
  echo "  (loading $model — first run can take a minute)"
  if curl -sf -m 300 http://localhost:11434/api/generate \
      -d "{\"model\":\"$model\",\"prompt\":\"hi\",\"stream\":false}" >/dev/null 2>&1; then
    ps_line=$(retry_grep "$model" ollama ps)
    if [ -z "$ps_line" ]; then
      fail "$model: generated, but 'ollama ps' does not list it"
    elif echo "$ps_line" | grep -qi "100% CPU"; then
      fail "$model: running on CPU, not GPU — check driver/Metal/CUDA setup ($ps_line)"
    else
      pass "$model: GPU-accelerated ($ps_line)"
    fi
  else
    fail "$model: generation request to Ollama failed"
  fi
done

echo "== 3. ffmpeg =="
if command -v ffmpeg >/dev/null 2>&1; then pass "ffmpeg on host ($(ffmpeg -version | head -1))"; else fail "ffmpeg not found on host"; fi
if command -v ffprobe >/dev/null 2>&1; then pass "ffprobe on host"; else fail "ffprobe not found on host (ships with ffmpeg)"; fi

echo "== 4. Open WebUI reachable =="
if curl -sf http://localhost:8080 >/dev/null; then pass "Open WebUI reachable at http://localhost:8080"; else fail "Open WebUI not reachable — run setup/start.sh"; fi

echo "== 5. Picker holds exactly the 2 PromptHub Functions =="
if [ -x "$VENV_DIR/bin/python" ]; then
  check_output="$("$VENV_DIR/bin/python" "$REPO_ROOT/setup/check_seeded.py")"
  echo "$check_output"
  while IFS= read -r line; do
    case "$line" in
      *PASS:*) PASS=$((PASS + 1)) ;;
      *FAIL:*) FAIL=$((FAIL + 1)) ;;
    esac
  done <<< "$check_output"
else
  fail "venv not found at $VENV_DIR — run setup/install_<os>.sh first"
fi

cat <<'EOF'

== 6. Manual checks (not automatable) ==
  Both entries accept typed text, an attachment, or both — check a few
  combinations, especially text + attachment together.
  [ ] Krea2 — a natural-language paragraph, no comma tags, no (word:1.3)
      weighting syntax. With text + image, the text drives the scene and
      the image supplies element detail.
  [ ] MiniMax H3 — three fields (integrated_multimodal_description,
      overall_soundscape, non_diegetic_music) and a footer naming the mode
      it inferred. Spot-check the inference:
        - nothing attached        -> T2VA, no alignment line
        - one image               -> I2VA, alignment line at 0.00s
        - two images              -> FL2VA, two-picture alignment line
        - one image + "ends on"   -> L2VA, alignment line at the end
        - saying "use L2VA"       -> L2VA regardless of attachments
  [ ] For any keyframe mode, the reported reference-frame path(s) exist on
      disk (default: ~/PromptHub/output); with no attachment it says none
      was saved rather than staying silent.
  [ ] No network activity while any of the above run (spot-check with a
      network monitor if you want to be sure).
EOF

echo ""
echo "== Summary: $PASS passed, $FAIL failed (automated checks only) =="
[ "$FAIL" -eq 0 ]
