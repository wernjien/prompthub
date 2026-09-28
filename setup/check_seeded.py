#!/usr/bin/env python3
"""Checks the picker holds exactly PromptHub's Functions; prints PASS/FAIL and exits non-zero on failure."""

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = os.environ.get("PROMPTHUB_OPENWEBUI_URL", "http://localhost:8080")
HOME_DIR = Path(os.environ.get("PROMPTHUB_HOME", str(Path.home() / "PromptHub")))
CREDS_FILE = HOME_DIR / ".admin_credentials.json"
START_SCRIPT = "setup\\start.ps1" if os.name == "nt" else "setup/start.sh"

EXPECTED_FUNCTIONS = ["krea2", "minimax_h3"]

EXPECTED_ABSENT = [
    os.environ.get("PROMPTHUB_TEXT_MODEL", "dolphin3:8b"),
    os.environ.get("PROMPTHUB_VISION_MODEL", "llava:13b"),
]


def api(method: str, path: str, token: str = None, payload: dict = None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(f"{BASE_URL}{path}", data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read()
            return resp.status, (json.loads(body) if body else None)
    except urllib.error.HTTPError as exc:
        return exc.code, None
    except urllib.error.URLError as exc:
        print(f"  FAIL: could not reach Open WebUI at {BASE_URL}: {exc}")
        sys.exit(1)


def _as_list(body) -> list:
    """Accepts both a bare list and an {"items": [...]} wrapper, which Open WebUI endpoints mix."""
    if isinstance(body, list):
        return body
    if isinstance(body, dict):
        return body.get("items", [])
    return []


def main() -> None:
    if not CREDS_FILE.exists():
        print(f"  FAIL: no saved admin credentials at {CREDS_FILE} — run {START_SCRIPT} first")
        sys.exit(1)

    creds = json.loads(CREDS_FILE.read_text())
    status, body = api("POST", "/api/v1/auths/signin", payload=creds)
    if not (200 <= status < 300):
        print(f"  FAIL: could not sign in with saved admin credentials (status {status})")
        sys.exit(1)
    token = body["token"]

    ok = True

    status, functions = api("GET", "/api/v1/functions/list", token=token)
    function_ids = {f["id"] for f in _as_list(functions)} if 200 <= status < 300 else set()
    for fid in EXPECTED_FUNCTIONS:
        if fid in function_ids:
            print(f"  PASS: Function present: {fid}")
        else:
            print(f"  FAIL: Function missing: {fid} — run {START_SCRIPT} to reseed")
            ok = False

    status, models = api("GET", "/api/models", token=token)
    entries = models.get("data", []) if isinstance(models, dict) else (models or [])
    visible_ids = {m.get("id") for m in entries} if 200 <= status < 300 else set()

    for mid in EXPECTED_ABSENT:
        if mid in visible_ids:
            print(f"  FAIL: raw model still in the picker: {mid} — run {START_SCRIPT} to reseed")
            ok = False
        else:
            print(f"  PASS: raw model hidden from picker: {mid}")

    unexpected = sorted(visible_ids - set(EXPECTED_FUNCTIONS))
    if unexpected:
        print(f"  FAIL: unexpected extra models in the picker: {', '.join(unexpected)}")
        ok = False
    else:
        print(f"  PASS: picker holds exactly the {len(EXPECTED_FUNCTIONS)} PromptHub entries")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
