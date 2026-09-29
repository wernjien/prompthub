#!/usr/bin/env python3
"""Checks the picker holds exactly PromptHub's Functions; prints PASS/FAIL and exits non-zero on failure."""

import os
import sys

from owui import admin_token, api

START_SCRIPT = "setup\\start.ps1" if os.name == "nt" else "setup/start.sh"

EXPECTED_FUNCTIONS = ["krea2", "minimax_h3"]

EXPECTED_ABSENT = [
    os.environ.get("PROMPTHUB_TEXT_MODEL", "dolphin3:8b"),
    os.environ.get("PROMPTHUB_VISION_MODEL", "llava:13b"),
]


def _as_list(body) -> list:
    """Accepts both a bare list and an {"items": [...]} wrapper, which Open WebUI endpoints mix."""
    if isinstance(body, list):
        return body
    if isinstance(body, dict):
        return body.get("items", [])
    return []


def main() -> None:
    token = admin_token(allow_signup=False)

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
