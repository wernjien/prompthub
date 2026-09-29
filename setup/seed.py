#!/usr/bin/env python3
"""Seeds a running Open WebUI through its REST API; safe to re-run after editing a Function."""

import os
import sys
from pathlib import Path

from owui import admin_token, api, ensure_login_saved

REPO_ROOT = Path(__file__).resolve().parent.parent
TEXT_MODEL = os.environ.get("PROMPTHUB_TEXT_MODEL", "dolphin3:8b")
VISION_MODEL = os.environ.get("PROMPTHUB_VISION_MODEL", "llava:13b")

# Pipe Functions, not Model presets: a preset breaks once the Ollama connection is disabled.
FUNCTIONS = [
    ("krea2", "Krea2", REPO_ROOT / "models" / "krea2.py"),
    ("minimax_h3", "MiniMax H3", REPO_ROOT / "models" / "minimax_h3.py"),
]

SHARED_HELPERS_MARKER = "# --- shared helpers"

# Pre-v0.4 Model presets, deleted so they don't linger as broken picker entries.
OBSOLETE_PRESETS = [
    "krea2-prompt-writer",
    "minimax-t2va-prompt-writer",
    "minimax-i2va-prompt-writer",
    "minimax-fl2va-prompt-writer",
    "minimax-l2va-prompt-writer",
]

# Functions from the v0.4 (split by input type) and v0.5 (one per MiniMax mode) layouts.
OBSOLETE_FUNCTIONS = [
    "krea2_from_text",
    "krea2_from_image",
    "minimax_t2va_from_text",
    "minimax_t2va_from_video",
    "minimax_i2va_from_text",
    "minimax_i2va_from_video",
    "minimax_fl2va_from_text",
    "minimax_fl2va_from_video",
    "minimax_l2va_from_text",
    "minimax_l2va_from_video",
    "minimax_t2va",
    "minimax_i2va",
    "minimax_fl2va",
    "minimax_l2va",
]


def check_shared_helpers() -> None:
    """Warns when the helper block duplicated across the Function files has drifted."""
    blocks = {path.name: path.read_text(encoding="utf-8").partition(SHARED_HELPERS_MARKER)[2] for _, _, path in FUNCTIONS}
    if len(set(blocks.values())) > 1:
        print(f"  WARNING: the shared helper block differs between {', '.join(blocks)}; keep it identical")


def seed_functions(token: str):
    """Creates or updates each Function; returns (created_any, failed_ids)."""
    created_any, failed = False, []
    for func_id, name, path in FUNCTIONS:
        payload = {
            "id": func_id,
            "name": name,
            "type": "pipe",
            "content": path.read_text(encoding="utf-8"),
            "meta": {"description": name, "manifest": {}},
            "is_active": True,
            "is_global": False,
        }
        status, body = api("POST", "/api/v1/functions/create", token=token, payload=payload)
        if 200 <= status < 300:
            print(f"  created function: {func_id}")
            created_any = True
        else:
            status, body = api("POST", f"/api/v1/functions/id/{func_id}/update", token=token, payload=payload)
            if 200 <= status < 300:
                print(f"  updated function: {func_id}")
            else:
                print(f"  FAILED to create/update function {func_id}: {status} {body}")
                failed.append(func_id)
                continue

        # create/update ignore `is_active`, so a new Function stays disabled until toggled.
        status, current = api("GET", f"/api/v1/functions/id/{func_id}", token=token)
        if 200 <= status < 300 and current and not current.get("is_active"):
            status, body = api("POST", f"/api/v1/functions/id/{func_id}/toggle", token=token)
            if 200 <= status < 300:
                print(f"  activated function: {func_id}")
            else:
                print(f"  FAILED to activate function {func_id}: {status} {body}")
                failed.append(func_id)
    return created_any, failed


def delete_obsolete(token: str) -> None:
    """Removes presets and Functions left over from earlier layouts."""
    for preset_id in OBSOLETE_PRESETS:
        status, _ = api("GET", f"/api/v1/models/model?id={preset_id}", token=token)
        if not (200 <= status < 300):
            continue
        status, body = api(
            "POST", f"/api/v1/models/model/delete?id={preset_id}", token=token, payload={"id": preset_id}
        )
        if 200 <= status < 300:
            print(f"  removed obsolete preset: {preset_id}")
        else:
            print(f"  FAILED to remove obsolete preset {preset_id}: {status} {body}")

    for func_id in OBSOLETE_FUNCTIONS:
        status, _ = api("GET", f"/api/v1/functions/id/{func_id}", token=token)
        if not (200 <= status < 300):
            continue
        status, body = api("DELETE", f"/api/v1/functions/id/{func_id}/delete", token=token)
        if 200 <= status < 300:
            print(f"  removed obsolete function: {func_id}")
        else:
            print(f"  FAILED to remove obsolete function {func_id}: {status} {body}")


def disable_ollama_connection(token: str) -> None:
    """Hides the raw Ollama models from the picker; the Functions call Ollama directly, so they keep working."""
    status, config = api("GET", "/ollama/config", token=token)
    if not (200 <= status < 300) or not config:
        print(f"  FAILED to read Ollama config: {status} {config}")
        return

    if config.get("ENABLE_OLLAMA_API") is False:
        print("  Ollama connection already disabled")
        return

    payload = {
        "ENABLE_OLLAMA_API": False,
        "OLLAMA_BASE_URLS": config.get("OLLAMA_BASE_URLS", []),
        "OLLAMA_API_CONFIGS": config.get("OLLAMA_API_CONFIGS", {}),
    }
    status, body = api("POST", "/ollama/config/update", token=token, payload=payload)
    if 200 <= status < 300:
        print(f"  disabled Ollama connection ({TEXT_MODEL}/{VISION_MODEL} out of the picker)")
    else:
        print(f"  FAILED to disable Ollama connection: {status} {body}")


def disable_arena_model(token: str) -> None:
    """Turns off Open WebUI's built-in Arena picker entry, where the release has one."""
    status, config = api("GET", "/api/v1/evaluations/config", token=token)
    if not (200 <= status < 300) or not isinstance(config, dict):
        print(f"  no evaluations config on this Open WebUI (status {status}) - nothing to disable")
        return

    if config.get("ENABLE_EVALUATION_ARENA_MODELS") is False:
        print("  Arena model already disabled")
        return

    config["ENABLE_EVALUATION_ARENA_MODELS"] = False
    status, body = api("POST", "/api/v1/evaluations/config", token=token, payload=config)
    if 200 <= status < 300:
        print("  disabled Arena model (Open WebUI's built-in evaluation entry)")
    else:
        print(f"  FAILED to disable Arena model: {status} {body}")


def sort_model_picker_alphabetically(token: str) -> None:
    """Sets MODEL_ORDER_LIST to every visible model sorted by display name."""
    status, models = api("GET", "/api/models", token=token)
    entries = models.get("data", []) if isinstance(models, dict) else (models if isinstance(models, list) else [])
    if not (200 <= status < 300) or not entries:
        print(f"  FAILED to read model list: {status}")
        return

    order_ids = [m["id"] for m in sorted(entries, key=lambda m: (m.get("name") or "").casefold())]

    status, config = api("GET", "/api/v1/configs/models", token=token)
    if not (200 <= status < 300) or not isinstance(config, dict):
        config = {}
    config["MODEL_ORDER_LIST"] = order_ids

    status, body = api("POST", "/api/v1/configs/models", token=token, payload=config)
    if 200 <= status < 300:
        print(f"  sorted {len(order_ids)} models alphabetically")
    else:
        print(f"  FAILED to set model order: {status} {body}")


def main() -> None:
    print("== Seeding Open WebUI ==")
    check_shared_helpers()
    token = admin_token()

    print("-- Functions --")
    created_new_function, failed = seed_functions(token)

    print("-- Cleaning up older layouts --")
    delete_obsolete(token)

    print("-- Ollama connection --")
    disable_ollama_connection(token)

    print("-- Arena model --")
    disable_arena_model(token)

    print("-- Sorting model picker --")
    sort_model_picker_alphabetically(token)

    ensure_login_saved(token)

    if created_new_function:
        # A new Function's `requirements:` are only pip-installed at server startup.
        print("NEEDS_RESTART")
    if failed:
        print(f"Seeding FAILED for: {', '.join(failed)}")
        sys.exit(1)
    print("Done.")


if __name__ == "__main__":
    main()
