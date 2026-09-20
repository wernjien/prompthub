#!/usr/bin/env python3
"""Seeds a running Open WebUI instance via its REST API: creates the admin
account on first run, deploys the two Pipe Functions from this repo,
removes entries from earlier layouts, disables the Ollama connection so raw
models stay out of the picker, and sorts the picker alphabetically.
No manual UI steps needed.

The picker ends up holding exactly PromptHub's two entries, all of the same
type, because every one of them is a Pipe Function that calls Ollama
directly over HTTP rather than resolving through an Open WebUI model
connection. That's what makes disabling the connection safe — see
disable_ollama_connection() for the verification behind that claim.

Safe to re-run — everything is created-or-updated, so re-running after
editing a function file syncs the change into Open WebUI.

Uses only the Python standard library so it has no dependency on anything
pip-installed into Open WebUI's own venv.
"""

import json
import os
import secrets
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_URL = os.environ.get("PROMPTHUB_OPENWEBUI_URL", "http://localhost:8080")
HOME_DIR = Path(os.environ.get("PROMPTHUB_HOME", str(Path.home() / "PromptHub")))
CREDS_FILE = HOME_DIR / ".admin_credentials.json"
TEXT_MODEL = os.environ.get("PROMPTHUB_TEXT_MODEL", "dolphin3:8b")
VISION_MODEL = os.environ.get("PROMPTHUB_VISION_MODEL", "llava:13b")

# One Function per target model. Each accepts a typed message, an
# attachment (image or video), or both together.
#
# MiniMax H3 is a single entry even though H3 has four modes
# (T2VA/I2VA/FL2VA/L2VA): which mode applies is a consequence of what you
# supplied — no attachment, one reference image, two, or an ending frame —
# not a preference, so the Function infers it and says which it used. Making
# the user choose meant understanding H3's reference-frame semantics before
# writing a single prompt.
#
# They're Pipe Functions rather than Open WebUI "Model presets" because a
# preset only works while its base model is exposed in the picker, which
# forced dolphin3:8b/llava:13b to sit there next to PromptHub's own entries.
# Functions call Ollama directly over HTTP, so nothing depends on the
# connection being listed and it can be switched off entirely (see
# disable_ollama_connection below).
FUNCTIONS = [
    ("krea2", "Krea2", REPO_ROOT / "models" / "krea2.py"),
    ("minimax_h3", "MiniMax H3", REPO_ROOT / "models" / "minimax_h3.py"),
]

# Model presets from before v0.4, deleted on sight so they don't linger in
# the picker as broken entries once the connection is disabled.
OBSOLETE_PRESETS = [
    "krea2-prompt-writer",
    "minimax-t2va-prompt-writer",
    "minimax-i2va-prompt-writer",
    "minimax-fl2va-prompt-writer",
    "minimax-l2va-prompt-writer",
]

# Functions from earlier layouts: v0.4 split every mode by input type, and
# v0.5 still had one entry per MiniMax mode.
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


def api(method: str, path: str, token: str = None, payload: dict = None):
    url = f"{BASE_URL}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read()
            return resp.status, (json.loads(body) if body else None)
    except urllib.error.HTTPError as exc:
        body = exc.read()
        try:
            return exc.code, json.loads(body)
        except json.JSONDecodeError:
            return exc.code, body.decode(errors="replace")
    except urllib.error.URLError as exc:
        print(f"Could not reach Open WebUI at {BASE_URL}: {exc}")
        sys.exit(1)


def get_token() -> str:
    HOME_DIR.mkdir(parents=True, exist_ok=True)

    if CREDS_FILE.exists():
        creds = json.loads(CREDS_FILE.read_text())
        status, body = api("POST", "/api/v1/auths/signin", payload=creds)
        if 200 <= status < 300:
            return body["token"]
        print(f"Saved admin credentials in {CREDS_FILE} no longer work (status {status}: {body}).")
        print("Delete that file to force re-creating an account, or fix it by hand, then re-run.")
        sys.exit(1)

    env_email = os.environ.get("PROMPTHUB_ADMIN_EMAIL")
    env_password = os.environ.get("PROMPTHUB_ADMIN_PASSWORD")
    if env_email and env_password:
        status, body = api("POST", "/api/v1/auths/signin", payload={"email": env_email, "password": env_password})
        if 200 <= status < 300:
            _save_creds(env_email, env_password)
            return body["token"]

    email = env_email or "admin@prompthub.local"
    password = env_password or secrets.token_urlsafe(16)
    status, body = api(
        "POST",
        "/api/v1/auths/signup",
        payload={"email": email, "password": password, "name": "PromptHub Admin"},
    )
    if 200 <= status < 300:
        _save_creds(email, password)
        print(f"Created Open WebUI admin account: {email}")
        print(f"Credentials saved to {CREDS_FILE} (used to seed on future runs too).")
        print("Use the same email/password to log in at http://localhost:8080 in your browser.")
        return body["token"]

    print(f"Signup failed (status {status}): {body}")
    print(
        "This usually means an admin account already exists from a previous run but "
        f"{CREDS_FILE} is missing or stale. Set PROMPTHUB_ADMIN_EMAIL and "
        "PROMPTHUB_ADMIN_PASSWORD to the existing account and re-run."
    )
    sys.exit(1)


def _save_creds(email: str, password: str) -> None:
    CREDS_FILE.write_text(json.dumps({"email": email, "password": password}))
    os.chmod(CREDS_FILE, 0o600)


def seed_functions(token: str) -> bool:
    created_any = False
    for func_id, name, path in FUNCTIONS:
        payload = {
            "id": func_id,
            "name": name,
            "type": "pipe",
            "content": path.read_text(),
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
                continue

        # create/update silently ignore `is_active` in the payload — it has
        # to be flipped explicitly via /toggle, or the function is created
        # disabled and never shows up as a selectable model.
        status, current = api("GET", f"/api/v1/functions/id/{func_id}", token=token)
        if 200 <= status < 300 and current and not current.get("is_active"):
            status, body = api("POST", f"/api/v1/functions/id/{func_id}/toggle", token=token)
            if 200 <= status < 300:
                print(f"  activated function: {func_id}")
            else:
                print(f"  FAILED to activate function {func_id}: {status} {body}")
    return created_any


def delete_obsolete(token: str) -> None:
    """Removes entries from earlier layouts so the picker doesn't accumulate
    stale duplicates across upgrades: pre-v0.4 Model presets (which break
    outright once the Ollama connection is disabled, since a preset can't
    resolve a base model that isn't listed) and v0.4's split-by-input-type
    Functions (superseded by the combined ones)."""
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
    """Turns the Ollama connection off inside Open WebUI, which takes
    TEXT_MODEL/VISION_MODEL out of the model picker.

    This is safe only because every PromptHub entry is a Pipe Function that
    POSTs to Ollama directly at OLLAMA_BASE_URL — none of them resolve
    through Open WebUI's connection layer, so switching it off removes the
    raw models from the picker without touching what they can call.
    Verified both ways: with the connection off a Function still reaches
    Ollama fine, while an old-style Model preset returns "Model not found"."""
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
    """Turns off Open WebUI's built-in Arena entry — the Evaluations feature
    that pits two models against each other for blind rating. Newer releases
    ship it enabled, where it shows up in the picker as a third entry
    alongside PromptHub's two. Older releases have no such endpoint, so a
    missing one is reported and skipped rather than treated as a failure."""
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
    """Sets Open WebUI's admin-configurable MODEL_ORDER_LIST (the same
    field behind Admin Settings -> Models' manual drag-to-reorder) to sort
    every currently-visible model alphabetically by display name. Runs
    last so newly (re)named entries are included, and re-derives the order
    from whatever's visible each time —
    so it stays correct if you add/remove/rename anything, including
    models this repo doesn't know about."""
    status, models = api("GET", "/api/models", token=token)
    entries = models.get("data", models if isinstance(models, list) else []) if models else []
    if not entries:
        print(f"  FAILED to read model list: {status}")
        return

    order_ids = [m["id"] for m in sorted(entries, key=lambda m: (m.get("name") or "").casefold())]

    status, config = api("GET", "/api/v1/configs/models", token=token)
    if not (200 <= status < 300) or config is None:
        config = {}
    config["MODEL_ORDER_LIST"] = order_ids

    status, body = api("POST", "/api/v1/configs/models", token=token, payload=config)
    if 200 <= status < 300:
        print(f"  sorted {len(order_ids)} models alphabetically")
    else:
        print(f"  FAILED to set model order: {status} {body}")


def main() -> None:
    print("== Seeding Open WebUI ==")
    token = get_token()

    print("-- Functions --")
    created_new_function = seed_functions(token)

    print("-- Cleaning up older layouts --")
    delete_obsolete(token)

    print("-- Ollama connection --")
    disable_ollama_connection(token)

    print("-- Arena model --")
    disable_arena_model(token)

    print("-- Sorting model picker --")
    sort_model_picker_alphabetically(token)

    if created_new_function:
        # A brand-new function's frontmatter `requirements:` only get
        # pip-installed on server startup, not at creation time.
        print("NEEDS_RESTART")
    print("Done.")


if __name__ == "__main__":
    main()
