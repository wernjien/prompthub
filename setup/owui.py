"""Open WebUI API access for the setup scripts, using only the standard library."""

import base64
import contextlib
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

BASE_URL = os.environ.get("PROMPTHUB_OPENWEBUI_URL", "http://localhost:8080")
HOME_DIR = Path(os.environ.get("PROMPTHUB_HOME", str(Path.home() / "PromptHub")))
CREDS_FILE = HOME_DIR / ".admin_credentials.json"
# start.sh / start.ps1 run Open WebUI from HOME_DIR with DATA_DIR=HOME_DIR/data.
SECRET_KEY_FILE = HOME_DIR / ".webui_secret_key"
DATABASE_FILE = HOME_DIR / "data" / "webui.db"
DEFAULT_ADMIN_EMAIL = "admin@prompthub.local"


def api(method: str, path: str, token: str = None, payload: dict = None, timeout: int = 30):
    """Calls Open WebUI and returns (status, parsed body); exits if the server is unreachable."""
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(f"{BASE_URL}{path}", data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            return resp.status, (json.loads(body) if body else None)
    except urllib.error.HTTPError as exc:
        body = exc.read()
        try:
            return exc.code, json.loads(body)
        except json.JSONDecodeError:
            return exc.code, body.decode(errors="replace")
    except urllib.error.URLError as exc:
        print(f"  FAIL: could not reach Open WebUI at {BASE_URL}: {exc}")
        sys.exit(1)


def admin_token(allow_signup: bool = True) -> str:
    """Returns an admin token, preferring one signed locally so no password is ever needed."""
    token = local_admin_token()
    if token:
        return token

    saved = json.loads(CREDS_FILE.read_text()) if CREDS_FILE.exists() else None
    env_email = os.environ.get("PROMPTHUB_ADMIN_EMAIL")
    env_password = os.environ.get("PROMPTHUB_ADMIN_PASSWORD")
    from_env = {"email": env_email, "password": env_password} if env_email and env_password else None
    for creds in (saved, from_env):
        if not creds:
            continue
        status, body = api("POST", "/api/v1/auths/signin", payload=creds)
        if 200 <= status < 300:
            if creds is from_env:
                save_creds(env_email, env_password)
            return body["token"]

    if allow_signup:
        email = env_email or DEFAULT_ADMIN_EMAIL
        password = env_password or secrets.token_urlsafe(16)
        status, body = api(
            "POST", "/api/v1/auths/signup", payload={"email": email, "password": password, "name": "PromptHub Admin"}
        )
        if 200 <= status < 300:
            save_creds(email, password)
            print(f"Created Open WebUI admin account: {email}")
            print(f"Credentials saved to {CREDS_FILE}.")
            print(f"Use the same email/password to log in at {BASE_URL} in your browser.")
            return body["token"]
        print(f"  Signup failed (status {status}): {body}")

    print(
        f"  FAIL: could not get admin access to Open WebUI at {BASE_URL}. PromptHub signs in with "
        f"Open WebUI's own key ({SECRET_KEY_FILE}) and database ({DATABASE_FILE}). If this instance "
        "keeps them elsewhere, set PROMPTHUB_ADMIN_EMAIL and PROMPTHUB_ADMIN_PASSWORD to an admin "
        "account and re-run."
    )
    sys.exit(1)


def local_admin_token():
    """Signs a short-lived session token for the first admin with Open WebUI's own key; None if unavailable."""
    secret = os.environ.get("WEBUI_SECRET_KEY")
    if secret is None and SECRET_KEY_FILE.exists():
        secret = SECRET_KEY_FILE.read_text()
    if not secret or not DATABASE_FILE.exists():
        return None
    try:
        with contextlib.closing(sqlite3.connect(DATABASE_FILE.as_uri() + "?mode=ro", uri=True)) as db:
            row = db.execute("SELECT id FROM user WHERE role = 'admin' ORDER BY rowid LIMIT 1").fetchone()
    except sqlite3.Error:
        return None
    if not row:
        return None

    now = int(time.time())
    token = _jwt({"id": row[0], "iat": now, "exp": now + 600, "jti": str(uuid.uuid4())}, secret)
    status, _ = api("GET", "/api/v1/auths/", token=token)
    return token if 200 <= status < 300 else None


def ensure_login_saved(token: str) -> None:
    """Restores a lost login for the PromptHub-created admin by resetting its generated password."""
    if CREDS_FILE.exists():
        return
    status, me = api("GET", "/api/v1/auths/", token=token)
    # Only the generated account is reset: nobody can know its password once the file is gone.
    if not (200 <= status < 300) or not isinstance(me, dict) or me.get("email") != DEFAULT_ADMIN_EMAIL:
        return

    env_password = os.environ.get("PROMPTHUB_ADMIN_PASSWORD")
    if env_password:
        status, _ = api("POST", "/api/v1/auths/signin", payload={"email": DEFAULT_ADMIN_EMAIL, "password": env_password})
        if 200 <= status < 300:
            save_creds(DEFAULT_ADMIN_EMAIL, env_password)
            return

    password = secrets.token_urlsafe(16)
    status, body = api("POST", f"/api/v1/users/{me['id']}/update", token=token, payload={"password": password})
    if 200 <= status < 300:
        save_creds(DEFAULT_ADMIN_EMAIL, password)
        print(f"The login for {DEFAULT_ADMIN_EMAIL} wasn't saved anywhere, so its password was reset.")
        print(f"New login saved to {CREDS_FILE}; use it at {BASE_URL} in your browser.")
    else:
        print(f"  WARNING: couldn't restore the {DEFAULT_ADMIN_EMAIL} login: {status} {body}")


def save_creds(email: str, password: str) -> None:
    """Writes the credentials file owner-only from the moment it exists."""
    HOME_DIR.mkdir(parents=True, exist_ok=True)
    fd = os.open(CREDS_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(json.dumps({"email": email, "password": password}))
    os.chmod(CREDS_FILE, 0o600)


def _jwt(payload: dict, secret: str) -> str:
    """Encodes an HS256 JWT, the format Open WebUI issues at sign-in."""
    def b64(data: bytes) -> bytes:
        return base64.urlsafe_b64encode(data).rstrip(b"=")

    signing_input = b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode()) + b"." + b64(json.dumps(payload).encode())
    signature = hmac.new(secret.encode(), signing_input, hashlib.sha256).digest()
    return (signing_input + b"." + b64(signature)).decode()
