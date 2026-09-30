#!/usr/bin/env python3
"""Copies shared/helpers.py into each models/*.py Function; --check exits 1 on drift instead."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE = REPO_ROOT / "shared" / "helpers.py"
TARGETS = [REPO_ROOT / "models" / "krea2.py", REPO_ROOT / "models" / "minimax_h3.py"]
MARKER = "# --- shared helpers"


def shared_block() -> str:
    """Returns the marker line and everything after it from shared/helpers.py."""
    text = SOURCE.read_text(encoding="utf-8")
    head, marker, body = text.partition(MARKER)
    if not marker:
        raise SystemExit(f"{SOURCE} has no '{MARKER}' marker line")
    return marker + body


def synced(target: Path) -> str:
    """Returns the target's content with its helper block replaced by the shared one."""
    head, marker, _ = target.read_text(encoding="utf-8").partition(MARKER)
    if not marker:
        raise SystemExit(f"{target} has no '{MARKER}' marker line")
    return head + shared_block()


def drifted() -> list:
    """Returns the target files whose helper block differs from shared/helpers.py."""
    return [t for t in TARGETS if t.read_text(encoding="utf-8") != synced(t)]


def main() -> None:
    if "--check" in sys.argv[1:]:
        stale = drifted()
        for t in stale:
            print(f"out of sync with shared/helpers.py: {t.relative_to(REPO_ROOT)}")
        if stale:
            print("run: python3 setup/sync_shared.py")
            sys.exit(1)
        return
    for t in TARGETS:
        new = synced(t)
        if t.read_text(encoding="utf-8") != new:
            t.write_text(new, encoding="utf-8", newline="\n")
            print(f"synced {t.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
