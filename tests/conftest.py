import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "setup"))
sys.path.insert(0, str(ROOT / "evals"))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def helpers():
    return _load("prompthub_helpers", ROOT / "shared" / "helpers.py")


@pytest.fixture(scope="session")
def minimax():
    return _load("prompthub_minimax", ROOT / "models" / "minimax_h3.py")


@pytest.fixture(scope="session")
def krea2():
    return _load("prompthub_krea2", ROOT / "models" / "krea2.py")
