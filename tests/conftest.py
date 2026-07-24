"""Shared pytest fixtures and path setup for the test suite.

The package lives under ``src/``; add it to ``sys.path`` so the tests can
``import finbot`` without an editable install.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the repository root."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def intents_path(repo_root: Path) -> Path:
    """Path to the real ``intents.json`` shipped with the project."""
    return repo_root / "intents.json"


@pytest.fixture(scope="session")
def words_path(repo_root: Path) -> Path:
    """Path to the pickled vocabulary."""
    return repo_root / "words.pkl"


@pytest.fixture(scope="session")
def classes_path(repo_root: Path) -> Path:
    """Path to the pickled class labels."""
    return repo_root / "classes.pkl"


@pytest.fixture
def sample_vocabulary() -> list[str]:
    """A small, fixed vocabulary for pure bag-of-words tests."""
    return ["account", "balance", "card", "hello", "payment"]
