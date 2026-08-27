"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest


@dataclass(frozen=True)
class GitRepoFixture:
    """A two-commit git repository used by RCA tool/sandbox tests."""

    repo: Path
    base_sha: str
    head_sha: str


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )


def _write(repo: Path, relative_path: str, content: str) -> None:
    path = repo / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


_INITIAL_CALC = '''\
def add(a, b):
    return a + b


def multiply(a, b):
    return a + b  # bug: should multiply
'''

_FIXED_CALC = '''\
def add(a, b):
    return a + b


def multiply(a, b):
    return a * b
'''

_INITIAL_TESTS = '''\
def test_add():
    from pkg.calc import add

    assert add(1, 2) == 3


def test_multiply():
    from pkg.calc import multiply

    assert multiply(3, 4) == 12
'''


@pytest.fixture
def git_repo(tmp_path: Path) -> GitRepoFixture:
    """Create a target repo with a base bug commit and a fixed HEAD commit."""
    repo = tmp_path / "target"
    repo.mkdir()
    _write(repo, "pkg/__init__.py", "")
    _write(repo, "pkg/calc.py", _INITIAL_CALC)
    _write(repo, "tests/__init__.py", "")
    _write(repo, "tests/test_calc.py", _INITIAL_TESTS)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "initial")
    base_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()

    _write(repo, "pkg/calc.py", _FIXED_CALC)
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "fix multiply")
    head_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()
    return GitRepoFixture(repo=repo, base_sha=base_sha, head_sha=head_sha)
