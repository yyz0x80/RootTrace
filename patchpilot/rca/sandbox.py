"""Ephemeral verification sandbox for RootTrace runtime tests.

The ``RuntimeVerificationSandbox`` is the only RCA component allowed to execute
test commands, and it executes them only inside a disposable copy of the target
repository derived from an explicit base commit (or the clone's HEAD). It
enforces:

- parsed command allowlist (``python -m pytest ...`` plus bounded pytest flags
  and sandbox-relative test targets);
- ``subprocess.run(..., shell=False)`` with no shell, pipes, or redirection;
- finite timeouts and bounded captured output;
- writes confined to the sandbox copy; the original repository is never
  modified, which is proven with a before/after repository fingerprint.

The disposable copy is destroyed by ``close()`` (or the context manager), and
the original target fingerprint is re-checked at teardown.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Self

from patchpilot.rca.context import RepositoryFingerprint
from patchpilot.rca.context_builder import (
    assert_fingerprint_unchanged,
    capture_repository_fingerprint,
)

DEFAULT_TIMEOUT_SECONDS = 120
MAX_TIMEOUT_SECONDS = 600
MAX_COMMAND_OUTPUT_CHARS = 200_000
MAX_COMMAND_TOKENS = 100
MAX_COMMAND_CHARS = 10_000

_VALID_REVISION = re.compile(r"^(?:[0-9a-fA-F]{4,64}|HEAD)$")
_SAFE_TARGET_CHARS = re.compile(r"^[A-Za-z0-9._/:+-]+$")

_ALLOWED_PYTEST_FLAGS = frozenset(
    {
        "-q",
        "--quiet",
        "-x",
        "--exitfirst",
        "--no-header",
        "--disable-warnings",
        "-v",
        "--verbose",
        "--tb=short",
        "--tb=long",
        "--tb=line",
    }
)
_PAIR_FLAGS = frozenset({"-p"})


def _validate_test_command(tokens: list[str], root: Path) -> list[str]:
    """Validate a pytest argv against the parsed allowlist."""
    if not tokens:
        raise ValueError("test command must not be empty")
    if len(tokens) > MAX_COMMAND_TOKENS:
        raise ValueError("test command has too many arguments")
    if sum(len(token) for token in tokens) > MAX_COMMAND_CHARS:
        raise ValueError("test command is too long")
    if tokens[0] != "python":
        raise ValueError("only 'python -m pytest ...' commands are allowed")
    if tokens[1:3] != ["-m", "pytest"]:
        raise ValueError("only 'python -m pytest ...' commands are allowed")

    index = 3
    while index < len(tokens):
        token = tokens[index]
        if token in _PAIR_FLAGS:
            if index + 1 >= len(tokens) or tokens[index + 1] != "no:cacheprovider":
                raise ValueError("-p only supports 'no:cacheprovider'")
            index += 2
            continue
        if token in _ALLOWED_PYTEST_FLAGS:
            index += 1
            continue
        if token.startswith("-"):
            raise ValueError(f"disallowed pytest option: {token}")
        _validate_target(token, root)
        index += 1
    return tokens


def _validate_target(token: str, root: Path) -> None:
    """Validate one sandbox-relative pytest target (file or directory)."""
    if not _SAFE_TARGET_CHARS.fullmatch(token):
        raise ValueError(f"test target contains unsafe characters: {token}")
    base = token.split("::", maxsplit=1)[0]
    path = PurePosixPath(base)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("test target must be a sandbox-relative path")
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("test target escapes the sandbox") from exc
    if not resolved.exists():
        raise ValueError(f"test target does not exist in sandbox: {base}")
    if resolved.is_file() and not base.endswith(".py"):
        raise ValueError("test file target must end with .py")


def _cap_output(text: str, limit: int = MAX_COMMAND_OUTPUT_CHARS) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit] + "\n... (output truncated)", True


@dataclass
class SandboxCommandResult:
    """Bounded result of one allowlisted test command in the sandbox."""

    command: str
    argv: list[str]
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool
    truncated: bool


class RuntimeVerificationSandbox:
    """Disposable repository copy for bounded, allowlisted runtime tests."""

    def __init__(
        self,
        repo: str | Path,
        base_commit: str | None = None,
        work_dir: str | Path | None = None,
    ) -> None:
        self.repo = Path(repo).resolve()
        self._require_git_repo(self.repo)
        if base_commit is not None and not _VALID_REVISION.fullmatch(base_commit):
            raise ValueError("base_commit must be a hex SHA or HEAD")
        self.base_commit = base_commit
        self.before: RepositoryFingerprint = capture_repository_fingerprint(self.repo)
        parent = (
            Path(work_dir).resolve()
            if work_dir is not None
            else Path(tempfile.gettempdir())
        )
        self._root = Path(tempfile.mkdtemp(prefix="roottrace-sandbox-", dir=parent))
        self.work_root = (self._root / "work").resolve()
        self.head_sha = ""
        self.closed = False
        try:
            self._clone_and_checkout()
        except Exception:
            shutil.rmtree(self._root, ignore_errors=True)
            self.closed = True
            raise

    @staticmethod
    def _require_git_repo(repo: Path) -> None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=repo,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except FileNotFoundError as exc:
            raise ValueError(f"target is not a valid git repository: {repo}") from exc
        if result.returncode != 0 or result.stdout.strip() != "true":
            raise ValueError(f"target is not a valid git repository: {repo}")

    def _clone_and_checkout(self) -> None:
        subprocess.run(
            ["git", "clone", "--quiet", "--no-hardlinks", str(self.repo), str(self.work_root)],
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )
        if self.base_commit is not None:
            subprocess.run(
                ["git", "checkout", "--quiet", self.base_commit],
                cwd=self.work_root,
                capture_output=True,
                text=True,
                timeout=60,
                check=True,
            )
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.work_root,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        self.head_sha = head.stdout.strip()

    def run(
        self,
        argv: list[str],
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> SandboxCommandResult:
        """Run one allowlisted ``python -m pytest`` command in the sandbox."""
        if self.closed:
            raise RuntimeError("verification sandbox is closed")
        if not isinstance(timeout_seconds, int) or not 1 <= timeout_seconds <= MAX_TIMEOUT_SECONDS:
            raise ValueError(f"timeout_seconds must be between 1 and {MAX_TIMEOUT_SECONDS}")
        tokens = [str(token) for token in argv]
        validated = _validate_test_command(tokens, self.work_root)
        executable = [sys.executable, *validated[1:]]
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        started = time.monotonic()
        try:
            result = subprocess.run(
                executable,
                cwd=self.work_root,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            return SandboxCommandResult(
                command=" ".join(validated),
                argv=list(validated),
                exit_code=124,
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
                duration_seconds=time.monotonic() - started,
                timed_out=True,
                truncated=False,
            )
        duration = time.monotonic() - started
        stdout, stdout_truncated = _cap_output(result.stdout)
        stderr, stderr_truncated = _cap_output(result.stderr)
        return SandboxCommandResult(
            command=" ".join(validated),
            argv=list(validated),
            exit_code=result.returncode,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=duration,
            timed_out=False,
            truncated=stdout_truncated or stderr_truncated,
        )

    def close(self) -> None:
        """Destroy the disposable copy and prove the target is unchanged."""
        if self.closed:
            return
        try:
            shutil.rmtree(self._root)
        finally:
            self.closed = True
        after = capture_repository_fingerprint(self.repo)
        assert_fingerprint_unchanged(self.before, after)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
