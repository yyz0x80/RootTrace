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
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Self
from xml.etree import ElementTree

from roottrace.runtime.workspace import (
    RepositoryFingerprint,
    assert_fingerprint_unchanged,
    capture_repository_fingerprint,
)

DEFAULT_TIMEOUT_SECONDS = 120
MAX_TIMEOUT_SECONDS = 600
MAX_COMMAND_OUTPUT_CHARS = 200_000
MAX_JUNIT_XML_BYTES = 1_000_000
MAX_COMMAND_TOKENS = 100
MAX_COMMAND_CHARS = 10_000

_JUNIT_XML_PATH = PurePosixPath(".roottrace") / "pytest-results.xml"

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


class PytestExecutionClassification(str, Enum):
    """Machine-derived classification of one pytest invocation."""

    PASSED = "passed"
    ASSERTION_FAILED = "assertion_failed"
    EXECUTION_ERROR = "execution_error"
    NO_TESTS = "no_tests"
    INVALID_RESULT = "invalid_result"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class _JUnitSummary:
    """Aggregate counters parsed from a bounded JUnit XML document."""

    tests: int
    failures: int
    errors: int
    skipped: int
    testcase_count: int
    failure_nodes: int
    error_nodes: int
    skipped_nodes: int


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


def _text_output(value: str | bytes | None) -> str:
    """Normalize subprocess output across text and timeout result variants."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _xml_tag(element: ElementTree.Element) -> str:
    """Return an XML element's local tag name without namespace details."""
    return element.tag.rsplit("}", maxsplit=1)[-1]


def _parse_counter(element: ElementTree.Element, name: str) -> int:
    """Parse one required non-negative JUnit counter."""
    value = element.attrib.get(name)
    if value is None:
        raise ValueError(f"missing JUnit counter: {name}")
    try:
        counter = int(value)
    except ValueError as exc:
        raise ValueError(f"invalid JUnit counter: {name}") from exc
    if counter < 0:
        raise ValueError(f"negative JUnit counter: {name}")
    return counter


def _parse_junit_xml(path: Path) -> tuple[_JUnitSummary | None, str]:
    """Parse bounded JUnit XML and reject structurally inconsistent results."""
    try:
        if not path.is_file():
            return None, "pytest JUnit result is missing"
        if path.stat().st_size > MAX_JUNIT_XML_BYTES:
            return None, "pytest JUnit result is too large"
        payload = path.read_bytes()
        if len(payload) > MAX_JUNIT_XML_BYTES:
            return None, "pytest JUnit result is too large"
        root = ElementTree.fromstring(payload)
    except (OSError, ElementTree.ParseError, ValueError) as exc:
        return None, f"pytest JUnit result is malformed: {type(exc).__name__}"

    root_name = _xml_tag(root)
    if root_name == "testsuite":
        suites = [root]
    elif root_name == "testsuites":
        suites = [element for element in root.iter() if _xml_tag(element) == "testsuite"]
    else:
        return None, "pytest JUnit result has an invalid root element"
    if not suites:
        return None, "pytest JUnit result contains no test suites"

    counters = {name: 0 for name in ("tests", "failures", "errors", "skipped")}
    for suite in suites:
        for name in counters:
            try:
                counters[name] += _parse_counter(suite, name)
            except ValueError as exc:
                return None, str(exc)

    testcases = [element for element in root.iter() if _xml_tag(element) == "testcase"]
    failure_nodes = 0
    error_nodes = 0
    skipped_nodes = 0
    for testcase in testcases:
        statuses = [_xml_tag(child) for child in testcase]
        failure_nodes += statuses.count("failure")
        error_nodes += statuses.count("error")
        skipped_nodes += statuses.count("skipped")
        if sum(
            statuses.count(status) for status in ("failure", "error", "skipped")
        ) > 1:
            return None, "pytest JUnit result has multiple statuses for one test"

    if counters["tests"] != len(testcases):
        return None, "pytest JUnit test count is inconsistent"
    if counters["failures"] != failure_nodes:
        return None, "pytest JUnit failure count is inconsistent"
    if counters["skipped"] != skipped_nodes:
        return None, "pytest JUnit skipped count is inconsistent"
    if counters["failures"] + counters["skipped"] > counters["tests"]:
        return None, "pytest JUnit result counters are inconsistent"
    if counters["tests"] == 0 and counters["failures"] > 0:
        return None, "pytest JUnit has failures without tests"

    return (
        _JUnitSummary(
            tests=counters["tests"],
            failures=counters["failures"],
            errors=counters["errors"],
            skipped=counters["skipped"],
            testcase_count=len(testcases),
            failure_nodes=failure_nodes,
            error_nodes=error_nodes,
            skipped_nodes=skipped_nodes,
        ),
        "valid pytest JUnit result",
    )


def _classify_pytest_result(
    exit_code: int,
    junit_path: Path,
) -> tuple[PytestExecutionClassification, str]:
    """Classify pytest from its exit code and machine-readable JUnit result."""
    summary, reason = _parse_junit_xml(junit_path)
    if summary is None:
        return PytestExecutionClassification.INVALID_RESULT, reason

    if exit_code in {2, 3, 4}:
        return PytestExecutionClassification.EXECUTION_ERROR, (
            f"pytest exited with infrastructure status {exit_code}"
        )

    if summary.tests == 0 or summary.skipped == summary.tests:
        if (
            exit_code == 5
            or (exit_code == 0 and summary.failures == 0 and summary.errors == 0)
        ):
            return PytestExecutionClassification.NO_TESTS, "no tests were executed"
        return PytestExecutionClassification.INVALID_RESULT, (
            "pytest reported no tests with an inconsistent exit code"
        )

    if exit_code == 0:
        if summary.failures or summary.errors:
            return PytestExecutionClassification.INVALID_RESULT, (
                "pytest passed with failing JUnit counters"
            )
        return PytestExecutionClassification.PASSED, "tests passed"

    if exit_code == 1:
        if summary.errors:
            return PytestExecutionClassification.EXECUTION_ERROR, (
                "pytest reported collection or execution errors"
            )
        if summary.failures > 0:
            return PytestExecutionClassification.ASSERTION_FAILED, (
                "pytest reported test failures"
            )
        return PytestExecutionClassification.INVALID_RESULT, (
            "pytest failed without a test failure in the JUnit result"
        )

    return PytestExecutionClassification.EXECUTION_ERROR, (
        f"pytest exited with infrastructure status {exit_code}"
    )


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
    classification: PytestExecutionClassification | None = None
    classification_reason: str | None = None


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
        self._junit_path = self.work_root.joinpath(*_JUNIT_XML_PATH.parts)
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
        self._prepare_junit_path()
        execution_argv = [
            *validated,
            "--junitxml",
            _JUNIT_XML_PATH.as_posix(),
        ]
        executable = [sys.executable, *execution_argv[1:]]
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
            stdout = self._sanitize_output(_text_output(exc.stdout))
            stderr = self._sanitize_output(_text_output(exc.stderr))
            stdout, stdout_truncated = _cap_output(stdout)
            stderr, stderr_truncated = _cap_output(stderr)
            self._remove_junit_path()
            return SandboxCommandResult(
                command=" ".join(validated),
                argv=list(validated),
                exit_code=124,
                stdout=stdout,
                stderr=stderr,
                duration_seconds=time.monotonic() - started,
                timed_out=True,
                truncated=stdout_truncated or stderr_truncated,
                classification=PytestExecutionClassification.TIMEOUT,
                classification_reason="pytest execution timed out",
            )
        duration = time.monotonic() - started
        stdout = self._sanitize_output(_text_output(result.stdout))
        stderr = self._sanitize_output(_text_output(result.stderr))
        stdout, stdout_truncated = _cap_output(stdout)
        stderr, stderr_truncated = _cap_output(stderr)
        classification, reason = _classify_pytest_result(
            result.returncode,
            self._junit_path,
        )
        self._remove_junit_path()
        return SandboxCommandResult(
            command=" ".join(validated),
            argv=list(validated),
            exit_code=result.returncode,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=duration,
            timed_out=False,
            truncated=stdout_truncated or stderr_truncated,
            classification=classification,
            classification_reason=reason,
        )

    def _prepare_junit_path(self) -> None:
        """Create the private, sandbox-relative directory for JUnit output."""
        result_dir = self._junit_path.parent
        try:
            result_dir.resolve().relative_to(self.work_root)
            if result_dir.exists() and result_dir.is_symlink():
                raise RuntimeError("pytest result directory must not be a symlink")
            result_dir.mkdir(mode=0o700, exist_ok=True)
            if self._junit_path.exists() and self._junit_path.is_dir():
                raise RuntimeError("pytest result path must not be a directory")
            self._junit_path.unlink(missing_ok=True)
        except OSError as exc:
            raise RuntimeError("could not prepare pytest result path") from exc

    def _remove_junit_path(self) -> None:
        """Remove the private machine-readable result after parsing or timeout."""
        try:
            self._junit_path.unlink(missing_ok=True)
        except OSError:
            return

    def _sanitize_output(self, text: str) -> str:
        """Replace disposable absolute paths before results leave the sandbox."""
        return text.replace(str(self._root), "<sandbox>").replace(
            str(self.work_root),
            "<sandbox>",
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
