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

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Self
from xml.etree import ElementTree

from roottrace.runtime.docker import DockerEnvironment
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

_JUNIT_XML_NAME = "pytest-results.xml"
_EVENT_PREFIX = "ROOTTRACE_TEST_EVENT:"
_CONTAINER_TOOL_DIR = "/roottrace-tools"
_MAX_TEST_EVENTS = 10_000

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


def validate_test_command(
    tokens: list[str],
    root: Path,
    *,
    allowed_targets: frozenset[str] | None = None,
    require_existing: bool = True,
) -> list[str]:
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
    targets = 0
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
        _validate_target(token, root, require_existing=require_existing)
        if allowed_targets is not None:
            target = token.split("::", maxsplit=1)[0]
            if target not in allowed_targets:
                raise ValueError(f"test target is not a tracked test file: {target}")
        targets += 1
        index += 1
    if targets == 0:
        raise ValueError("pytest command must name an existing test target")
    return tokens


def _validate_target(token: str, root: Path, *, require_existing: bool = True) -> None:
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
    if require_existing and not resolved.exists():
        raise ValueError(f"test target does not exist in sandbox: {base}")
    if (not require_existing or resolved.is_file()) and not base.endswith(".py"):
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
        if path.is_symlink() or path.parent.is_symlink():
            return None, "pytest JUnit result must not be a symlink"
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


def _record_test_events(stdout: str, token: str, destination: Path) -> tuple[str, str | None]:
    """Validate a complete event stream and build JUnit on the host."""
    clean_lines = []
    cases: dict[str, str] = {}
    ended = False
    collected = None
    reported = None
    sequence = 0
    error = None
    rank = {"passed": 0, "skipped": 1, "failure": 2, "error": 3}
    for line in stdout.splitlines(keepends=True):
        marker = line.find(_EVENT_PREFIX)
        if marker < 0:
            clean_lines.append(line)
            continue
        if marker:
            clean_lines.append(line[:marker])
        line = line[marker:]
        if len(line) > 2_000 or not line.startswith(_EVENT_PREFIX + token + ":"):
            error = "test event stream contains an invalid record"
            continue
        try:
            record = json.loads(line[len(_EVENT_PREFIX + token + ":"):])
        except (ValueError, TypeError):
            error = "test event stream contains malformed JSON"
            continue
        if not isinstance(record, dict) or ended:
            error = "test event stream has an invalid order"
            continue
        if record.get("seq") != sequence + 1:
            error = "test event stream has a missing or repeated record"
            continue
        sequence += 1
        if record.get("kind") == "case":
            name, status = record.get("name"), record.get("status")
            if (not isinstance(name, str) or not name or len(name) > 300
                    or status not in rank or len(cases) >= _MAX_TEST_EVENTS):
                error = "test event stream has an invalid case"
                continue
            if name not in cases or rank[status] > rank[cases[name]]:
                cases[name] = status
        elif record.get("kind") == "end":
            ended = True
            collected = record.get("collected")
            reported = record.get("reported")
            if not isinstance(record.get("exit_code"), int):
                error = "test event stream has an invalid completion record"
        else:
            error = "test event stream has an unknown record"
    if error is not None:
        return "".join(clean_lines), error
    if not ended:
        return "".join(clean_lines), "test event stream has no completion record"
    if not isinstance(reported, int) or reported != len(cases):
        return "".join(clean_lines), "test event stream has an inconsistent case count"
    if isinstance(collected, int) and collected > 0 and not cases:
        return "".join(clean_lines), "collected tests have no outcome events"
    counts = {name: sum(status == name for status in cases.values())
              for name in ("failure", "error", "skipped")}
    root = ElementTree.Element("testsuite", {
        "tests": str(len(cases)), "failures": str(counts["failure"]),
        "errors": str(counts["error"]), "skipped": str(counts["skipped"]),
    })
    for name, status in cases.items():
        testcase = ElementTree.SubElement(root, "testcase", {"name": name})
        if status != "passed":
            ElementTree.SubElement(testcase, status)
    payload = ElementTree.tostring(root, encoding="utf-8")
    if len(payload) > MAX_JUNIT_XML_BYTES:
        return "".join(clean_lines), "test event result is too large"
    try:
        destination.write_bytes(payload)
    except OSError:
        return "".join(clean_lines), "could not save host test result"
    return "".join(clean_lines), None


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
    executed_command: str | None = None


class RuntimeVerificationSandbox:
    """Disposable repository copy for bounded, allowlisted runtime tests."""

    def __init__(
        self,
        repo: str | Path,
        base_commit: str | None = None,
        work_dir: str | Path | None = None,
        docker_environment: DockerEnvironment | None = None,
        unavailable_reason: str | None = None,
    ) -> None:
        self.repo = Path(repo).resolve()
        self._require_git_repo(self.repo)
        if base_commit is not None and not _VALID_REVISION.fullmatch(base_commit):
            raise ValueError("base_commit must be a hex SHA or HEAD")
        self.base_commit = base_commit
        self.docker_environment = docker_environment
        self.unavailable_reason = unavailable_reason
        self.before: RepositoryFingerprint = capture_repository_fingerprint(self.repo)
        parent = (
            Path(work_dir).resolve()
            if work_dir is not None
            else Path(tempfile.gettempdir())
        )
        self._root = Path(tempfile.mkdtemp(prefix="roottrace-sandbox-", dir=parent))
        self.work_root = (self._root / "work").resolve()
        self._junit_path = self._root / "results" / _JUNIT_XML_NAME
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
        validated = validate_test_command(tokens, self.work_root)
        if self.unavailable_reason is not None:
            raise RuntimeError(self.unavailable_reason)
        self._prepare_junit_path()
        django_labels = self._django_test_labels(validated)
        event_token = uuid.uuid4().hex
        if self.docker_environment is not None:
            self._stage_test_helpers()
            execution_argv = [*validated, "-p", "roottrace_pytest_events"]
        else:
            execution_argv = [*validated, "--junitxml", str(self._junit_path)]
        executable = [sys.executable, *execution_argv[1:]]
        container_name = None
        if self.docker_environment is not None:
            container_name = f"roottrace-test-{uuid.uuid4().hex}"
            container_args = (
                ["tests/runtests.py", "--settings=roottrace_django_settings",
                 "--noinput", "--parallel=1", *django_labels]
                if django_labels else execution_argv[1:]
            )
            python_path = (
                f"{_CONTAINER_TOOL_DIR}:/roottrace-deps:/work/tests:/work"
                if django_labels else "/roottrace-deps"
            )
            if not django_labels:
                python_path = f"{_CONTAINER_TOOL_DIR}:{python_path}"
            executable = [
                "docker", "run", "--rm", "--name", container_name,
                "--pull", "never", "--network", "none", "--read-only",
                "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                "--pids-limit", "64", "--memory", "1g", "--cpus", "1",
                "--user", f"{os.getuid()}:{os.getgid()}",
                "--platform", self.docker_environment.platform,
                "--entrypoint", self.docker_environment.python_executable,
                "--tmpfs", "/tmp:rw,nosuid,nodev,size=128m",
                "--env", "HOME=/tmp", "--env", "PYTHONDONTWRITEBYTECODE=1",
                "--env", f"PYTHONPATH={python_path}",
                "--env", f"ROOTTRACE_EVENT_TOKEN={event_token}",
                *(
                    ["--env", f"PATH={self.docker_environment.execution_path}"]
                    if self.docker_environment.execution_path else []
                ),
                "--mount", f"type=bind,src={self.work_root},dst=/work",
                "--mount", (
                    f"type=bind,src={self._root / 'tools'},"
                    f"dst={_CONTAINER_TOOL_DIR},readonly"
                ),
                "--workdir", "/work", self.docker_environment.image,
                *container_args,
            ]
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        started = time.monotonic()
        try:
            result = subprocess.run(
                executable,
                cwd=self.work_root,
                # A spawned RQ workhorse runs in a background process group;
                # terminal reads by pytest would stop the entire group.
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
                env=env,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            if container_name is not None:
                subprocess.run(
                    ["docker", "rm", "-f", container_name],
                    capture_output=True, timeout=15, check=False,
                )
            if isinstance(exc, OSError):
                raise RuntimeError("Docker execution unavailable") from exc  # noqa: TRY004
            raw_stdout = _text_output(exc.stdout)
            if self.docker_environment is not None:
                raw_stdout, _ = _record_test_events(
                    raw_stdout, event_token, self._junit_path,
                )
            stdout = self._sanitize_output(raw_stdout)
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
                executed_command=(
                    "python " + " ".join(container_args) if django_labels else None
                ),
            )
        duration = time.monotonic() - started
        raw_stdout = _text_output(result.stdout)
        event_error = None
        if self.docker_environment is not None:
            raw_stdout, event_error = _record_test_events(
                raw_stdout, event_token, self._junit_path,
            )
        stdout = self._sanitize_output(raw_stdout)
        stderr = self._sanitize_output(_text_output(result.stderr))
        stdout, stdout_truncated = _cap_output(stdout)
        stderr, stderr_truncated = _cap_output(stderr)
        if event_error is None:
            classification, reason = _classify_pytest_result(
                result.returncode, self._junit_path,
            )
        else:
            classification = PytestExecutionClassification.INVALID_RESULT
            reason = event_error
        if self.docker_environment is not None and result.returncode in {125, 126, 127}:
            classification = PytestExecutionClassification.EXECUTION_ERROR
            reason = f"Docker execution failed with status {result.returncode}"
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
            executed_command=(
                "python " + " ".join(container_args) if django_labels else None
            ),
        )

    def _django_test_labels(self, tokens: list[str]) -> list[str]:
        """Map validated pytest files to Django's native runner when present."""
        if not (
            (self.work_root / "django" / "__init__.py").is_file()
            and (self.work_root / "tests" / "runtests.py").is_file()
            and (self.work_root / "tests" / "test_sqlite.py").is_file()
        ):
            return []
        targets = [
            token for token in tokens[3:]
            if token.endswith(".py") or ".py::" in token
        ]
        if not targets or any(not target.startswith("tests/") for target in targets):
            return []
        labels = []
        for target in targets:
            path, *selectors = target.split("::")
            label = path.removeprefix("tests/").removesuffix(".py").replace("/", ".")
            labels.append(".".join([label, *selectors]))
        return labels

    def _stage_test_helpers(self) -> None:
        """Mount trusted event reporters separately from the repository copy."""
        source = Path(__file__).resolve().parent.parent / "verification"
        tools_dir = self._root / "tools"
        tools_dir.mkdir(mode=0o700, exist_ok=True)
        for name in (
            "roottrace_event_protocol.py", "roottrace_pytest_events.py",
            "roottrace_django_settings.py", "roottrace_django_runner.py",
        ):
            (tools_dir / name).write_bytes((source / name).read_bytes())

    def _prepare_junit_path(self) -> None:
        """Create a private result directory outside the repository copy."""
        result_dir = self._junit_path.parent
        try:
            result_dir.resolve().relative_to(self._root)
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
        return text.replace(str(self.repo), "<target>").replace(str(self._root), "<sandbox>").replace(
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
