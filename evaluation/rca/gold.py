"""Evaluator-only gold data access (never passed to the RCA runtime).

Gold parsing happens strictly after a RootTrace run completes. The parsed
output is the ordered set of non-test source files changed by the gold patch;
the patch content itself is never given to RootTrace.
"""

from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict, Field

_DIFF_HEADER = re.compile(r"^diff --git a/(\S+) b/(\S+)$")
_NEW_FILE = re.compile(r"^\+\+\+ b/(\S+)$")
_DEV_NULL_VALUES = frozenset({"/dev/null", "dev/null"})

_TEST_DIR_NAMES = frozenset(
    {"test", "tests", "testing", "testsuite", "testdata"}
)
_TEST_BASE_NAMES = frozenset(
    {"test.py", "tests.py", "conftest.py", "test_runner.py"}
)


class GoldCase(BaseModel):
    """One gold record. Evaluator-only by construction."""

    model_config = ConfigDict(extra="forbid")

    instance_id: str = Field(min_length=1, max_length=200)
    patch: str
    test_patch: str = ""
    FAIL_TO_PASS: list[str] = Field(default_factory=list)
    PASS_TO_PASS: list[str] = Field(default_factory=list)


def _normalize_patch_path(value: str) -> str:
    if value in _DEV_NULL_VALUES:
        raise ValueError("dev/null is not a repository path")
    if not value or "\\" in value or value.startswith(("/", "~")):
        raise ValueError(f"invalid patch path: {value}")
    path = PurePosixPath(value)
    if any(part in {".", ".."} for part in path.parts):
        raise ValueError(f"invalid patch path: {value}")
    return path.as_posix()


def is_test_file(path: str) -> bool:
    """True when ``path`` looks like a test or test-support file."""
    parts = PurePosixPath(path).parts
    if any(part in _TEST_DIR_NAMES for part in parts):
        return True
    name = parts[-1]
    if name.startswith(("test_", "tests_")):
        return True
    if name.endswith(("_test.py", "_tests.py")):
        return True
    return name in _TEST_BASE_NAMES


def parse_gold_patch(patch: str) -> list[str]:
    """Extract non-test source files changed by a gold patch.

    Returns repository-relative, deduplicated, sorted paths. Deleted files
    (``/dev/null`` targets) are skipped.
    """
    changed: set[str] = set()
    for line in patch.splitlines():
        stripped = line.strip()
        match = _DIFF_HEADER.match(stripped)
        if match:
            new_path = match.group(2)
            if new_path not in _DEV_NULL_VALUES:
                changed.add(_normalize_patch_path(new_path))
            continue
        match = _NEW_FILE.match(stripped)
        if match:
            new_path = match.group(1)
            if new_path not in _DEV_NULL_VALUES:
                changed.add(_normalize_patch_path(new_path))
    return sorted(path for path in changed if not is_test_file(path))


def _load_gold_cases(path: str | Path) -> dict[str, GoldCase]:
    gold_path = Path(path)
    cases: dict[str, GoldCase] = {}
    with gold_path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                case = GoldCase(**record)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    f"invalid gold record at line {line_no}: {exc}"
                ) from exc
            if case.instance_id in cases:
                raise ValueError(
                    f"duplicate gold instance_id: {case.instance_id}"
                )
            cases[case.instance_id] = case
    return cases


class GoldStore:
    """Lazy gold store; the file is only read after an RCA run completes."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._cases: dict[str, GoldCase] | None = None

    def load(self) -> dict[str, GoldCase]:
        if self._cases is None:
            self._cases = _load_gold_cases(self._path)
        return self._cases

    def gold_files(self, instance_id: str) -> list[str]:
        case = self.load().get(instance_id)
        if case is None:
            raise ValueError(f"gold record missing for {instance_id}")
        return parse_gold_patch(case.patch)
