"""Validated case manifest for the live GitHub smoke evaluation.

The smoke suite deliberately keeps its source metadata in the repository.  A
case contains only public incident/provenance fields and evaluator-only file
labels; no patch or test patch is stored here.  File labels are never passed
to RootTrace and are used only after a run has completed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from roottrace.github import parse_github_resource_url
from roottrace.incident.schema import validate_commit_sha
from roottrace.runtime.paths import validate_relative_path

SUITE_NAME = "github_smoke10"
DEFAULT_MANIFEST_PATH = Path(__file__).with_name("github_smoke10.json")

GitHubSmokeSourceType = Literal["github_issue", "github_pr"]

_CONTROL_PATTERN = re.compile(r"[\x00-\x1f\x7f]")


def _validate_paths(values: list[str], field_name: str) -> list[str]:
    """Normalize and validate a bounded list of repository-relative paths."""
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        try:
            path = validate_relative_path(value)
        except ValueError as exc:
            raise ValueError(f"{field_name} contains an unsafe path: {value!r}") from exc
        if path in seen:
            raise ValueError(f"{field_name} contains a duplicate path: {path}")
        seen.add(path)
        normalized.append(path)
    return normalized


class GitHubSmokeCase(BaseModel):
    """One pinned GitHub issue or regression-PR smoke case."""

    model_config = ConfigDict(extra="forbid")

    instance_id: str = Field(min_length=1, max_length=200)
    source_type: GitHubSmokeSourceType
    source_url: str = Field(min_length=1, max_length=500)
    repo: str = Field(min_length=1, max_length=200)
    # ``base_commit`` is the revision RootTrace analyzes.  For issue cases it
    # is the SWE-bench historical base; for PR cases it is the pinned bad-PR
    # merge commit (the post-regression state).
    base_commit: str = Field(min_length=7, max_length=64)
    gold_files: list[str] = Field(default_factory=list, max_length=100)
    expected_files: list[str] = Field(default_factory=list, max_length=100)
    regression_issue_url: str | None = Field(default=None, max_length=500)
    bad_pr_base_commit: str | None = Field(default=None, min_length=7, max_length=64)
    bad_pr_head_commit: str | None = Field(default=None, min_length=7, max_length=64)
    fix_evidence_url: str | None = Field(default=None, max_length=500)
    manual_review_required: bool = False

    @field_validator("instance_id", "repo")
    @classmethod
    def _reject_control_chars(cls, value: str) -> str:
        if _CONTROL_PATTERN.search(value):
            raise ValueError("must not contain control characters")
        if value != value.strip():
            raise ValueError("must not have leading/trailing whitespace")
        return value

    @field_validator("repo")
    @classmethod
    def _validate_repo(cls, value: str) -> str:
        parts = value.split("/")
        if len(parts) != 2 or any(not part for part in parts):
            raise ValueError("repo must be an owner/name identifier")
        if any(part in {".", ".."} for part in parts):
            raise ValueError("repo must not contain '.' or '..' segments")
        return value

    @field_validator("base_commit", "bad_pr_base_commit", "bad_pr_head_commit")
    @classmethod
    def _validate_sha(cls, value: str | None) -> str | None:
        if value is not None:
            return validate_commit_sha(value)
        return value

    @field_validator("gold_files")
    @classmethod
    def _validate_gold_files(cls, values: list[str]) -> list[str]:
        return _validate_paths(values, "gold_files")

    @field_validator("expected_files")
    @classmethod
    def _validate_expected_files(cls, values: list[str]) -> list[str]:
        return _validate_paths(values, "expected_files")

    @staticmethod
    def _validate_url(value: str, *, expected_kind: str, repo: str, name: str):
        try:
            reference = parse_github_resource_url(value)
        except ValueError as exc:
            raise ValueError(f"{name} is not a canonical GitHub URL: {exc}") from exc
        if reference.kind != expected_kind:
            raise ValueError(
                f"{name} must identify a GitHub {expected_kind}, got {reference.kind}"
            )
        if reference.repository.full_name != repo:
            raise ValueError(
                f"{name} repository {reference.repository.full_name} does not match {repo}"
            )
        return reference

    @classmethod
    def _validate_optional_url(
        cls,
        value: str | None,
        *,
        expected_kind: str,
        repo: str,
        name: str,
    ) -> None:
        if value is not None:
            cls._validate_url(value, expected_kind=expected_kind, repo=repo, name=name)

    def model_post_init(self, __context: object, /) -> None:
        """Validate URL identity and issue/PR-specific field invariants."""
        del __context
        self._validate_url(
            self.source_url,
            expected_kind="issue" if self.source_type == "github_issue" else "pull_request",
            repo=self.repo,
            name="source_url",
        )
        self._validate_optional_url(
            self.regression_issue_url,
            expected_kind="issue",
            repo=self.repo,
            name="regression_issue_url",
        )
        self._validate_optional_url(
            self.fix_evidence_url,
            expected_kind="pull_request",
            repo=self.repo,
            name="fix_evidence_url",
        )

        if self.source_type == "github_issue":
            if not self.gold_files:
                raise ValueError("github_issue cases require non-empty gold_files")
            if self.expected_files:
                raise ValueError("github_issue cases must not define expected_files")
            if self.regression_issue_url or self.fix_evidence_url:
                raise ValueError(
                    "github_issue cases must not define PR regression/fix metadata"
                )
            if self.bad_pr_base_commit or self.bad_pr_head_commit:
                raise ValueError("github_issue cases must not define bad PR commits")
            if self.manual_review_required:
                raise ValueError("github_issue cases are machine-scored, not manual")
        else:
            if not self.regression_issue_url:
                raise ValueError("github_pr cases require regression_issue_url")
            if not self.expected_files:
                raise ValueError("github_pr cases require expected_files")
            if self.gold_files:
                raise ValueError(
                    "github_pr cases must not define gold_files; use expected_files"
                )
            if not self.manual_review_required:
                raise ValueError("github_pr cases must set manual_review_required")

    @property
    def expected_root_cause_files(self) -> list[str]:
        """Return informational expected files for either case kind."""
        return list(self.gold_files or self.expected_files)

    @property
    def is_pull_request(self) -> bool:
        """Whether this is a regression PR case."""
        return self.source_type == "github_pr"


class GitHubSmokeManifest(BaseModel):
    """The fixed, exactly ten-case ``github_smoke10`` manifest."""

    model_config = ConfigDict(extra="forbid")

    name: Literal[SUITE_NAME] = SUITE_NAME
    seed: int = 42
    instances: list[GitHubSmokeCase] = Field(min_length=10, max_length=10)

    @classmethod
    def validate_suite(cls, value: GitHubSmokeManifest) -> GitHubSmokeManifest:
        """Validate the suite composition after Pydantic field parsing."""
        issue_count = sum(case.source_type == "github_issue" for case in value.instances)
        pr_count = sum(case.source_type == "github_pr" for case in value.instances)
        if issue_count != 8 or pr_count != 2:
            raise ValueError(
                "github_smoke10 must contain exactly 8 github_issue and 2 github_pr cases"
            )
        ids = [case.instance_id for case in value.instances]
        if len(ids) != len(set(ids)):
            raise ValueError("github_smoke10 contains duplicate instance_id values")
        checkouts = [(case.repo, case.base_commit) for case in value.instances]
        if len(checkouts) != len(set(checkouts)):
            raise ValueError("github_smoke10 contains duplicate repo/base_commit checkouts")
        return value

    # Pydantic v2 permits a class method validator, but keeping the check in
    # ``model_post_init`` gives us the same strictness for direct construction.
    def model_post_init(self, __context: object, /) -> None:
        del __context
        self.validate_suite(self)


def load_manifest(path: str | Path = DEFAULT_MANIFEST_PATH) -> GitHubSmokeManifest:
    """Load and validate the checked-in smoke manifest."""
    manifest_path = Path(path)
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid GitHub smoke manifest JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise TypeError("GitHub smoke manifest must be a JSON object")
    try:
        return GitHubSmokeManifest.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"invalid GitHub smoke manifest {manifest_path.name}: {exc}") from exc


def select_cases(
    manifest: GitHubSmokeManifest,
    case_id: str | None = None,
) -> list[GitHubSmokeCase]:
    """Select all cases or one manifest case, preserving manifest order."""
    if case_id is None:
        return list(manifest.instances)
    for case in manifest.instances:
        if case.instance_id == case_id:
            return [case]
    raise ValueError(f"unknown github_smoke10 case: {case_id}")


__all__ = [
    "DEFAULT_MANIFEST_PATH",
    "SUITE_NAME",
    "GitHubSmokeCase",
    "GitHubSmokeManifest",
    "GitHubSmokeSourceType",
    "load_manifest",
    "select_cases",
]
