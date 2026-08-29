"""Public input contracts for normalized RootTrace incidents."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from roottrace.runtime.paths import validate_relative_path

MAX_TITLE_CHARS = 200
MAX_PROBLEM_CHARS = 20_000
MAX_LOG_CHARS = 20_000
MAX_DIFF_CHARS = 100_000
MAX_SOURCE_CHARS = 1_000
MAX_TOOL_CHARS = 200
MAX_COMMAND_CHARS = 500
MAX_LOGS = 10
MAX_GIT_HISTORY_DEPTH = 50
DEFAULT_GIT_SEARCH_DEPTHS = (8, 16, 32, 50)
MAX_GIT_CANDIDATE_COMMITS = 20
MAX_GIT_CANDIDATE_PATHS = 20
MAX_GIT_POLICY_REASONS = 10

StableId = Annotated[
    str,
    StringConstraints(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$",
        max_length=128,
    ),
]
BoundedLog = Annotated[str, StringConstraints(max_length=MAX_LOG_CHARS)]

_SHA_PATTERN = re.compile(r"^[0-9a-fA-F]{7,64}$")
_TEXT_SHA_PATTERN = re.compile(
    r"(?<![0-9A-Fa-f])([0-9A-Fa-f]{7,64})(?![0-9A-Fa-f])"
)
_COMMIT_CONTEXT_PATTERN = re.compile(
    r"\b(?:commit|sha|revision|changeset|regression)\b",
    re.IGNORECASE,
)
_REGRESSION_SIGNAL_PATTERN = re.compile(
    r"\b(?:regression|regressed|introduced\s+by|since\s+(?:the\s+)?commit|"
    r"after\s+(?:the\s+)?commit|git\s+bisect(?:ed|ing)?)\b",
    re.IGNORECASE,
)
_GENERATED_PR_COMMIT_PATTERN = re.compile(
    r"^\s*PR commit\s+[0-9A-Fa-f]{7,64}(?:\s|:)",
    re.IGNORECASE,
)

ResourceKind = Literal["issue", "pull_request"]


def validate_commit_sha(value: str) -> str:
    if not isinstance(value, str) or not _SHA_PATTERN.fullmatch(value):
        raise ValueError("commit must be a 7-64 character hexadecimal SHA")
    return value


def extract_diff_paths(diff: str | None) -> list[str]:
    """Extract bounded, validated repository paths from a unified diff."""
    if not diff:
        return []
    paths: set[str] = set()
    for line in diff.splitlines():
        candidate: str | None = None
        if line.startswith("+++ b/"):
            candidate = line[6:].strip()
        elif line.startswith("diff --git a/"):
            fields = line.split(" ")
            if len(fields) >= 4 and fields[3].startswith("b/"):
                candidate = fields[3][2:].strip()
        if not candidate or candidate == "/dev/null":
            continue
        try:
            paths.add(validate_relative_path(candidate))
        except ValueError:
            continue
    return sorted(paths)


def _candidate_text_sha(text: str) -> set[str]:
    """Extract syntactically valid SHAs from user-provided incident text."""
    candidates: set[str] = set()
    for line in text.splitlines():
        if _GENERATED_PR_COMMIT_PATTERN.match(line):
            continue
        has_commit_context = _COMMIT_CONTEXT_PATTERN.search(line) is not None
        for match in _TEXT_SHA_PATTERN.finditer(line):
            candidate = match.group(1)
            if len(candidate) == 40 or has_commit_context:
                candidates.add(candidate.lower())
    return candidates


class GitVerificationPolicy(BaseModel):
    """Deterministic policy controlling bounded repository history access."""

    enabled: bool
    reasons: list[str] = Field(default_factory=list, max_length=MAX_GIT_POLICY_REASONS)
    history_depth: int = Field(ge=1, le=MAX_GIT_HISTORY_DEPTH)
    search_depths: list[int] = Field(default_factory=list, max_length=4)
    candidate_commits: list[str] = Field(
        default_factory=list,
        max_length=MAX_GIT_CANDIDATE_COMMITS,
    )
    candidate_paths: list[str] = Field(
        default_factory=list,
        max_length=MAX_GIT_CANDIDATE_PATHS,
    )
    max_tool_calls: int = Field(default=1, ge=1, le=5)

    @field_validator("reasons")
    @classmethod
    def _validate_reasons(cls, values: list[str]) -> list[str]:
        if any(
            not isinstance(value, str)
            or not value.strip()
            or len(value) > 100
            for value in values
        ):
            raise ValueError("Git verification reasons must be short and non-empty")
        return sorted({value.strip() for value in values})

    @field_validator("candidate_commits")
    @classmethod
    def _validate_candidate_commits(cls, values: list[str]) -> list[str]:
        normalized = [validate_commit_sha(value).lower() for value in values]
        if len(set(normalized)) != len(normalized):
            raise ValueError("Git verification candidate commits must be unique")
        return normalized

    @field_validator("candidate_paths")
    @classmethod
    def _validate_candidate_paths(cls, values: list[str]) -> list[str]:
        normalized = [validate_relative_path(value) for value in values]
        if len(set(normalized)) != len(normalized):
            raise ValueError("Git verification candidate paths must be unique")
        return normalized

    @field_validator("search_depths")
    @classmethod
    def _validate_search_depths(cls, values: list[int]) -> list[int]:
        if any(type(value) is not int or value < 1 for value in values):
            raise ValueError("Git search depths must be positive integers")
        if len(set(values)) != len(values) or values != sorted(values):
            raise ValueError("Git search depths must be unique and ascending")
        return values

    @model_validator(mode="after")
    def _validate_mode(self) -> GitVerificationPolicy:
        if not self.search_depths:
            self.search_depths = (
                [
                    depth
                    for depth in DEFAULT_GIT_SEARCH_DEPTHS
                    if depth <= self.history_depth
                ]
                if self.enabled
                else [1]
            )
            if self.enabled and not self.search_depths:
                self.search_depths = [self.history_depth]
        if self.search_depths[-1] > self.history_depth:
            raise ValueError("Git search depths must not exceed history depth")
        if self.enabled and self.history_depth <= 1:
            raise ValueError("enabled Git verification requires history depth greater than 1")
        if not self.enabled and self.history_depth != 1:
            raise ValueError("disabled Git verification must use history depth 1")
        if self.enabled and not self.reasons:
            raise ValueError("enabled Git verification requires a reason")
        if not self.enabled and (self.candidate_commits or self.candidate_paths):
            raise ValueError("disabled Git verification must not contain candidates")
        if not self.enabled and self.search_depths != [1]:
            raise ValueError("disabled Git verification must use search depth 1")
        return self


def build_git_verification_policy(
    *,
    resource_kind: ResourceKind,
    title: str | None,
    problem: str,
    logs: Iterable[str] = (),
    labels: Iterable[str] = (),
    related_commits: Iterable[str] = (),
    changed_files: Iterable[str] = (),
    diff: str | None = None,
) -> GitVerificationPolicy:
    """Build the bounded Git verification policy without model inference."""
    if resource_kind not in {"issue", "pull_request"}:
        raise ValueError("resource_kind must be issue or pull_request")

    incident_text = "\n".join(
        [title or "", problem, *(value for value in logs if value)]
    )
    candidate_commits = _candidate_text_sha(incident_text)
    for value in related_commits:
        candidate_commits.add(validate_commit_sha(value).lower())
    candidate_commits = set(sorted(candidate_commits)[:MAX_GIT_CANDIDATE_COMMITS])

    candidate_paths: set[str] = set()
    for value in changed_files:
        candidate_paths.add(validate_relative_path(value))
    candidate_paths.update(extract_diff_paths(diff))

    reasons: list[str] = []
    if resource_kind == "pull_request":
        reasons.append("pull_request")
    normalized_labels = {
        label.strip().casefold() for label in labels if label and label.strip()
    }
    if "regression" in normalized_labels:
        reasons.append("regression_label")
    if _REGRESSION_SIGNAL_PATTERN.search(incident_text):
        reasons.append("regression_signal")
    if candidate_commits:
        reasons.append("commit_sha")

    enabled = bool(reasons)
    return GitVerificationPolicy(
        enabled=enabled,
        reasons=sorted(set(reasons)) if enabled else ["not_triggered"],
        history_depth=MAX_GIT_HISTORY_DEPTH if enabled else 1,
        search_depths=list(DEFAULT_GIT_SEARCH_DEPTHS) if enabled else [1],
        candidate_commits=sorted(candidate_commits) if enabled else [],
        candidate_paths=sorted(candidate_paths)[:MAX_GIT_CANDIDATE_PATHS]
        if enabled
        else [],
        max_tool_calls=5 if enabled else 1,
    )


class Provenance(BaseModel):
    """Reproducible origin of an evidence item or incident input."""

    source: str = Field(max_length=MAX_SOURCE_CHARS)
    tool: str | None = Field(default=None, max_length=MAX_TOOL_CHARS)
    command: str | None = Field(default=None, max_length=MAX_COMMAND_CHARS)
    commit: str | None = None

    @field_validator("commit")
    @classmethod
    def _validate_commit(cls, value: str | None) -> str | None:
        if value is not None:
            validate_commit_sha(value)
        return value


class IncidentInput(BaseModel):
    """Normalized incident input for one RootTrace run."""

    id: StableId
    repo: str = Field(max_length=MAX_SOURCE_CHARS)
    base_commit: str
    resource_kind: ResourceKind = "issue"
    title: str | None = Field(default=None, max_length=MAX_TITLE_CHARS)
    problem: str = Field(max_length=MAX_PROBLEM_CHARS)
    logs: list[BoundedLog] = Field(default_factory=list, max_length=MAX_LOGS)
    diff: str | None = Field(default=None, max_length=MAX_DIFF_CHARS)
    labels: list[str] = Field(default_factory=list, max_length=100)
    related_commits: list[str] = Field(default_factory=list, max_length=50)
    changed_files: list[str] = Field(default_factory=list, max_length=3_000)
    git_verification_policy: GitVerificationPolicy = Field(
        default_factory=lambda: GitVerificationPolicy(
            enabled=False,
            reasons=["not_triggered"],
            history_depth=1,
        )
    )
    provenance: Provenance

    @model_validator(mode="before")
    @classmethod
    def _derive_policy(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        values = dict(data)
        resource_kind = values.get("resource_kind")
        if resource_kind is None or (
            resource_kind == "issue" and values.get("diff")
        ):
            resource_kind = "pull_request" if values.get("diff") else "issue"
            values["resource_kind"] = resource_kind
        if values.get("git_verification_policy") is None:
            title = values.get("title")
            problem = values.get("problem")
            diff = values.get("diff")
            if (
                (title is not None and not isinstance(title, str))
                or not isinstance(problem, str)
                or (diff is not None and not isinstance(diff, str))
            ):
                return values
            labels = values.get("labels", [])
            related_commits = values.get("related_commits", [])
            changed_files = values.get("changed_files", [])
            logs = values.get("logs", [])
            if all(
                isinstance(items, list)
                and all(isinstance(item, str) for item in items)
                for items in (labels, related_commits, changed_files, logs)
            ):
                values["git_verification_policy"] = build_git_verification_policy(
                    resource_kind=resource_kind,
                    title=title,
                    problem=problem,
                    logs=logs,
                    labels=labels,
                    related_commits=related_commits,
                    changed_files=changed_files,
                    diff=diff,
                )
        return values

    @field_validator("labels")
    @classmethod
    def _normalize_labels(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("labels must not contain empty values")
        return sorted({value.strip() for value in values})

    @field_validator("related_commits")
    @classmethod
    def _normalize_related_commits(cls, values: list[str]) -> list[str]:
        normalized = [validate_commit_sha(value).lower() for value in values]
        if len(set(normalized)) != len(normalized):
            raise ValueError("related commits must be unique")
        return sorted(normalized)

    @field_validator("changed_files")
    @classmethod
    def _normalize_changed_files(cls, values: list[str]) -> list[str]:
        normalized = [validate_relative_path(value) for value in values]
        if len(set(normalized)) != len(normalized):
            raise ValueError("changed files must be unique")
        return sorted(normalized)

    @field_validator("repo")
    @classmethod
    def _validate_repo(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("repo must be a non-empty repository identifier")
        if "\\" in value or re.match(r"^[A-Za-z]:", value):
            raise ValueError("repo must be a repository identifier, not a host path")
        try:
            path = PurePosixPath(value)
        except ValueError as exc:
            raise ValueError("invalid repository identifier") from exc
        if (
            path.is_absolute()
            or not path.parts
            or any(part in (".", "..") for part in path.parts)
        ):
            raise ValueError("repo must not contain '.' or '..' segments")
        return value

    @field_validator("base_commit")
    @classmethod
    def _validate_base_commit(cls, value: str) -> str:
        return validate_commit_sha(value)

    @model_validator(mode="after")
    def _validate_derived_policy(self) -> IncidentInput:
        expected = build_git_verification_policy(
            resource_kind=self.resource_kind,
            title=self.title,
            problem=self.problem,
            logs=self.logs,
            labels=self.labels,
            related_commits=self.related_commits,
            changed_files=self.changed_files,
            diff=self.diff,
        )
        if self.git_verification_policy != expected:
            raise ValueError(
                "git_verification_policy must be deterministically derived "
                "from incident metadata"
            )
        return self
