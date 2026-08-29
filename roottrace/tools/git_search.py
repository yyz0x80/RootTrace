"""Deterministic, bounded layered Git search for RootTrace.

The searcher materializes the commit set for each depth before applying path,
message, or content filters. This keeps a ``--max-count`` option from being
mistaken for a history boundary and gives later model-facing tools an explicit
set of commits that has already been opened.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, Field, field_validator, model_validator

from roottrace.incident.context import IncidentContext
from roottrace.incident.schema import (
    DEFAULT_GIT_SEARCH_DEPTHS,
    MAX_GIT_HISTORY_DEPTH,
    validate_commit_sha,
)
from roottrace.runtime.paths import validate_relative_path

MAX_GIT_SEARCH_CANDIDATES = 5
MAX_GIT_SEARCH_PATHS = 6
MAX_GIT_SEARCH_CONTENT_SIGNALS = 4
MAX_GIT_SEARCH_MESSAGE_SIGNALS = 4
MAX_GIT_SEARCH_COMMANDS = 16
MAX_GIT_SEARCH_SIGNAL_CHARS = 80
MAX_GIT_SEARCH_MATCHES = 10

GitSignalKind = Literal[
    "explicit_commit",
    "path",
    "symbol",
    "failure_signature",
    "message",
]
GitQueryKind = Literal["path", "content", "message"]

_STACK_PATH_PATTERN = re.compile(r"File\s+[\"']([^\"']+)[\"']")
_CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x1f\x7f]")
_GENERIC_MESSAGE_TERMS = frozenset(
    {
        "after",
        "before",
        "error",
        "failed",
        "failure",
        "issue",
        "problem",
        "regression",
        "returns",
        "should",
        "test",
        "tests",
        "when",
    }
)


class GitSearchQuery(BaseModel):
    """One bounded deterministic query used by the layered search."""

    kind: GitQueryKind
    signal: GitSignalKind
    source: str = Field(min_length=1, max_length=40)
    value: str = Field(min_length=1, max_length=MAX_GIT_SEARCH_SIGNAL_CHARS)
    weight: int = Field(ge=1, le=8)

    @field_validator("source", "value")
    @classmethod
    def _validate_text(cls, value: str) -> str:
        if _CONTROL_CHAR_PATTERN.search(value):
            raise ValueError("Git search query text must not contain control characters")
        normalized = value.strip()
        if not normalized:
            raise ValueError("Git search query text must not be empty")
        return normalized

    @field_validator("value")
    @classmethod
    def _validate_query_value(cls, value: str) -> str:
        if not value:
            raise ValueError("Git search query value must not be empty")
        return value


class GitSearchPlan(BaseModel):
    """The deterministic search plan prepared before the Git model call."""

    enabled: bool
    max_depth: int = Field(ge=1, le=MAX_GIT_HISTORY_DEPTH)
    search_depths: list[int] = Field(default_factory=list, max_length=4)
    candidate_commits: list[str] = Field(default_factory=list, max_length=20)
    queries: list[GitSearchQuery] = Field(default_factory=list, max_length=20)
    max_candidates: int = Field(default=MAX_GIT_SEARCH_CANDIDATES, ge=1, le=10)
    max_commands: int = Field(default=MAX_GIT_SEARCH_COMMANDS, ge=1, le=32)

    @field_validator("search_depths")
    @classmethod
    def _validate_depths(cls, values: list[int]) -> list[int]:
        if any(type(value) is not int or value < 1 for value in values):
            raise ValueError("Git search depths must be positive integers")
        if len(set(values)) != len(values) or values != sorted(values):
            raise ValueError("Git search depths must be unique and ascending")
        return values

    @field_validator("candidate_commits")
    @classmethod
    def _validate_commits(cls, values: list[str]) -> list[str]:
        normalized = [validate_commit_sha(value).lower() for value in values]
        if len(set(normalized)) != len(normalized):
            raise ValueError("Git search candidate commits must be unique")
        return normalized

    @model_validator(mode="after")
    def _validate_plan(self) -> GitSearchPlan:
        if not self.search_depths:
            raise ValueError("Git search plan requires at least one search depth")
        if self.search_depths[-1] > self.max_depth:
            raise ValueError("Git search depth must not exceed the maximum depth")
        if self.enabled and self.search_depths == [1]:
            raise ValueError("enabled Git search must include a layered depth")
        if not self.enabled and (
            self.search_depths != [1] or self.candidate_commits or self.queries
        ):
            raise ValueError("disabled Git search must not contain search inputs")
        return self


class GitSearchCandidate(BaseModel):
    """One ranked commit candidate produced by deterministic Git search."""

    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    depth: int = Field(ge=1, le=MAX_GIT_HISTORY_DEPTH)
    score: int = Field(ge=1, le=100)
    subject: str = Field(default="", max_length=200)
    matched_paths: list[str] = Field(default_factory=list, max_length=MAX_GIT_SEARCH_PATHS)
    matched_signals: list[str] = Field(default_factory=list, max_length=MAX_GIT_SEARCH_MATCHES)
    signal_kinds: list[GitSignalKind] = Field(default_factory=list, max_length=5)
    strong_match: bool = False
    command: str = Field(max_length=500)

    @field_validator("commit")
    @classmethod
    def _normalize_commit(cls, value: str) -> str:
        return value.lower()

    @field_validator("matched_paths")
    @classmethod
    def _validate_paths(cls, values: list[str]) -> list[str]:
        normalized = [validate_relative_path(value) for value in values]
        if len(set(normalized)) != len(normalized):
            raise ValueError("Git search candidate paths must be unique")
        return normalized

    @model_validator(mode="after")
    def _validate_signals(self) -> GitSearchCandidate:
        if not self.signal_kinds or not self.matched_signals:
            raise ValueError("Git search candidate requires a matching signal")
        kinds = set(self.signal_kinds)
        expected_strong_match = (
            "explicit_commit" in kinds
            or (
                "path" in kinds
                and bool(kinds.intersection({"symbol", "failure_signature"}))
            )
            or {"symbol", "failure_signature"}.issubset(kinds)
        )
        if self.strong_match != expected_strong_match:
            raise ValueError("Git search candidate strength must match its signals")
        return self


class GitSearchStage(BaseModel):
    """Bounded outcome for one opened history layer."""

    depth: int = Field(ge=1, le=MAX_GIT_HISTORY_DEPTH)
    opened_commits: int = Field(ge=0, le=MAX_GIT_HISTORY_DEPTH)
    matched_commits: list[str] = Field(default_factory=list, max_length=MAX_GIT_SEARCH_CANDIDATES)
    candidate_count: int = Field(ge=0, le=MAX_GIT_HISTORY_DEPTH)
    command_count: int = Field(ge=0, le=MAX_GIT_SEARCH_COMMANDS)
    duration_seconds: float = Field(ge=0)
    stopped: bool = False
    stop_reason: str | None = Field(default=None, max_length=80)

    @field_validator("matched_commits")
    @classmethod
    def _validate_commits(cls, values: list[str]) -> list[str]:
        normalized = [validate_commit_sha(value).lower() for value in values]
        if len(set(normalized)) != len(normalized):
            raise ValueError("Git search stage commits must be unique")
        return normalized


class GitSearchSummary(BaseModel):
    """Auditable summary of the layered Git search."""

    enabled: bool
    attempted_depths: list[int] = Field(default_factory=list, max_length=4)
    reached_depth: int = Field(ge=1, le=MAX_GIT_HISTORY_DEPTH)
    candidate_commits: list[str] = Field(default_factory=list, max_length=MAX_GIT_SEARCH_CANDIDATES)
    candidates: list[GitSearchCandidate] = Field(
        default_factory=list,
        max_length=MAX_GIT_SEARCH_CANDIDATES,
    )
    stages: list[GitSearchStage] = Field(default_factory=list, max_length=4)
    stop_reason: str = Field(min_length=1, max_length=80)
    commands_executed: int = Field(ge=0, le=MAX_GIT_SEARCH_COMMANDS)
    errors: list[str] = Field(default_factory=list, max_length=4)

    @field_validator("attempted_depths")
    @classmethod
    def _validate_attempted_depths(cls, values: list[int]) -> list[int]:
        if any(type(value) is not int or value < 1 for value in values):
            raise ValueError("attempted Git search depths must be positive integers")
        if len(set(values)) != len(values) or values != sorted(values):
            raise ValueError("attempted Git search depths must be unique and ascending")
        return values

    @field_validator("candidate_commits")
    @classmethod
    def _validate_candidate_commits(cls, values: list[str]) -> list[str]:
        normalized = [validate_commit_sha(value).lower() for value in values]
        if len(set(normalized)) != len(normalized):
            raise ValueError("Git search summary commits must be unique")
        return normalized


@dataclass(frozen=True)
class GitSearchCommandResult:
    """Minimal command result consumed by the deterministic searcher."""

    ok: bool
    output: str
    command: str
    duration_seconds: float


class GitSearchCommandRunner(Protocol):
    """Read-only command runner used by ``GitSearchExecutor``."""

    def __call__(self, argv: list[str]) -> GitSearchCommandResult: ...


def _safe_query(value: str, *, message: bool = False) -> str | None:
    normalized = value.strip()
    if (
        not normalized
        or len(normalized) > MAX_GIT_SEARCH_SIGNAL_CHARS
        or _CONTROL_CHAR_PATTERN.search(normalized)
    ):
        return None
    if message and (
        len(normalized) < 4 or normalized.casefold() in _GENERIC_MESSAGE_TERMS
    ):
        return None
    return normalized


def _stack_paths(context: IncidentContext) -> list[str]:
    paths: set[str] = set()
    for log in context.incident.logs:
        for match in _STACK_PATH_PATTERN.finditer(log):
            try:
                paths.add(validate_relative_path(match.group(1).strip()))
            except ValueError:
                continue
    return sorted(paths)


def build_git_search_plan(context: IncidentContext) -> GitSearchPlan:
    """Build a deterministic layered plan from the bounded incident context."""
    policy = context.incident.git_verification_policy
    if not policy.enabled:
        return GitSearchPlan(
            enabled=False,
            max_depth=1,
            search_depths=[1],
        )

    queries: dict[tuple[str, str, str], GitSearchQuery] = {}

    def add_query(
        *,
        kind: GitQueryKind,
        signal: GitSignalKind,
        source: str,
        value: str,
        weight: int,
    ) -> None:
        safe_value = _safe_query(value, message=kind == "message")
        if safe_value is None:
            return
        key = (kind, signal, safe_value.casefold() if kind != "path" else safe_value)
        candidate = GitSearchQuery(
            kind=kind,
            signal=signal,
            source=source,
            value=safe_value,
            weight=weight,
        )
        current = queries.get(key)
        if current is None or (candidate.weight, candidate.source) > (
            current.weight,
            current.source,
        ):
            queries[key] = candidate

    for path in policy.candidate_paths:
        add_query(
            kind="path",
            signal="path",
            source="pr_diff_path" if context.incident.resource_kind == "pull_request" else "changed_file",
            value=path,
            weight=4,
        )
    for path in context.signals.diff_paths:
        add_query(
            kind="path",
            signal="path",
            source="pr_diff_path",
            value=path,
            weight=4,
        )
    for path in _stack_paths(context):
        add_query(
            kind="path",
            signal="path",
            source="stack_trace_path",
            value=path,
            weight=3,
        )
    for snippet in context.snippets[:MAX_GIT_SEARCH_PATHS]:
        add_query(
            kind="path",
            signal="path",
            source="prepared_snippet_path",
            value=snippet.path,
            weight=2,
        )
    for symbol in context.signals.stack_symbols:
        add_query(
            kind="content",
            signal="symbol",
            source="stack_symbol",
            value=symbol,
            weight=3,
        )
    for exception_name in context.signals.exception_names:
        add_query(
            kind="content",
            signal="failure_signature",
            source="exception_name",
            value=exception_name,
            weight=3,
        )
    for term in context.signals.terms:
        add_query(
            kind="message",
            signal="message",
            source="incident_term",
            value=term,
            weight=1,
        )

    ordered_queries = sorted(
        queries.values(),
        key=lambda query: (-query.weight, query.kind, query.signal, query.value),
    )
    path_queries = [query for query in ordered_queries if query.kind == "path"][:MAX_GIT_SEARCH_PATHS]
    content_queries = [query for query in ordered_queries if query.kind == "content"][
        :MAX_GIT_SEARCH_CONTENT_SIGNALS
    ]
    message_queries = [query for query in ordered_queries if query.kind == "message"][
        :MAX_GIT_SEARCH_MESSAGE_SIGNALS
    ]
    selected_queries = sorted(
        (*path_queries, *content_queries, *message_queries),
        key=lambda query: (-query.weight, query.kind, query.signal, query.value),
    )
    search_depths = list(context.incident.git_verification_policy.search_depths)
    if not search_depths:
        search_depths = list(DEFAULT_GIT_SEARCH_DEPTHS)
    return GitSearchPlan(
        enabled=True,
        max_depth=policy.history_depth,
        search_depths=search_depths,
        candidate_commits=policy.candidate_commits,
        queries=selected_queries,
    )


def _parse_commit_list(output: str) -> list[str]:
    commits: list[str] = []
    for line in output.splitlines():
        value = line.strip().lower()
        if re.fullmatch(r"[0-9a-f]{40}", value) and value not in commits:
            commits.append(value)
    return commits


def _parse_log_metadata(output: str) -> dict[str, tuple[str, set[str]]]:
    metadata: dict[str, tuple[str, set[str]]] = {}
    current_commit: str | None = None
    for line in output.splitlines():
        if line.startswith("ROOTTRACE_COMMIT:"):
            header = line.removeprefix("ROOTTRACE_COMMIT:")
            commit, _, subject = header.partition("\t")
            if re.fullmatch(r"[0-9a-fA-F]{40}", commit):
                current_commit = commit.lower()
                metadata[current_commit] = (subject[:200], set())
            else:
                current_commit = None
            continue
        if current_commit is None or not line.strip():
            continue
        try:
            path = validate_relative_path(line.strip())
        except ValueError:
            continue
        metadata[current_commit][1].add(path)
    return metadata


def _compact_command(command: str) -> str:
    if len(command) <= 500:
        return command
    return command[:480] + "..."


def _combined_pattern(values: list[str]) -> str:
    escaped = [re.escape(value) for value in values]
    return "(" + "|".join(escaped) + ")"


class GitSearchExecutor:
    """Execute a bounded layered search using only explicit commit sets."""

    def __init__(
        self,
        *,
        base_commit: str,
        run_command: GitSearchCommandRunner,
        set_visible_depth: Callable[[int], None] | None = None,
    ) -> None:
        self._base_commit = validate_commit_sha(base_commit).lower()
        self._run_command = run_command
        self._set_visible_depth = set_visible_depth

    def _set_depth(self, depth: int) -> None:
        if self._set_visible_depth is not None:
            self._set_visible_depth(depth)

    @staticmethod
    def _stop_reason(candidates: list[GitSearchCandidate]) -> str | None:
        for candidate in candidates:
            if "explicit_commit" in candidate.signal_kinds:
                return "explicit_commit_verified"
        for candidate in candidates:
            kinds = set(candidate.signal_kinds)
            if "path" in kinds and (
                "symbol" in kinds or "failure_signature" in kinds
            ):
                return "path_and_content_match"
            if len(kinds.intersection({"symbol", "failure_signature"})) >= 2:
                return "independent_content_match"
        return None

    def _candidate(
        self,
        *,
        commit: str,
        depth: int,
        subject: str,
        changed_paths: set[str],
        metadata_command: str,
        plan: GitSearchPlan,
        content_matches: dict[GitSignalKind, set[str]],
    ) -> GitSearchCandidate | None:
        matched_paths: list[str] = []
        matched_signals: list[str] = []
        signal_kinds: list[GitSignalKind] = []
        score = 0

        explicit_matches = [
            value
            for value in plan.candidate_commits
            if commit.startswith(value)
        ]
        if len(explicit_matches) == 1:
            signal_kinds.append("explicit_commit")
            matched_signals.append(f"explicit_commit:{explicit_matches[0]}")
            score += 8

        matched_content_kinds: set[GitSignalKind] = set()
        for query in plan.queries:
            matched = False
            if query.kind == "path":
                matched = query.value in changed_paths
                if matched and query.value not in matched_paths:
                    matched_paths.append(query.value)
            elif query.kind == "message":
                matched = query.value.casefold() in subject.casefold()
            else:
                matched = commit in content_matches.get(query.signal, set())
                if matched and query.signal in matched_content_kinds:
                    continue
            if not matched:
                continue
            if query.kind == "content":
                matched_content_kinds.add(query.signal)
            if query.signal not in signal_kinds:
                signal_kinds.append(query.signal)
            matched_signals.append(
                f"pickaxe:{query.signal}"
                if query.kind == "content"
                else f"{query.source}:{query.value}"
            )
            score += query.weight

        if not matched_signals:
            return None
        if "path" in signal_kinds:
            matched_paths.sort()
        unique_signal_kinds = sorted(set(signal_kinds))
        kinds = set(unique_signal_kinds)
        strong_match = (
            "explicit_commit" in kinds
            or (
                "path" in kinds
                and bool(kinds.intersection({"symbol", "failure_signature"}))
            )
            or {"symbol", "failure_signature"}.issubset(kinds)
        )
        return GitSearchCandidate(
            commit=commit,
            depth=depth,
            score=min(score, 100),
            subject=subject,
            matched_paths=matched_paths[:MAX_GIT_SEARCH_PATHS],
            matched_signals=sorted(set(matched_signals))[:MAX_GIT_SEARCH_MATCHES],
            signal_kinds=unique_signal_kinds,
            strong_match=strong_match,
            command=_compact_command(metadata_command),
        )

    def run(self, plan: GitSearchPlan) -> GitSearchSummary:
        """Run the plan, stopping only on a strong match or a hard bound."""
        if not plan.enabled:
            return GitSearchSummary(
                enabled=False,
                reached_depth=1,
                stop_reason="disabled",
                commands_executed=0,
            )

        commands_executed = 0
        reached_depth = 1
        opened_commits: list[str] = []
        candidates_by_commit: dict[str, GitSearchCandidate] = {}
        stages: list[GitSearchStage] = []
        errors: list[str] = []
        attempted_depths: list[int] = []
        stop_reason = "max_depth_reached"

        for depth in plan.search_depths:
            if depth > plan.max_depth:
                break
            if commands_executed >= plan.max_commands:
                stop_reason = "command_budget_exhausted"
                break
            stage_started = time.monotonic()
            stage_commands = 0
            attempted_depths.append(depth)
            rev_result = self._run_command(
                [
                    "git",
                    "rev-list",
                    "--topo-order",
                    f"--max-count={depth}",
                    self._base_commit,
                ]
            )
            commands_executed += 1
            stage_commands += 1
            if not rev_result.ok:
                errors.append(f"depth {depth}: rev-list failed")
                stages.append(
                    GitSearchStage(
                        depth=depth,
                        opened_commits=len(opened_commits),
                        candidate_count=0,
                        command_count=stage_commands,
                        duration_seconds=time.monotonic() - stage_started,
                        stop_reason="git_command_failed",
                    )
                )
                stop_reason = "git_command_failed"
                break

            layer_commits = _parse_commit_list(rev_result.output)
            reached_depth = max(reached_depth, len(layer_commits))
            self._set_depth(depth)
            new_commits = [commit for commit in layer_commits if commit not in opened_commits]
            opened_commits.extend(new_commits)
            if not new_commits:
                stages.append(
                    GitSearchStage(
                        depth=depth,
                        opened_commits=len(layer_commits),
                        candidate_count=0,
                        command_count=stage_commands,
                        duration_seconds=time.monotonic() - stage_started,
                        stopped=True,
                        stop_reason="history_exhausted",
                    )
                )
                stop_reason = "history_exhausted"
                break

            metadata_result = self._run_command(
                [
                    "git",
                    "log",
                    "--no-walk",
                    "--no-ext-diff",
                    "--no-decorate",
                    "--no-abbrev",
                    "--date=short",
                    "--format=ROOTTRACE_COMMIT:%H%x09%s",
                    "--name-only",
                    *new_commits,
                    "--",
                ]
            )
            commands_executed += 1
            stage_commands += 1
            metadata = _parse_log_metadata(metadata_result.output) if metadata_result.ok else {}
            if not metadata_result.ok:
                errors.append(f"depth {depth}: metadata lookup failed")

            content_matches: dict[str, set[str]] = {}
            for signal in ("symbol", "failure_signature"):
                if commands_executed >= plan.max_commands:
                    stop_reason = "command_budget_exhausted"
                    break
                values = [
                    query.value
                    for query in plan.queries
                    if query.kind == "content" and query.signal == signal
                ]
                if not values:
                    continue
                content_result = self._run_command(
                    [
                        "git",
                        "log",
                        "--no-walk",
                        "--no-ext-diff",
                        "--no-decorate",
                        "--no-abbrev",
                        "--format=%H",
                        f"-G{_combined_pattern(values)}",
                        *new_commits,
                        "--",
                    ]
                )
                commands_executed += 1
                stage_commands += 1
                if content_result.ok:
                    content_matches[signal] = set(_parse_commit_list(content_result.output))
                else:
                    errors.append(f"depth {depth}: {signal} lookup failed")

            stage_candidates: list[GitSearchCandidate] = []
            for commit in new_commits:
                subject, changed_paths = metadata.get(commit, ("", set()))
                candidate = self._candidate(
                    commit=commit,
                    depth=depth,
                    subject=subject,
                    changed_paths=changed_paths,
                    metadata_command=metadata_result.command or rev_result.command,
                    plan=plan,
                    content_matches=content_matches,
                )
                if candidate is None:
                    continue
                candidates_by_commit[commit] = candidate
                stage_candidates.append(candidate)

            ranked_stage = sorted(
                stage_candidates,
                key=lambda candidate: (-candidate.score, candidate.depth, candidate.commit),
            )
            strong_reason = self._stop_reason(stage_candidates)
            stage = GitSearchStage(
                depth=depth,
                opened_commits=len(layer_commits),
                matched_commits=[candidate.commit for candidate in ranked_stage[:MAX_GIT_SEARCH_CANDIDATES]],
                candidate_count=len(stage_candidates),
                command_count=stage_commands,
                duration_seconds=time.monotonic() - stage_started,
                stopped=strong_reason is not None,
                stop_reason=strong_reason,
            )
            stages.append(stage)
            if strong_reason is not None:
                stop_reason = strong_reason
                break
            if len(layer_commits) < depth:
                stage.stopped = True
                stage.stop_reason = "history_exhausted"
                stop_reason = "history_exhausted"
                break
            if commands_executed >= plan.max_commands:
                stop_reason = "command_budget_exhausted"
                break

        ranked_candidates = sorted(
            candidates_by_commit.values(),
            key=lambda candidate: (-candidate.score, candidate.depth, candidate.commit),
        )[: plan.max_candidates]
        if not stages and stop_reason == "max_depth_reached":
            stop_reason = "no_search_layer"
        return GitSearchSummary(
            enabled=True,
            attempted_depths=attempted_depths,
            reached_depth=reached_depth,
            candidate_commits=[candidate.commit for candidate in ranked_candidates],
            candidates=ranked_candidates,
            stages=stages,
            stop_reason=stop_reason,
            commands_executed=commands_executed,
            errors=errors[:4],
        )
