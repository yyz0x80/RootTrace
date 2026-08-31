"""Online GitHub end-to-end smoke evaluation for RootTrace.

This module intentionally stays at the evaluation boundary.  GitHub URLs are
ingested by :mod:`roottrace.github`, repositories are prepared by the shared
revision-pinned checkout helper, and the existing in-process evaluation client
and localization metrics do the RCA work and scoring.  The evaluator never
passes gold labels or fix-PR metadata to the RootTrace client.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from evaluation.gold import GoldStore
from evaluation.manifest import load_manifest as load_swebench_manifest
from evaluation.metrics import CaseMetrics, compute_case_metrics
from evaluation.runner import (
    InProcessRootTraceClient,
    RootTraceClient,
    RootTraceOutcome,
    extract_predicted_files,
)
from roottrace.diagnostics import PipelineDiagnostic
from roottrace.github import (
    GitHubClient,
    GitHubIngestionResult,
    GitHubIngestor,
    GitHubRepositoryRef,
    PreparedGitHubRepository,
    map_review_comment_threads_to_revision,
    parse_github_resource_url,
    prepare_github_repository,
)
from roottrace.incident.schema import (
    IncidentInput,
    Provenance,
    ReviewCommentThread,
)

from .manifest import (
    DEFAULT_MANIFEST_PATH,
    SUITE_NAME,
    GitHubSmokeCase,
    GitHubSmokeManifest,
    load_manifest,
    select_cases,
)

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "roottrace" / "github-smoke"
DEFAULT_RESULTS_DIR = Path(__file__).with_name("results")
DEFAULT_DATA_ROOT = Path.home() / "Datasets" / "roottrace-swebench"
MAX_ERROR_CHARS = 500
MAX_ARTIFACTS = 100

StageStatus = Literal["success", "error", "not_run"]
SmokeStatus = Literal["completed", "error", "manual_review_required"]


class PreparedRepositoryFactory(Protocol):
    """Callable seam used by tests to replace online repository preparation."""

    def __call__(
        self,
        reference: GitHubRepositoryRef,
        revision: str,
        *,
        cache_dir: str | Path,
        clone_url: str | None = None,
        token: str | None = None,
        work_dir: str | Path | None = None,
        history_depth: int = 1,
    ) -> PreparedGitHubRepository:
        """Prepare one disposable checkout."""


class SmokeStage(BaseModel):
    """Auditable status for repository acquisition or checkout."""

    model_config = ConfigDict(extra="forbid")

    status: StageStatus = "not_run"
    operation: Literal["clone", "fetch"] | None = None
    success: bool = False
    final_head: str | None = None
    error: str | None = None


class GitHubSmokeCaseResult(BaseModel):
    """Persisted result for one smoke case."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    suite: str = SUITE_NAME
    case_id: str
    source_type: str
    source_url: str
    regression_issue_url: str | None = None
    repo: str
    base_commit: str
    expected_root_cause_files: list[str] = Field(default_factory=list)
    # Gold is evaluator-only.  It is populated for issue cases after RCA and
    # remains empty for manual PR cases.
    gold_files: list[str] = Field(default_factory=list)
    predicted_files: list[str] = Field(default_factory=list)
    predicted_top5: list[str] = Field(default_factory=list)
    status: SmokeStatus
    result: SmokeStatus
    manual_review_required: bool = False
    ingestion_status: StageStatus = "not_run"
    ingestion_error: str | None = None
    repo_acquisition_status: StageStatus = "not_run"
    repo_acquisition_success: bool = False
    acquisition_operation: Literal["clone", "fetch"] | None = None
    checkout_status: StageStatus = "not_run"
    checkout_success: bool = False
    final_checkout_commit: str | None = None
    acquisition: SmokeStage = Field(default_factory=SmokeStage)
    checkout: SmokeStage = Field(default_factory=SmokeStage)
    error: str | None = None
    diagnostics: list[PipelineDiagnostic] = Field(default_factory=list, max_length=20)
    rca_errors: list[str] = Field(default_factory=list, max_length=20)
    latency_seconds: float | None = Field(default=None, ge=0)
    rca_latency_seconds: float | None = Field(default=None, ge=0)
    llm_calls: int | None = Field(default=None, ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    metrics: CaseMetrics | None = None
    artifacts: list[str] = Field(default_factory=list, max_length=MAX_ARTIFACTS)
    notes: list[str] = Field(default_factory=list, max_length=20)


class PreflightCheck(BaseModel):
    """One no-LLM preflight check."""

    name: str
    case_id: str | None = None
    ok: bool
    warning: bool = False
    detail: str


class GitHubSmokePreflightReport(BaseModel):
    """Machine-readable output from ``--preflight``."""

    schema_version: str = "1.0"
    suite: str = SUITE_NAME
    timestamp: str
    manifest: str
    cache_dir: str
    token_configured: bool
    ok: bool
    checks: list[PreflightCheck]


class GitHubSmokeSummary(BaseModel):
    """Aggregate smoke metrics and operational success counts."""

    schema_version: str = "1.0"
    suite: str = SUITE_NAME
    timestamp: str
    manifest: str
    manifest_sha256: str | None = None
    total_cases: int
    issue_cases: int
    pr_cases: int
    ingestion_success: int
    repo_acquisition_success: int
    checkout_success: int
    issue_top1_correct: int
    issue_top1_total: int
    issue_recall_at_5_correct: int
    issue_recall_at_5_total: int
    issue_top1_file_accuracy: float | None = None
    issue_any_file_recall_at_5: float | None = None
    mean_latency_seconds: float | None = None
    mean_tokens: float | None = None
    exact_token_cases: int = 0
    null_token_cases: int = 0
    manual_review_cases: list[str] = Field(default_factory=list)
    cases_file: str
    summary_file: str | None = None


@dataclass(frozen=True)
class CaseContext:
    """Normalized incident and repository identity used by one RCA run."""

    incident: IncidentInput
    repository: GitHubRepositoryRef
    repository_url: str
    notes: tuple[str, ...] = ()


@dataclass
class PreparedCase:
    """Prepared checkout plus auditable acquisition/checkout statuses."""

    prepared: Any | None
    acquisition: SmokeStage
    checkout: SmokeStage


def _bounded_error(value: object, limit: int = MAX_ERROR_CHARS) -> str:
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "..."


def _timestamp(now: datetime | None = None) -> str:
    current = now or datetime.now(UTC)
    return current.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def _iso_timestamp(now: datetime | None = None) -> str:
    current = now or datetime.now(UTC)
    return current.astimezone(UTC).isoformat()


def _arg(args: argparse.Namespace, name: str, default: Any = None) -> Any:
    return getattr(args, name, default)


def default_cache_dir() -> Path:
    """Return the dedicated online smoke cache path."""
    return Path.home() / ".cache" / "roottrace" / "github-smoke"


def _canonical_clone_url(repository: GitHubRepositoryRef) -> str:
    """Return the only clone origin accepted by the smoke runner."""
    return f"https://github.com/{repository.full_name}.git"


def _cache_mirror_path(cache_dir: Path, repository: GitHubRepositoryRef) -> Path:
    return cache_dir / f"{repository.owner}__{repository.repo}.git"


def _validate_existing_cache_origin(mirror: Path, canonical: str) -> None:
    """Reject a pre-existing mirror whose origin is not canonical GitHub HTTPS."""
    if not mirror.exists():
        return
    if not mirror.is_dir():
        raise ValueError(f"online cache path is not a directory: {mirror.name}")
    try:
        completed = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=mirror,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"cannot inspect existing online cache origin: {exc}") from exc
    if completed.returncode != 0:
        # Let the shared preparation helper report malformed git directories;
        # this branch does not accidentally accept a local repository as an
        # online mirror.
        return
    origin = completed.stdout.strip()
    if origin != canonical:
        raise ValueError(
            f"online cache origin must be {canonical}, got {origin or 'empty'}"
        )


def _list_artifacts(path: Path) -> list[str]:
    if not path.is_dir():
        return []
    files = sorted(
        item.relative_to(path).as_posix()
        for item in path.rglob("*")
        if item.is_file()
    )
    return files[:MAX_ARTIFACTS]


def _read_head(repo: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    head = completed.stdout.strip()
    return head or None


def _same_revision(actual: str | None, expected: str) -> bool:
    if not actual:
        return False
    left = actual.lower()
    right = expected.lower()
    return left == right or left.startswith(right) or right.startswith(left)


def _close_prepared(prepared: Any | None) -> None:
    close = getattr(prepared, "close", None)
    if callable(close):
        close()


def _assert_fetched_reference(
    fetched: Any,
    *,
    case: GitHubSmokeCase,
    expected_kind: Literal["issue", "pull_request"],
    label: str,
) -> None:
    reference = getattr(fetched, "reference", None)
    if reference is None or reference.kind != expected_kind:
        raise ValueError(f"{label} did not return a GitHub {expected_kind}")
    if reference.repository.full_name != case.repo:
        raise ValueError(
            f"{label} repository {reference.repository.full_name} does not match {case.repo}"
        )


def _redact_text(value: str | None, forbidden: str | None) -> str | None:
    """Remove evaluator-only fix-PR references from incident evidence."""
    if value is None or not forbidden:
        return value
    reference = parse_github_resource_url(forbidden)
    marker = "[evaluator-only fix evidence omitted]"
    redacted = value.replace(forbidden, marker)
    # GitHub comments commonly refer to a PR using only ``#123`` or
    # ``owner/repo#123``.  Those spellings must not bypass the evaluator/Agent
    # boundary simply because the manifest stores the canonical URL.
    number = str(reference.number)
    patterns = (
        rf"(?<![A-Za-z0-9_])#{re.escape(number)}\b",
        rf"(?<![A-Za-z0-9_.-]){re.escape(reference.repository.full_name)}#{re.escape(number)}\b",
        rf"(?<![A-Za-z0-9_.-])pull/{re.escape(number)}\b",
    )
    for pattern in patterns:
        redacted = re.sub(pattern, marker, redacted)
    return redacted


def _redact_review_threads(
    threads: list[ReviewCommentThread],
    forbidden: str | None,
) -> list[ReviewCommentThread]:
    """Copy review threads while removing evaluator-only references from text."""
    redacted_threads: list[ReviewCommentThread] = []
    for thread in threads:
        comments = []
        for comment in thread.comments:
            excerpt = _redact_text(comment.excerpt, forbidden) or ""
            source = _redact_text(comment.provenance.source, forbidden)
            provenance = (
                comment.provenance
                if source == comment.provenance.source
                else comment.provenance.model_copy(update={"source": source or ""})
            )
            comments.append(
                comment.model_copy(
                    update={
                        "excerpt": excerpt,
                        "provenance": provenance,
                    }
                )
            )
        redacted_threads.append(thread.model_copy(update={"comments": comments}))
    return redacted_threads


def _compose_pr_incident(
    case: GitHubSmokeCase,
    issue_result: GitHubIngestionResult,
    pr_result: GitHubIngestionResult,
) -> IncidentInput:
    """Combine regression issue evidence with bad-PR context only.

    The resulting incident deliberately excludes ``expected_files`` and the
    fix-PR URL.  The bad PR diff/comments/changed files are useful evidence;
    the later regression issue supplies the observed failure narrative.
    """
    forbidden = case.fix_evidence_url
    logs = [
        _redact_text(log, forbidden) or ""
        for log in [*issue_result.incident.logs, *pr_result.incident.logs]
    ]
    logs = [log for log in logs if log][:10]
    related_commits: list[str] = []
    for value in (pr_result.head_commit, case.bad_pr_head_commit):
        if value and not any(_same_revision(value, existing) for existing in related_commits):
            related_commits.append(value)
    labels = sorted(set(issue_result.incident.labels) | set(pr_result.incident.labels))
    diff = _redact_text(pr_result.incident.diff, forbidden)
    changed_files = list(dict.fromkeys(pr_result.changed_files))
    review_threads = map_review_comment_threads_to_revision(
        _redact_review_threads(pr_result.incident.review_threads, forbidden),
        base_commit=case.base_commit,
    )
    review_comment_truncation = pr_result.incident.review_comment_truncation.model_copy(
        update={
            "locations_unmapped": sum(
                comment.location_mapping == "unmapped"
                for thread in review_threads
                for comment in thread.comments
            )
        }
    )
    return IncidentInput(
        id=issue_result.incident.id,
        repo=case.repo,
        base_commit=case.base_commit,
        resource_kind="pull_request",
        title=_redact_text(issue_result.incident.title, forbidden),
        problem=_redact_text(issue_result.incident.problem, forbidden) or "",
        logs=logs,
        diff=diff,
        labels=labels,
        related_commits=related_commits,
        changed_files=changed_files,
        review_threads=review_threads,
        review_comment_truncation=review_comment_truncation,
        provenance=Provenance(
            source=case.regression_issue_url or issue_result.incident.provenance.source,
            tool="github_smoke10",
            commit=case.base_commit,
        ),
    )


def build_case_context(
    case: GitHubSmokeCase,
    ingestor: GitHubIngestor,
) -> CaseContext:
    """Fetch and normalize one case's bounded GitHub context."""
    if case.source_type == "github_issue":
        fetched = ingestor.fetch(case.source_url)
        _assert_fetched_reference(
            fetched,
            case=case,
            expected_kind="issue",
            label="source_url",
        )
        normalized = ingestor.normalize(fetched, base_commit=case.base_commit)
        if normalized.base_commit.lower() != case.base_commit.lower():
            raise ValueError("issue normalization selected a different base commit")
        canonical = _canonical_clone_url(normalized.reference.repository)
        if normalized.repository_url != canonical:
            raise ValueError("GitHub ingestion returned a non-canonical clone URL")
        return CaseContext(
            incident=normalized.incident,
            repository=normalized.reference.repository,
            repository_url=canonical,
            notes=tuple(normalized.notes),
        )

    pr_fetched = ingestor.fetch(case.source_url)
    _assert_fetched_reference(
        pr_fetched,
        case=case,
        expected_kind="pull_request",
        label="source_url",
    )
    pr_normalized = ingestor.normalize(pr_fetched)
    if case.bad_pr_base_commit and not _same_revision(
        pr_normalized.base_commit, case.bad_pr_base_commit
    ):
        raise ValueError("bad PR base commit does not match the pinned manifest prefix")
    if case.bad_pr_head_commit and not _same_revision(
        pr_normalized.head_commit, case.bad_pr_head_commit
    ):
        raise ValueError("bad PR head commit does not match the pinned manifest prefix")
    detail = getattr(pr_fetched, "detail", None)
    merge_commit = getattr(detail, "merge_commit_sha", None)
    if merge_commit and not _same_revision(merge_commit, case.base_commit):
        raise ValueError("bad PR merge commit does not match the pinned analysis commit")

    if not case.regression_issue_url:
        raise ValueError("PR case has no regression issue URL")
    issue_fetched = ingestor.fetch(case.regression_issue_url)
    _assert_fetched_reference(
        issue_fetched,
        case=case,
        expected_kind="issue",
        label="regression_issue_url",
    )
    issue_normalized = ingestor.normalize(
        issue_fetched,
        base_commit=case.base_commit,
    )
    if issue_normalized.base_commit.lower() != case.base_commit.lower():
        raise ValueError("regression issue normalization selected a different analysis commit")
    incident = _compose_pr_incident(case, issue_normalized, pr_normalized)
    canonical = _canonical_clone_url(pr_normalized.reference.repository)
    if pr_normalized.repository_url != canonical:
        raise ValueError("GitHub ingestion returned a non-canonical clone URL")
    return CaseContext(
        incident=incident,
        repository=pr_normalized.reference.repository,
        repository_url=canonical,
        notes=(
            *issue_normalized.notes,
            f"bad PR context fetched from {case.source_url}",
            f"analysis checkout pinned to {case.base_commit}",
        ),
    )


def prepare_case_repository(
    case: GitHubSmokeCase,
    context: CaseContext,
    *,
    cache_dir: str | Path,
    prepare_fn: PreparedRepositoryFactory = prepare_github_repository,
) -> PreparedCase:
    """Prepare an anonymous public clone and verify the final checkout SHA."""
    cache = Path(cache_dir).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    mirror = _cache_mirror_path(cache, context.repository)
    operation: Literal["clone", "fetch"] = "fetch" if mirror.exists() else "clone"
    canonical = _canonical_clone_url(context.repository)
    if context.repository_url != canonical:
        raise ValueError("repository URL is not the canonical HTTPS GitHub clone URL")
    _validate_existing_cache_origin(mirror, canonical)

    prepared: Any | None = None
    try:
        policy = getattr(context.incident, "git_verification_policy", None)
        history_depth = getattr(policy, "history_depth", 1)
        prepared = prepare_fn(
            context.repository,
            case.base_commit,
            cache_dir=cache,
            clone_url=canonical,
            token=None,
            history_depth=history_depth,
        )
    except Exception as exc:  # noqa: BLE001 - persisted as per-case status
        message = _bounded_error(exc)
        acquisition_failed = (
            "git clone --bare" in message
            or "git fetch --prune origin" in message
        )
        acquisition_success = False
        if not acquisition_failed and mirror.is_dir():
            try:
                _validate_existing_cache_origin(mirror, canonical)
            except ValueError:
                acquisition_success = False
            else:
                acquisition_success = True
        acquisition = SmokeStage(
            status="success" if acquisition_success else "error",
            operation=operation,
            success=acquisition_success,
            error=None if acquisition_success else message,
        )
        checkout = SmokeStage(
            status="error",
            operation=operation,
            success=False,
            error=message,
        )
        return PreparedCase(None, acquisition, checkout)

    acquisition = SmokeStage(
        status="success",
        operation=operation,
        success=True,
    )
    final_head = getattr(prepared, "revision", None)
    if not isinstance(final_head, str) or not final_head:
        repo_path = getattr(prepared, "repo", None)
        if isinstance(repo_path, Path):
            final_head = _read_head(repo_path)
    if not _same_revision(final_head, case.base_commit):
        error = (
            f"prepared checkout HEAD mismatch: {final_head or 'unknown'} != "
            f"{case.base_commit}"
        )
        checkout = SmokeStage(
            status="error",
            operation=operation,
            success=False,
            final_head=final_head,
            error=error,
        )
        _close_prepared(prepared)
        return PreparedCase(None, acquisition, checkout)
    checkout = SmokeStage(
        status="success",
        operation=operation,
        success=True,
        final_head=final_head,
    )
    return PreparedCase(prepared, acquisition, checkout)


def _result_base(
    case: GitHubSmokeCase,
    *,
    status: SmokeStatus,
    context: CaseContext | None = None,
    error: str | None = None,
    notes: list[str] | None = None,
) -> GitHubSmokeCaseResult:
    expected = case.expected_root_cause_files
    return GitHubSmokeCaseResult(
        case_id=case.instance_id,
        source_type=case.source_type,
        source_url=case.source_url,
        regression_issue_url=case.regression_issue_url,
        repo=case.repo,
        base_commit=case.base_commit,
        expected_root_cause_files=expected,
        status=status,
        result=status,
        manual_review_required=case.manual_review_required,
        error=error,
        notes=[*(context.notes if context else ()), *(notes or [])],
    )


def _outcome_from_value(value: Any) -> RootTraceOutcome:
    if isinstance(value, RootTraceOutcome):
        return value
    return RootTraceOutcome.model_validate(value)


def _resolve_gold_files(
    case: GitHubSmokeCase,
    gold_path: Path | None,
) -> list[str]:
    """Resolve issue gold only after the RootTrace invocation has returned."""
    if case.source_type != "github_issue":
        return []
    if gold_path is not None and gold_path.is_file():
        return GoldStore(gold_path).gold_files(case.instance_id)
    return list(case.gold_files)


def run_case(
    case: GitHubSmokeCase,
    *,
    ingestor: GitHubIngestor,
    roottrace_client: RootTraceClient,
    cache_dir: str | Path,
    case_dir: str | Path,
    model: str | None,
    gold_path: str | Path | None = None,
    prepare_fn: PreparedRepositoryFactory = prepare_github_repository,
) -> GitHubSmokeCaseResult:
    """Run one case with independent ingestion, checkout, RCA, and cleanup."""
    started = time.monotonic()
    artifact_root = Path(case_dir)
    artifact_root.mkdir(parents=True, exist_ok=True)
    context: CaseContext | None = None
    try:
        context = build_case_context(case, ingestor)
    except Exception as exc:  # noqa: BLE001 - isolate one online case
        elapsed = time.monotonic() - started
        result = _result_base(
            case,
            status="error",
            error=_bounded_error(exc),
            notes=["ingestion failed"],
        )
        result.ingestion_status = "error"
        result.ingestion_error = result.error
        result.latency_seconds = elapsed
        return result

    (artifact_root / "incident.json").write_text(
        json.dumps(context.incident.model_dump(mode="json"), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    try:
        prepared_case = prepare_case_repository(
            case,
            context,
            cache_dir=cache_dir,
            prepare_fn=prepare_fn,
        )
    except Exception as exc:  # noqa: BLE001 - isolate preparation failures
        message = _bounded_error(exc)
        prepared_case = PreparedCase(
            None,
            SmokeStage(status="error", operation=None, error=message),
            SmokeStage(status="error", operation=None, error=message),
        )
    result = _result_base(
        case,
        status="error",
        context=context,
    )
    result.ingestion_status = "success"
    result.acquisition = prepared_case.acquisition
    result.checkout = prepared_case.checkout
    result.repo_acquisition_status = prepared_case.acquisition.status
    result.repo_acquisition_success = prepared_case.acquisition.success
    result.acquisition_operation = prepared_case.acquisition.operation
    result.checkout_status = prepared_case.checkout.status
    result.checkout_success = prepared_case.checkout.success
    result.final_checkout_commit = prepared_case.checkout.final_head
    if prepared_case.prepared is None:
        result.error = prepared_case.checkout.error or prepared_case.acquisition.error
        result.latency_seconds = time.monotonic() - started
        return result

    prepared = prepared_case.prepared
    try:
        repo_path = getattr(prepared, "repo", None)
        if not isinstance(repo_path, Path):
            outcome = RootTraceOutcome(
                status="error",
                error="prepared repository did not expose a local repo path",
                latency_seconds=0.0,
            )
        else:
            rca_started = time.monotonic()
            try:
                raw_outcome = roottrace_client.run(
                    case_id=case.instance_id,
                    repo=repo_path,
                    incident=context.incident,
                    output_dir=artifact_root / "roottrace",
                    model=model,
                )
                outcome = _outcome_from_value(raw_outcome)
            except Exception as exc:  # noqa: BLE001 - isolate RootTrace failures
                outcome = RootTraceOutcome(
                    status="error",
                    error=_bounded_error(exc),
                    latency_seconds=time.monotonic() - rca_started,
                )
        result.rca_latency_seconds = outcome.latency_seconds
        result.llm_calls = outcome.llm_calls
        result.prompt_tokens = outcome.prompt_tokens
        result.completion_tokens = outcome.completion_tokens
        result.reasoning_tokens = outcome.reasoning_tokens
        result.diagnostics = list(outcome.diagnostics)
        result.rca_errors = list(outcome.errors)
        result.predicted_files = extract_predicted_files(outcome.report)
        result.predicted_top5 = result.predicted_files[:5]
        result.artifacts = _list_artifacts(artifact_root)
        if outcome.status == "error":
            result.error = outcome.error or "RootTrace invocation failed"
        if case.source_type == "github_pr":
            if outcome.status == "error":
                result.status = "error"
                result.result = "error"
            else:
                result.status = "manual_review_required"
                result.result = "manual_review_required"
                result.manual_review_required = True
        else:
            try:
                resolved_gold = _resolve_gold_files(
                    case,
                    Path(gold_path).expanduser().resolve()
                    if gold_path is not None
                    else None,
                )
            except Exception as exc:  # noqa: BLE001 - scoring failure is explicit
                result.status = "error"
                result.result = "error"
                result.error = f"gold resolution failed: {_bounded_error(exc)}"
                resolved_gold = []
            result.gold_files = resolved_gold
            metric_status = (
                "completed"
                if outcome.status == "completed" and result.error is None
                else "error"
            )
            result.metrics = compute_case_metrics(
                instance_id=case.instance_id,
                predicted_files=result.predicted_files,
                gold_files=resolved_gold,
                status=metric_status,
                error=result.error,
                latency_seconds=None,
                llm_calls=result.llm_calls,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                reasoning_tokens=result.reasoning_tokens,
            )
            if metric_status == "completed":
                result.status = "completed"
                result.result = "completed"
            else:
                result.status = "error"
                result.result = "error"
        result.total_tokens = (
            result.prompt_tokens + result.completion_tokens
            if result.prompt_tokens is not None and result.completion_tokens is not None
            else None
        )
        result.latency_seconds = time.monotonic() - started
        if result.metrics is not None:
            result.metrics.latency_seconds = result.latency_seconds
    finally:
        _close_prepared(prepared)
    return result


def _find_data_file(data_root: Path | None, candidates: tuple[str, ...]) -> Path | None:
    if data_root is None:
        return None
    for relative in candidates:
        candidate = data_root / relative
        if candidate.is_file():
            return candidate
    return None


def external_data_paths(data_root: str | Path | None) -> tuple[Path | None, Path | None]:
    """Find optional dev50 and verified-gold files without guessing mirrors."""
    if data_root is None:
        root = DEFAULT_DATA_ROOT if DEFAULT_DATA_ROOT.is_dir() else None
    else:
        root = Path(data_root).expanduser().resolve()
    return (
        _find_data_file(root, ("manifests/dev50.json", "dev50.json")),
        _find_data_file(root, ("gold/verified_gold.jsonl", "verified_gold.jsonl")),
    )


def validate_external_data(
    manifest: GitHubSmokeManifest,
    *,
    data_root: str | Path | None,
) -> list[PreflightCheck]:
    """Validate issue IDs/revisions and copied gold labels when data exists."""
    dev50_path, gold_path = external_data_paths(data_root)
    checks: list[PreflightCheck] = []
    explicit_root = data_root is not None
    if dev50_path is None:
        checks.append(
            PreflightCheck(
                name="dev50 manifest",
                ok=not explicit_root,
                warning=not explicit_root,
                detail=(
                    "not found; checked-in gold labels will be used"
                    if not explicit_root
                    else "dev50.json not found under --data-root"
                ),
            )
        )
    else:
        try:
            dev50 = load_swebench_manifest(dev50_path)
            public_by_id = {item.instance_id: item for item in dev50.instances}
            mismatches = [
                case.instance_id
                for case in manifest.instances
                if case.source_type == "github_issue"
                and (
                    case.instance_id not in public_by_id
                    or (public_by_id[case.instance_id].repo, public_by_id[case.instance_id].base_commit)
                    != (case.repo, case.base_commit)
                )
            ]
            checks.append(
                PreflightCheck(
                    name="dev50 issue coverage",
                    ok=not mismatches,
                    detail="ok" if not mismatches else f"mismatch={sorted(mismatches)}",
                )
            )
        except Exception as exc:  # noqa: BLE001 - report data validation failure
            checks.append(
                PreflightCheck(
                    name="dev50 issue coverage",
                    ok=False,
                    detail=_bounded_error(exc),
                )
            )

    if gold_path is None:
        checks.append(
            PreflightCheck(
                name="verified gold",
                ok=not explicit_root,
                warning=not explicit_root,
                detail=(
                    "not found; checked-in gold labels will be used"
                    if not explicit_root
                    else "verified_gold.jsonl not found under --data-root"
                ),
            )
        )
    else:
        mismatches: list[str] = []
        try:
            store = GoldStore(gold_path)
            for case in manifest.instances:
                if case.source_type != "github_issue":
                    continue
                if sorted(store.gold_files(case.instance_id)) != sorted(case.gold_files):
                    mismatches.append(case.instance_id)
            checks.append(
                PreflightCheck(
                    name="verified gold labels",
                    ok=not mismatches,
                    detail="ok" if not mismatches else f"mismatch={sorted(mismatches)}",
                )
            )
        except Exception as exc:  # noqa: BLE001 - report data validation failure
            checks.append(
                PreflightCheck(
                    name="verified gold labels",
                    ok=False,
                    detail=_bounded_error(exc),
                )
            )
    return checks


def run_preflight(
    manifest: GitHubSmokeManifest,
    *,
    selected: list[GitHubSmokeCase],
    github_client: GitHubClient,
    cache_dir: str | Path,
    token: str | None,
    data_root: str | Path | None = None,
    prepare_fn: PreparedRepositoryFactory = prepare_github_repository,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    now: datetime | None = None,
    include_review_comments: bool = False,
) -> GitHubSmokePreflightReport:
    """Run manifest/API/cache/checkout checks without creating a provider."""
    checks: list[PreflightCheck] = []
    checks.extend(
        [
            PreflightCheck(
                name="manifest schema",
                ok=True,
                detail=f"{len(manifest.instances)} case(s); 8 issues + 2 PRs",
            ),
            PreflightCheck(
                name="manifest URL/repo identity",
                ok=True,
                detail="canonical GitHub resource URLs match their repository and kind",
            ),
        ]
    )
    token_configured = bool(token)
    checks.append(
        PreflightCheck(
            name="GITHUB_TOKEN",
            ok=True,
            warning=not token_configured,
            detail=(
                "configured"
                if token_configured
                else "not set; public GitHub API access may hit the unauthenticated rate limit"
            ),
        )
    )
    cache = Path(cache_dir).expanduser().resolve()
    try:
        cache.mkdir(parents=True, exist_ok=True)
        cache_ok = cache.is_dir() and os.access(cache, os.W_OK)
    except OSError as exc:
        cache_ok = False
        cache_error = _bounded_error(exc)
    else:
        cache_error = "ok"
    checks.append(
        PreflightCheck(
            name="online cache directory",
            ok=cache_ok,
            detail=cache_error,
        )
    )
    checks.extend(validate_external_data(manifest, data_root=data_root))

    if include_review_comments:
        ingestor = GitHubIngestor(
            github_client,
            include_review_comments=True,
        )
    else:
        ingestor = GitHubIngestor(github_client)
    for case in selected:
        context: CaseContext | None = None
        try:
            context = build_case_context(case, ingestor)
            checks.append(
                PreflightCheck(
                    name="GitHub API ingestion",
                    case_id=case.instance_id,
                    ok=True,
                    detail="source URL fetched and normalized",
                )
            )
        except Exception as exc:  # noqa: BLE001 - continue checking other cases
            checks.append(
                PreflightCheck(
                    name="GitHub API ingestion",
                    case_id=case.instance_id,
                    ok=False,
                    detail=_bounded_error(exc),
                )
            )
            continue
        try:
            prepared_case = prepare_case_repository(
                case,
                context,
                cache_dir=cache,
                prepare_fn=prepare_fn,
            )
        except Exception as exc:  # noqa: BLE001 - continue checking other cases
            message = _bounded_error(exc)
            prepared_case = PreparedCase(
                None,
                SmokeStage(status="error", error=message),
                SmokeStage(status="error", error=message),
            )
        checks.append(
            PreflightCheck(
                name="online repository acquisition",
                case_id=case.instance_id,
                ok=prepared_case.acquisition.success,
                detail=(
                    f"{prepared_case.acquisition.operation or 'unknown'} succeeded"
                    if prepared_case.acquisition.success
                    else prepared_case.acquisition.error or "acquisition failed"
                ),
            )
        )
        checks.append(
            PreflightCheck(
                name="historical checkout",
                case_id=case.instance_id,
                ok=prepared_case.checkout.success,
                detail=(
                    f"HEAD={prepared_case.checkout.final_head}"
                    if prepared_case.checkout.success
                    else prepared_case.checkout.error or "checkout failed"
                ),
            )
        )
        _close_prepared(prepared_case.prepared)

    ok = all(check.ok for check in checks)
    return GitHubSmokePreflightReport(
        timestamp=_iso_timestamp(now),
        manifest=str(Path(manifest_path)),
        cache_dir=str(cache),
        token_configured=token_configured,
        ok=ok,
        checks=checks,
    )


def _summary_for_results(
    results: list[GitHubSmokeCaseResult],
    *,
    manifest_path: Path,
    timestamp: str,
    cases_file: Path,
) -> GitHubSmokeSummary:
    issues = [item for item in results if item.source_type == "github_issue"]
    prs = [item for item in results if item.source_type == "github_pr"]
    metric_items = [item.metrics for item in issues if item.metrics is not None]
    top1_correct = sum(1 for metric in metric_items if metric and metric.top_1_file_accuracy)
    recall_correct = sum(1 for metric in metric_items if metric and metric.any_file_recall_at_5)
    token_values = [item.total_tokens for item in results if item.total_tokens is not None]
    latencies = [item.latency_seconds for item in results if item.latency_seconds is not None]
    return GitHubSmokeSummary(
        timestamp=timestamp,
        manifest=str(manifest_path),
        total_cases=len(results),
        issue_cases=len(issues),
        pr_cases=len(prs),
        ingestion_success=sum(item.ingestion_status == "success" for item in results),
        repo_acquisition_success=sum(item.repo_acquisition_success for item in results),
        checkout_success=sum(item.checkout_success for item in results),
        issue_top1_correct=top1_correct,
        issue_top1_total=len(issues),
        issue_recall_at_5_correct=recall_correct,
        issue_recall_at_5_total=len(issues),
        issue_top1_file_accuracy=(top1_correct / len(issues)) if issues else None,
        issue_any_file_recall_at_5=(recall_correct / len(issues)) if issues else None,
        mean_latency_seconds=fmean(latencies) if latencies else None,
        mean_tokens=fmean(token_values) if token_values else None,
        exact_token_cases=len(token_values),
        null_token_cases=sum(item.total_tokens is None for item in results),
        manual_review_cases=[item.case_id for item in prs],
        cases_file=str(cases_file),
    )


def _print_case_result(result: GitHubSmokeCaseResult) -> None:
    print(
        f"[{result.case_id}] {result.source_type} url={result.source_url} "
        f"repo={result.repo} expected={result.expected_root_cause_files or '-'} "
        f"result={result.result} "
        f"ingestion={result.ingestion_status} checkout={result.checkout_status} "
        f"predicted_top5={result.predicted_top5 or '-'} "
        f"latency={result.latency_seconds if result.latency_seconds is not None else 'n/a'} "
        f"tokens={result.total_tokens if result.total_tokens is not None else 'n/a'}"
    )


def _print_summary(summary: GitHubSmokeSummary) -> None:
    print("github_smoke10")
    print(f"\nCases: {summary.total_cases}")
    print(f"\nIngestion: {summary.ingestion_success}/{summary.total_cases}")
    print(
        f"Repo acquisition: {summary.repo_acquisition_success}/{summary.total_cases}"
    )
    print(f"Checkout: {summary.checkout_success}/{summary.total_cases}")
    print("\nIssue cases:")
    print(f"Top-1: {summary.issue_top1_correct}/{summary.issue_top1_total}")
    print(
        "Recall@5: "
        f"{summary.issue_recall_at_5_correct}/{summary.issue_recall_at_5_total}"
    )
    print("\nPR smoke:")
    for case_id in summary.manual_review_cases:
        print(f"{case_id}: manual_review_required")
    print(
        f"\nMean latency: {summary.mean_latency_seconds if summary.mean_latency_seconds is not None else 'n/a'}"
    )
    print(f"Mean tokens: {summary.mean_tokens if summary.mean_tokens is not None else 'n/a'}")
    print(f"Cases JSONL: {summary.cases_file}")
    if summary.summary_file:
        print(f"Summary: {summary.summary_file}")


def _write_preflight_report(report: GitHubSmokePreflightReport, path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _args_to_paths(args: argparse.Namespace) -> tuple[Path, Path, Path | None]:
    manifest_path = Path(
        _arg(args, "manifest", DEFAULT_MANIFEST_PATH) or DEFAULT_MANIFEST_PATH
    ).expanduser().resolve()
    cache_dir = Path(
        _arg(args, "cache_dir", None) or default_cache_dir()
    ).expanduser().resolve()
    data_root_arg = _arg(args, "data_root", None)
    data_root = (
        Path(data_root_arg).expanduser().resolve() if data_root_arg is not None else None
    )
    return manifest_path, cache_dir, data_root


def run_from_args(
    args: argparse.Namespace,
    *,
    github_client: GitHubClient | None = None,
    roottrace_client: RootTraceClient | None = None,
    client: RootTraceClient | None = None,
    prepare_fn: PreparedRepositoryFactory = prepare_github_repository,
    now: datetime | None = None,
) -> int:
    """Run ``github_smoke10`` from parsed CLI arguments."""
    suite = _arg(args, "suite", SUITE_NAME)
    if suite != SUITE_NAME:
        print(f"evaluation error: unsupported suite: {suite}", file=sys.stderr)
        return 2
    manifest_path, cache_dir, data_root = _args_to_paths(args)
    try:
        manifest = load_manifest(manifest_path)
        selected = select_cases(manifest, _arg(args, "case", None))
    except (FileNotFoundError, TypeError, ValueError) as exc:
        print(f"evaluation error: {exc}", file=sys.stderr)
        return 2
    token = _arg(args, "github_token", None)
    if token is None:
        token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print(
            "warning: GITHUB_TOKEN is not set; public GitHub API access is rate-limited",
            file=sys.stderr,
        )
    if github_client is not None:
        api_client = github_client
    else:
        try:
            api_client = GitHubClient(
                token=token,
                timeout=float(_arg(args, "github_timeout", 30.0)),
            )
        except (TypeError, ValueError) as exc:
            print(f"evaluation error: invalid GitHub client configuration: {exc}", file=sys.stderr)
            return 2

    if bool(_arg(args, "preflight", False)):
        report = run_preflight(
            manifest,
            selected=selected,
            github_client=api_client,
            cache_dir=cache_dir,
            token=token,
            data_root=data_root,
            prepare_fn=prepare_fn,
            manifest_path=manifest_path,
            now=now,
            include_review_comments=bool(
                _arg(args, "include_review_comments", False)
            ),
        )
        _write_preflight_report(
            report,
            Path(_arg(args, "preflight_json")).expanduser().resolve()
            if _arg(args, "preflight_json")
            else None,
        )
        for check in report.checks:
            marker = "PASS" if check.ok and not check.warning else "WARN" if check.ok else "FAIL"
            suffix = f" [{check.case_id}]" if check.case_id else ""
            print(f"[{marker}] {check.name}{suffix}: {check.detail}")
        print(f"preflight {'passed' if report.ok else 'failed'}")
        return 0 if report.ok else 2

    output_root = Path(
        _arg(args, "output_dir", None) or DEFAULT_RESULTS_DIR
    ).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = _timestamp(now)
    run_root = output_root / f"{SUITE_NAME}_{stamp}"
    case_root = run_root / "cases"
    case_root.mkdir(parents=True, exist_ok=True)
    active_client = roottrace_client or client or InProcessRootTraceClient()
    _dev50_path, gold_path = external_data_paths(data_root)
    results: list[GitHubSmokeCaseResult] = []
    if bool(_arg(args, "include_review_comments", False)):
        ingestor = GitHubIngestor(api_client, include_review_comments=True)
    else:
        ingestor = GitHubIngestor(api_client)
    for case in selected:
        try:
            result = run_case(
                case,
                ingestor=ingestor,
                roottrace_client=active_client,
                cache_dir=cache_dir,
                case_dir=case_root / case.instance_id,
                model=_arg(args, "model", None),
                gold_path=gold_path,
                prepare_fn=prepare_fn,
            )
        except Exception as exc:  # noqa: BLE001 - one case cannot abort the suite
            result = _result_base(case, status="error", error=_bounded_error(exc))
        results.append(result)
        _print_case_result(result)

    cases_file = output_root / f"{SUITE_NAME}_{stamp}.jsonl"
    with cases_file.open("w", encoding="utf-8") as stream:
        for result in results:
            stream.write(json.dumps(result.model_dump(mode="json"), sort_keys=True) + "\n")
    summary = _summary_for_results(
        results,
        manifest_path=manifest_path,
        timestamp=stamp,
        cases_file=cases_file,
    )
    manifest_hash: str | None = None
    try:
        import hashlib

        manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    except OSError:
        pass
    summary.manifest_sha256 = manifest_hash
    summary_path = output_root / f"{SUITE_NAME}_{stamp}.summary.json"
    summary.summary_file = str(summary_path)
    summary_path.write_text(
        json.dumps(summary.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _print_summary(summary)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the lightweight live-smoke CLI parser."""
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.github.run",
        description="RootTrace online GitHub github_smoke10 evaluation",
    )
    parser.add_argument("--suite", default=SUITE_NAME, choices=[SUITE_NAME])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--case", help="run one manifest case by ID")
    parser.add_argument("--preflight", action="store_true", help="check API/cache/checkout without LLM calls")
    parser.add_argument("--preflight-json", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None, help="optional SWE-bench root for dev50/gold validation")
    parser.add_argument("--cache-dir", type=Path, default=None, help="dedicated online clone cache (default: ~/.cache/roottrace/github-smoke)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--github-token", default=None, help="GitHub token (otherwise GITHUB_TOKEN)")
    parser.add_argument("--github-timeout", type=float, default=30.0)
    parser.add_argument("--model", default=None, help="RootTrace model override")
    parser.add_argument(
        "--include-review-comments",
        action="store_true",
        help="Include bounded pull request review-comment evidence",
    )
    return parser


__all__ = [
    "DEFAULT_CACHE_DIR",
    "DEFAULT_DATA_ROOT",
    "DEFAULT_MANIFEST_PATH",
    "DEFAULT_RESULTS_DIR",
    "CaseContext",
    "GitHubSmokeCaseResult",
    "GitHubSmokePreflightReport",
    "GitHubSmokeSummary",
    "PreflightCheck",
    "SmokeStage",
    "build_case_context",
    "build_parser",
    "default_cache_dir",
    "external_data_paths",
    "prepare_case_repository",
    "run_case",
    "run_from_args",
    "run_preflight",
    "validate_external_data",
]
