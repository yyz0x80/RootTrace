"""Normalize GitHub issue and pull request metadata into RootTrace inputs."""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, Field

from roottrace.incident.loader import LoadedIncident
from roottrace.incident.schema import (
    MAX_DIFF_CHARS,
    MAX_LOG_CHARS,
    MAX_PROBLEM_CHARS,
    MAX_TITLE_CHARS,
    IncidentInput,
    Provenance,
    validate_commit_sha,
)

from .client import GitHubClient
from .models import (
    GitHubComment,
    GitHubCommit,
    GitHubIssueDetail,
    GitHubPullRequestDetail,
    GitHubPullRequestFile,
    GitHubPullRequestReview,
    GitHubPullRequestReviewComment,
    GitHubResource,
    GitHubResourceRef,
    GitHubUser,
    parse_github_resource_url,
)

MAX_INGESTION_COMMENTS = 10
MAX_INGESTION_COMMENT_CHARS = 4_000
MAX_INGESTION_COMMIT_CHARS = 2_000
MAX_INGESTION_FILE_NAMES = 300
_TRUNCATED_MARKER = "\n...[truncated: {n} chars omitted]"


class GitHubFetchedResource(BaseModel):
    """Bounded GitHub API data collected before repository preparation."""

    reference: GitHubResource
    detail: GitHubIssueDetail | GitHubPullRequestDetail
    comments: list[GitHubComment] = Field(default_factory=list)
    review_comments: list[GitHubPullRequestReviewComment] = Field(default_factory=list)
    reviews: list[GitHubPullRequestReview] = Field(default_factory=list)
    files: list[GitHubPullRequestFile] = Field(default_factory=list)
    commits: list[GitHubCommit] = Field(default_factory=list)


class GitHubIngestionResult(BaseModel):
    """Normalized RootTrace input and the selected repository revision."""

    reference: GitHubResource
    incident: IncidentInput
    repository_url: str
    base_commit: str
    head_commit: str | None = None
    state: str | None = None
    labels: list[str] = Field(default_factory=list, max_length=100)
    changed_files: list[str] = Field(default_factory=list, max_length=3_000)
    revision_kind: Literal["default_branch", "pull_request_base"]
    notes: list[str] = Field(default_factory=list, max_length=20)

    @property
    def loaded_incident(self) -> LoadedIncident:
        """Expose the normalized input at the existing pipeline boundary."""
        return LoadedIncident(incident=self.incident, notes=list(self.notes))


def github_clone_url(reference: GitHubResourceRef) -> str:
    """Return the HTTPS clone URL for a GitHub repository reference."""
    return f"https://github.com/{reference.repository.full_name}.git"


def _bounded_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    omitted = len(value) - limit
    marker = _TRUNCATED_MARKER.format(n=omitted)
    return value[: max(0, limit - len(marker))] + marker


def _bounded_title(value: str) -> str:
    return _bounded_text(value.strip(), MAX_TITLE_CHARS)


def _bounded_problem(value: str) -> str:
    return _bounded_text(value.strip(), MAX_PROBLEM_CHARS)


def _user_name(user: GitHubUser | None) -> str:
    return user.login if user and user.login else "unknown"


def _comment_log(comment: GitHubComment) -> str:
    body = _bounded_text((comment.body or "").strip(), MAX_INGESTION_COMMENT_CHARS)
    return f"GitHub comment by @{_user_name(comment.user)}:\n{body}".strip()


def _review_comment_log(comment: GitHubPullRequestReviewComment) -> str:
    """Render bounded code-review text with source file/line provenance."""
    body = _bounded_text((comment.body or "").strip(), MAX_INGESTION_COMMENT_CHARS)
    path = (comment.path or "").strip()
    line = comment.line if comment.line is not None else comment.original_line
    location = f" on {path}:{line}" if path and line is not None else f" on {path}" if path else ""
    return f"GitHub code review comment by @{_user_name(comment.user)}{location}:\n{body}".strip()


def _commit_log(commit: GitHubCommit) -> str:
    message = _bounded_text(
        (commit.commit.message if commit.commit else "").strip(),
        MAX_INGESTION_COMMIT_CHARS,
    )
    return f"PR commit {commit.sha}: {message}".strip()


def _pr_diff(files: list[GitHubPullRequestFile]) -> str | None:
    sections: list[str] = []
    for file in files:
        if not file.patch:
            continue
        filename = file.filename
        sections.append(
            f"diff --git a/{filename} b/{filename}\n"
            f"--- a/{filename}\n+++ b/{filename}\n{file.patch}"
        )
    if not sections:
        return None
    return _bounded_text("\n".join(sections), MAX_DIFF_CHARS)


def _changed_files(files: list[GitHubPullRequestFile]) -> list[str]:
    return sorted({file.filename for file in files})


def _labels(detail: GitHubIssueDetail | GitHubPullRequestDetail) -> list[str]:
    raw_labels = detail.model_extra.get("labels") if detail.model_extra else None
    if not isinstance(raw_labels, list):
        return []
    labels = {
        item.get("name")
        for item in raw_labels
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    return sorted(labels)[:100]


def _detail_sha(detail: GitHubPullRequestDetail, key: str) -> str | None:
    value = getattr(detail, key, None)
    if value is None:
        value = detail.model_extra.get(key) if detail.model_extra else None
    if not isinstance(value, dict):
        return None
    sha = value.get("sha")
    if not isinstance(sha, str):
        return None
    try:
        return validate_commit_sha(sha)
    except ValueError:
        return None


def _incident_id(reference: GitHubResourceRef) -> str:
    raw = (
        f"github-{reference.repository.owner}-{reference.repository.repo}-"
        f"{reference.kind}-{reference.number}"
    )
    if len(raw) <= 128:
        return raw
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"github-{reference.kind}-{reference.number}-{digest}"


class GitHubIngestor:
    """Fetch and normalize one supported GitHub issue or pull request."""

    def __init__(self, client: GitHubClient) -> None:
        self.client = client

    def fetch(self, url: str) -> GitHubFetchedResource:
        """Fetch the bounded metadata required by the RCA context."""
        reference = parse_github_resource_url(url)
        detail = self.client.get_resource_detail(reference)
        comments = self.client.list_comments(reference)
        if reference.kind == "pull_request":
            review_comments = self.client.list_pull_request_review_comments(reference)
            files = self.client.list_pull_request_files(reference)
            commits = self.client.list_pull_request_commits(reference)
            reviews = self.client.list_pull_request_reviews(reference)
        else:
            review_comments = []
            reviews = []
            files = []
            commits = []
        return GitHubFetchedResource(
            reference=reference,
            detail=detail,
            comments=comments,
            review_comments=review_comments,
            reviews=reviews,
            files=files,
            commits=commits,
        )

    def normalize(
        self,
        fetched: GitHubFetchedResource,
        *,
        base_commit: str | None = None,
    ) -> GitHubIngestionResult:
        """Convert fetched data into the existing ``IncidentInput`` contract."""
        reference = fetched.reference
        detail = fetched.detail
        if detail.number != reference.number:
            raise ValueError("GitHub response number does not match the requested URL")
        labels = _labels(detail)

        if reference.kind == "pull_request":
            if not isinstance(detail, GitHubPullRequestDetail):
                raise ValueError("GitHub pull request response has an invalid type")
            selected_base = _detail_sha(detail, "base")
            if selected_base is None:
                raise ValueError("GitHub pull request response has no valid base SHA")
            if base_commit is not None and base_commit != selected_base:
                raise ValueError("provided base commit does not match the pull request base SHA")
            selected_revision = selected_base
            revision_kind: Literal["default_branch", "pull_request_base"] = "pull_request_base"
            head_commit = _detail_sha(detail, "head")
            title = detail.title
            body = detail.body or ""
            diff = _pr_diff(fetched.files)
            logs = [_comment_log(item) for item in fetched.comments if item.body]
            logs.extend(
                _bounded_text(
                    f"GitHub review by @{_user_name(item.user)} "
                    f"({item.state or 'unknown'}): {(item.body or '').strip()}",
                    MAX_LOG_CHARS,
                )
                for item in fetched.reviews
                if item.body
            )
            logs.extend(
                _review_comment_log(item)
                for item in fetched.review_comments
                if item.body and item.body.strip()
            )
            logs.extend(_commit_log(item) for item in fetched.commits if item.commit)
            filenames = _changed_files(fetched.files)
            if filenames:
                names = ", ".join(filenames[:MAX_INGESTION_FILE_NAMES])
                if len(filenames) > MAX_INGESTION_FILE_NAMES:
                    names += f", ... ({len(filenames) - MAX_INGESTION_FILE_NAMES} more)"
                logs.append(_bounded_text(f"PR changed files: {names}", MAX_LOG_CHARS))
            notes = [f"PR base SHA selected: {selected_revision}"]
        else:
            if not isinstance(detail, GitHubIssueDetail):
                raise ValueError("GitHub issue response has an invalid type")
            if base_commit is None:
                raise ValueError("an issue requires the prepared default-branch commit")
            selected_revision = validate_commit_sha(base_commit)
            revision_kind = "default_branch"
            head_commit = None
            title = detail.title
            body = detail.body or ""
            diff = None
            logs = [_comment_log(item) for item in fetched.comments if item.body]
            filenames = []
            notes = [f"Issue analyzed at default-branch commit: {selected_revision}"]

        problem = _bounded_problem(body or title)
        if not problem:
            raise ValueError("GitHub resource has no title or body")
        logs = [_bounded_text(log, MAX_LOG_CHARS) for log in logs[:10]]
        incident = IncidentInput(
            id=_incident_id(reference),
            repo=reference.repository.full_name,
            base_commit=selected_revision,
            resource_kind=reference.kind,
            title=_bounded_title(title),
            problem=problem,
            logs=logs,
            diff=diff,
            labels=labels,
            changed_files=filenames,
            provenance=Provenance(
                source=reference.url,
                tool="github_rest_client",
                commit=selected_revision,
            ),
        )
        return GitHubIngestionResult(
            reference=reference,
            incident=incident,
            repository_url=github_clone_url(reference),
            base_commit=selected_revision,
            head_commit=head_commit,
            state=detail.state,
            labels=labels,
            changed_files=filenames,
            revision_kind=revision_kind,
            notes=notes,
        )


def ingest_github_resource(
    url: str,
    client: GitHubClient,
    *,
    base_commit: str | None = None,
) -> GitHubIngestionResult:
    """Fetch and normalize a GitHub URL without touching a repository."""
    ingestor = GitHubIngestor(client)
    return ingestor.normalize(ingestor.fetch(url), base_commit=base_commit)


__all__ = [
    "GitHubFetchedResource",
    "GitHubIngestionResult",
    "GitHubIngestor",
    "github_clone_url",
    "ingest_github_resource",
]
