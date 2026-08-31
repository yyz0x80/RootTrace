"""Typed models and strict URL parsing for GitHub resources."""

from __future__ import annotations

import re
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

MAX_GITHUB_OWNER_LENGTH = 39
MAX_GITHUB_REPOSITORY_LENGTH = 100
MAX_GITHUB_NUMBER_DIGITS = 10
MAX_GITHUB_RESOURCE_NUMBER = 2_147_483_647
MAX_GITHUB_TEXT_CHARS = 200_000

_OWNER_PATTERN = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?$"
)
_REPOSITORY_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
_NUMBER_PATTERN = re.compile(r"^[1-9][0-9]{0,9}$")


def _resource_path_segment(kind: str) -> str:
    """Return the canonical web path segment for a resource kind."""
    return "issues" if kind == "issue" else "pull"


class GitHubRepositoryRef(BaseModel):
    """Canonical owner/repository identity used by GitHub API requests."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    owner: str = Field(min_length=1, max_length=MAX_GITHUB_OWNER_LENGTH)
    repo: str = Field(min_length=1, max_length=MAX_GITHUB_REPOSITORY_LENGTH)

    @field_validator("owner")
    @classmethod
    def _validate_owner(cls, value: str) -> str:
        """Validate a GitHub user or organization name."""
        if not _OWNER_PATTERN.fullmatch(value):
            raise ValueError(
                "GitHub owner must be 1-39 characters of letters, numbers, or "
                "hyphens and cannot start or end with a hyphen"
            )
        return value

    @field_validator("repo")
    @classmethod
    def _validate_repo(cls, value: str) -> str:
        """Validate a GitHub repository name."""
        if value in {".", ".."} or not _REPOSITORY_PATTERN.fullmatch(value):
            raise ValueError(
                "GitHub repository must contain only letters, numbers, dots, "
                "hyphens, or underscores"
            )
        return value

    @property
    def name(self) -> str:
        """Return the repository name under the ``name`` spelling."""
        return self.repo

    @property
    def full_name(self) -> str:
        """Return the GitHub ``owner/repository`` identifier."""
        return f"{self.owner}/{self.repo}"


class GitHubResourceRef(BaseModel):
    """Typed reference to one GitHub issue or pull request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    repository: GitHubRepositoryRef
    number: StrictInt = Field(gt=0, le=MAX_GITHUB_RESOURCE_NUMBER)
    kind: Literal["issue", "pull_request"]
    canonical_url: str | None = None

    @model_validator(mode="after")
    def _set_and_validate_canonical_url(self) -> GitHubResourceRef:
        """Ensure the stored URL is exactly the supported canonical form."""
        expected = (
            f"https://github.com/{self.repository.full_name}/"
            f"{_resource_path_segment(self.kind)}/{self.number}"
        )
        if self.canonical_url is not None and self.canonical_url != expected:
            raise ValueError(
                "canonical_url must exactly match the canonical GitHub resource URL"
            )
        object.__setattr__(self, "canonical_url", expected)
        return self

    @property
    def owner(self) -> str:
        """Return the owning user or organization name."""
        return self.repository.owner

    @property
    def repo(self) -> str:
        """Return the repository name."""
        return self.repository.repo

    @property
    def url(self) -> str:
        """Return the canonical web URL."""
        assert self.canonical_url is not None
        return self.canonical_url

    @property
    def resource_type(self) -> str:
        """Return ``issue`` or ``pull_request`` for generic callers."""
        return self.kind


class GitHubIssueRef(GitHubResourceRef):
    """Typed reference to a GitHub issue."""

    kind: Literal["issue"] = "issue"


class GitHubPullRequestRef(GitHubResourceRef):
    """Typed reference to a GitHub pull request."""

    kind: Literal["pull_request"] = "pull_request"


GitHubResource = GitHubIssueRef | GitHubPullRequestRef


def parse_github_resource_url(url: str) -> GitHubResource:
    """Parse one strict canonical GitHub issue or pull request URL.

    Only URLs of the form ``https://github.com/{owner}/{repo}/issues/{number}``
    and ``https://github.com/{owner}/{repo}/pull/{number}`` are accepted. Query
    strings, fragments, trailing slashes, encoded path segments, and additional
    path components are rejected.

    Args:
        url: URL to validate and parse.

    Returns:
        A typed issue or pull request reference.

    Raises:
        ValueError: If ``url`` is not a supported canonical GitHub URL.
    """
    if not isinstance(url, str) or not url:
        raise ValueError("GitHub resource URL must be a non-empty string")
    if url != url.strip():
        raise ValueError("GitHub resource URL must not have surrounding whitespace")
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in url):
        raise ValueError("GitHub resource URL must not contain control characters")

    try:
        parsed = urlsplit(url)
    except ValueError as exc:
        raise ValueError(f"invalid GitHub resource URL: {exc}") from exc

    if parsed.scheme != "https" or parsed.netloc != "github.com":
        raise ValueError(
            "GitHub resource URL must use the exact https://github.com host"
        )
    if parsed.query:
        raise ValueError("GitHub resource URL must not contain a query string")
    if parsed.fragment:
        raise ValueError("GitHub resource URL must not contain a fragment")

    parts = parsed.path.split("/")
    if len(parts) != 5 or parts[0] != "" or any(not part for part in parts[1:]):
        raise ValueError(
            "GitHub resource URL must have exactly /{owner}/{repo}/"
            "issues/{number} or /{owner}/{repo}/pull/{number} path components"
        )

    owner, repo, resource_kind, number_text = parts[1:]
    if any("%" in part for part in parts[1:]):
        raise ValueError("GitHub resource URL must not contain encoded path segments")
    if resource_kind not in {"issues", "pull"}:
        raise ValueError("GitHub resource URL path must use /issues/ or /pull/")
    if not _NUMBER_PATTERN.fullmatch(number_text):
        raise ValueError(
            "GitHub issue or pull request number must be a positive decimal "
            "integer without leading zeroes"
        )

    try:
        repository = GitHubRepositoryRef(owner=owner, repo=repo)
        number = int(number_text)
        if number > MAX_GITHUB_RESOURCE_NUMBER:
            raise ValueError
        if resource_kind == "issues":
            reference: GitHubResource = GitHubIssueRef(
                repository=repository,
                number=number,
                canonical_url=url,
            )
        else:
            reference = GitHubPullRequestRef(
                repository=repository,
                number=number,
                canonical_url=url,
            )
    except ValueError as exc:
        if str(exc) == "":
            raise ValueError("GitHub issue or pull request number is too large") from exc
        raise ValueError(f"invalid GitHub repository reference: {exc}") from exc

    if reference.url != url:
        raise ValueError("GitHub resource URL is not canonical")
    return reference


def parse_github_url(url: str) -> GitHubResource:
    """Backward-compatible short name for :func:`parse_github_resource_url`."""
    return parse_github_resource_url(url)


def parse_github_ref(url: str) -> GitHubResource:
    """Parse a GitHub issue or pull request URL into a typed reference."""
    return parse_github_resource_url(url)


class GitHubUser(BaseModel):
    """Small typed projection of a GitHub user object."""

    model_config = ConfigDict(extra="allow", frozen=True)

    login: StrictStr | None = Field(default=None, max_length=MAX_GITHUB_TEXT_CHARS)


class GitHubIssueDetail(BaseModel):
    """Typed projection of the GitHub issue detail response."""

    model_config = ConfigDict(extra="allow", frozen=True)

    number: StrictInt = Field(gt=0, le=MAX_GITHUB_RESOURCE_NUMBER)
    title: StrictStr = Field(max_length=MAX_GITHUB_TEXT_CHARS)
    body: StrictStr | None = Field(default=None, max_length=MAX_GITHUB_TEXT_CHARS)
    state: StrictStr | None = Field(default=None, max_length=64)
    html_url: StrictStr | None = Field(default=None, max_length=MAX_GITHUB_TEXT_CHARS)
    user: GitHubUser | None = None
    comments: StrictInt | None = Field(default=None, ge=0)


class GitHubPullRequestDetail(BaseModel):
    """Typed projection of the GitHub pull request detail response."""

    model_config = ConfigDict(extra="allow", frozen=True)

    number: StrictInt = Field(gt=0, le=MAX_GITHUB_RESOURCE_NUMBER)
    title: StrictStr = Field(max_length=MAX_GITHUB_TEXT_CHARS)
    body: StrictStr | None = Field(default=None, max_length=MAX_GITHUB_TEXT_CHARS)
    state: StrictStr | None = Field(default=None, max_length=64)
    html_url: StrictStr | None = Field(default=None, max_length=MAX_GITHUB_TEXT_CHARS)
    user: GitHubUser | None = None
    comments: StrictInt | None = Field(default=None, ge=0)
    review_comments: StrictInt | None = Field(default=None, ge=0)
    commits: StrictInt | None = Field(default=None, ge=0)
    additions: StrictInt | None = Field(default=None, ge=0)
    deletions: StrictInt | None = Field(default=None, ge=0)
    changed_files: StrictInt | None = Field(default=None, ge=0)
    base: dict[str, Any] | None = None
    head: dict[str, Any] | None = None
    merge_commit_sha: StrictStr | None = Field(default=None, max_length=128)
    merged: bool | None = None


class GitHubComment(BaseModel):
    """Typed projection of an issue or pull request comment."""

    model_config = ConfigDict(extra="allow", frozen=True)

    id: StrictInt = Field(gt=0)
    body: StrictStr | None = Field(default=None, max_length=MAX_GITHUB_TEXT_CHARS)
    user: GitHubUser | None = None
    html_url: StrictStr | None = Field(default=None, max_length=MAX_GITHUB_TEXT_CHARS)
    created_at: StrictStr | None = Field(default=None, max_length=128)
    updated_at: StrictStr | None = Field(default=None, max_length=128)


class GitHubPullRequestReviewComment(BaseModel):
    """Typed projection of a pull request code review comment."""

    model_config = ConfigDict(extra="allow", frozen=True)

    id: StrictInt = Field(gt=0)
    body: StrictStr | None = Field(default=None, max_length=MAX_GITHUB_TEXT_CHARS)
    user: GitHubUser | None = None
    html_url: StrictStr | None = Field(default=None, max_length=MAX_GITHUB_TEXT_CHARS)
    path: StrictStr | None = Field(default=None, max_length=MAX_GITHUB_TEXT_CHARS)
    line: StrictInt | None = Field(default=None, ge=1)
    side: StrictStr | None = Field(default=None, max_length=16)
    start_line: StrictInt | None = Field(default=None, ge=1)
    start_side: StrictStr | None = Field(default=None, max_length=16)
    original_line: StrictInt | None = Field(default=None, ge=1)
    original_start_line: StrictInt | None = Field(default=None, ge=1)
    position: StrictInt | None = Field(default=None, ge=1)
    original_position: StrictInt | None = Field(default=None, ge=1)
    commit_id: StrictStr | None = Field(default=None, max_length=128)
    in_reply_to_id: StrictInt | None = Field(default=None, gt=0)
    diff_hunk: StrictStr | None = Field(default=None, max_length=MAX_GITHUB_TEXT_CHARS)
    created_at: StrictStr | None = Field(default=None, max_length=128)
    updated_at: StrictStr | None = Field(default=None, max_length=128)


class GitHubPullRequestReview(BaseModel):
    """Typed projection of a pull request review."""

    model_config = ConfigDict(extra="allow", frozen=True)

    id: StrictInt = Field(gt=0)
    body: StrictStr | None = Field(default=None, max_length=MAX_GITHUB_TEXT_CHARS)
    user: GitHubUser | None = None
    state: StrictStr | None = Field(default=None, max_length=64)
    commit_id: StrictStr | None = Field(default=None, max_length=128)
    submitted_at: StrictStr | None = Field(default=None, max_length=128)


class GitHubPullRequestFile(BaseModel):
    """Typed projection of a pull request file entry."""

    model_config = ConfigDict(extra="allow", frozen=True)

    filename: StrictStr = Field(min_length=1, max_length=MAX_GITHUB_TEXT_CHARS)
    status: StrictStr | None = Field(default=None, max_length=64)
    additions: StrictInt | None = Field(default=None, ge=0)
    deletions: StrictInt | None = Field(default=None, ge=0)
    changes: StrictInt | None = Field(default=None, ge=0)
    blob_url: StrictStr | None = Field(default=None, max_length=MAX_GITHUB_TEXT_CHARS)
    raw_url: StrictStr | None = Field(default=None, max_length=MAX_GITHUB_TEXT_CHARS)
    contents_url: StrictStr | None = Field(default=None, max_length=MAX_GITHUB_TEXT_CHARS)
    sha: StrictStr | None = Field(default=None, max_length=128)
    patch: StrictStr | None = Field(default=None, max_length=MAX_GITHUB_TEXT_CHARS)


class GitHubCommitMetadata(BaseModel):
    """Typed projection of the nested commit metadata object."""

    model_config = ConfigDict(extra="allow", frozen=True)

    message: StrictStr | None = Field(default=None, max_length=MAX_GITHUB_TEXT_CHARS)


class GitHubCommit(BaseModel):
    """Typed projection of a pull request commit entry."""

    model_config = ConfigDict(extra="allow", frozen=True)

    sha: StrictStr = Field(min_length=1, max_length=128)
    commit: GitHubCommitMetadata | None = None
    html_url: StrictStr | None = Field(default=None, max_length=MAX_GITHUB_TEXT_CHARS)
    author: GitHubUser | None = None
    committer: GitHubUser | None = None


GitHubIssue = GitHubIssueDetail
GitHubPullRequest = GitHubPullRequestDetail
