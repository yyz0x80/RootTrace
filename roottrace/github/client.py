"""Bounded, read-only GitHub REST client for RootTrace ingestion."""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urljoin, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from pydantic import BaseModel, ValidationError

from .models import (
    GitHubComment,
    GitHubCommit,
    GitHubIssueDetail,
    GitHubIssueRef,
    GitHubPullRequestDetail,
    GitHubPullRequestFile,
    GitHubPullRequestRef,
    GitHubPullRequestReview,
    GitHubPullRequestReviewComment,
    GitHubResourceRef,
)

DEFAULT_GITHUB_API_BASE_URL = "https://api.github.com"
DEFAULT_TIMEOUT = 30.0
DEFAULT_PER_PAGE = 100
DEFAULT_MAX_PAGES = 20
DEFAULT_MAX_ITEMS = 1_000
MAX_PER_PAGE = 100
MAX_ALLOWED_PAGES = 100
MAX_ALLOWED_ITEMS = 10_000
MAX_DEFAULT_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_ALLOWED_RESPONSE_BYTES = 50 * 1024 * 1024
MAX_ERROR_DETAIL_CHARS = 240

_T = TypeVar("_T", bound=BaseModel)
_NEXT_LINK_PATTERN = re.compile(r"^<([^>]+)>\s*;(.*)$")


class GitHubTransport(Protocol):
    """Callable transport contract used by :class:`GitHubClient`.

    The request is a standard-library ``urllib.request.Request`` and the
    timeout is passed separately so tests can provide a small deterministic
    fake without opening a network connection.
    """

    def __call__(self, request: Request, timeout: float) -> Any:
        """Execute a request and return a response-like object."""


@dataclass(frozen=True)
class GitHubTransportResponse:
    """Minimal response object convenient for mocked transports."""

    status_code: int
    body: bytes | str
    headers: Mapping[str, str] = field(default_factory=dict)


class GitHubClientError(RuntimeError):
    """Base class for all GitHub adapter failures."""


class GitHubHTTPError(GitHubClientError):
    """HTTP response error with bounded metadata and status code."""

    def __init__(
        self,
        status_code: int,
        url: str,
        *,
        method: str = "GET",
        message: str | None = None,
        response_body: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.url = url
        self.method = method
        self.response_body = response_body
        detail = message or f"GitHub request failed with HTTP {status_code}"
        super().__init__(f"{detail} ({method} {url})")


class GitHubAuthenticationError(GitHubHTTPError):
    """Raised when GitHub rejects credentials with HTTP 401."""


class GitHubPermissionError(GitHubHTTPError):
    """Raised for a non-rate-limit HTTP 403 response."""


class GitHubRateLimitError(GitHubHTTPError):
    """Raised when GitHub reports a primary or secondary rate limit."""


class GitHubNotFoundError(GitHubHTTPError):
    """Raised when GitHub returns HTTP 404."""


class GitHubResponseError(GitHubClientError):
    """Raised for malformed, oversized, or structurally invalid responses."""

    def __init__(self, message: str, url: str | None = None) -> None:
        self.url = url
        super().__init__(f"{message} ({url})" if url else message)


class GitHubNetworkError(GitHubClientError):
    """Raised when the transport cannot reach GitHub."""

    def __init__(self, url: str, cause: BaseException) -> None:
        self.url = url
        self.cause = cause
        super().__init__(f"GitHub network request failed for {url}: {cause}")


class GitHubPaginationError(GitHubClientError):
    """Raised when a bounded pagination policy cannot safely continue."""


# Common names for callers that prefer protocol-oriented terminology.
GitHubUnauthorizedError = GitHubAuthenticationError
GitHubForbiddenError = GitHubPermissionError


def _urllib_transport(request: Request, timeout: float) -> Any:
    """Open a request using the Python standard library."""
    return urlopen(request, timeout=timeout)


def _header(headers: Mapping[str, str], name: str) -> str | None:
    """Read one HTTP header case-insensitively."""
    wanted = name.lower()
    for key, value in headers.items():
        if str(key).lower() == wanted:
            return str(value)
    return None


def _normalise_headers(headers: Any) -> dict[str, str]:
    """Convert standard or fake response headers to a small string mapping."""
    if headers is None:
        return {}
    try:
        return {str(key): str(value) for key, value in headers.items()}
    except AttributeError as exc:
        raise GitHubResponseError("GitHub response headers are not a mapping") from exc


def _response_status(response: Any) -> int:
    """Extract and validate a status code from a response-like object."""
    status = getattr(response, "status", None)
    if status is None:
        status = getattr(response, "status_code", None)
    if status is None:
        getcode = getattr(response, "getcode", None)
        if callable(getcode):
            status = getcode()
    if isinstance(status, bool) or not isinstance(status, int) or not 100 <= status <= 599:
        raise GitHubResponseError("GitHub response has no valid HTTP status code")
    return status


class GitHubClient:
    """Read-only client for the GitHub REST endpoints needed by RootTrace.

    Args:
        token: Optional token. If omitted, ``GITHUB_TOKEN`` is read once at
            construction time. An empty token means no authorization header.
        transport: Callable receiving ``(Request, timeout)``. The default uses
            ``urllib.request.urlopen``; tests can inject a fake transport.
        base_url: GitHub API origin, optionally with a path such as
            ``https://github.example/api/v3``.
        timeout: Per-request network timeout in seconds.
        max_response_bytes: Maximum response body size accepted before JSON
            parsing.
    """

    def __init__(
        self,
        token: str | None = None,
        *,
        transport: GitHubTransport | None = None,
        base_url: str = DEFAULT_GITHUB_API_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        max_response_bytes: int = MAX_DEFAULT_RESPONSE_BYTES,
    ) -> None:
        self.base_url = self._validate_base_url(base_url)
        self.timeout = self._validate_timeout(timeout)
        self.max_response_bytes = self._validate_response_limit(max_response_bytes)
        self.token = os.environ.get("GITHUB_TOKEN") if token is None else token
        if self.token is not None and any(
            ord(char) < 0x20 or ord(char) == 0x7F for char in self.token
        ):
            raise ValueError("GitHub token must not contain control characters")
        self._transport = transport or _urllib_transport

    @staticmethod
    def _validate_base_url(base_url: str) -> str:
        """Validate and normalize an API base URL without network access."""
        if not isinstance(base_url, str) or not base_url:
            raise ValueError("GitHub API base_url must be a non-empty URL")
        if base_url != base_url.strip():
            raise ValueError("GitHub API base_url must not have surrounding whitespace")
        if any(ord(char) < 0x20 or ord(char) == 0x7F for char in base_url):
            raise ValueError("GitHub API base_url must not contain control characters")
        try:
            parsed = urlsplit(base_url)
            hostname = parsed.hostname
            _port = parsed.port
        except ValueError as exc:
            raise ValueError(f"invalid GitHub API base_url: {exc}") from exc
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or not hostname:
            raise ValueError("GitHub API base_url must use an http or https URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("GitHub API base_url must not contain user credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("GitHub API base_url must not contain query or fragment")
        path = parsed.path.rstrip("/")
        if any(part in {".", ".."} for part in path.split("/")):
            raise ValueError("GitHub API base_url must not contain dot path segments")
        return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))

    @staticmethod
    def _validate_timeout(timeout: float) -> float:
        """Validate a finite and bounded per-request timeout."""
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise TypeError("GitHub timeout must be a finite number of seconds")
        if not math.isfinite(float(timeout)) or timeout <= 0 or timeout > 300:
            raise ValueError("GitHub timeout must be greater than 0 and at most 300 seconds")
        return float(timeout)

    @staticmethod
    def _validate_response_limit(max_response_bytes: int) -> int:
        """Validate the maximum response body size."""
        if (
            isinstance(max_response_bytes, bool)
            or not isinstance(max_response_bytes, int)
            or max_response_bytes <= 0
            or max_response_bytes > MAX_ALLOWED_RESPONSE_BYTES
        ):
            raise ValueError(
                "max_response_bytes must be a positive integer of at most "
                f"{MAX_ALLOWED_RESPONSE_BYTES}"
            )
        return max_response_bytes

    def _headers(self) -> dict[str, str]:
        """Build headers for one request without exposing the token in logs."""
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "RootTrace-GitHub-Adapter/1.0",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _url(self, path: str) -> str:
        """Join a validated API-relative path to the configured base URL."""
        if not path.startswith("/") or "?" in path or "#" in path:
            raise ValueError("GitHub API path must be an absolute relative path without query")
        return f"{self.base_url}{path}"

    @staticmethod
    def _repository_path(reference: GitHubResourceRef, resource: str) -> str:
        """Build a repository resource path from a validated reference."""
        owner = quote(reference.repository.owner, safe="")
        repo = quote(reference.repository.repo, safe="")
        return f"/repos/{owner}/{repo}/{resource}/{reference.number}"

    @staticmethod
    def _require_reference(reference: GitHubResourceRef) -> GitHubResourceRef:
        """Ensure public methods receive a typed GitHub resource reference."""
        if not isinstance(reference, GitHubResourceRef):
            raise TypeError("GitHub client methods require a GitHubResourceRef")
        return reference

    @staticmethod
    def _require_pull_request(reference: GitHubResourceRef) -> GitHubResourceRef:
        """Ensure a reference identifies a pull request."""
        reference = GitHubClient._require_reference(reference)
        if reference.kind != "pull_request":
            raise ValueError("GitHub pull request endpoint requires a GitHubPullRequestRef")
        return reference

    def _request_json(self, url: str) -> tuple[Any, Mapping[str, str]]:
        """Perform one GET and parse a bounded JSON response."""
        request = Request(url, headers=self._headers(), method="GET")
        try:
            response = self._transport(request, self.timeout)
        except HTTPError as exc:
            status = self._http_error_status(exc)
            headers = _normalise_headers(getattr(exc, "headers", None))
            try:
                body = self._read_body(exc, url, headers)
            except GitHubResponseError:
                body = b""
            self._raise_http_error(status, url, headers, body)
        except (URLError, TimeoutError, OSError) as exc:
            raise GitHubNetworkError(url, exc) from exc

        try:
            if isinstance(response, tuple):
                if len(response) != 3:
                    raise GitHubResponseError(
                        "mock GitHub transport tuple must contain status, headers, and body",
                        url,
                    )
                status, headers_value, body_value = response
                if isinstance(status, bool) or not isinstance(status, int):
                    raise GitHubResponseError("GitHub response has an invalid status code", url)
                headers = _normalise_headers(headers_value)
                body = self._coerce_body(body_value, url)
            else:
                status = _response_status(response)
                headers = _normalise_headers(getattr(response, "headers", None))
                body = self._read_body(response, url, headers)
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()

        if status < 200 or status >= 300:
            self._raise_http_error(status, url, headers, body)
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GitHubResponseError("GitHub response body is not valid UTF-8", url) from exc
        try:
            return json.loads(text), headers
        except json.JSONDecodeError as exc:
            raise GitHubResponseError("GitHub response body is not valid JSON", url) from exc

    @staticmethod
    def _http_error_status(error: HTTPError) -> int:
        """Extract an HTTPError status code without assuming a concrete mock type."""
        status = getattr(error, "code", None)
        if isinstance(status, int) and not isinstance(status, bool):
            return status
        return _response_status(error)

    def _read_body(
        self,
        response: Any,
        url: str,
        headers: Mapping[str, str],
    ) -> bytes:
        """Read at most one byte beyond the configured response limit."""
        content_length = _header(headers, "Content-Length")
        if content_length is not None:
            try:
                if int(content_length) > self.max_response_bytes:
                    raise GitHubResponseError(
                        f"GitHub response exceeds {self.max_response_bytes} bytes",
                        url,
                    )
            except ValueError as exc:
                raise GitHubResponseError("GitHub response has an invalid Content-Length", url) from exc

        read = getattr(response, "read", None)
        if callable(read):
            try:
                body = read(self.max_response_bytes + 1)
            except TypeError:
                body = read()
        elif hasattr(response, "body"):
            body = response.body
        else:
            raise GitHubResponseError("GitHub response has no readable body", url)
        return self._coerce_body(body, url)

    def _coerce_body(self, body: Any, url: str) -> bytes:
        """Convert response content to bytes and enforce the body limit."""
        if isinstance(body, str):
            body_bytes = body.encode("utf-8")
        elif isinstance(body, (bytes, bytearray, memoryview)):
            body_bytes = bytes(body)
        else:
            raise GitHubResponseError("GitHub response body must be bytes or text", url)
        if len(body_bytes) > self.max_response_bytes:
            raise GitHubResponseError(
                f"GitHub response exceeds {self.max_response_bytes} bytes",
                url,
            )
        return body_bytes

    def _raise_http_error(
        self,
        status: int,
        url: str,
        headers: Mapping[str, str],
        body: bytes,
    ) -> None:
        """Map GitHub status responses to distinct public exception classes."""
        detail = self._error_detail(body)
        suffix = f": {detail}" if detail else ""
        if status == 401:
            raise GitHubAuthenticationError(
                status,
                url,
                message=f"GitHub authentication failed{suffix}; check GITHUB_TOKEN",
                response_body=detail,
            )
        if status == 403:
            if self._looks_rate_limited(headers, body):
                raise GitHubRateLimitError(
                    status,
                    url,
                    message=f"GitHub API rate limit exceeded{suffix}",
                    response_body=detail,
                )
            raise GitHubPermissionError(
                status,
                url,
                message=f"GitHub request was forbidden{suffix}",
                response_body=detail,
            )
        if status == 404:
            raise GitHubNotFoundError(
                status,
                url,
                message=f"GitHub resource was not found{suffix}",
                response_body=detail,
            )
        if status == 429:
            raise GitHubRateLimitError(
                status,
                url,
                message=f"GitHub API rate limit exceeded{suffix}",
                response_body=detail,
            )
        raise GitHubHTTPError(
            status,
            url,
            message=f"GitHub request failed{suffix}",
            response_body=detail,
        )

    @staticmethod
    def _error_detail(body: bytes) -> str | None:
        """Extract only a small safe message from an error body."""
        if not body:
            return None
        try:
            decoded = body.decode("utf-8")
        except UnicodeDecodeError:
            return None
        try:
            payload = json.loads(decoded)
        except json.JSONDecodeError:
            detail = decoded.strip()
        else:
            detail = payload.get("message") if isinstance(payload, dict) else None
            if not isinstance(detail, str):
                detail = None
        if not detail:
            return None
        return detail[:MAX_ERROR_DETAIL_CHARS]

    @staticmethod
    def _looks_rate_limited(headers: Mapping[str, str], body: bytes) -> bool:
        """Recognize GitHub's primary and secondary rate-limit responses."""
        remaining = _header(headers, "X-RateLimit-Remaining")
        if remaining is not None and remaining.strip() == "0":
            return True
        retry_after = _header(headers, "Retry-After")
        if retry_after is not None:
            return True
        try:
            message = body.decode("utf-8", errors="replace").lower()
        except AttributeError:
            return False
        return "rate limit" in message or "abuse detection" in message

    def _detail(
        self,
        path: str,
        model: type[_T],
        label: str,
    ) -> _T:
        """Fetch and validate one object response."""
        url = self._url(path)
        payload, _ = self._request_json(url)
        if not isinstance(payload, dict):
            raise GitHubResponseError(f"GitHub {label} response must be a JSON object", url)
        try:
            return model.model_validate(payload)
        except ValidationError as exc:
            raise GitHubResponseError(
                f"GitHub {label} response has an invalid shape: {exc}",
                url,
            ) from exc

    @staticmethod
    def _validate_pagination(per_page: int, max_pages: int, max_items: int) -> None:
        """Validate caller-provided pagination limits."""
        if isinstance(per_page, bool) or not isinstance(per_page, int) or not 1 <= per_page <= MAX_PER_PAGE:
            raise ValueError(f"per_page must be an integer between 1 and {MAX_PER_PAGE}")
        if isinstance(max_pages, bool) or not isinstance(max_pages, int) or not 1 <= max_pages <= MAX_ALLOWED_PAGES:
            raise ValueError(f"max_pages must be an integer between 1 and {MAX_ALLOWED_PAGES}")
        if isinstance(max_items, bool) or not isinstance(max_items, int) or not 1 <= max_items <= MAX_ALLOWED_ITEMS:
            raise ValueError(f"max_items must be an integer between 1 and {MAX_ALLOWED_ITEMS}")

    def _list_endpoint(
        self,
        path: str,
        model: type[_T],
        label: str,
        *,
        per_page: int = DEFAULT_PER_PAGE,
        max_pages: int = DEFAULT_MAX_PAGES,
        max_items: int = DEFAULT_MAX_ITEMS,
    ) -> list[_T]:
        """Fetch a bounded GitHub list endpoint using Link-header pagination."""
        self._validate_pagination(per_page, max_pages, max_items)
        first_url = self._url(path)
        next_url = f"{first_url}?{urlencode({'per_page': per_page, 'page': 1})}"
        seen_urls: set[str] = set()
        results: list[_T] = []

        for page_index in range(1, max_pages + 1):
            if next_url in seen_urls:
                raise GitHubPaginationError("GitHub pagination link repeated the same URL")
            seen_urls.add(next_url)
            payload, headers = self._request_json(next_url)
            if not isinstance(payload, list):
                raise GitHubResponseError(f"GitHub {label} response must be a JSON list", next_url)
            if len(results) + len(payload) > max_items:
                raise GitHubPaginationError(
                    f"GitHub {label} pagination exceeded max_items={max_items}"
                )
            for item_index, item in enumerate(payload):
                if not isinstance(item, dict):
                    raise GitHubResponseError(
                        f"GitHub {label} item {item_index} must be a JSON object",
                        next_url,
                    )
                try:
                    results.append(model.model_validate(item))
                except ValidationError as exc:
                    raise GitHubResponseError(
                        f"GitHub {label} item {item_index} has an invalid shape: {exc}",
                        next_url,
                    ) from exc

            following = self._next_link(headers, next_url)
            if following is None:
                return results
            if page_index >= max_pages:
                raise GitHubPaginationError(
                    f"GitHub {label} pagination exceeded max_pages={max_pages}"
                )
            if len(results) >= max_items:
                raise GitHubPaginationError(
                    f"GitHub {label} pagination exceeded max_items={max_items}"
                )
            next_url = self._safe_next_url(following, next_url)

        raise GitHubPaginationError(f"GitHub {label} pagination exceeded max_pages={max_pages}")

    def _next_link(self, headers: Mapping[str, str], current_url: str) -> str | None:
        """Extract the RFC-style ``rel=next`` URL from a Link header."""
        del current_url
        link = _header(headers, "Link")
        if not link:
            return None
        for entry in link.split(","):
            match = _NEXT_LINK_PATTERN.match(entry.strip())
            if not match:
                continue
            target, attributes = match.groups()
            for attribute in attributes.split(";"):
                if "=" not in attribute:
                    continue
                key, value = attribute.split("=", 1)
                if key.strip().lower() != "rel":
                    continue
                relations = value.strip().strip('"').split()
                if "next" in relations:
                    return target
        return None

    def _safe_next_url(self, target: str, current_url: str) -> str:
        """Allow pagination links only on the configured API origin and path."""
        candidate = urljoin(current_url, target)
        base = urlsplit(self.base_url)
        parsed = urlsplit(candidate)
        if (
            parsed.scheme != base.scheme
            or parsed.hostname != base.hostname
            or parsed.port != base.port
        ):
            raise GitHubResponseError("GitHub pagination link points outside the API host", candidate)
        base_path = base.path.rstrip("/")
        if base_path and not (parsed.path == base_path or parsed.path.startswith(f"{base_path}/")):
            raise GitHubResponseError("GitHub pagination link points outside the API path", candidate)
        if parsed.fragment:
            raise GitHubResponseError("GitHub pagination link must not contain a fragment", candidate)
        return candidate

    def get_issue(self, reference: GitHubIssueRef | GitHubResourceRef) -> GitHubIssueDetail:
        """Fetch issue detail from ``GET /repos/{owner}/{repo}/issues/{number}``."""
        reference = self._require_reference(reference)
        if reference.kind != "issue":
            raise ValueError("get_issue requires a GitHubIssueRef")
        return self._detail(
            self._repository_path(reference, "issues"),
            GitHubIssueDetail,
            "issue detail",
        )

    def get_pull_request(
        self,
        reference: GitHubPullRequestRef | GitHubResourceRef,
    ) -> GitHubPullRequestDetail:
        """Fetch pull request detail from the GitHub pulls endpoint."""
        reference = self._require_pull_request(reference)
        return self._detail(
            self._repository_path(reference, "pulls"),
            GitHubPullRequestDetail,
            "pull request detail",
        )

    def get_resource_detail(
        self,
        reference: GitHubIssueRef | GitHubPullRequestRef | GitHubResourceRef,
    ) -> GitHubIssueDetail | GitHubPullRequestDetail:
        """Fetch detail using the endpoint appropriate for an issue or PR."""
        reference = self._require_reference(reference)
        if reference.kind == "issue":
            return self.get_issue(reference)
        return self.get_pull_request(reference)

    def list_comments(
        self,
        reference: GitHubIssueRef | GitHubPullRequestRef | GitHubResourceRef,
        *,
        per_page: int = DEFAULT_PER_PAGE,
        max_pages: int = DEFAULT_MAX_PAGES,
        max_items: int = DEFAULT_MAX_ITEMS,
    ) -> list[GitHubComment]:
        """List issue or pull request comments with bounded pagination."""
        reference = self._require_reference(reference)
        return self._list_endpoint(
            f"{self._repository_path(reference, 'issues')}/comments",
            GitHubComment,
            "comments",
            per_page=per_page,
            max_pages=max_pages,
            max_items=max_items,
        )

    def list_pull_request_files(
        self,
        reference: GitHubPullRequestRef | GitHubResourceRef,
        *,
        per_page: int = DEFAULT_PER_PAGE,
        max_pages: int = DEFAULT_MAX_PAGES,
        max_items: int = DEFAULT_MAX_ITEMS,
    ) -> list[GitHubPullRequestFile]:
        """List changed files for a pull request with bounded pagination."""
        reference = self._require_pull_request(reference)
        return self._list_endpoint(
            f"{self._repository_path(reference, 'pulls')}/files",
            GitHubPullRequestFile,
            "pull request files",
            per_page=per_page,
            max_pages=max_pages,
            max_items=max_items,
        )

    def list_pull_request_commits(
        self,
        reference: GitHubPullRequestRef | GitHubResourceRef,
        *,
        per_page: int = DEFAULT_PER_PAGE,
        max_pages: int = DEFAULT_MAX_PAGES,
        max_items: int = DEFAULT_MAX_ITEMS,
    ) -> list[GitHubCommit]:
        """List commits in a pull request with bounded pagination."""
        reference = self._require_pull_request(reference)
        return self._list_endpoint(
            f"{self._repository_path(reference, 'pulls')}/commits",
            GitHubCommit,
            "pull request commits",
            per_page=per_page,
            max_pages=max_pages,
            max_items=max_items,
        )

    def list_pull_request_reviews(
        self,
        reference: GitHubPullRequestRef | GitHubResourceRef,
        *,
        per_page: int = DEFAULT_PER_PAGE,
        max_pages: int = DEFAULT_MAX_PAGES,
        max_items: int = DEFAULT_MAX_ITEMS,
    ) -> list[GitHubPullRequestReview]:
        """List pull request reviews with bounded pagination."""
        reference = self._require_pull_request(reference)
        return self._list_endpoint(
            f"{self._repository_path(reference, 'pulls')}/reviews",
            GitHubPullRequestReview,
            "pull request reviews",
            per_page=per_page,
            max_pages=max_pages,
            max_items=max_items,
        )

    def list_pull_request_review_comments(
        self,
        reference: GitHubPullRequestRef | GitHubResourceRef,
        *,
        per_page: int = DEFAULT_PER_PAGE,
        max_pages: int = DEFAULT_MAX_PAGES,
        max_items: int = DEFAULT_MAX_ITEMS,
    ) -> list[GitHubPullRequestReviewComment]:
        """List deduplicated code review comments from a pull request.

        Pagination and the raw item budget are enforced by the shared list
        collector before duplicate IDs are removed.  The first occurrence of
        each immutable numeric GitHub comment ID retains API order.
        """
        reference = self._require_pull_request(reference)
        raw_comments = self._list_endpoint(
            f"{self._repository_path(reference, 'pulls')}/comments",
            GitHubPullRequestReviewComment,
            "pull request review comments",
            per_page=per_page,
            max_pages=max_pages,
            max_items=max_items,
        )
        seen_ids: set[int] = set()
        comments: list[GitHubPullRequestReviewComment] = []
        for comment in raw_comments:
            if comment.id in seen_ids:
                continue
            seen_ids.add(comment.id)
            comments.append(comment)
        return comments

    def get_issue_detail(self, reference: GitHubIssueRef | GitHubResourceRef) -> GitHubIssueDetail:
        """Alias for :meth:`get_issue`."""
        return self.get_issue(reference)

    def get_pull_request_detail(
        self,
        reference: GitHubPullRequestRef | GitHubResourceRef,
    ) -> GitHubPullRequestDetail:
        """Alias for :meth:`get_pull_request`."""
        return self.get_pull_request(reference)

    def get_issue_comments(
        self,
        reference: GitHubIssueRef | GitHubPullRequestRef | GitHubResourceRef,
        **pagination: int,
    ) -> list[GitHubComment]:
        """Alias for :meth:`list_comments`."""
        return self.list_comments(reference, **pagination)

    def list_pull_request_comments(
        self,
        reference: GitHubPullRequestRef | GitHubResourceRef,
        **pagination: int,
    ) -> list[GitHubComment]:
        """List comments for a pull request using the issues comments endpoint."""
        reference = self._require_pull_request(reference)
        return self.list_comments(reference, **pagination)

    def get_pull_request_files(
        self,
        reference: GitHubPullRequestRef | GitHubResourceRef,
        **pagination: int,
    ) -> list[GitHubPullRequestFile]:
        """Alias for :meth:`list_pull_request_files`."""
        return self.list_pull_request_files(reference, **pagination)

    def get_pull_request_commits(
        self,
        reference: GitHubPullRequestRef | GitHubResourceRef,
        **pagination: int,
    ) -> list[GitHubCommit]:
        """Alias for :meth:`list_pull_request_commits`."""
        return self.list_pull_request_commits(reference, **pagination)

    def get_pull_request_reviews(
        self,
        reference: GitHubPullRequestRef | GitHubResourceRef,
        **pagination: int,
    ) -> list[GitHubPullRequestReview]:
        """Alias for :meth:`list_pull_request_reviews`."""
        return self.list_pull_request_reviews(reference, **pagination)

    def get_pull_request_review_comments(
        self,
        reference: GitHubPullRequestRef | GitHubResourceRef,
        **pagination: int,
    ) -> list[GitHubPullRequestReviewComment]:
        """Alias for :meth:`list_pull_request_review_comments`."""
        return self.list_pull_request_review_comments(reference, **pagination)

    fetch_issue = get_issue
    fetch_pull_request = get_pull_request
    fetch_comments = list_comments
    fetch_pull_request_files = list_pull_request_files
    fetch_pull_request_commits = list_pull_request_commits
    fetch_pull_request_reviews = list_pull_request_reviews
    fetch_pull_request_review_comments = list_pull_request_review_comments
