"""Focused mock-only tests for the bounded GitHub REST client."""

import json
from urllib.error import URLError
from urllib.parse import parse_qs, urlsplit

import pytest

from roottrace.github import (
    GitHubAuthenticationError,
    GitHubClient,
    GitHubForbiddenError,
    GitHubHTTPError,
    GitHubNetworkError,
    GitHubNotFoundError,
    GitHubPaginationError,
    GitHubPermissionError,
    GitHubPullRequestFile,
    GitHubPullRequestRef,
    GitHubPullRequestReviewComment,
    GitHubRateLimitError,
    GitHubResponseError,
    GitHubTransportResponse,
    parse_github_resource_url,
)


class FakeTransport:
    """Queue response values and record every request without network access."""

    def __init__(self, *responses) -> None:
        self.responses = list(responses)
        self.requests = []
        self.timeouts = []

    def __call__(self, request, timeout):
        self.requests.append(request)
        self.timeouts.append(timeout)
        if not self.responses:
            raise AssertionError("fake transport received more requests than expected")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def response(
    payload,
    *,
    status_code: int = 200,
    headers=None,
) -> GitHubTransportResponse:
    """Build a JSON response for the fake transport."""
    return GitHubTransportResponse(
        status_code=status_code,
        body=json.dumps(payload),
        headers=headers or {},
    )


@pytest.fixture
def issue_ref():
    """Return a valid issue reference for client tests."""
    return parse_github_resource_url("https://github.com/acme/widget/issues/7")


@pytest.fixture
def pull_ref() -> GitHubPullRequestRef:
    """Return a valid pull request reference for client tests."""
    return parse_github_resource_url("https://github.com/acme/widget/pull/8")


def test_client_uses_token_headers_base_url_and_timeout(issue_ref, monkeypatch) -> None:
    """The client reads GITHUB_TOKEN and builds standard urllib requests."""
    monkeypatch.setenv("GITHUB_TOKEN", "env-secret")
    transport = FakeTransport(response({"number": 7, "title": "Bug"}))
    client = GitHubClient(
        transport=transport,
        base_url="https://api.example.test/api/v3/",
        timeout=12.5,
    )

    detail = client.get_issue(issue_ref)

    assert detail.number == 7
    assert detail.title == "Bug"
    request = transport.requests[0]
    assert request.full_url == (
        "https://api.example.test/api/v3/repos/acme/widget/issues/7"
    )
    assert request.get_header("Authorization") == "Bearer env-secret"
    assert request.get_header("Accept") == "application/vnd.github+json"
    assert transport.timeouts == [12.5]


def test_client_fetches_issue_and_pull_request_detail(issue_ref, pull_ref) -> None:
    """Issue and PR details use their distinct GitHub REST endpoints."""
    transport = FakeTransport(
        response({"number": 7, "title": "Issue", "body": "details"}),
        response({"number": 8, "title": "Pull request", "body": "changes"}),
    )
    client = GitHubClient(transport=transport)

    issue = client.get_issue(issue_ref)
    pull_request = client.get_pull_request(pull_ref)

    assert issue.title == "Issue"
    assert pull_request.title == "Pull request"
    assert [request.full_url for request in transport.requests] == [
        "https://api.github.com/repos/acme/widget/issues/7",
        "https://api.github.com/repos/acme/widget/pulls/8",
    ]


def test_client_paginates_comments_from_link_header(issue_ref) -> None:
    """List endpoints follow only bounded RFC-style rel=next links."""
    next_url = (
        "https://api.github.com/repos/acme/widget/issues/7/comments"
        "?per_page=2&page=2"
    )
    transport = FakeTransport(
        response(
            [{"id": 1, "body": "first"}, {"id": 2, "body": "second"}],
            headers={"Link": f'<{next_url}>; rel="next"'},
        ),
        response([{"id": 3, "body": "third"}]),
    )
    client = GitHubClient(transport=transport)

    comments = client.list_comments(issue_ref, per_page=2, max_pages=2)

    assert [comment.id for comment in comments] == [1, 2, 3]
    assert transport.requests[0].full_url.endswith("?per_page=2&page=1")
    assert transport.requests[1].full_url == next_url


def test_client_fetches_pr_files_and_commits(pull_ref) -> None:
    """Files and commits are typed and use their paginated PR endpoints."""
    transport = FakeTransport(
        response([{"filename": "src/app.py", "status": "modified"}]),
        response([{"sha": "abc123", "commit": {"message": "fix"}}]),
    )
    client = GitHubClient(transport=transport)

    files = client.list_pull_request_files(pull_ref)
    commits = client.list_pull_request_commits(pull_ref)

    assert isinstance(files[0], GitHubPullRequestFile)
    assert files[0].filename == "src/app.py"
    assert commits[0].sha == "abc123"
    assert [urlsplit(request.full_url).path for request in transport.requests] == [
        "/repos/acme/widget/pulls/8/files",
        "/repos/acme/widget/pulls/8/commits",
    ]


def test_client_deduplicates_review_comments_by_id_and_keeps_first_order(pull_ref) -> None:
    """Review comments keep first-seen IDs and retain same-body distinct IDs."""
    transport = FakeTransport(
        response(
            [
                {"id": 2, "body": "second"},
                {"id": 1, "body": "first"},
                {"id": 2, "body": "later duplicate"},
                {"id": 3, "body": "same body"},
                {"id": 4, "body": "same body"},
            ]
        )
    )
    client = GitHubClient(transport=transport)

    comments = client.list_pull_request_review_comments(pull_ref)

    assert [comment.id for comment in comments] == [2, 1, 3, 4]
    assert [comment.body for comment in comments] == [
        "second",
        "first",
        "same body",
        "same body",
    ]
    assert all(isinstance(comment, GitHubPullRequestReviewComment) for comment in comments)


def test_client_deduplicates_review_comments_across_pages(pull_ref) -> None:
    """Duplicate review IDs across API pages retain the first page's value."""
    next_url = (
        "https://api.github.com/repos/acme/widget/pulls/8/comments"
        "?per_page=2&page=2"
    )
    transport = FakeTransport(
        response(
            [{"id": 10, "body": "first"}, {"id": 20, "body": "page one"}],
            headers={"Link": f'<{next_url}>; rel="next"'},
        ),
        response(
            [
                {"id": 20, "body": "duplicate from page two"},
                {"id": 30, "body": "page two"},
            ]
        ),
    )
    client = GitHubClient(transport=transport)

    comments = client.list_pull_request_review_comments(pull_ref, per_page=2)

    assert [comment.id for comment in comments] == [10, 20, 30]
    assert [comment.body for comment in comments] == ["first", "page one", "page two"]
    assert [request.full_url for request in transport.requests] == [
        "https://api.github.com/repos/acme/widget/pulls/8/comments?per_page=2&page=1",
        next_url,
    ]


def test_client_returns_empty_review_comment_response(pull_ref) -> None:
    """An empty review-comment endpoint response normalizes to an empty list."""
    client = GitHubClient(transport=FakeTransport(response([])))

    assert client.list_pull_request_review_comments(pull_ref) == []


def test_client_rejects_malformed_review_comment_response(pull_ref) -> None:
    """Review-comment responses must be lists of objects with numeric IDs."""
    with pytest.raises(GitHubResponseError, match="response must be a JSON list"):
        GitHubClient(transport=FakeTransport(response({"id": 1}))).list_pull_request_review_comments(
            pull_ref
        )
    with pytest.raises(GitHubResponseError, match="item 0 has an invalid shape"):
        GitHubClient(transport=FakeTransport(response([{"body": "missing id"}]))).list_pull_request_review_comments(
            pull_ref
        )


def test_client_preserves_review_comment_thread_and_location_fields(pull_ref) -> None:
    """The dedicated model exposes threaded and source-location metadata."""
    payload = {
        "id": 42,
        "body": "Please use the helper.",
        "path": "src/app.py",
        "line": 12,
        "side": "RIGHT",
        "start_line": 10,
        "start_side": "RIGHT",
        "original_line": 11,
        "original_start_line": 9,
        "position": 33,
        "original_position": 30,
        "commit_id": "a" * 40,
        "in_reply_to_id": 41,
        "diff_hunk": "@@ -9,4 +10,4 @@",
    }
    client = GitHubClient(transport=FakeTransport(response([payload])))

    comment = client.list_pull_request_review_comments(pull_ref)[0]

    assert comment.path == "src/app.py"
    assert comment.line == 12
    assert comment.side == "RIGHT"
    assert comment.start_line == 10
    assert comment.original_line == 11
    assert comment.original_start_line == 9
    assert comment.position == 33
    assert comment.original_position == 30
    assert comment.commit_id == "a" * 40
    assert comment.in_reply_to_id == 41
    assert comment.diff_hunk == "@@ -9,4 +10,4 @@"


def test_client_enforces_raw_review_comment_max_items_before_deduplication(pull_ref) -> None:
    """Duplicate IDs cannot make a raw paginated response fit the item budget."""
    next_url = "https://api.github.com/repos/acme/widget/pulls/8/comments?page=2"
    transport = FakeTransport(
        response(
            [{"id": 1, "body": "first"}, {"id": 1, "body": "duplicate"}],
            headers={"Link": f"<{next_url}>; rel=next"},
        )
    )
    client = GitHubClient(transport=transport)

    with pytest.raises(GitHubPaginationError, match="max_items=2"):
        client.list_pull_request_review_comments(pull_ref, max_items=2)

    assert len(transport.requests) == 1


@pytest.mark.parametrize(
    ("status_code", "error_type", "payload", "headers"),
    [
        (401, GitHubAuthenticationError, {"message": "Bad credentials"}, {}),
        (403, GitHubRateLimitError, {"message": "rate limit exceeded"}, {}),
        (403, GitHubPermissionError, {"message": "Resource forbidden"}, {}),
        (403, GitHubRateLimitError, {}, {"X-RateLimit-Remaining": "0"}),
        (404, GitHubNotFoundError, {"message": "Not Found"}, {}),
    ],
)
def test_client_classifies_http_errors(
    issue_ref,
    status_code,
    error_type,
    payload,
    headers,
) -> None:
    """Authentication, permissions, rate limits, and missing resources differ."""
    client = GitHubClient(
        transport=FakeTransport(
            response(payload, status_code=status_code, headers=headers)
        )
    )

    with pytest.raises(error_type) as raised:
        client.get_issue(issue_ref)

    assert raised.value.status_code == status_code
    assert "GITHUB_TOKEN" not in str(raised.value.response_body or "")


def test_forbidden_alias_and_generic_http_error(issue_ref) -> None:
    """Compatibility aliases retain distinct forbidden and generic errors."""
    assert GitHubForbiddenError is GitHubPermissionError
    client = GitHubClient(transport=FakeTransport(response({}, status_code=500)))

    with pytest.raises(GitHubHTTPError) as raised:
        client.get_issue(issue_ref)

    assert type(raised.value) is GitHubHTTPError


def test_client_classifies_network_failure(issue_ref) -> None:
    """urllib network failures are wrapped without being HTTP errors."""
    client = GitHubClient(transport=FakeTransport(URLError("offline")))

    with pytest.raises(GitHubNetworkError) as raised:
        client.get_issue(issue_ref)

    assert "api.github.com" in str(raised.value)


def test_client_rejects_invalid_json_and_shapes(issue_ref) -> None:
    """Success responses must be bounded JSON objects with required fields."""
    invalid_json = GitHubTransportResponse(status_code=200, body="not-json")
    invalid_shape = response({"number": 7})

    with pytest.raises(GitHubResponseError, match="valid JSON"):
        GitHubClient(transport=FakeTransport(invalid_json)).get_issue(issue_ref)
    with pytest.raises(GitHubResponseError, match="invalid shape"):
        GitHubClient(transport=FakeTransport(invalid_shape)).get_issue(issue_ref)


def test_client_enforces_response_size_limit(issue_ref) -> None:
    """A response larger than the configured byte limit is rejected before parsing."""
    large = GitHubTransportResponse(status_code=200, body="x" * 11)
    client = GitHubClient(transport=FakeTransport(large), max_response_bytes=10)

    with pytest.raises(GitHubResponseError, match="exceeds 10 bytes"):
        client.get_issue(issue_ref)


def test_client_enforces_pagination_limits(issue_ref) -> None:
    """Pagination cannot silently exceed its page or item budget."""
    next_url = "https://api.github.com/repos/acme/widget/issues/7/comments?page=2"
    transport = FakeTransport(
        response(
            [{"id": 1}],
            headers={"Link": f"<{next_url}>; rel=next"},
        )
    )
    client = GitHubClient(transport=transport)

    with pytest.raises(GitHubPaginationError, match="max_pages=1"):
        client.list_comments(issue_ref, max_pages=1)
    with pytest.raises(ValueError, match="per_page"):
        client.list_comments(issue_ref, per_page=101)


def test_client_rejects_external_pagination_link(issue_ref) -> None:
    """A server cannot make the adapter follow a pagination link elsewhere."""
    transport = FakeTransport(
        response(
            [{"id": 1}],
            headers={"Link": '<https://evil.example/items?page=2>; rel="next"'},
        )
    )
    client = GitHubClient(transport=transport)

    with pytest.raises(GitHubResponseError, match="outside the API host"):
        client.list_comments(issue_ref)


def test_client_pagination_query_is_well_formed(issue_ref) -> None:
    """Initial pagination parameters are explicit and bounded."""
    transport = FakeTransport(response([]))
    GitHubClient(transport=transport).list_comments(issue_ref, per_page=3)

    query = parse_qs(urlsplit(transport.requests[0].full_url).query)
    assert query == {"per_page": ["3"], "page": ["1"]}
