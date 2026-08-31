"""Tests for GitHub-to-RootTrace normalization and revision selection."""

from roottrace.github import (
    GitHubComment,
    GitHubCommit,
    GitHubCommitMetadata,
    GitHubFetchedResource,
    GitHubIngestor,
    GitHubIssueDetail,
    GitHubPullRequestDetail,
    GitHubPullRequestFile,
    GitHubPullRequestReviewComment,
    parse_github_resource_url,
)


def test_issue_normalization_preserves_body_and_bounded_comments() -> None:
    reference = parse_github_resource_url(
        "https://github.com/acme/widget/issues/7"
    )
    fetched = GitHubFetchedResource(
        reference=reference,
        detail=GitHubIssueDetail(
            number=7,
            title="Crash on empty config",
            body="The loader crashes when the config path is absent.",
            state="open",
            labels=[{"name": "bug"}, {"name": "regression"}],
        ),
        comments=[GitHubComment(id=1, body="This is reproducible.")],
    )

    result = GitHubIngestor(None).normalize(
        fetched,
        base_commit="a" * 40,
    )

    assert result.incident.repo == "acme/widget"
    assert result.incident.title == "Crash on empty config"
    assert "config path is absent" in result.incident.problem
    assert "reproducible" in result.incident.logs[0]
    assert result.revision_kind == "default_branch"
    assert result.state == "open"
    assert result.labels == ["bug", "regression"]
    assert result.incident.resource_kind == "issue"
    assert result.incident.git_verification_policy.enabled is True
    assert "regression_label" in result.incident.git_verification_policy.reasons
    assert result.repository_url == "https://github.com/acme/widget.git"


def test_pull_request_uses_base_sha_and_keeps_head_as_context() -> None:
    reference = parse_github_resource_url(
        "https://github.com/acme/widget/pull/8"
    )
    base_sha = "a" * 40
    head_sha = "b" * 40
    fetched = GitHubFetchedResource(
        reference=reference,
        detail=GitHubPullRequestDetail(
            number=8,
            title="Fix config loading",
            body="Handle a missing config path.",
            base={"sha": base_sha},
            head={"sha": head_sha},
        ),
        files=[
            GitHubPullRequestFile(
                filename="src/config.py",
                patch="@@ -1 +1 @@\n-old\n+new",
            )
        ],
        commits=[
            GitHubCommit(
                sha=head_sha,
                commit=GitHubCommitMetadata(message="fix config loading"),
            )
        ],
    )

    result = GitHubIngestor(None).normalize(fetched)

    assert result.base_commit == base_sha
    assert result.head_commit == head_sha
    assert result.incident.base_commit == base_sha
    assert result.incident.resource_kind == "pull_request"
    assert result.incident.git_verification_policy.enabled is True
    assert result.incident.git_verification_policy.history_depth > 1
    assert result.incident.git_verification_policy.candidate_paths == [
        "src/config.py"
    ]
    assert result.changed_files == ["src/config.py"]
    assert "diff --git a/src/config.py" in (result.incident.diff or "")
    assert result.revision_kind == "pull_request_base"


def test_pull_request_review_comments_use_a_typed_channel() -> None:
    """Opted-in review comments stay out of logs and retain provenance."""
    reference = parse_github_resource_url(
        "https://github.com/acme/widget/pull/8"
    )
    base_commit = "a" * 40
    fetched = GitHubFetchedResource(
        reference=reference,
        detail=GitHubPullRequestDetail(
            number=8,
            title="Fix config loading",
            body="Handle a missing config path.",
            base={"sha": base_commit},
        ),
        review_comments=[
            GitHubPullRequestReviewComment(
                id=1,
                body="x" * 5_000,
                path="src/config.py",
                line=12,
                start_line=10,
                original_line=11,
                original_start_line=9,
                commit_id=base_commit,
                html_url="https://github.com/acme/widget/pull/8#discussion_r1",
                diff_hunk="@@ unbounded diff context should not be included",
            ),
            GitHubPullRequestReviewComment(
                id=2,
                body="Please keep the reply in the same thread.",
                path="src/config.py",
                line=99,
                commit_id=base_commit,
                in_reply_to_id=1,
                html_url="https://github.com/acme/widget/pull/8#discussion_r2",
            ),
            GitHubPullRequestReviewComment(id=3, body="   ", path="src/empty.py"),
            GitHubPullRequestReviewComment(
                id=4,
                body="This has no source line.",
                path="src/other.py",
                html_url="https://github.com/acme/widget/pull/8#discussion_r4",
            ),
        ],
    )

    result = GitHubIngestor(
        None,
        include_review_comments=True,
    ).normalize(fetched)

    assert result.incident.logs == []
    assert [
        [comment.comment_id for comment in thread.comments]
        for thread in result.incident.review_threads
    ] == [[1, 2], [4]]
    first, reply = result.incident.review_threads[0].comments
    assert len(first.excerpt) <= 1_500
    assert "@@ unbounded diff context" not in first.model_dump_json()
    assert first.provenance.source.endswith("discussion_r1")
    assert first.provenance.tool == "github_rest_client"
    assert first.location is not None
    assert first.location.path == "src/config.py"
    assert first.location.start_line == 10
    assert first.location.end_line == 12
    assert first.location_mapping == "analysis_revision"
    assert reply.parent_comment_id == 1
    assert reply.location == first.location
    assert reply.location_source_comment_id == 1
    assert reply.line == 99
    second = result.incident.review_threads[1].comments[0]
    assert second.location is not None
    assert second.location.path == "src/other.py"
    assert second.location.start_line is None
    assert second.location_mapping == "unmapped"
    assert result.incident.review_comment_truncation.comments_omitted == 1


def test_pull_request_normalization_is_opt_out_even_for_preloaded_comments() -> None:
    """The default normalizer does not activate a preloaded review channel."""
    reference = parse_github_resource_url(
        "https://github.com/acme/widget/pull/8"
    )
    fetched = GitHubFetchedResource(
        reference=reference,
        detail=GitHubPullRequestDetail(
            number=8,
            title="Fix config loading",
            body="Handle a missing config path.",
            base={"sha": "a" * 40},
        ),
        review_comments=[
            GitHubPullRequestReviewComment(id=1, body="Do not enter logs."),
        ],
    )

    result = GitHubIngestor(None).normalize(fetched)

    assert result.incident.review_threads == []
    assert result.incident.review_comment_truncation.threads_considered == 0
    assert result.incident.logs == []


def test_pull_request_review_comment_fetching_is_opt_in() -> None:
    """Default PR ingestion avoids review-comment API cost and context noise."""
    reference = parse_github_resource_url(
        "https://github.com/acme/widget/pull/8"
    )
    detail = GitHubPullRequestDetail(
        number=8,
        title="Fix config loading",
        body="Handle a missing config path.",
        base={"sha": "a" * 40},
    )

    class FakeClient:
        def __init__(self) -> None:
            self.review_comment_calls = 0

        def get_resource_detail(self, requested_reference):
            assert requested_reference == reference
            return detail

        def list_comments(self, requested_reference):
            assert requested_reference == reference
            return []

        def list_pull_request_review_comments(self, requested_reference):
            assert requested_reference == reference
            self.review_comment_calls += 1
            return [
                GitHubPullRequestReviewComment(
                    id=1,
                    body="Use the shared helper.",
                    path="src/config.py",
                    line=12,
                )
            ]

        def list_pull_request_files(self, requested_reference):
            assert requested_reference == reference
            return []

        def list_pull_request_commits(self, requested_reference):
            assert requested_reference == reference
            return []

        def list_pull_request_reviews(self, requested_reference):
            assert requested_reference == reference
            return []

    client = FakeClient()

    default_result = GitHubIngestor(client).fetch(reference.url)
    opted_in_result = GitHubIngestor(
        client,
        include_review_comments=True,
    ).fetch(reference.url)

    assert default_result.review_comments == []
    assert [comment.id for comment in opted_in_result.review_comments] == [1]
    assert client.review_comment_calls == 1


def test_issue_normalization_ignores_pull_request_review_comments() -> None:
    """Review comments are PR-only and cannot change ordinary issue results."""
    reference = parse_github_resource_url(
        "https://github.com/acme/widget/issues/9"
    )
    base = GitHubFetchedResource(
        reference=reference,
        detail=GitHubIssueDetail(
            number=9,
            title="Crash on empty config",
            body="The loader crashes when the config path is absent.",
        ),
    )
    with_review_comments = base.model_copy(
        update={
            "review_comments": [
                GitHubPullRequestReviewComment(id=1, body="Must not be included.")
            ]
        }
    )

    without_result = GitHubIngestor(None).normalize(base, base_commit="a" * 40)
    with_result = GitHubIngestor(None).normalize(
        with_review_comments,
        base_commit="a" * 40,
    )

    assert with_result.incident.logs == without_result.incident.logs
    assert with_result.incident.model_dump() == without_result.incident.model_dump()


def test_ordinary_github_issue_does_not_expand_history() -> None:
    reference = parse_github_resource_url(
        "https://github.com/acme/widget/issues/9"
    )
    fetched = GitHubFetchedResource(
        reference=reference,
        detail=GitHubIssueDetail(
            number=9,
            title="Crash on empty config",
            body="The loader crashes when the config path is absent.",
            state="open",
            labels=[{"name": "bug"}],
        ),
    )

    result = GitHubIngestor(None).normalize(
        fetched,
        base_commit="a" * 40,
    )

    assert result.incident.resource_kind == "issue"
    assert result.incident.git_verification_policy.enabled is False
    assert result.incident.git_verification_policy.history_depth == 1
    assert result.incident.git_verification_policy.max_tool_calls == 1


def test_pull_request_rejects_mismatched_explicit_revision() -> None:
    reference = parse_github_resource_url(
        "https://github.com/acme/widget/pull/8"
    )
    base_sha = "a" * 40
    fetched = GitHubFetchedResource(
        reference=reference,
        detail=GitHubPullRequestDetail(
            number=8,
            title="Fix",
            body="Details",
            base={"sha": base_sha},
        ),
    )

    try:
        GitHubIngestor(None).normalize(fetched, base_commit="b" * 40)
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("mismatched PR revision was accepted")
