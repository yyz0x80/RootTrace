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
    assert result.changed_files == ["src/config.py"]
    assert "diff --git a/src/config.py" in (result.incident.diff or "")
    assert result.revision_kind == "pull_request_base"


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
