"""Focused tests for strict GitHub issue and pull request references."""

import pytest

from roottrace.github import (
    GitHubIssueRef,
    GitHubPullRequestRef,
    GitHubResourceRef,
    parse_github_resource_url,
)


@pytest.mark.parametrize(
    ("url", "reference_type", "kind"),
    [
        (
            "https://github.com/acme/widget/issues/42",
            GitHubIssueRef,
            "issue",
        ),
        (
            "https://github.com/acme/widget/pull/7",
            GitHubPullRequestRef,
            "pull_request",
        ),
    ],
)
def test_parse_canonical_issue_and_pull_request_urls(
    url,
    reference_type,
    kind,
) -> None:
    """Supported URLs produce typed references and canonical URLs."""
    reference = parse_github_resource_url(url)

    assert isinstance(reference, reference_type)
    assert isinstance(reference, GitHubResourceRef)
    assert reference.repository.owner == "acme"
    assert reference.repository.repo == "widget"
    assert reference.number in {7, 42}
    assert reference.kind == kind
    assert reference.url == url
    assert reference.canonical_url == url


@pytest.mark.parametrize(
    "url",
    [
        "",
        " https://github.com/acme/widget/issues/1",
        "https://github.com/acme/widget/issues/1 ",
        "http://github.com/acme/widget/issues/1",
        "https://www.github.com/acme/widget/issues/1",
        "https://github.com:443/acme/widget/issues/1",
        "https://user:pass@github.com/acme/widget/issues/1",
        "https://github.com/acme/widget/issues/1?preview=1",
        "https://github.com/acme/widget/issues/1#comments",
        "https://github.com/acme/widget/issues/1/extra",
        "https://github.com/acme/widget/issues/1/",
        "https://github.com/acme/widget/pulls/1",
        "https://github.com/acme/widget/issues/0",
        "https://github.com/acme/widget/issues/01",
        "https://github.com/acme/widget/issues/not-a-number",
        "https://github.com/acme/widget/issues/1%2F2",
        "https://github.com/acme_/widget/issues/1",
        "https://github.com/acme/widget name/issues/1",
        "https://github.com/acme//issues/1",
    ],
)
def test_parse_rejects_noncanonical_or_invalid_urls(url: str) -> None:
    """Malformed, unsafe, and noncanonical paths raise clear ValueErrors."""
    with pytest.raises(ValueError):
        parse_github_resource_url(url)


def test_reference_model_derives_pull_canonical_url() -> None:
    """The model maps pull request kinds to GitHub's singular /pull/ route."""
    reference = GitHubPullRequestRef(
        repository={"owner": "acme", "repo": "widget"},
        number=7,
    )

    assert reference.canonical_url == "https://github.com/acme/widget/pull/7"


def test_reference_model_rejects_mismatched_canonical_url() -> None:
    """A caller cannot attach a URL that disagrees with the typed reference."""
    with pytest.raises(ValueError, match="canonical_url"):
        GitHubIssueRef(
            repository={"owner": "acme", "repo": "widget"},
            number=7,
            canonical_url="https://github.com/acme/widget/pull/7",
        )
