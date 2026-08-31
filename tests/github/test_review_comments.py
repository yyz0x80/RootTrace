"""Tests for deterministic pull request review-comment normalization."""

from __future__ import annotations

from roottrace.github import (
    GitHubPullRequestReviewComment,
    build_review_comment_threads,
    map_review_comment_threads_to_revision,
    select_review_comment_threads,
)
from roottrace.incident.schema import ReviewCommentThread, ReviewCommentTruncation

BASE_COMMIT = "a" * 40
CURRENT_COMMIT = "b" * 40
RESOURCE_URL = "https://github.com/acme/widget/pull/8"


def _comment(
    comment_id: int,
    body: str,
    *,
    created_at: str | None = None,
    **fields: object,
) -> GitHubPullRequestReviewComment:
    return GitHubPullRequestReviewComment(
        id=comment_id,
        body=body,
        created_at=created_at,
        html_url=f"{RESOURCE_URL}#discussion_r{comment_id}",
        **fields,
    )


def _select(
    comments: list[GitHubPullRequestReviewComment],
    **kwargs: object,
) -> tuple[list[ReviewCommentThread], ReviewCommentTruncation]:
    return select_review_comment_threads(
        comments,
        base_commit=BASE_COMMIT,
        resource_url=RESOURCE_URL,
        **kwargs,
    )


def test_empty_review_comments_are_bounded_and_typed() -> None:
    threads, truncation = _select([])

    assert threads == []
    assert truncation.model_dump() == {
        "threads_considered": 0,
        "comments_considered": 0,
        "threads_omitted": 0,
        "comments_omitted": 0,
        "chars_omitted": 0,
        "locations_unmapped": 0,
        "invalid_paths": 0,
    }


def test_threads_keep_root_and_latest_replies_with_inherited_location() -> None:
    root = _comment(
        20,
        "Root review comment.",
        created_at="2024-01-01T00:00:00Z",
        path="src/app.py",
        line=12,
        start_line=10,
        commit_id=BASE_COMMIT,
    )
    older_reply = _comment(
        21,
        "Older reply.",
        created_at="2024-01-02T00:00:00Z",
        path="src/app.py",
        line=13,
        commit_id=CURRENT_COMMIT,
        in_reply_to_id=20,
    )
    newest_reply = _comment(
        22,
        "Newest reply.",
        created_at="2024-01-03T00:00:00Z",
        path="src/app.py",
        line=14,
        commit_id=CURRENT_COMMIT,
        in_reply_to_id=20,
    )

    threads, _ = _select(
        [newest_reply, root, older_reply],
        max_comments_per_thread=2,
    )

    assert len(threads) == 1
    assert [comment.comment_id for comment in threads[0].comments] == [20, 22]
    assert threads[0].root_comment_id == 20
    assert threads[0].comments[1].parent_comment_id == 20
    assert threads[0].comments[1].location == threads[0].comments[0].location
    assert threads[0].comments[1].location_source_comment_id == 20
    assert threads[0].comments[0].location_mapping == "analysis_revision"
    assert threads[0].comments[1].location_mapping == "analysis_revision"


def test_location_mapping_is_conservative_across_revisions() -> None:
    current = _comment(
        1,
        "Current revision comment.",
        path="src/current.py",
        line=8,
        commit_id=CURRENT_COMMIT,
    )
    original = _comment(
        2,
        "Original revision comment.",
        path="src/original.py",
        line=20,
        start_line=18,
        original_line=12,
        original_start_line=10,
        commit_id=CURRENT_COMMIT,
        original_commit_id=BASE_COMMIT,
    )
    unmapped = _comment(
        3,
        "Revision is unavailable.",
        path="src/unknown.py",
        line=4,
    )

    threads, truncation = _select([current, original, unmapped])

    by_id = {
        thread.comments[0].comment_id: thread.comments[0]
        for thread in threads
    }
    assert by_id[1].location_mapping == "current_comment_revision"
    assert by_id[1].location is not None
    assert by_id[1].location.path == "src/current.py"
    assert by_id[1].location.start_line is None
    assert by_id[1].location.end_line is None
    assert by_id[2].location_mapping == "analysis_revision"
    assert by_id[2].location is not None
    assert by_id[2].location.start_line == 10
    assert by_id[2].location.end_line == 12
    assert by_id[3].location_mapping == "unmapped"
    assert by_id[3].location is not None
    assert by_id[3].location.path == "src/unknown.py"
    assert by_id[3].location.start_line is None
    assert truncation.locations_unmapped == 1


def test_missing_root_does_not_assign_reply_line_to_the_checkout() -> None:
    reply = _comment(
        2,
        "Reply received without its root.",
        path="src/app.py",
        line=20,
        commit_id=BASE_COMMIT,
        in_reply_to_id=1,
    )

    threads, _ = _select([reply])

    evidence = threads[0].comments[0]
    assert threads[0].root_comment_id == 1
    assert evidence.location_mapping == "unmapped"
    assert evidence.location is not None
    assert evidence.location.path == "src/app.py"
    assert evidence.location.start_line is None
    assert evidence.location_source_comment_id == 1


def test_ranking_is_stable_and_prefers_changed_files_and_issue_terms() -> None:
    cold = _comment(
        10,
        "Unrelated reviewer observation.",
        created_at="2024-01-01T00:00:00Z",
        path="src/cold.py",
        commit_id=CURRENT_COMMIT,
    )
    hot = _comment(
        11,
        "The cache timeout path is affected.",
        created_at="2024-01-01T00:00:00Z",
        path="src/hot.py",
        line=5,
        commit_id=BASE_COMMIT,
    )

    first, _ = _select(
        [cold, hot],
        changed_files=["src/hot.py"],
        incident_text="cache timeout failure",
    )
    second, _ = _select(
        [hot, cold],
        changed_files=["src/hot.py"],
        incident_text="cache timeout failure",
    )

    assert [thread.root_comment_id for thread in first] == [11, 10]
    assert [thread.id for thread in first] == [thread.id for thread in second]
    assert first[0].rank == 1
    assert "changed_file" in first[0].score_reasons
    assert any(reason.startswith("issue_terms:") for reason in first[0].score_reasons)


def test_selection_deduplicates_ids_and_records_budget_omissions() -> None:
    duplicate = _comment(1, "duplicate API entry")
    comments = [duplicate, _comment(1, "later duplicate"), _comment(2, "second")]
    threads, truncation = _select(
        comments,
        max_threads=1,
        max_comments_per_thread=1,
    )

    assert [comment.comment_id for comment in threads[0].comments] == [1]
    assert truncation.comments_considered == 2
    assert truncation.comments_omitted == 1

    many = [
        _comment(
            index,
            "x" * 200,
            created_at=f"2024-01-{index:02d}T00:00:00Z",
        )
        for index in range(1, 7)
    ]
    limited, limited_truncation = _select(
        many,
        max_comment_chars=100,
        max_total_chars=250,
    )
    assert len(limited) <= 5
    assert sum(len(comment.excerpt) for thread in limited for comment in thread.comments) <= 250
    assert limited_truncation.threads_omitted > 0
    assert limited_truncation.comments_omitted > 0
    assert limited_truncation.chars_omitted > 0


def test_invalid_paths_are_not_exposed_as_source_locations() -> None:
    threads, truncation = _select(
        [_comment(1, "Do not trust this path.", path="../outside.py", line=3)]
    )

    assert threads[0].comments[0].location is None
    assert threads[0].comments[0].location_mapping == "unmapped"
    assert truncation.invalid_paths == 1


def test_malformed_line_range_keeps_only_the_safe_path() -> None:
    threads, _ = _select(
        [
            _comment(
                1,
                "Invalid range from API.",
                path="src/app.py",
                line=3,
                start_line=5,
                commit_id=BASE_COMMIT,
            )
        ]
    )

    location = threads[0].comments[0].location
    assert location is not None
    assert location.path == "src/app.py"
    assert location.start_line is None
    assert location.end_line is None


def test_compatibility_alias_matches_selector() -> None:
    comment = _comment(1, "same result", path="src/app.py")

    assert build_review_comment_threads(
        [comment],
        base_commit=BASE_COMMIT,
        resource_url=RESOURCE_URL,
    ) == _select([comment])


def test_selected_threads_can_be_remapped_for_composed_analysis_revision() -> None:
    comment = _comment(
        1,
        "Review on another revision.",
        path="src/app.py",
        line=7,
        commit_id=CURRENT_COMMIT,
    )
    threads, _ = _select([comment])

    remapped = map_review_comment_threads_to_revision(
        threads,
        base_commit="c" * 40,
    )

    evidence = remapped[0].comments[0]
    assert evidence.location_mapping == "current_comment_revision"
    assert evidence.location is not None
    assert evidence.location.path == "src/app.py"
    assert evidence.location.start_line is None
    assert evidence.provenance.commit == CURRENT_COMMIT
