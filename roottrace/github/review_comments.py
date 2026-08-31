"""Deterministic selection and normalization of GitHub review comments."""

from __future__ import annotations

import re
from dataclasses import dataclass

from roottrace.incident.schema import (
    MAX_REVIEW_AUTHOR_CHARS,
    MAX_REVIEW_COMMENT_CHARS,
    MAX_REVIEW_COMMENT_TOTAL_CHARS,
    MAX_REVIEW_COMMENTS_PER_THREAD,
    MAX_REVIEW_SIDE_CHARS,
    MAX_REVIEW_THREADS,
    MAX_SOURCE_CHARS,
    Provenance,
    ReviewCommentEvidence,
    ReviewCommentLocationMapping,
    ReviewCommentThread,
    ReviewCommentTruncation,
    SourceLocation,
    validate_commit_sha,
)
from roottrace.runtime.paths import validate_relative_path

from .models import GitHubPullRequestReviewComment

_TERM_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_STOPWORDS = frozenset(
    {
        "and",
        "are",
        "but",
        "for",
        "from",
        "has",
        "have",
        "into",
        "not",
        "that",
        "the",
        "this",
        "was",
        "with",
    }
)
_TRUNCATION_MARKER = "\n...[truncated]"


@dataclass(frozen=True)
class _LocationMetadata:
    """Revision-aware location information for one comment anchor."""

    location: SourceLocation | None
    mapping: ReviewCommentLocationMapping
    applicable_commit: str | None
    path_valid: bool


@dataclass(frozen=True)
class _ThreadCandidate:
    """Raw thread data used by the deterministic selector."""

    root_comment_id: int
    comments: tuple[GitHubPullRequestReviewComment, ...]
    nonempty_comments: tuple[GitHubPullRequestReviewComment, ...]
    score: int
    score_reasons: tuple[str, ...]
    location: _LocationMetadata


def _path_only_location(location: _LocationMetadata) -> _LocationMetadata:
    """Retain only a valid path when the thread root is not available."""
    if location.location is None:
        return _LocationMetadata(None, "unmapped", None, location.path_valid)
    return _LocationMetadata(
        SourceLocation(path=location.location.path),
        "unmapped",
        None,
        location.path_valid,
    )


def _valid_commit(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        return validate_commit_sha(value)
    except ValueError:
        return None


def _same_revision(left: str | None, right: str) -> bool:
    return bool(left and left.lower() == right.lower())


def _safe_path(value: str | None) -> tuple[str | None, bool]:
    """Return a normalized repository path and whether a supplied path was invalid."""
    if value is None or not value.strip():
        return None, False
    try:
        return validate_relative_path(value.strip()), False
    except ValueError:
        return None, True


def _location_range(
    start_line: int | None,
    end_line: int | None,
) -> tuple[int | None, int | None]:
    if end_line is None:
        return None, None
    start = start_line or end_line
    if start > end_line:
        return None, None
    return start, end_line


def _location_metadata(
    comment: GitHubPullRequestReviewComment,
    *,
    base_commit: str,
) -> _LocationMetadata:
    path, path_invalid = _safe_path(comment.path)
    if path is None:
        return _LocationMetadata(None, "unmapped", None, not path_invalid)

    current_commit = _valid_commit(comment.commit_id)
    original_commit = _valid_commit(comment.original_commit_id)
    if _same_revision(current_commit, base_commit):
        mapping: ReviewCommentLocationMapping = "analysis_revision"
        applicable_commit = current_commit
        start_line, end_line = _location_range(comment.start_line, comment.line)
    elif _same_revision(original_commit, base_commit):
        mapping = "analysis_revision"
        applicable_commit = original_commit
        start_line, end_line = _location_range(
            comment.original_start_line,
            comment.original_line,
        )
    elif current_commit is not None:
        mapping = "current_comment_revision"
        applicable_commit = current_commit
        start_line, end_line = None, None
    elif original_commit is not None:
        mapping = "original_comment_revision"
        applicable_commit = original_commit
        start_line, end_line = None, None
    else:
        # A line without a known revision is not safe to apply to the checkout.
        return _LocationMetadata(
            SourceLocation(path=path),
            "unmapped",
            None,
            True,
        )

    return _LocationMetadata(
        SourceLocation(path=path, start_line=start_line, end_line=end_line),
        mapping,
        applicable_commit,
        True,
    )


def _comment_sort_key(comment: GitHubPullRequestReviewComment) -> tuple[str, int]:
    return (comment.created_at or "", comment.id)


def _issue_terms(text: str) -> set[str]:
    return {
        match.group(0).lower()
        for match in _TERM_PATTERN.finditer(text)
        if len(match.group(0)) <= 200
        and match.group(0).lower() not in _STOPWORDS
    }


def _thread_score(
    comments: tuple[GitHubPullRequestReviewComment, ...],
    *,
    location: _LocationMetadata,
    changed_files: set[str],
    issue_terms: set[str],
) -> tuple[int, tuple[str, ...]]:
    score = 0
    reasons: list[str] = []
    paths: set[str] = set()
    for comment in comments:
        path, _invalid = _safe_path(comment.path)
        if path is not None:
            paths.add(path)
    if paths & changed_files:
        score += 8
        reasons.append("changed_file")
    if location.mapping == "analysis_revision":
        score += 6
        reasons.append("analysis_revision")
    if any(
        comment.line is not None or comment.original_line is not None
        for comment in comments
    ):
        score += 2
        reasons.append("source_line")
    body_terms = _issue_terms(
        "\n".join((comment.body or "") for comment in comments)
    )
    overlap = sorted(issue_terms & body_terms)
    if overlap:
        score += min(6, 2 * len(overlap))
        reasons.append(
            "issue_terms:" + ",".join(term[:64] for term in overlap[:3])
        )
    if len(comments) > 1:
        score += 1
        reasons.append("threaded")
    return score, tuple(reasons)


def _thread_root_id(
    comment: GitHubPullRequestReviewComment,
    by_id: dict[int, GitHubPullRequestReviewComment],
) -> int:
    """Resolve a reply chain while remaining safe on missing or cyclic parents."""
    current = comment.id
    visited: set[int] = set()
    while current not in visited:
        visited.add(current)
        parent_id = by_id.get(current).in_reply_to_id if current in by_id else None
        if parent_id is None or parent_id not in by_id:
            return parent_id if parent_id is not None else current
        current = parent_id
    return current


def _comment_url(comment: GitHubPullRequestReviewComment, resource_url: str) -> str:
    if comment.html_url and len(comment.html_url) <= MAX_SOURCE_CHARS:
        return comment.html_url
    return f"{resource_url}#discussion_r{comment.id}"


def _truncate_text(text: str, limit: int) -> tuple[str, int]:
    """Return bounded text and the number of original characters omitted."""
    if len(text) <= limit:
        return text, 0
    if limit <= len(_TRUNCATION_MARKER):
        return text[:limit], len(text) - limit
    keep = limit - len(_TRUNCATION_MARKER)
    return text[:keep] + _TRUNCATION_MARKER, len(text) - keep


def _selected_reply_comments(
    candidate: _ThreadCandidate,
    max_comments_per_thread: int,
) -> list[GitHubPullRequestReviewComment]:
    root = next(
        (comment for comment in candidate.comments if comment.id == candidate.root_comment_id),
        None,
    )
    replies = [
        comment
        for comment in candidate.comments
        if comment.id != candidate.root_comment_id and (comment.body or "").strip()
    ]
    slots = max_comments_per_thread - (1 if root and (root.body or "").strip() else 0)
    latest = sorted(replies, key=_comment_sort_key, reverse=True)[:slots]
    latest.sort(key=_comment_sort_key)
    selected: list[GitHubPullRequestReviewComment] = []
    if root is not None and (root.body or "").strip():
        selected.append(root)
    selected.extend(latest)
    return selected[:max_comments_per_thread]


def _build_comment_evidence(
    comment: GitHubPullRequestReviewComment,
    *,
    thread_id: str,
    location: _LocationMetadata,
    location_source_comment_id: int,
    excerpt: str,
    resource_url: str,
) -> ReviewCommentEvidence:
    current_commit = _valid_commit(comment.commit_id)
    original_commit = _valid_commit(comment.original_commit_id)
    author = comment.user.login if comment.user and comment.user.login else None
    return ReviewCommentEvidence(
        id=f"ev-github-review-comment-{comment.id}",
        comment_id=comment.id,
        thread_id=thread_id,
        parent_comment_id=comment.in_reply_to_id,
        author=author[:MAX_REVIEW_AUTHOR_CHARS] if author else None,
        excerpt=excerpt,
        provenance=Provenance(
            source=_comment_url(comment, resource_url),
            tool="github_rest_client",
            commit=location.applicable_commit,
        ),
        location=location.location,
        location_source_comment_id=location_source_comment_id,
        location_mapping=location.mapping,
        side=comment.side[:MAX_REVIEW_SIDE_CHARS] if comment.side else None,
        start_side=(
            comment.start_side[:MAX_REVIEW_SIDE_CHARS]
            if comment.start_side
            else None
        ),
        commit_id=current_commit,
        original_commit_id=original_commit,
        pull_request_review_id=comment.pull_request_review_id,
        subject_type=(
            comment.subject_type[:MAX_REVIEW_SIDE_CHARS]
            if comment.subject_type
            else None
        ),
        created_at=comment.created_at,
        line=comment.line,
        start_line=comment.start_line,
        original_line=comment.original_line,
        original_start_line=comment.original_start_line,
        position=comment.position,
        original_position=comment.original_position,
    )


def _location_for_normalized_comment(
    comment: ReviewCommentEvidence,
    *,
    base_commit: str,
) -> _LocationMetadata:
    """Map persisted comment coordinates to a different analysis revision."""
    path = comment.location.path if comment.location is not None else None
    if path is None:
        return _LocationMetadata(None, "unmapped", None, False)

    current_commit = _valid_commit(comment.commit_id)
    original_commit = _valid_commit(comment.original_commit_id)
    if _same_revision(current_commit, base_commit):
        start_line, end_line = _location_range(comment.start_line, comment.line)
        return _LocationMetadata(
            SourceLocation(path=path, start_line=start_line, end_line=end_line),
            "analysis_revision",
            current_commit,
            True,
        )
    if _same_revision(original_commit, base_commit):
        start_line, end_line = _location_range(
            comment.original_start_line,
            comment.original_line,
        )
        return _LocationMetadata(
            SourceLocation(path=path, start_line=start_line, end_line=end_line),
            "analysis_revision",
            original_commit,
            True,
        )
    if current_commit is not None:
        return _LocationMetadata(
            SourceLocation(path=path),
            "current_comment_revision",
            current_commit,
            True,
        )
    if original_commit is not None:
        return _LocationMetadata(
            SourceLocation(path=path),
            "original_comment_revision",
            original_commit,
            True,
        )
    return _LocationMetadata(SourceLocation(path=path), "unmapped", None, True)


def map_review_comment_threads_to_revision(
    threads: list[ReviewCommentThread],
    *,
    base_commit: str,
) -> list[ReviewCommentThread]:
    """Re-map selected comments while preserving thread and evidence identity."""
    base_commit = validate_commit_sha(base_commit)
    remapped: list[ReviewCommentThread] = []
    for thread in threads:
        root = next(
            (
                comment
                for comment in thread.comments
                if comment.comment_id == thread.root_comment_id
            ),
            None,
        )
        root_location = (
            _location_for_normalized_comment(root, base_commit=base_commit)
            if root is not None
            else _path_only_location(
                _location_for_normalized_comment(
                    thread.comments[0],
                    base_commit=base_commit,
                )
            )
        )
        comments: list[ReviewCommentEvidence] = []
        for comment in thread.comments:
            location = (
                root_location
                if comment.comment_id != thread.root_comment_id
                else _location_for_normalized_comment(
                    comment,
                    base_commit=base_commit,
                )
            )
            provenance = comment.provenance.model_copy(
                update={"commit": location.applicable_commit}
            )
            comments.append(
                comment.model_copy(
                    update={
                        "location": location.location,
                        "location_mapping": location.mapping,
                        "location_source_comment_id": thread.root_comment_id,
                        "provenance": provenance,
                    }
                )
            )
        remapped.append(thread.model_copy(update={"comments": comments}))
    return remapped


def select_review_comment_threads(
    comments: list[GitHubPullRequestReviewComment],
    *,
    base_commit: str,
    changed_files: list[str] | tuple[str, ...] = (),
    incident_text: str = "",
    resource_url: str,
    max_threads: int = MAX_REVIEW_THREADS,
    max_comments_per_thread: int = MAX_REVIEW_COMMENTS_PER_THREAD,
    max_comment_chars: int = MAX_REVIEW_COMMENT_CHARS,
    max_total_chars: int = MAX_REVIEW_COMMENT_TOTAL_CHARS,
) -> tuple[list[ReviewCommentThread], ReviewCommentTruncation]:
    """Select bounded review threads without model inference.

    Threads are ranked by deterministic metadata and issue-term overlap. The
    root comment is retained when it has content, followed by the newest
    replies in chronological order. Every retained comment keeps its own
    provenance, even when its location is inherited from the thread root.
    """
    base_commit = validate_commit_sha(base_commit)
    if not 1 <= max_threads <= MAX_REVIEW_THREADS:
        raise ValueError("max_threads is outside the supported review budget")
    if not 1 <= max_comments_per_thread <= MAX_REVIEW_COMMENTS_PER_THREAD:
        raise ValueError("max_comments_per_thread is outside the supported review budget")
    if not 1 <= max_comment_chars <= MAX_REVIEW_COMMENT_CHARS:
        raise ValueError("max_comment_chars is outside the supported review budget")
    if not 1 <= max_total_chars <= MAX_REVIEW_COMMENT_TOTAL_CHARS:
        raise ValueError("max_total_chars is outside the supported review budget")

    # Keep the first occurrence as a second defensive boundary for callers
    # constructing fetched resources without the GitHub client.
    unique_comments: list[GitHubPullRequestReviewComment] = []
    seen_ids: set[int] = set()
    for comment in comments:
        if comment.id in seen_ids:
            continue
        seen_ids.add(comment.id)
        unique_comments.append(comment)

    by_id = {comment.id: comment for comment in unique_comments}
    grouped: dict[int, list[GitHubPullRequestReviewComment]] = {}
    for comment in unique_comments:
        grouped.setdefault(_thread_root_id(comment, by_id), []).append(comment)

    normalized_changed_files: set[str] = set()
    for value in changed_files:
        try:
            normalized_changed_files.add(validate_relative_path(value))
        except ValueError:
            continue

    candidates: list[_ThreadCandidate] = []
    invalid_paths = 0
    for root_id, grouped_comments in grouped.items():
        ordered_comments = tuple(sorted(grouped_comments, key=_comment_sort_key))
        root = by_id.get(root_id)
        anchor = root or ordered_comments[0]
        anchor_location = _location_metadata(anchor, base_commit=base_commit)
        for comment in ordered_comments:
            _path, path_invalid = _safe_path(comment.path)
            invalid_paths += int(path_invalid)
        nonempty = tuple(
            comment for comment in ordered_comments if (comment.body or "").strip()
        )
        score, reasons = _thread_score(
            ordered_comments,
            location=anchor_location,
            changed_files=normalized_changed_files,
            issue_terms=_issue_terms(incident_text),
        )
        candidates.append(
            _ThreadCandidate(
                root_comment_id=root_id,
                comments=ordered_comments,
                nonempty_comments=nonempty,
                score=score,
                score_reasons=reasons,
                location=anchor_location,
            )
        )

    candidates.sort(key=lambda item: (-item.score, item.root_comment_id))
    considered_comments = len(unique_comments)
    selected_threads: list[ReviewCommentThread] = []
    selected_comment_ids: set[int] = set()
    retained_lengths: dict[int, int] = {}
    total_chars_omitted = 0
    remaining_chars = max_total_chars

    for candidate in candidates[:max_threads]:
        selected_raw = _selected_reply_comments(candidate, max_comments_per_thread)
        if not selected_raw or remaining_chars <= 0:
            continue

        root = next(
            (
                comment
                for comment in candidate.comments
                if comment.id == candidate.root_comment_id
            ),
            None,
        )
        root_location = (
            _location_metadata(root, base_commit=base_commit)
            if root is not None
            else _path_only_location(candidate.location)
        )
        thread_id = f"review-thread-{candidate.root_comment_id}"
        evidence: list[ReviewCommentEvidence] = []
        for comment in selected_raw:
            if remaining_chars <= 0:
                break
            location = (
                root_location
                if comment.id != candidate.root_comment_id
                else _location_metadata(comment, base_commit=base_commit)
            )
            body = (comment.body or "").strip()
            bounded, omitted = _truncate_text(body, max_comment_chars)
            if len(bounded) > remaining_chars:
                bounded, global_omitted = _truncate_text(body, remaining_chars)
                omitted = max(omitted, global_omitted)
            if not bounded:
                continue
            total_chars_omitted += omitted
            retained_lengths[comment.id] = len(body) - omitted
            remaining_chars -= len(bounded)
            selected_comment_ids.add(comment.id)
            evidence.append(
                _build_comment_evidence(
                    comment,
                    thread_id=thread_id,
                    location=location,
                    location_source_comment_id=candidate.root_comment_id,
                    excerpt=bounded,
                    resource_url=resource_url,
                )
            )

        if evidence:
            omitted_in_thread = max(0, len(candidate.nonempty_comments) - len(evidence))
            selected_threads.append(
                ReviewCommentThread(
                    id=thread_id,
                    root_comment_id=candidate.root_comment_id,
                    rank=len(selected_threads) + 1,
                    score=candidate.score,
                    score_reasons=list(candidate.score_reasons),
                    comments=evidence,
                    comments_omitted=omitted_in_thread,
                )
            )

    all_body_chars = sum(
        len((comment.body or "").strip()) for comment in unique_comments
    )
    retained_chars = sum(retained_lengths.values())
    if all_body_chars >= retained_chars:
        total_chars_omitted = max(total_chars_omitted, all_body_chars - retained_chars)
    comments_omitted = max(0, considered_comments - len(selected_comment_ids))
    truncation = ReviewCommentTruncation(
        threads_considered=len(candidates),
        comments_considered=considered_comments,
        threads_omitted=max(0, len(candidates) - len(selected_threads)),
        comments_omitted=comments_omitted,
        chars_omitted=total_chars_omitted,
        locations_unmapped=sum(
            1
            for thread in selected_threads
            for comment in thread.comments
            if comment.location_mapping == "unmapped"
        ),
        invalid_paths=invalid_paths,
    )
    return selected_threads, truncation


build_review_comment_threads = select_review_comment_threads


__all__ = [
    "build_review_comment_threads",
    "map_review_comment_threads_to_revision",
    "select_review_comment_threads",
]
