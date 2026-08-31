"""SWE-bench adapter tests: gold data must never reach RootTrace input."""

from __future__ import annotations

import json

import pytest
from fixtures import gold_patch, gold_record, public_record

from evaluation.adapter import (
    PublicCase,
    build_incident_input,
    case_from_public_record,
    write_root_trace_input,
)

_REPO = "acme/demo"
_BASE = "a" * 40


def _public(problem: str = "Something is broken.", **extra: object) -> dict:
    return public_record("acme__demo-1", _REPO, _BASE, problem, **extra)


def test_gold_record_rejected_by_public_parser() -> None:
    record = gold_record(
        "acme__demo-1",
        patch=gold_patch(["src/mod.py"]),
        test_patch=gold_patch([], ["tests/test_mod.py"]),
        fail_to_pass=["tests/test_mod.py::test_x"],
        pass_to_pass=["tests/test_mod.py::test_y"],
    )
    with pytest.raises(ValueError, match="disallowed fields"):
        case_from_public_record(record)


def test_extra_unknown_field_rejected() -> None:
    record = _public()
    record["pr_url"] = "https://example.invalid/pr/1"
    with pytest.raises(ValueError, match="disallowed fields"):
        case_from_public_record(record)


def test_build_incident_input_forwards_only_allowed_fields() -> None:
    case = case_from_public_record(_public())
    result = build_incident_input(case)
    incident = result.incident
    assert incident.id == "acme__demo-1"
    assert incident.repo == _REPO
    assert incident.base_commit == _BASE
    assert incident.problem == "Something is broken."
    assert incident.logs == []
    assert incident.diff is None
    assert "patch" not in incident.model_dump()
    assert result.problem_chars_omitted == 0


def test_public_metadata_fields_never_forwarded() -> None:
    case = case_from_public_record(
        _public(created_at="2022-01-01T00:00:00Z", difficulty="easy")
    )
    incident = build_incident_input(case).incident
    assert set(incident.model_dump()) == {
        "id",
        "repo",
        "base_commit",
        "resource_kind",
        "title",
        "problem",
        "logs",
        "diff",
        "labels",
        "related_commits",
        "changed_files",
        "review_threads",
        "review_comment_truncation",
        "git_verification_policy",
        "provenance",
    }
    assert not {
        "created_at",
        "difficulty",
        "version",
        "patch",
        "test_patch",
    }.intersection(incident.model_dump())


def test_write_root_trace_input_contains_exactly_four_fields(tmp_path) -> None:
    case = case_from_public_record(_public())
    incident = build_incident_input(case).incident
    path = tmp_path / "input.json"
    write_root_trace_input(incident, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload) == {"instance_id", "repo", "base_commit", "problem_statement"}
    assert payload["problem_statement"] == "Something is broken."


def test_long_problem_truncated_with_marker() -> None:
    from roottrace.incident.schema import MAX_PROBLEM_CHARS

    problem = "x" * (MAX_PROBLEM_CHARS + 500)
    case = case_from_public_record(_public(problem=problem))
    result = build_incident_input(case)
    assert result.problem_chars_omitted == 500
    assert result.incident.problem.endswith("...[truncated: 500 chars omitted]")
    assert len(result.incident.problem) <= MAX_PROBLEM_CHARS
    assert result.notes


def test_invalid_repo_and_commit_rejected() -> None:
    record = _public()
    record["repo"] = "/absolute/path"
    with pytest.raises(ValueError):
        case_from_public_record(record)
    record = _public()
    record["base_commit"] = "HEAD"
    with pytest.raises(ValueError):
        case_from_public_record(record)
    with pytest.raises(ValueError):
        PublicCase(instance_id="x", repo="acme/demo", base_commit=_BASE, problem="")
