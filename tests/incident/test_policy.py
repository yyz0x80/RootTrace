"""Tests for deterministic Git verification policy derivation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from roottrace.incident.schema import (
    MAX_GIT_HISTORY_DEPTH,
    IncidentInput,
    Provenance,
    build_git_verification_policy,
)


def _incident(**overrides) -> IncidentInput:
    fields = {
        "id": "inc-policy",
        "repo": "owner/repo",
        "base_commit": "a" * 40,
        "title": "Configuration failure",
        "problem": "The loader raises ValueError.",
        "provenance": Provenance(source="issue.md"),
    }
    fields.update(overrides)
    return IncidentInput(**fields)


def test_ordinary_issue_keeps_git_history_at_depth_one() -> None:
    incident = _incident()

    assert incident.git_verification_policy.model_dump(mode="json") == {
        "enabled": False,
        "reasons": ["not_triggered"],
        "history_depth": 1,
        "search_depths": [1],
        "candidate_commits": [],
        "candidate_paths": [],
        "max_tool_calls": 1,
    }


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"title": "Regression in configuration loading"}, "regression_signal"),
        ({"labels": ["bug", "Regression"]}, "regression_label"),
        ({"problem": "Fails after commit abc1234."}, "commit_sha"),
        ({"problem": f"Fails after commit {'b' * 40}."}, "commit_sha"),
    ],
)
def test_issue_regression_signals_enable_bounded_git_verification(
    overrides: dict[str, object],
    reason: str,
) -> None:
    incident = _incident(**overrides)
    policy = incident.git_verification_policy

    assert policy.enabled is True
    assert reason in policy.reasons
    assert policy.history_depth == MAX_GIT_HISTORY_DEPTH
    assert policy.search_depths == [8, 16, 32, 50]
    assert policy.max_tool_calls == 5


def test_issue_with_generic_commit_word_does_not_expand_history() -> None:
    incident = _incident(problem="Please review the commit message.")

    assert incident.git_verification_policy.enabled is False
    assert incident.git_verification_policy.history_depth == 1


def test_issue_error_code_does_not_look_like_a_commit() -> None:
    incident = _incident(problem="The service returns error code 1234567.")

    assert incident.git_verification_policy.enabled is False
    assert incident.git_verification_policy.history_depth == 1


def test_pull_request_always_enables_policy_and_collects_paths() -> None:
    incident = _incident(
        resource_kind="pull_request",
        changed_files=["src/config.py"],
        diff="diff --git a/tests/test_config.py b/tests/test_config.py\n",
    )

    assert incident.git_verification_policy.enabled is True
    assert incident.git_verification_policy.reasons == ["pull_request"]
    assert incident.git_verification_policy.candidate_paths == [
        "src/config.py",
        "tests/test_config.py",
    ]


def test_policy_is_derived_and_cannot_be_overridden() -> None:
    with pytest.raises(ValidationError, match="deterministically derived"):
        _incident(
            title="Regression in configuration loading",
            git_verification_policy={
                "enabled": False,
                "reasons": ["not_triggered"],
                "history_depth": 1,
                "max_tool_calls": 1,
            },
        )


def test_policy_builder_is_deterministic_and_caps_candidates() -> None:
    policy = build_git_verification_policy(
        resource_kind="issue",
        title="Regression report",
        problem="The failure was introduced by a commit.",
        related_commits=["C" * 40, "d" * 40],
        changed_files=[f"src/module_{index}.py" for index in range(30)],
    )

    assert policy.enabled is True
    assert policy.candidate_commits == ["c" * 40, "d" * 40]
    assert len(policy.candidate_paths) == 20
    assert policy.candidate_paths == sorted(policy.candidate_paths)
