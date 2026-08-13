"""Tests for the plan validator module."""

import pytest

from patchpilot.planning.schema import ChangePlan, PlannedChange
from patchpilot.planning.validator import (
    validate_plan,
    validate_plan_against_repository,
)
from patchpilot.repository.schema import RepositoryContext


def test_validate_plan_success():
    """Test validation passes for a valid plan with proper repository context."""
    plan = ChangePlan(
        repository_match=True,
        relevant_files=["src/main.py", "src/utils.py"],
        planned_changes=[
            PlannedChange(
                path="src/main.py",
                action="modify",
                description="Fix bug"
            ),
            PlannedChange(
                path="src/utils.py",
                action="modify",
                description="Update utility function"
            )
        ],
        risk_level="low"
    )

    repo_context = RepositoryContext(
        base_commit="abc123",
        tracked_files=["src/main.py", "src/utils.py", "tests/test_main.py"],
        python_files=["src/main.py", "src/utils.py"],
        test_files=["tests/test_main.py"],
        config_files=[],
        keyword_matches=[]
    )

    result = validate_plan(plan, repo_context)

    assert result.allowed is True
    assert len(result.violations) == 0
    assert len(result.warnings) == 0


def test_validate_plan_with_scope_violation():
    """Test validation catches scope violations."""
    plan = ChangePlan(
        repository_match=True,
        relevant_files=[".env"],
        planned_changes=[
            PlannedChange(
                path=".env",
                action="modify",
                description="Update config"
            )
        ],
        risk_level="low"
    )

    repo_context = RepositoryContext(
        base_commit="abc123",
        tracked_files=[".env", "src/main.py"],
        python_files=["src/main.py"],
        test_files=[],
        config_files=[],
        keyword_matches=[]
    )

    result = validate_plan(plan, repo_context)

    assert result.allowed is False
    assert len(result.violations) > 0
    assert any(".env" in v for v in result.violations)


def test_validate_plan_with_repository_mismatch():
    """Test validation raises error for repository mismatch."""
    plan = ChangePlan(
        repository_match=False,
        repository_mismatch_reason="No existing Task model found",
        planned_changes=[],
        risk_level="low"
    )

    repo_context = RepositoryContext(
        base_commit="abc123",
        tracked_files=["src/main.py"],
        python_files=["src/main.py"],
        test_files=[],
        config_files=[],
        keyword_matches=[]
    )

    with pytest.raises(ValueError) as exc_info:
        validate_plan(plan, repo_context)

    assert "Issue does not match repository" in str(exc_info.value)
    assert "No existing Task model found" in str(exc_info.value)


def test_validate_plan_with_inconsistent_files():
    """Test validation raises error for inconsistent file references."""
    plan = ChangePlan(
        repository_match=True,
        relevant_files=["src/missing.py"],
        planned_changes=[
            PlannedChange(
                path="src/missing.py",
                action="modify",
                description="Fix bug"
            )
        ],
        risk_level="low"
    )

    repo_context = RepositoryContext(
        base_commit="abc123",
        tracked_files=["src/main.py"],
        python_files=["src/main.py"],
        test_files=[],
        config_files=[],
        keyword_matches=[]
    )

    with pytest.raises(ValueError) as exc_info:
        validate_plan(plan, repo_context)

    assert "non-existing file" in str(exc_info.value)
    assert "modify" in str(exc_info.value)


def test_validate_plan_against_repository_success():
    """Test validation passes for a valid plan."""
    plan = ChangePlan(
        repository_match=True,
        planned_changes=[
            PlannedChange(
                path="src/main.py",
                action="modify",
                description="Fix bug"
            ),
            PlannedChange(
                path="tests/test_new.py",
                action="create",
                description="Add new test"
            )
        ],
        risk_level="low"
    )

    repo_context = RepositoryContext(
        base_commit="abc123",
        tracked_files=["src/main.py", "src/utils.py", "tests/test_main.py"],
        python_files=["src/main.py", "src/utils.py"],
        test_files=["tests/test_main.py"],
        config_files=[],
        keyword_matches=[]
    )

    # Should not raise
    validate_plan_against_repository(plan, repo_context)


def test_validate_plan_against_repository_mismatch():
    """Test validation fails when repository_match is False."""
    plan = ChangePlan(
        repository_match=False,
        repository_mismatch_reason="No existing Task model found",
        planned_changes=[],
        risk_level="low"
    )

    repo_context = RepositoryContext(
        base_commit="abc123",
        tracked_files=["src/main.py"],
        python_files=["src/main.py"],
        test_files=[],
        config_files=[],
        keyword_matches=[]
    )

    with pytest.raises(ValueError) as exc_info:
        validate_plan_against_repository(plan, repo_context)

    assert "Issue does not match repository" in str(exc_info.value)
    assert "No existing Task model found" in str(exc_info.value)


def test_validate_plan_against_repository_modify_nonexistent():
    """Test validation fails when modifying a non-existent file."""
    plan = ChangePlan(
        repository_match=True,
        planned_changes=[
            PlannedChange(
                path="src/missing.py",
                action="modify",
                description="Fix bug"
            )
        ],
        risk_level="low"
    )

    repo_context = RepositoryContext(
        base_commit="abc123",
        tracked_files=["src/main.py"],
        python_files=["src/main.py"],
        test_files=[],
        config_files=[],
        keyword_matches=[]
    )

    with pytest.raises(ValueError) as exc_info:
        validate_plan_against_repository(plan, repo_context)

    assert "Plan references a non-existing file" in str(exc_info.value)
    assert "modify" in str(exc_info.value)
    assert "src/missing.py" in str(exc_info.value)


def test_validate_plan_against_repository_delete_nonexistent():
    """Test validation fails when deleting a non-existent file."""
    plan = ChangePlan(
        repository_match=True,
        planned_changes=[
            PlannedChange(
                path="src/missing.py",
                action="delete",
                description="Remove file"
            )
        ],
        risk_level="low"
    )

    repo_context = RepositoryContext(
        base_commit="abc123",
        tracked_files=["src/main.py"],
        python_files=["src/main.py"],
        test_files=[],
        config_files=[],
        keyword_matches=[]
    )

    with pytest.raises(ValueError) as exc_info:
        validate_plan_against_repository(plan, repo_context)

    assert "Plan references a non-existing file" in str(exc_info.value)
    assert "delete" in str(exc_info.value)
    assert "src/missing.py" in str(exc_info.value)


def test_validate_plan_against_repository_create_existing():
    """Test validation fails when creating a file that already exists."""
    plan = ChangePlan(
        repository_match=True,
        planned_changes=[
            PlannedChange(
                path="src/main.py",
                action="create",
                description="Add new file"
            )
        ],
        risk_level="low"
    )

    repo_context = RepositoryContext(
        base_commit="abc123",
        tracked_files=["src/main.py"],
        python_files=["src/main.py"],
        test_files=[],
        config_files=[],
        keyword_matches=[]
    )

    with pytest.raises(ValueError) as exc_info:
        validate_plan_against_repository(plan, repo_context)

    assert "Plan attempts to create an existing file" in str(exc_info.value)
    assert "src/main.py" in str(exc_info.value)


def test_validate_plan_against_repository_mixed_actions():
    """Test validation with mixed create, modify, and delete actions."""
    plan = ChangePlan(
        repository_match=True,
        planned_changes=[
            PlannedChange(
                path="src/main.py",
                action="modify",
                description="Update existing file"
            ),
            PlannedChange(
                path="src/deprecated.py",
                action="delete",
                description="Remove deprecated file"
            ),
            PlannedChange(
                path="src/new_feature.py",
                action="create",
                description="Add new feature"
            )
        ],
        risk_level="low"
    )

    repo_context = RepositoryContext(
        base_commit="abc123",
        tracked_files=["src/main.py", "src/deprecated.py", "src/utils.py"],
        python_files=["src/main.py", "src/deprecated.py", "src/utils.py"],
        test_files=[],
        config_files=[],
        keyword_matches=[]
    )

    # Should not raise
    validate_plan_against_repository(plan, repo_context)
