"""Tests for the scope_gate module."""

import pytest

from patchpilot.planning.schema import ChangePlan, PlannedChange
from patchpilot.planning.scope_gate import (
    check_scope,
    validate_plan_against_repository,
)
from patchpilot.repository.schema import RepositoryContext


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


def test_check_scope_basic():
    """Test basic scope check functionality."""
    plan = ChangePlan(
        repository_match=True,
        relevant_files=["src/main.py"],
        planned_changes=[
            PlannedChange(
                path="src/main.py",
                action="modify",
                description="Fix bug"
            )
        ],
        risk_level="low"
    )

    result = check_scope(plan)

    assert result.allowed is True
    assert len(result.violations) == 0
    assert len(result.warnings) == 0


def test_check_scope_too_many_files():
    """Test scope check rejects plans with too many file changes."""
    plan = ChangePlan(
        repository_match=True,
        relevant_files=["src/1.py", "src/2.py", "src/3.py", "src/4.py", "src/5.py", "src/6.py", "src/7.py"],
        planned_changes=[
            PlannedChange(path=f"src/{i}.py", action="modify", description=f"Change {i}")
            for i in range(1, 8)
        ],
        risk_level="low"
    )

    result = check_scope(plan, max_modified_files=6)

    assert result.allowed is False
    assert len(result.violations) == 1
    assert "7 files" in result.violations[0]
    assert "maximum allowed is 6" in result.violations[0]


def test_check_scope_forbidden_file():
    """Test scope check rejects forbidden files."""
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

    result = check_scope(plan)

    assert result.allowed is False
    assert len(result.violations) == 1
    assert ".env" in result.violations[0]
    assert "forbidden" in result.violations[0]


def test_check_scope_forbidden_prefix():
    """Test scope check rejects forbidden path prefixes."""
    plan = ChangePlan(
        repository_match=True,
        relevant_files=[".github/workflows/ci.yml"],
        planned_changes=[
            PlannedChange(
                path=".github/workflows/ci.yml",
                action="modify",
                description="Update CI"
            )
        ],
        risk_level="low"
    )

    result = check_scope(plan)

    assert result.allowed is False
    assert len(result.violations) == 1
    assert "CI/CD modification is forbidden" in result.violations[0]


def test_check_scope_database_migration():
    """Test scope check warns about database migrations."""
    plan = ChangePlan(
        repository_match=True,
        relevant_files=["alembic/versions/001_migration.py"],
        planned_changes=[
            PlannedChange(
                path="alembic/versions/001_migration.py",
                action="create",
                description="Add migration"
            )
        ],
        risk_level="low"
    )

    result = check_scope(plan)

    assert result.allowed is False
    assert len(result.violations) == 1
    assert "Database migration" in result.violations[0]


def test_check_scope_not_in_relevant_files():
    """Test scope check warns about files not in relevant_files."""
    plan = ChangePlan(
        repository_match=True,
        relevant_files=["src/main.py"],
        planned_changes=[
            PlannedChange(
                path="src/utils.py",
                action="modify",
                description="Fix bug"
            )
        ],
        risk_level="low"
    )

    result = check_scope(plan)

    assert result.allowed is True
    assert len(result.violations) == 0
    assert len(result.warnings) == 1
    assert "src/utils.py" in result.warnings[0]
    assert "not listed in relevant_files" in result.warnings[0]


def test_check_scope_high_risk():
    """Test scope check rejects high-risk plans."""
    plan = ChangePlan(
        repository_match=True,
        relevant_files=["src/main.py"],
        planned_changes=[
            PlannedChange(
                path="src/main.py",
                action="modify",
                description="Major refactor"
            )
        ],
        risk_level="high"
    )

    result = check_scope(plan)

    assert result.allowed is False
    assert len(result.violations) == 1
    assert "High-risk plan" in result.violations[0]


def test_check_scope_multiple_violations():
    """Test scope check accumulates multiple violations."""
    plan = ChangePlan(
        repository_match=True,
        relevant_files=[".env", ".github/workflows/ci.yml"],
        planned_changes=[
            PlannedChange(path=".env", action="modify", description="Update config"),
            PlannedChange(path=".github/workflows/ci.yml", action="modify", description="Update CI")
        ],
        risk_level="high"
    )

    result = check_scope(plan)

    assert result.allowed is False
    assert len(result.violations) == 3  # .env, CI/CD, high risk
