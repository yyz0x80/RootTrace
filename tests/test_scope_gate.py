"""Tests for the scope_gate module."""

import pytest

from patchpilot.planning.schema import ChangePlan, PlannedChange
from patchpilot.planning.scope_gate import check_scope, validate_actual_changes
from patchpilot.tools import WorkspaceChange


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


class TestValidateActualChanges:
    """Tests for runtime scope validation of actual changes."""

    def test_validate_actual_changes_success(self):
        """Test validation passes when actual changes match the plan."""
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

        actual_changes = [
            WorkspaceChange(path="src/main.py", action="modify")
        ]

        # Should not raise any exception
        validate_actual_changes(plan, actual_changes)

    def test_validate_actual_changes_env_forbidden(self):
        """Test validation rejects .env modifications."""
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

        actual_changes = [
            WorkspaceChange(path=".env", action="modify")
        ]

        with pytest.raises(RuntimeError) as exc_info:
            validate_actual_changes(plan, actual_changes)

        assert ".env" in str(exc_info.value)
        assert "forbidden" in str(exc_info.value)

    def test_validate_actual_changes_ci_workflow_forbidden(self):
        """Test validation rejects CI workflow modifications."""
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

        actual_changes = [
            WorkspaceChange(path=".github/workflows/ci.yml", action="modify")
        ]

        with pytest.raises(RuntimeError) as exc_info:
            validate_actual_changes(plan, actual_changes)

        assert "CI workflow" in str(exc_info.value)
        assert "forbidden" in str(exc_info.value)

    def test_validate_actual_changes_unplanned_file(self):
        """Test validation rejects changes to files not in the plan."""
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

        actual_changes = [
            WorkspaceChange(path="src/unplanned.py", action="modify")
        ]

        with pytest.raises(RuntimeError) as exc_info:
            validate_actual_changes(plan, actual_changes)

        assert "outside the approved plan" in str(exc_info.value)
        assert "src/unplanned.py" in str(exc_info.value)

    def test_validate_actual_changes_action_mismatch(self):
        """Test validation rejects action type mismatches."""
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

        actual_changes = [
            WorkspaceChange(path="src/main.py", action="create")  # Wrong action
        ]

        with pytest.raises(RuntimeError) as exc_info:
            validate_actual_changes(plan, actual_changes)

        assert "Unexpected change action" in str(exc_info.value)
        assert "expected modify" in str(exc_info.value)
        assert "got create" in str(exc_info.value)

    def test_validate_actual_changes_multiple_files(self):
        """Test validation with multiple planned changes."""
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
                    action="create",
                    description="Add utility"
                )
            ],
            risk_level="low"
        )

        actual_changes = [
            WorkspaceChange(path="src/main.py", action="modify"),
            WorkspaceChange(path="src/utils.py", action="create")
        ]

        # Should not raise any exception
        validate_actual_changes(plan, actual_changes)

    def test_validate_actual_changes_delete_action(self):
        """Test validation handles delete actions correctly."""
        plan = ChangePlan(
            repository_match=True,
            relevant_files=["src/old.py"],
            planned_changes=[
                PlannedChange(
                    path="src/old.py",
                    action="delete",
                    description="Remove deprecated file"
                )
            ],
            risk_level="low"
        )

        actual_changes = [
            WorkspaceChange(path="src/old.py", action="delete")
        ]

        # Should not raise any exception
        validate_actual_changes(plan, actual_changes)
