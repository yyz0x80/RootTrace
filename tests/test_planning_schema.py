import pytest
from pydantic import ValidationError

from patchpilot.planning.schema import (
    ChangeAction,
    ChangePlan,
    PlannedChange,
    PlannedTest,
)


def test_planned_change_basic():
    """Test creating a basic PlannedChange."""
    change = PlannedChange(
        path="src/auth.py",
        action="modify",
        description="Add password validation for special characters"
    )

    assert change.path == "src/auth.py"
    assert change.action == ChangeAction.MODIFY
    assert change.description == "Add password validation for special characters"
    assert change.acceptance_criteria == []


def test_planned_change_with_criteria():
    """Test PlannedChange with acceptance criteria."""
    change = PlannedChange(
        path="src/auth.py",
        action="modify",
        description="Add password validation for special characters",
        acceptance_criteria=[
            "Special characters are allowed in passwords",
            "Validation error is raised for invalid patterns"
        ]
    )

    assert len(change.acceptance_criteria) == 2
    assert "Special characters are allowed in passwords" in change.acceptance_criteria


def test_planned_change_create_action():
    """Test PlannedChange with create action."""
    change = PlannedChange(
        path="tests/test_auth.py",
        action="create",
        description="Add authentication tests"
    )

    assert change.action == ChangeAction.CREATE
    assert change.path == "tests/test_auth.py"


def test_planned_change_delete_action():
    """Test PlannedChange with delete action."""
    change = PlannedChange(
        path="src/deprecated.py",
        action="delete",
        description="Remove deprecated module"
    )

    assert change.action == ChangeAction.DELETE
    assert change.path == "src/deprecated.py"


def test_planned_change_invalid_action():
    """Test that invalid action is rejected."""
    with pytest.raises(ValidationError):
        PlannedChange(
            path="src/test.py",
            action="invalid",
            description="Test"
        )


def test_planned_test_basic():
    """Test creating a basic PlannedTest."""
    test = PlannedTest(
        command="pytest tests/test_auth.py",
        purpose="Verify authentication changes work correctly"
    )

    assert test.command == "pytest tests/test_auth.py"
    assert test.purpose == "Verify authentication changes work correctly"
    assert test.acceptance_criteria == []


def test_planned_test_with_criteria():
    """Test PlannedTest with acceptance criteria."""
    test = PlannedTest(
        command="pytest tests/test_auth.py -v",
        purpose="Verify authentication changes work correctly",
        acceptance_criteria=[
            "All tests pass",
            "No new warnings are introduced"
        ]
    )

    assert len(test.acceptance_criteria) == 2
    assert "All tests pass" in test.acceptance_criteria


def test_change_plan_basic():
    """Test creating a basic ChangePlan."""
    plan = ChangePlan(
        risk_level="low"
    )

    assert plan.risk_level == "low"
    assert plan.relevant_files == []
    assert plan.planned_changes == []
    assert plan.planned_tests == []
    assert plan.out_of_scope == []


def test_change_plan_full():
    """Test creating a complete ChangePlan with all fields."""
    plan = ChangePlan(
        base_commit="abc123",
        repository_match=True,
        relevant_files=["src/auth.py", "tests/test_auth.py"],
        planned_changes=[
            PlannedChange(
                path="src/auth.py",
                action="modify",
                description="Add password validation",
                acceptance_criteria=["Validation works correctly"]
            )
        ],
        planned_tests=[
            PlannedTest(
                command="pytest tests/test_auth.py",
                purpose="Verify authentication changes"
            )
        ],
        out_of_scope=["UI changes", "Database migration"],
        risk_level="medium"
    )

    assert len(plan.relevant_files) == 2
    assert len(plan.planned_changes) == 1
    assert len(plan.planned_tests) == 1
    assert len(plan.out_of_scope) == 2
    assert plan.risk_level == "medium"
    assert plan.base_commit == "abc123"
    assert plan.repository_match is True


def test_change_plan_risk_levels():
    """Test all valid risk levels."""
    for risk_level in ["low", "medium", "high"]:
        plan = ChangePlan(risk_level=risk_level)
        assert plan.risk_level == risk_level


def test_change_plan_invalid_risk_level():
    """Test that invalid risk levels are rejected."""
    with pytest.raises(ValidationError):
        ChangePlan(risk_level="critical")


def test_change_plan_serialization():
    """Test that ChangePlan can be serialized to JSON."""
    plan = ChangePlan(
        base_commit="abc123",
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

    json_str = plan.model_dump_json(indent=2)
    assert "src/main.py" in json_str
    assert "Fix bug" in json_str
    assert "low" in json_str
    assert "abc123" in json_str
    assert '"action"' in json_str
    assert '"path"' in json_str


def test_change_plan_round_trip():
    """Test that ChangePlan can be serialized and deserialized."""
    plan = ChangePlan(
        base_commit="abc123",
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

    json_str = plan.model_dump_json()
    restored = ChangePlan.model_validate_json(json_str)

    assert restored.base_commit == plan.base_commit
    assert restored.repository_match == plan.repository_match
    assert len(restored.planned_changes) == 1
    assert restored.planned_changes[0].path == "src/main.py"
    assert restored.planned_changes[0].action == ChangeAction.MODIFY
    assert restored.planned_changes[0].description == "Fix bug"


def test_planned_change_required_fields():
    """Test that required fields are enforced."""
    with pytest.raises(ValidationError):
        PlannedChange(description="Missing path and action fields")

    with pytest.raises(ValidationError):
        PlannedChange(path="src/test.py", description="Missing action field")

    with pytest.raises(ValidationError):
        PlannedChange(action="modify", description="Missing path field")


def test_planned_test_required_fields():
    """Test that required fields are enforced."""
    with pytest.raises(ValidationError):
        PlannedTest(purpose="Missing command field")

    with pytest.raises(ValidationError):
        PlannedTest(command="pytest")


def test_change_plan_risk_level_required():
    """Test that risk_level is required."""
    with pytest.raises(ValidationError):
        ChangePlan()


def test_change_plan_repository_mismatch():
    """Test ChangePlan with repository mismatch."""
    plan = ChangePlan(
        base_commit="abc123",
        repository_match=False,
        repository_mismatch_reason="No existing Task model found in repository",
        relevant_files=[],
        planned_changes=[],
        planned_tests=[],
        out_of_scope=[],
        risk_level="low"
    )

    assert plan.repository_match is False
    assert plan.repository_mismatch_reason == "No existing Task model found in repository"
    assert plan.base_commit == "abc123"


def test_change_action_enum_values():
    """Test that ChangeAction enum has correct values."""
    assert ChangeAction.CREATE.value == "create"
    assert ChangeAction.MODIFY.value == "modify"
    assert ChangeAction.DELETE.value == "delete"


def test_planned_change_serialization_action_lowercase():
    """Test that action serializes as lowercase string."""
    change = PlannedChange(
        path="src/test.py",
        action="modify",
        description="Test"
    )

    json_str = change.model_dump_json()
    assert '"action":"modify"' in json_str
    assert '"action":"MODIFY"' not in json_str


def test_planned_change_all_actions():
    """Test that all action types are valid."""
    for action in ["create", "modify", "delete"]:
        change = PlannedChange(
            path="src/test.py",
            action=action,
            description="Test"
        )
        assert change.action.value == action
