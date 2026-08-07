import pytest
from pydantic import ValidationError

from patchpilot.planning.schema import ChangePlan, PlannedChange, PlannedTest


def test_planned_change_basic():
    """Test creating a basic PlannedChange."""
    change = PlannedChange(
        file="src/auth.py",
        description="Add password validation for special characters"
    )

    assert change.file == "src/auth.py"
    assert change.description == "Add password validation for special characters"
    assert change.acceptance_criteria == []


def test_planned_change_with_criteria():
    """Test PlannedChange with acceptance criteria."""
    change = PlannedChange(
        file="src/auth.py",
        description="Add password validation for special characters",
        acceptance_criteria=[
            "Special characters are allowed in passwords",
            "Validation error is raised for invalid patterns"
        ]
    )

    assert len(change.acceptance_criteria) == 2
    assert "Special characters are allowed in passwords" in change.acceptance_criteria


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
        relevant_files=["src/auth.py", "tests/test_auth.py"],
        planned_changes=[
            PlannedChange(
                file="src/auth.py",
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
        relevant_files=["src/main.py"],
        planned_changes=[
            PlannedChange(
                file="src/main.py",
                description="Fix bug"
            )
        ],
        risk_level="low"
    )

    json_str = plan.model_dump_json(indent=2)
    assert "src/main.py" in json_str
    assert "Fix bug" in json_str
    assert "low" in json_str


def test_planned_change_required_fields():
    """Test that required fields are enforced."""
    with pytest.raises(ValidationError):
        PlannedChange(description="Missing file field")

    with pytest.raises(ValidationError):
        PlannedChange(file="src/test.py")


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
