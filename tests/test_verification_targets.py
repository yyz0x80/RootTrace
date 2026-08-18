from __future__ import annotations

import pytest

from patchpilot.planning.schema import ChangePlan, PlannedTest
from patchpilot.verification.targets import select_target_tests


def make_plan(planned_tests: list[PlannedTest]) -> ChangePlan:
    return ChangePlan(
        relevant_files=[],
        planned_changes=[],
        planned_tests=planned_tests,
        out_of_scope=[],
        risk_level="low",
    )


def test_select_pytest_target_and_acceptance_criteria() -> None:
    plan = make_plan(
        [
            PlannedTest(
                command=(
                    "pytest "
                    "tests/test_task.py::test_priority -q"
                ),
                purpose="Verify priority",
                acceptance_criteria=["AC-1"],
            )
        ]
    )

    selection = select_target_tests(plan)

    assert selection.tests == [
        "tests/test_task.py::test_priority"
    ]
    assert selection.acceptance_criteria == ["AC-1"]
    assert selection.direct_acceptance_criteria == ["AC-1"]


def test_select_python_module_pytest_command() -> None:
    plan = make_plan(
        [
            PlannedTest(
                command="python -m pytest tests/test_task.py -q",
                purpose="Verify task behavior",
                acceptance_criteria=["AC-1", "AC-2"],
            )
        ]
    )

    selection = select_target_tests(plan)

    assert selection.tests == ["tests/test_task.py"]
    assert selection.acceptance_criteria == ["AC-1", "AC-2"]
    assert selection.direct_acceptance_criteria == []


def test_file_level_test_with_single_ac_is_direct() -> None:
    """File-level test mapped to a single AC should be considered direct evidence."""
    plan = make_plan(
        [
            PlannedTest(
                command="python -m pytest tests/test_task.py -q",
                purpose="Verify task behavior",
                acceptance_criteria=["AC-1"],
            )
        ]
    )

    selection = select_target_tests(plan)

    assert selection.tests == ["tests/test_task.py"]
    assert selection.acceptance_criteria == ["AC-1"]
    assert selection.direct_acceptance_criteria == ["AC-1"]


def test_ignore_ruff_planned_check() -> None:
    plan = make_plan(
        [
            PlannedTest(
                command="ruff check .",
                purpose="Run lint",
                acceptance_criteria=[],
            )
        ]
    )

    selection = select_target_tests(plan)

    assert selection.tests == []
    assert selection.acceptance_criteria == []
    assert selection.direct_acceptance_criteria == []


def test_reject_path_traversal_target() -> None:
    plan = make_plan(
        [
            PlannedTest(
                command="pytest ../outside/test_secret.py -q",
                purpose="Unsafe test",
                acceptance_criteria=["AC-1"],
            )
        ]
    )

    with pytest.raises(
        ValueError,
        match="Unsafe planned test target",
    ):
        select_target_tests(plan)


def test_reject_unsupported_command() -> None:
    plan = make_plan(
        [
            PlannedTest(
                command="pip install pytest",
                purpose="Install dependency",
                acceptance_criteria=[],
            )
        ]
    )

    with pytest.raises(
        ValueError,
        match="Unsupported planned test command",
    ):
        select_target_tests(plan)
