from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import PurePosixPath

from patchpilot.planning.schema import ChangePlan


@dataclass(frozen=True)
class TargetTestSelection:
    """Target tests and their mapped acceptance criteria."""

    tests: list[str]
    acceptance_criteria: list[str]
    direct_acceptance_criteria: list[str]


def _append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _pytest_arguments(command: str) -> list[str] | None:
    """Return pytest arguments or None for a non-pytest command."""
    args = shlex.split(command)

    if not args:
        raise ValueError("Planned test command must not be empty")

    if args[0] == "pytest":
        return args[1:]

    if args[:3] == ["python", "-m", "pytest"]:
        return args[3:]

    if args[:2] == ["ruff", "check"]:
        # Ruff is always executed by Verifier Level 1.
        return None

    raise ValueError(
        "Unsupported planned test command: "
        f"{command}. Only pytest, python -m pytest, "
        "and ruff check are supported."
    )


def _is_test_target(argument: str) -> bool:
    if argument.startswith("-"):
        return False

    file_part = argument.split("::", maxsplit=1)[0]
    path = PurePosixPath(file_part)

    if path.is_absolute() or ".." in path.parts:
        raise ValueError(
            f"Unsafe planned test target: {argument}"
        )

    return (
        file_part.endswith(".py")
        or "tests" in path.parts
        or path.name.startswith("test_")
    )


def select_target_tests(
    plan: ChangePlan | None,
) -> TargetTestSelection:
    """Extract safe pytest targets and mapped AC IDs from a plan."""
    if plan is None:
        return TargetTestSelection(
            tests=[],
            acceptance_criteria=[],
            direct_acceptance_criteria=[],
        )

    targets: list[str] = []
    criterion_ids: list[str] = []
    direct_criterion_ids: list[str] = []

    for planned_test in plan.planned_tests:
        pytest_args = _pytest_arguments(planned_test.command)

        if pytest_args is None:
            continue

        planned_targets = [
            argument
            for argument in pytest_args
            if _is_test_target(argument)
        ]

        for target in planned_targets:
            _append_unique(targets, target)

        if planned_targets:
            for criterion_id in planned_test.acceptance_criteria:
                _append_unique(criterion_ids, criterion_id)

        if (
            len(planned_targets) == 1
            and "::" in planned_targets[0]
            and len(planned_test.acceptance_criteria) == 1
        ):
            _append_unique(
                direct_criterion_ids,
                planned_test.acceptance_criteria[0],
            )

    return TargetTestSelection(
        tests=targets,
        acceptance_criteria=criterion_ids,
        direct_acceptance_criteria=direct_criterion_ids,
    )
