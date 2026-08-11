"""Scope Gate module for validating ChangePlan safety and compliance.

This module provides functionality to validate change plans against security
and scope restrictions before execution, as well as runtime validation of
actual changes against approved plans.
"""

from pathlib import PurePosixPath

from pydantic import BaseModel, Field

from patchpilot.planning.schema import ChangePlan
from patchpilot.tools import WorkspaceChange


class ScopeGateResult(BaseModel):
    """Result of scope gate validation.

    Attributes:
        allowed: Whether the plan passes all security checks.
        violations: List of security violations that block execution.
        warnings: List of warnings about potentially risky changes.
    """

    allowed: bool
    violations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# Files that are never allowed to be modified
FORBIDDEN_FILES = {
    ".env",
}

# Path prefixes that are forbidden to modify
FORBIDDEN_PREFIXES = (
    ".github/workflows/",
)

# Path hints that indicate database migrations requiring manual handling
DATABASE_MIGRATION_HINTS = (
    "migrations/",
    "alembic/versions/",
)


def check_scope(
    plan: ChangePlan,
    max_modified_files: int = 6,
) -> ScopeGateResult:
    """Validate a change plan against security and scope restrictions.

    Args:
        plan: The change plan to validate.
        max_modified_files: Maximum number of files allowed to be modified.

    Returns:
        ScopeGateResult indicating whether the plan is allowed to proceed,
        along with any violations or warnings.
    """
    violations = []
    warnings = []

    planned_files = [
        change.path
        for change in plan.planned_changes
    ]

    # 1. Maximum file count check
    if len(set(planned_files)) > max_modified_files:
        violations.append(
            f"Plan modifies {len(set(planned_files))} files, "
            f"maximum allowed is {max_modified_files}."
        )

    # 2. File-level security checks
    for file in planned_files:
        normalized = str(PurePosixPath(file))

        if normalized in FORBIDDEN_FILES:
            violations.append(
                f"Modification of {normalized} is forbidden."
            )

        if normalized.startswith(FORBIDDEN_PREFIXES):
            violations.append(
                f"CI/CD modification is forbidden: {normalized}"
            )

        if any(
            hint in normalized
            for hint in DATABASE_MIGRATION_HINTS
        ):
            violations.append(
                f"Database migration requires manual handling: "
                f"{normalized}"
            )

    # 3. Verify modified files are in relevant_files
    relevant_files = set(plan.relevant_files)

    for file in planned_files:
        if file not in relevant_files:
            warnings.append(
                f"{file} is modified but not listed in relevant_files."
            )

    # 4. High-risk plans cannot be automatically executed
    if plan.risk_level == "high":
        violations.append(
            "High-risk plan cannot be automatically executed."
        )

    return ScopeGateResult(
        allowed=len(violations) == 0,
        violations=violations,
        warnings=warnings,
    )


def validate_actual_changes(
    plan: ChangePlan,
    actual_changes: list[WorkspaceChange],
) -> None:
    """Validate actual workspace changes against the approved plan.

    This function enforces runtime scope validation by comparing the actual
    changes made by the agent against the approved change plan. It ensures that:

    1. Forbidden files (like .env) are never modified
    2. CI/CD workflows are never modified
    3. All changed files were in the approved plan
    4. The action type matches what was approved

    Args:
        plan: The approved ChangePlan containing planned changes
        actual_changes: List of WorkspaceChange objects representing actual changes

    Raises:
        RuntimeError: If any security violation is detected:
            - .env modification attempt
            - CI workflow modification attempt
            - File modification outside approved plan
            - Action type mismatch between plan and actual change
    """
    # Build a lookup of planned changes for quick validation
    planned = {
        change.path: change.action.value
        for change in plan.planned_changes
    }

    for actual in actual_changes:
        # .env files are always forbidden
        if actual.path == ".env":
            raise RuntimeError(
                "Modification of .env is forbidden."
            )

        # CI workflow modifications are always forbidden
        if actual.path.startswith(".github/workflows/"):
            raise RuntimeError(
                "CI workflow modification is forbidden."
            )

        # Check if the file was in the approved plan
        expected_action = planned.get(actual.path)

        # File modified outside the approved plan
        if expected_action is None:
            raise RuntimeError(
                "Agent modified a file outside "
                f"the approved plan: {actual.path}"
            )

        # Action type mismatch between plan and actual
        if expected_action != actual.action:
            raise RuntimeError(
                "Unexpected change action for "
                f"{actual.path}: "
                f"expected {expected_action}, "
                f"got {actual.action}"
            )
