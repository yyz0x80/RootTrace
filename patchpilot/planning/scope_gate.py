"""Scope Gate module for validating ChangePlan safety and compliance.

This module provides functionality to validate change plans against security
and scope restrictions before execution, as well as runtime validation of
actual changes against approved plans using the unified PolicySet system.
"""

from pathlib import PurePosixPath

from pydantic import BaseModel, Field

from patchpilot.planning.schema import ChangePlan
from patchpilot.policy.evaluator import PolicyEvaluator
from patchpilot.policy.schema import PolicySet
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


# Directory names to ignore during runtime validation
IGNORED_DIRECTORIES = (
    "__pycache__",
    ".pytest_cache",
)

# File patterns to ignore during runtime validation
IGNORED_FILE_PATTERNS = (
    "*.pyc",
    "*.pyo",
    "*.pyd",
    ".DS_Store",
    "*.so",
)

# Path hints that indicate database migrations requiring manual handling
DATABASE_MIGRATION_HINTS = (
    "migrations/",
    "alembic/versions/",
)


def _should_ignore_file(file_path: str) -> bool:
    """Check if a file should be ignored during runtime validation.

    Args:
        file_path: The file path to check

    Returns:
        True if the file should be ignored, False otherwise
    """
    # Check if any part of the path matches ignored directories
    parts = file_path.split("/")
    for part in parts:
        # Check for exact matches (directory names like __pycache__)
        if part in IGNORED_DIRECTORIES:
            return True
        # Check for file extension patterns (like *.pyc)
        for pattern in IGNORED_FILE_PATTERNS:
            if pattern.startswith("*") and part.endswith(pattern[1:]):
                return True
    return False


def check_scope(
    plan: ChangePlan,
    policy_set: PolicySet,
    max_modified_files: int = 6,
) -> ScopeGateResult:
    """Validate a change plan against security and scope restrictions.

    Args:
        plan: The change plan to validate.
        policy_set: The compiled PolicySet containing security policies.
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

    # Create policy evaluator for checking policies
    policy_evaluator = PolicyEvaluator(policy_set)

    # 1. Maximum file count check
    if len(set(planned_files)) > max_modified_files:
        violations.append(
            f"Plan modifies {len(set(planned_files))} files, "
            f"maximum allowed is {max_modified_files}."
        )

    # 2. File-level security checks using PolicySet
    for file in planned_files:
        normalized = str(PurePosixPath(file))

        # Check write permissions using PolicyEvaluator
        try:
            policy_evaluator.assert_write_allowed(normalized)
        except PermissionError as e:
            violations.append(str(e))

        # Check for database migration hints
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
    policy_set: PolicySet,
) -> None:
    """Validate actual workspace changes against the approved plan.

    This function enforces runtime scope validation by comparing the actual
    changes made by the agent against the approved change plan and PolicySet.
    It ensures that:

    1. All security policies from PolicySet are enforced
    2. All changed files were in the approved plan
    3. The action type matches what was approved

    Args:
        plan: The approved ChangePlan containing planned changes
        actual_changes: List of WorkspaceChange objects representing actual changes
        policy_set: The compiled PolicySet containing security policies

    Raises:
        RuntimeError: If any security violation is detected:
            - Policy violation (sensitive files, test files, etc.)
            - File modification outside approved plan
            - Action type mismatch between plan and actual change
    """
    # Create policy evaluator for checking policies
    policy_evaluator = PolicyEvaluator(policy_set)

    # Build a lookup of planned changes for quick validation
    planned = {
        change.path: change.action.value
        for change in plan.planned_changes
    }

    for actual in actual_changes:
        # Skip ignored files (cache files, build artifacts, etc.)
        if _should_ignore_file(actual.path):
            continue

        # Check write permissions using PolicyEvaluator
        try:
            policy_evaluator.assert_write_allowed(actual.path)
        except PermissionError as e:
            raise RuntimeError(str(e))

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
