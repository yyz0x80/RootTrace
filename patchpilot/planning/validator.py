"""Plan Validator module for validating change plans.

This module provides functionality to validate change plans against
repository context and ensure consistency before execution.
"""

from patchpilot.planning.schema import ChangePlan
from patchpilot.planning.scope_gate import ScopeGateResult, check_scope
from patchpilot.repository.schema import RepositoryContext


def validate_plan_against_repository(
    plan: ChangePlan,
    repository_context: RepositoryContext,
) -> None:
    """Validate that a change plan is consistent with the repository state.

    Ensures that:
    - The plan's repository_match flag is true
    - Modify/delete actions reference existing tracked files
    - Create actions do not reference existing tracked files

    Args:
        plan: The change plan to validate.
        repository_context: Repository context with tracked files.

    Raises:
        ValueError: If the plan is inconsistent with the repository state.
    """
    if not plan.repository_match:
        raise ValueError(
            "Issue does not match repository: "
            f"{plan.repository_mismatch_reason}"
        )

    tracked = set(repository_context.tracked_files)

    for change in plan.planned_changes:
        if change.action.value in {
            "modify",
            "delete",
        }:
            if change.path not in tracked:
                raise ValueError(
                    "Plan references a non-existing file "
                    f"for {change.action.value}: "
                    f"{change.path}"
                )

        elif change.action.value == "create" and change.path in tracked:
            raise ValueError(
                "Plan attempts to create an existing file: "
                f"{change.path}"
            )


def validate_plan(
    plan: ChangePlan,
    repository_context: RepositoryContext,
) -> ScopeGateResult:
    """Validate a change plan against repository context and scope restrictions.

    This function performs comprehensive plan validation:
    1. Validates plan consistency with repository state
    2. Checks scope restrictions and security boundaries
    3. Returns detailed validation results

    Args:
        plan: The change plan to validate.
        repository_context: Repository context with tracked files and structure.

    Returns:
        ScopeGateResult indicating whether the plan is allowed to proceed,
        along with any violations or warnings.

    Raises:
        ValueError: If the plan is inconsistent with the repository state.
    """
    # Step 1: Validate plan consistency with repository
    validate_plan_against_repository(plan, repository_context)

    # Step 2: Check scope restrictions
    scope_result = check_scope(plan)

    return scope_result
