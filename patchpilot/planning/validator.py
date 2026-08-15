"""Plan Validator module for validating change plans.

This module provides functionality to validate change plans against
repository context and ensure consistency before execution.
"""

from patchpilot.issue.schema import NormalizedIssue
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


def validate_acceptance_coverage(
    plan: ChangePlan,
    issue: NormalizedIssue,
) -> None:
    """Validate that the plan completely and consistently maps issue ACs.

    Every acceptance criterion must map to at least one planned source change
    and one planned deterministic test. Constraints are intentionally excluded
    because Workspace and Scope Gate enforce execution boundaries separately.

    Args:
        plan: Change plan whose AC references should be validated.
        issue: Normalized issue that defines the authoritative AC IDs.

    Raises:
        ValueError: If AC IDs are duplicated, unknown, or incompletely mapped.
    """
    criterion_ids = [
        criterion.id
        for criterion in issue.acceptance_criteria
    ]
    known_ids = set(criterion_ids)

    if len(criterion_ids) != len(known_ids):
        raise ValueError(
            "Normalized issue contains duplicate acceptance criterion IDs."
        )

    change_ids = {
        criterion_id
        for change in plan.planned_changes
        for criterion_id in change.acceptance_criteria
    }
    test_ids = {
        criterion_id
        for test in plan.planned_tests
        for criterion_id in test.acceptance_criteria
    }
    referenced_ids = change_ids | test_ids

    unknown_ids = sorted(referenced_ids - known_ids)
    if unknown_ids:
        raise ValueError(
            "Plan references unknown acceptance criterion IDs: "
            + ", ".join(unknown_ids)
        )

    missing_change_ids = sorted(known_ids - change_ids)
    if missing_change_ids:
        raise ValueError(
            "Acceptance criteria missing planned source changes: "
            + ", ".join(missing_change_ids)
        )

    missing_test_ids = sorted(known_ids - test_ids)
    if missing_test_ids:
        raise ValueError(
            "Acceptance criteria missing planned verification: "
            + ", ".join(missing_test_ids)
        )


def validate_plan(
    plan: ChangePlan,
    repository_context: RepositoryContext,
    issue: NormalizedIssue | None = None,
) -> ScopeGateResult:
    """Validate a change plan against repository context and scope restrictions.

    This function performs comprehensive plan validation:
    1. Validates plan consistency with repository state
    2. Checks scope restrictions and security boundaries
    3. Returns detailed validation results

    Args:
        plan: The change plan to validate.
        repository_context: Repository context with tracked files and structure.
        issue: Optional normalized issue used for deterministic AC coverage
            validation. Legacy callers may omit it.

    Returns:
        ScopeGateResult indicating whether the plan is allowed to proceed,
        along with any violations or warnings.

    Raises:
        ValueError: If the plan is inconsistent with the repository state.
    """
    # Step 1: Validate plan consistency with repository
    validate_plan_against_repository(plan, repository_context)

    # Step 2: Validate acceptance-criterion coverage when issue context exists
    if issue is not None:
        validate_acceptance_coverage(plan, issue)

    # Step 3: Check scope restrictions
    scope_result = check_scope(plan)

    return scope_result
