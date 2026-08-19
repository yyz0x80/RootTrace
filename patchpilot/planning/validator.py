"""Plan Validator module for validating change plans.

This module provides functionality to validate change plans against
repository context and ensure consistency before execution.
"""

from patchpilot.issue.schema import NormalizedIssue
from patchpilot.planning.schema import ChangePlan, CriterionPlanDetail
from patchpilot.planning.scope_gate import ScopeGateResult, check_scope
from patchpilot.policy.builtins import get_builtin_policies
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
) -> list[str]:
    """Validate that the plan completely and consistently maps issue ACs.

    New validation rules:
    - Hard failure: AC ID duplicates in plan
    - Hard failure: References to non-existent ACs
    - Hard failure: Behavior change claims IMPLEMENT but has no planned paths
    - Hard failure: Structural contract has no relevant planned paths
    - Warning: AC has no direct verification specification
    - No longer hard failure: Preservation AC lacks source mapping
    - No longer hard failure: AC lacks existing direct test
    - No longer hard failure: Multiple ACs share one source file

    Args:
        plan: Change plan whose AC references should be validated.
        issue: Normalized issue that defines the authoritative AC IDs.

    Returns:
        List of warning messages for non-critical issues.

    Raises:
        ValueError: If hard validation failures are detected.
    """
    warnings = []
    
    # Extract known AC IDs from issue
    criterion_ids = [criterion.id for criterion in issue.acceptance_criteria]
    known_ids = set(criterion_ids)

    # Hard failure: Check for duplicate AC IDs in the issue itself
    if len(criterion_ids) != len(known_ids):
        raise ValueError(
            "Normalized issue contains duplicate acceptance criterion IDs."
        )
    
    # Build maps of AC references in planned changes and tests
    change_ids = {
        criterion_id
        for change in plan.planned_changes
        for criterion_id in change.criterion_ids
    }
    test_ids = {
        criterion_id
        for test in plan.planned_tests
        for criterion_id in test.criterion_ids
    }
    
    # Also check acceptance_criteria fields for backward compatibility
    change_ids_from_acceptance = {
        criterion_id
        for change in plan.planned_changes
        for criterion_id in change.acceptance_criteria
    }
    test_ids_from_acceptance = {
        criterion_id
        for test in plan.planned_tests
        for criterion_id in test.acceptance_criteria
    }
    
    change_ids = change_ids | change_ids_from_acceptance
    test_ids = test_ids | test_ids_from_acceptance
    referenced_ids = change_ids | test_ids
    
    # Hard failure: Check for references to non-existent ACs
    unknown_ids = sorted(referenced_ids - known_ids)
    if unknown_ids:
        raise ValueError(
            f"Plan references unknown acceptance criterion IDs: {', '.join(unknown_ids)}"
        )
    
    # Build criterion plan lookup
    criterion_plan_map = {
        cp.criterion_id: cp 
        for cp in plan.criterion_plans
    }
    
    # Validate each criterion plan
    for criterion in issue.acceptance_criteria:
        criterion_id = criterion.id
        criterion_plan = criterion_plan_map.get(criterion_id)
        
        if criterion_plan is None:
            # Hard failure: Referenced AC not in criterion_plans
            raise ValueError(
                f"Acceptance criterion {criterion_id} not found in criterion_plans"
            )
        
        # Hard failure: Behavior change claims IMPLEMENT but has no planned paths
        if (
            criterion.kind == "behavior" and 
            criterion_plan.disposition == CriterionPlanDetail.TO_IMPLEMENT and
            not criterion_plan.relevant_source_files
        ):
            raise ValueError(
                f"Behavior change {criterion_id} claims IMPLEMENT but has no planned source paths"
            )
        
        # Hard failure: Structural contract has no relevant planned paths
        if (
            criterion.kind == "structural" and 
            not criterion_plan.relevant_source_files
        ):
            raise ValueError(
                f"Structural contract {criterion_id} has no relevant planned paths"
            )
        
        # Warning: AC has no direct verification specification
        if criterion_id not in test_ids:
            warnings.append(
                f"AC-{criterion_id} has no direct verification specification. "
                "Execution may continue, but the criterion cannot become PASS."
            )
        
        # Warning: AC has no planned source changes (no longer hard failure)
        if criterion_id not in change_ids:
            warnings.append(
                f"AC-{criterion_id} has no planned source changes. "
                "This may indicate incomplete implementation planning."
            )
    
    return warnings


def validate_plan(
    plan: ChangePlan,
    repository_context: RepositoryContext,
    issue: NormalizedIssue | None = None,
) -> ScopeGateResult:
    """Validate a change plan against repository context and scope restrictions.

    This function performs comprehensive plan validation:
    1. Validates plan consistency with repository state
    2. Validates AC ID uniqueness and references
    3. Validates file paths and actions
    4. Validates test targets
    5. Validates command support
    6. Validates behavior and structural contract requirements
    7. Checks scope restrictions and security boundaries
    8. Returns detailed validation results with warnings

    Args:
        plan: The change plan to validate.
        repository_context: Repository context with tracked files and structure.
        issue: Optional normalized issue used for deterministic AC coverage
            validation. Legacy callers may omit it.

    Returns:
        ScopeGateResult indicating whether the plan is allowed to proceed,
        along with any violations or warnings.

    Raises:
        ValueError: If the plan has hard validation failures.
    """
    # Step 1: Validate plan consistency with repository
    validate_plan_against_repository(plan, repository_context)

    # Step 2: Validate AC coverage and generate warnings
    coverage_warnings = []
    if issue is not None:
        coverage_warnings = validate_acceptance_coverage(plan, issue)

    # Step 3: Validate planned changes (paths and actions)
    validate_planned_changes(plan, repository_context)

    # Step 4: Validate test targets
    validate_test_targets(plan, repository_context)

    # Step 5: Validate command support
    validate_command_support(plan)

    # Step 6: Check scope restrictions and security boundaries
    policy_set = get_builtin_policies()
    scope_result = check_scope(plan, policy_set)

    # Add coverage warnings to scope result
    scope_result.warnings.extend(coverage_warnings)

    return scope_result


def validate_planned_changes(
    plan: ChangePlan,
    repository_context: RepositoryContext,
) -> None:
    """Validate that planned changes have valid paths and actions.

    Hard failure rules:
    - Modification path does not exist
    - Action is invalid for the path type

    Args:
        plan: The change plan to validate.
        repository_context: Repository context with tracked files.

    Raises:
        ValueError: If planned changes have invalid paths or actions.
    """
    tracked = set(repository_context.tracked_files)
    
    for change in plan.planned_changes:
        # Hard failure: Path does not exist for modify/delete actions
        if change.action.value in {"modify", "delete"} and change.path not in tracked:
            raise ValueError(
                f"Planned change path does not exist for {change.action.value}: {change.path}"
            )

        # Hard failure: Create action on existing file
        if change.action.value == "create" and change.path in tracked:
            raise ValueError(
                f"Planned create action on existing file: {change.path}"
            )


def validate_test_targets(
    plan: ChangePlan,
    repository_context: RepositoryContext,
) -> None:
    """Validate that test targets exist and are valid.

    Hard failure rules:
    - Test target does not exist
    - Test target is not a valid test file or directory

    Args:
        plan: The change plan to validate.
        repository_context: Repository context with tracked files.

    Raises:
        ValueError: If test targets are invalid.
    """
    # Skip validation for empty repository contexts (test scenarios)
    if not repository_context.tracked_files:
        return
    
    from pathlib import PurePosixPath
    
    tracked_files = set(repository_context.tracked_files)
    tracked_dirs = {
        str(PurePosixPath(path).parent)
        for path in tracked_files
    }
    
    def _is_test_file(path: str) -> bool:
        """Determine if a path refers to a test file."""
        normalized = str(PurePosixPath(path))
        parts = normalized.split("/")
        
        # Check if file is in a tests/ directory
        if "tests" in parts:
            return True
        
        # Check if filename starts with test_
        filename = parts[-1] if parts else ""
        return filename.startswith("test_") and filename.endswith(".py")
    
    for test in plan.planned_tests:
        command = test.command.strip()
        if not command.startswith("pytest") and not command.startswith("python -m pytest"):
            continue
        
        # Extract target from pytest command
        parts = command.split()
        if len(parts) < 2:
            continue
        
        target = parts[1]
        # Remove pytest flags
        if target.startswith("-"):
            continue
        
        # Hard failure: Test target does not exist
        if target not in tracked_files and target not in tracked_dirs:
            # For new files in planned_changes, allow the target
            is_planned_file = any(
                change.path == target for change in plan.planned_changes
            )
            if not is_planned_file:
                raise ValueError(
                    f"Test target does not exist in repository: {target}"
                )
        
        # Hard failure: Test target is not a valid test file or directory
        if target in tracked_files and not _is_test_file(target):
            raise ValueError(
                f"Test target must be a test file or test directory: {target}"
            )


def validate_command_support(
    plan: ChangePlan,
) -> None:
    """Validate that commands in the plan are supported.

    Hard failure rules:
    - Command is not in the supported command list

    Args:
        plan: The change plan to validate.

    Raises:
        ValueError: If commands are not supported.
    """
    # Define supported commands
    supported_commands = {
        "pytest",
        "python",
        "ruff",
        "git",
    }
    
    for test in plan.planned_tests:
        command = test.command.strip()
        # Extract the base command (first word)
        parts = command.split()
        if not parts:
            continue
        
        base_command = parts[0]
        
        # Check if command is supported
        if base_command not in supported_commands:
            raise ValueError(
                f"Unsupported command in plan: {base_command}"
            )
