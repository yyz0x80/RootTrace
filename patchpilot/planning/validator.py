"""Plan Validator module for validating change plans.

This module provides functionality to validate change plans against
repository context and ensure consistency before execution.
"""

from patchpilot.issue.schema import NormalizedIssue
from patchpilot.planning.schema import (
    ChangePlan,
    CriterionPlanDetail,
    PlanDisposition,
)
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

    Validation rules:
    - Hard failure: AC ID duplicates in plan
    - Warning: Acceptance metadata is incomplete or inconsistent
    - Hard failure: Explicit verification requirements lack direct checks
    - Warning: AC has no direct specialized verification otherwise

    Acceptance metadata controls evidence quality, not write authorization. A safe
    source plan remains executable when optional evidence cannot be compiled.

    Args:
        plan: Change plan whose AC references should be validated.
        issue: Normalized issue that defines the authoritative AC IDs.

    Returns:
        List of warning messages for non-critical issues.

    Raises:
        ValueError: If hard validation failures are detected.
    """
    warnings = list(plan.validation_warnings)
    
    # Extract known AC IDs from issue
    criterion_ids = [criterion.id for criterion in issue.acceptance_criteria]
    known_ids = set(criterion_ids)

    # Hard failure: Check for duplicate AC IDs in the issue itself
    if len(criterion_ids) != len(known_ids):
        raise ValueError(
            "Normalized issue contains duplicate acceptance criterion IDs."
        )

    if (
        plan.plan_disposition == PlanDisposition.CHANGE_REQUIRED
        and any(criterion.required for criterion in issue.acceptance_criteria)
        and not plan.planned_changes
    ):
        raise ValueError(
            "A change_required plan must include at least one planned source change"
        )
    
    # Build maps of AC references in planned changes and regression tests.
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

    probe_ids = {
        criterion_id
        for probe in plan.acceptance_probes
        for criterion_id in probe.criterion_ids
    }
    structural_ids = {
        criterion_id
        for check in plan.structural_checks
        for criterion_id in check.criterion_ids
    }
    direct_ids = probe_ids | structural_ids
    referenced_ids |= direct_ids
    
    # Unknown mappings cannot grant write access, so keep the safe source plan and
    # exclude the mappings from acceptance evidence.
    unknown_ids = sorted(referenced_ids - known_ids)
    if unknown_ids:
        warnings.append(
            "Plan references unknown acceptance criterion IDs: "
            + ", ".join(unknown_ids)
        )

    if plan.verification_specs:
        warnings.append(
            "verification_specs are not executable; use acceptance_probes or "
            "structural_checks"
        )

    for probe in plan.acceptance_probes:
        if not probe.criterion_ids:
            warnings.append(
                f"Acceptance probe {probe.probe_id} has no criterion_ids"
            )

    for check in plan.structural_checks:
        if not check.criterion_ids:
            warnings.append(
                f"Structural check {check.check_id} has no criterion_ids"
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
            warnings.append(
                f"Acceptance criterion {criterion_id} has no criterion plan"
            )
            criterion_plan = None
        
        if (
            criterion_plan is not None
            and criterion.kind == "behavior"
            and criterion_plan.disposition == CriterionPlanDetail.TO_IMPLEMENT
            and not criterion_plan.relevant_source_files
        ):
            warnings.append(
                f"Behavior change {criterion_id} claims IMPLEMENT but has no planned source paths"
            )
        
        if (
            criterion_plan is not None
            and criterion.kind == "structural"
            and not criterion_plan.relevant_source_files
        ):
            warnings.append(
                f"Structural contract {criterion_id} has no relevant planned paths"
            )
        
        if criterion.required and criterion_id not in direct_ids:
            message = (
                f"Required acceptance criterion {criterion_id} has no direct "
                "acceptance check"
            )
            if issue.verification_requirements:
                raise ValueError(message)
            warnings.append(message)
        
        if (
            criterion_plan is not None
            and criterion_plan.disposition == CriterionPlanDetail.TO_IMPLEMENT
            and criterion_id not in change_ids
        ):
            warnings.append(
                f"Required implementation criterion {criterion_id} has no "
                "planned source change"
            )
    
    return list(dict.fromkeys(warnings))


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

    # Step 2: Validate planned changes (paths and actions)
    validate_planned_changes(plan, repository_context)

    # Step 3: Validate test targets
    validate_test_targets(plan, repository_context)

    # Step 4: Validate command support
    validate_command_support(plan)

    # Step 5: Check scope restrictions before acceptance completeness.
    policy_set = get_builtin_policies()
    scope_result = check_scope(plan, policy_set)
    if not scope_result.allowed:
        return scope_result

    # Step 6: Validate direct acceptance coverage for an otherwise safe plan.
    coverage_warnings = []
    if issue is not None:
        coverage_warnings = validate_acceptance_coverage(plan, issue)

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
