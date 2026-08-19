"""Plan Post-Processor for programmatic security and validation rules.

This module enforces security and validation rules programmatically rather than
relying on LLM compliance with prompt instructions. It implements:

1. Automatic test file migration from planned_changes to planned_tests
2. Pytest target existence and test file validation
3. Merging of multiple planned changes for the same file
4. NEEDS_CLARIFICATION detection for non-empty ambiguous_points
5. Deterministic ambiguity classification for undefined semantics
6. Local AC mapping error completion instead of task failure

The post-processor runs after LLM plan generation but before plan validation,
ensuring that security boundaries are enforced regardless of model behavior.
"""

from __future__ import annotations

from pathlib import PurePosixPath

from patchpilot.issue.schema import NormalizedIssue
from patchpilot.planning.schema import (
    ChangePlan,
    CriterionPlanDetail,
    PlannedChange,
    PlannedTest,
)
from patchpilot.repository.schema import RepositoryContext


class PlanPostProcessError(ValueError):
    """Raised when plan post-processing fails with an unrecoverable error."""


def _is_test_file(path: str) -> bool:
    """Determine if a path refers to a test file.

    Args:
        path: File path to check.

    Returns:
        True if the path is a test file, False otherwise.
    """
    normalized = str(PurePosixPath(path))
    parts = normalized.split("/")

    # Check if file is in a tests/ directory
    if "tests" in parts:
        return True

    # Check if filename starts with test_
    filename = parts[-1] if parts else ""
    return filename.startswith("test_") and filename.endswith(".py")


def _migrate_test_files_to_tests(plan: ChangePlan) -> ChangePlan:
    """Move test files from planned_changes to planned_tests.

    This rule enforces the Day 1 restriction that test files are read-only.
    Any test file that appears in planned_changes is automatically converted
    to a pytest verification command in planned_tests.

    Args:
        plan: The change plan to process.

    Returns:
        Modified plan with test files migrated to planned_tests.
    """
    source_changes: list[PlannedChange] = []
    test_commands: list[PlannedTest] = list(plan.planned_tests)

    for change in plan.planned_changes:
        if _is_test_file(change.path):
            # Convert to a pytest command
            command = f"pytest {change.path} -q"
            test_command = PlannedTest(
                command=command,
                purpose=f"Verify test file: {change.path}",
                acceptance_criteria=change.acceptance_criteria,
                criterion_ids=change.criterion_ids,
            )
            test_commands.append(test_command)
        else:
            source_changes.append(change)

    return plan.model_copy(
        update={
            "planned_changes": source_changes,
            "planned_tests": test_commands,
        }
    )


def _merge_duplicate_file_changes(plan: ChangePlan) -> ChangePlan:
    """Merge multiple planned changes for the same file.

    When the LLM generates multiple changes for the same file, this function
    merges them into a single change with combined acceptance criteria and
    a merged description.

    Args:
        plan: The change plan to process.

    Returns:
        Modified plan with duplicate file changes merged.
    """
    changes_by_path: dict[str, dict[str, list[str]]] = {}

    for change in plan.planned_changes:
        if change.path not in changes_by_path:
            changes_by_path[change.path] = {
                "descriptions": [],
                "acceptance_criteria": [],
                "criterion_ids": [],
                "actions": [],
            }

        changes_by_path[change.path]["descriptions"].append(change.description)
        changes_by_path[change.path]["acceptance_criteria"].extend(
            change.acceptance_criteria
        )
        changes_by_path[change.path]["criterion_ids"].extend(
            change.criterion_ids
        )
        changes_by_path[change.path]["actions"].append(change.action.value)

    merged_changes: list[PlannedChange] = []

    for path, data in changes_by_path.items():
        # Deduplicate acceptance criteria while preserving order
        unique_criteria: list[str] = []
        seen: set[str] = set()
        for ac in data["acceptance_criteria"]:
            if ac not in seen:
                seen.add(ac)
                unique_criteria.append(ac)

        # Deduplicate criterion_ids while preserving order
        unique_criterion_ids: list[str] = []
        seen_ids: set[str] = set()
        for cid in data["criterion_ids"]:
            if cid not in seen_ids:
                seen_ids.add(cid)
                unique_criterion_ids.append(cid)

        # Merge descriptions
        merged_description = "; ".join(data["descriptions"])

        # Use the most recent action (last in list)
        final_action = data["actions"][-1]

        merged_changes.append(
            PlannedChange(
                path=path,
                action=final_action,
                description=merged_description,
                acceptance_criteria=unique_criteria,
                criterion_ids=unique_criterion_ids,
            )
        )

    return plan.model_copy(update={"planned_changes": merged_changes})


def _validate_pytest_targets(
    plan: ChangePlan,
    repository_context: RepositoryContext,
) -> ChangePlan:
    """Validate that pytest targets exist and are test files or directories.

    This function checks that all pytest targets in planned_tests:
    1. Reference files or directories that exist in the repository
    2. Are test files (under tests/ or starting with test_) or test directories

    Args:
        plan: The change plan to process.
        repository_context: Repository context with tracked files.

    Returns:
        The original plan if validation passes.

    Raises:
        PlanPostProcessError: If pytest targets are invalid.
    """
    # Skip validation for empty repository contexts (test scenarios)
    if not repository_context.tracked_files:
        return plan

    tracked_files = set(repository_context.tracked_files)
    tracked_dirs = {
        str(PurePosixPath(path).parent)
        for path in tracked_files
    }

    for test in plan.planned_tests:
        command = test.command.strip()
        if not command.startswith("pytest") and not command.startswith(
            "python -m pytest"
        ):
            continue

        # Extract target from pytest command
        parts = command.split()
        if len(parts) < 2:
            continue

        target = parts[1]
        # Remove pytest flags
        if target.startswith("-"):
            continue

        # Check if target exists
        if target not in tracked_files and target not in tracked_dirs:
            # For new files in planned_changes, allow the target
            is_planned_file = any(
                change.path == target for change in plan.planned_changes
            )
            if not is_planned_file:
                raise PlanPostProcessError(
                    f"Pytest target does not exist in repository: {target}"
                )

        # Check if target is a test file or directory
        if target in tracked_files and not _is_test_file(target):
            raise PlanPostProcessError(
                f"Pytest target must be a test file or test directory: {target}"
            )

    return plan


def _check_ambiguity(
    plan: ChangePlan,
    issue: NormalizedIssue,
) -> ChangePlan:
    """Check for ambiguous points that require clarification.

    If the normalized issue has non-empty ambiguous_points, the plan should
    be marked as needing clarification regardless of other validation results.

    Note: The CLI layer handles this check before plan generation. This function
    provides defense-in-depth for other code paths (e.g., evaluation, testing).

    Args:
        plan: The change plan to process.
        issue: The normalized issue.

    Returns:
        The original plan (ambiguity is handled by workflow layer).

    Raises:
        PlanPostProcessError: If ambiguous_points is non-empty.
    """
    if issue.ambiguous_points:
        raise PlanPostProcessError(
            "Issue contains ambiguous points that require clarification: "
            + "; ".join(issue.ambiguous_points)
        )

    return plan


def _classify_ambiguities(
    plan: ChangePlan,
    issue: NormalizedIssue,
) -> ChangePlan:
    """Run deterministic ambiguity classifier for undefined semantics.

    This function applies a conservative standard for blocking ambiguities:
    Only when missing information would lead to at least two incompatible but
    reasonable externally observable behaviors should it be considered a blocking
    ambiguity.

    The following are NOT considered blocking ambiguities:
    - Multiple valid implementation approaches
    - Hypothetical scenarios not required by the issue
    - Details that can be determined from existing code or tests
    - Background information the model wants for context
    - Modifications already prohibited by security policies

    Actual ambiguity detection is primarily handled by the LLM-based normalizer
    through the ambiguous_points field. This function provides defense-in-depth
    for cases where programmatic detection is feasible and reliable.

    Args:
        plan: The change plan to process.
        issue: The normalized issue.

    Returns:
        The original plan (no programmatic ambiguity classification applied).

    Raises:
        PlanPostProcessError: If semantic ambiguities are detected.
    """
    # Keyword-based ambiguity classification removed as it was too restrictive
    # and caused false positives on legitimate tasks involving priority/ordering.
    # Ambiguity detection now relies on LLM-based classification through
    # ambiguous_points in the normalized issue.
    return plan


def _validate_criterion_plans(
    plan: ChangePlan,
    issue: NormalizedIssue,
) -> ChangePlan:
    """Validate criterion plans according to the updated requirements.

    This function ensures that:
    - ALREADY_SATISFIED criteria have baseline_evidence
    - Unknown criterion IDs are rejected (hard failure)
    - Preservation criteria without source mapping are allowed (no longer hard failure)
    - Multiple ACs sharing one source file are allowed (no longer hard failure)

    Args:
        plan: The change plan to process.
        issue: The normalized issue.

    Returns:
        The original plan if validation passes.

    Raises:
        PlanPostProcessError: If criterion plan validation fails with hard errors.
    """
    # Skip validation for issues without acceptance criteria
    if not issue.acceptance_criteria:
        return plan

    known_ids = {criterion.id for criterion in issue.acceptance_criteria}

    for criterion_plan in plan.criterion_plans:
        # Hard failure: Unknown criterion ID
        if criterion_plan.criterion_id not in known_ids:
            raise PlanPostProcessError(
                f"Unknown criterion ID in criterion_plans: {criterion_plan.criterion_id}"
            )

        # Hard failure: ALREADY_SATISFIED must have baseline_evidence
        if (
            criterion_plan.disposition == CriterionPlanDetail.ALREADY_SATISFIED
            and not criterion_plan.baseline_evidence
        ):
            raise PlanPostProcessError(
                f"ALREADY_SATISFIED criterion {criterion_plan.criterion_id} must have baseline_evidence"
            )

    # No longer hard failure: Allow preservation criteria without planned changes
    # No longer hard failure: Allow multiple ACs to share one source file
    # These are now handled as warnings or allowed by the validator

    return plan


def _complete_ac_mapping(
    plan: ChangePlan,
    issue: NormalizedIssue,
) -> ChangePlan:
    """Locally complete AC mapping errors instead of failing the entire task.

    Updated rules:
    - Hard failure: AC ID typos (unknown references) are not auto-corrected
    - Hard failure: Fabricated test references are rejected
    - No longer hard failure: Missing test mappings (GLM median case)
    - Warning: AC without direct verification (handled by validator)

    Args:
        plan: The change plan to process.
        issue: The normalized issue.

    Returns:
        Plan with completed AC mappings.

    Raises:
        PlanPostProcessError: If AC mapping has hard failures.
    """
    from patchpilot.planning.validator import validate_acceptance_coverage

    # Skip AC validation for issues without acceptance criteria
    if not issue.acceptance_criteria:
        return plan

    try:
        validate_acceptance_coverage(plan, issue)
        # Warnings are now handled by the validator, not as post-process errors
    except ValueError as e:
        error_msg = str(e)

        # Hard failure: Unknown AC references (no auto-correction)
        if "unknown acceptance criterion" in error_msg or "unknown acceptance criterion IDs" in error_msg:
            raise PlanPostProcessError(
                f"Plan references unknown acceptance criteria: {e}"
            ) from e

        # Hard failure: Behavior change claims IMPLEMENT but has no planned paths
        if "claims IMPLEMENT but has no planned source paths" in error_msg:
            raise PlanPostProcessError(
                f"Behavior change validation failed: {e}"
            ) from e

        # Hard failure: Structural contract has no relevant planned paths
        if "has no relevant planned paths" in error_msg:
            raise PlanPostProcessError(
                f"Structural contract validation failed: {e}"
            ) from e

        # Hard failure: Fabricated test references
        if "not found in criterion_plans" in error_msg:
            raise PlanPostProcessError(
                f"Fabricated test reference detected: {e}"
            ) from e

        # Other validation errors should fail
        raise PlanPostProcessError(f"AC validation error: {e}") from e

    return plan


def post_process_plan(
    plan: ChangePlan,
    issue: NormalizedIssue,
    repository_context: RepositoryContext,
    skip_ac_validation: bool = False,
) -> ChangePlan:
    """Apply all programmatic post-processing rules to a change plan.

    This function enforces security and validation rules programmatically,
    ensuring that LLM non-compliance with prompt instructions does not
    compromise security or correctness.

    Updated processing order:
    1. Migrate test files from planned_changes to planned_tests
    2. Merge duplicate file changes
    3. Validate pytest targets
    4. Check for ambiguous points
    5. Classify semantic ambiguities (currently no-op, relies on LLM)
    6. Validate criterion plans (if skip_ac_validation is False)
    7. Complete AC mapping validation (if skip_ac_validation is False)

    Args:
        plan: The change plan generated by the LLM.
        issue: The normalized issue.
        repository_context: Repository context with tracked files.
        skip_ac_validation: If True, skip AC mapping validation. This is used
            when the plan may have scope violations that should be checked first.

    Returns:
        Post-processed change plan.

    Raises:
        PlanPostProcessError: If post-processing fails with an unrecoverable error.
    """
    # Step 1: Migrate test files to planned_tests
    plan = _migrate_test_files_to_tests(plan)

    # Step 2: Merge duplicate file changes
    plan = _merge_duplicate_file_changes(plan)

    # Step 3: Validate pytest targets
    plan = _validate_pytest_targets(plan, repository_context)

    # Step 4: Check for ambiguous points
    plan = _check_ambiguity(plan, issue)

    # Step 5: Classify semantic ambiguities
    plan = _classify_ambiguities(plan, issue)

    # Step 6: Validate criterion plans (skip if requested)
    if not skip_ac_validation:
        plan = _validate_criterion_plans(plan, issue)

    # Step 7: Complete AC mapping validation (skip if requested)
    if not skip_ac_validation:
        plan = _complete_ac_mapping(plan, issue)

    return plan
