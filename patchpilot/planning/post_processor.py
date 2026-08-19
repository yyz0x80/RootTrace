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
from patchpilot.planning.schema import ChangePlan, PlannedChange, PlannedTest
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
                "actions": [],
            }

        changes_by_path[change.path]["descriptions"].append(change.description)
        changes_by_path[change.path]["acceptance_criteria"].extend(
            change.acceptance_criteria
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


def _complete_ac_mapping(
    plan: ChangePlan,
    issue: NormalizedIssue,
) -> ChangePlan:
    """Locally complete AC mapping errors instead of failing the entire task.

    If the planner mapped ACs incorrectly (e.g., typos, missing mappings),
    this function attempts to complete the mapping locally using deterministic
    rules rather than failing the entire task.

    Args:
        plan: The change plan to process.
        issue: The normalized issue.

    Returns:
        Plan with completed AC mappings.

    Raises:
        PlanPostProcessError: If AC mapping cannot be completed locally.
    """
    from patchpilot.planning.validator import validate_acceptance_coverage

    # Skip AC validation for issues without acceptance criteria
    if not issue.acceptance_criteria:
        return plan

    try:
        validate_acceptance_coverage(plan, issue)
    except ValueError as e:
        # Attempt local completion for specific error types
        error_msg = str(e)

        if "unknown acceptance criterion" in error_msg:
            # Try to correct typos in AC IDs
            known_ids = {criterion.id for criterion in issue.acceptance_criteria}

            corrected_changes = []
            for change in plan.planned_changes:
                corrected_criteria: list[str] = []
                for ac_id in change.acceptance_criteria:
                    if ac_id in known_ids:
                        corrected_criteria.append(ac_id)
                    else:
                        # Try to find similar AC ID (simple typo correction)
                        for known_id in known_ids:
                            if (
                                ac_id.lower() == known_id.lower()
                                or ac_id.replace("-", "") == known_id.replace("-", "")
                            ):
                                corrected_criteria.append(known_id)
                                break

                if corrected_criteria != change.acceptance_criteria:
                    # Create a new change with corrected AC IDs
                    corrected_changes.append(
                        change.model_copy(update={"acceptance_criteria": corrected_criteria})
                    )
                else:
                    corrected_changes.append(change)

            corrected_tests = []
            for test in plan.planned_tests:
                corrected_criteria: list[str] = []
                for ac_id in test.acceptance_criteria:
                    if ac_id in known_ids:
                        corrected_criteria.append(ac_id)
                    else:
                        for known_id in known_ids:
                            if (
                                ac_id.lower() == known_id.lower()
                                or ac_id.replace("-", "") == known_id.replace("-", "")
                            ):
                                corrected_criteria.append(known_id)
                                break

                if corrected_criteria != test.acceptance_criteria:
                    # Create a new test with corrected AC IDs
                    corrected_tests.append(
                        test.model_copy(update={"acceptance_criteria": corrected_criteria})
                    )
                else:
                    corrected_tests.append(test)

            # Update plan with corrected changes and tests
            plan = plan.model_copy(
                update={
                    "planned_changes": corrected_changes,
                    "planned_tests": corrected_tests,
                }
            )

            # Re-validate after correction
            try:
                validate_acceptance_coverage(plan, issue)
            except ValueError:
                # If still failing, raise the original error
                raise PlanPostProcessError(
                    f"AC mapping error cannot be completed locally: {e}"
                ) from e

        elif "missing planned source changes" in error_msg or "missing planned verification" in error_msg:
            # For missing mappings, we cannot complete locally without LLM
            raise PlanPostProcessError(
                f"AC mapping requires LLM intervention: {e}"
            ) from e
        else:
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

    Processing order:
    1. Migrate test files from planned_changes to planned_tests
    2. Merge duplicate file changes
    3. Validate pytest targets
    4. Check for ambiguous points
    5. Classify semantic ambiguities (currently no-op, relies on LLM)
    6. Complete AC mapping errors locally (if skip_ac_validation is False)

    Args:
        plan: The change plan generated by the LLM.
        issue: The normalized issue.
        repository_context: Repository context with tracked files.
        skip_ac_validation: If True, skip AC mapping completion. This is used
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

    # Step 6: Complete AC mapping errors locally (skip if requested)
    if not skip_ac_validation:
        plan = _complete_ac_mapping(plan, issue)

    return plan
