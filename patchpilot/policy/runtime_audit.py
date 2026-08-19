"""Runtime audit for post-execution Git diff validation.

This module provides functionality to validate the actual Git diff produced
by agent execution against the approved change plan and PolicySet. This ensures
that no unauthorized changes were made during execution.
"""

import subprocess
from pathlib import Path

from patchpilot.policy.evaluator import PolicyEvaluator
from patchpilot.policy.schema import PolicySet
from patchpilot.policy.tracing import TraceDecision, record_permission_decision
from patchpilot.tools import WorkspaceChange


class RuntimeAuditResult:
    """Result of runtime audit validation.

    Attributes:
        passed: Whether the audit passed all checks
        violations: List of security violations detected
        actual_changes: List of actual changes made to the workspace
    """

    def __init__(
        self,
        passed: bool,
        violations: list[str],
        actual_changes: list[WorkspaceChange],
    ):
        self.passed = passed
        self.violations = violations
        self.actual_changes = actual_changes


def get_git_diff_changes(workspace_root: Path) -> list[WorkspaceChange]:
    """Get the actual changes from Git diff.

    Args:
        workspace_root: Path to the workspace directory

    Returns:
        List of WorkspaceChange objects representing actual changes

    Raises:
        subprocess.CalledProcessError: If git command fails
    """
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=workspace_root,
        capture_output=True,
        text=True,
        check=True,
    )

    changes = []

    for line in result.stdout.splitlines():
        if not line.strip():
            continue

        status = line[:2]
        path = line[3:].strip()

        # Skip ignored files (cache files, build artifacts, etc.)
        if _should_ignore_file(path):
            continue

        if status == "??":
            action = "create"
        elif "D" in status:
            action = "delete"
        elif "A" in status:
            action = "create"
        else:
            action = "modify"

        changes.append(
            WorkspaceChange(
                path=path,
                action=action,
            )
        )

    return changes


def _should_ignore_file(file_path: str) -> bool:
    """Check if a file should be ignored during runtime audit.

    Args:
        file_path: The file path to check

    Returns:
        True if the file should be ignored, False otherwise
    """
    # Check if any part of the path matches ignored directories
    ignored_directories = {
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
    }

    parts = file_path.split("/")
    for part in parts:
        if part in ignored_directories:
            return True

    # Check for file extension patterns
    ignored_extensions = {
        ".pyc",
        ".pyo",
        ".pyd",
        ".so",
        ".DS_Store",
    }

    for ext in ignored_extensions:
        if file_path.endswith(ext):
            return True

    return False


def audit_git_diff(
    workspace_root: Path,
    policy_set: PolicySet,
    planned_files: set[str] | None = None,
) -> RuntimeAuditResult:
    """Audit the actual Git diff against policies and planned changes.

    This function validates the actual changes made by the agent against:
    1. The PolicySet security policies
    2. The approved change plan (if provided)

    Args:
        workspace_root: Path to the workspace directory
        policy_set: The compiled PolicySet containing security policies
        planned_files: Optional set of planned file paths for validation

    Returns:
        RuntimeAuditResult with validation results
    """
    violations = []

    # Get actual changes from Git diff
    try:
        actual_changes = get_git_diff_changes(workspace_root)
    except subprocess.CalledProcessError as e:
        return RuntimeAuditResult(
            passed=False,
            violations=[f"Failed to get Git diff: {e}"],
            actual_changes=[],
        )

    # Create policy evaluator for checking policies
    policy_evaluator = PolicyEvaluator(policy_set)

    # Validate each actual change
    for change in actual_changes:
        # Record the permission decision
        record_permission_decision(
            operation="runtime_audit",
            target=change.path,
            decision=TraceDecision.ALLOW,  # Tentative, will update if denied
            rule_id="RUNTIME_AUDIT",
            reason=f"Runtime audit checking {change.action} on {change.path}",
            metadata={"action": change.action},
        )

        # Check write permissions using PolicyEvaluator
        try:
            policy_evaluator.assert_write_allowed(change.path)
        except PermissionError as e:
            violations.append(str(e))
            record_permission_decision(
                operation="runtime_audit",
                target=change.path,
                decision=TraceDecision.DENY,
                rule_id="RUNTIME_AUDIT",
                reason=str(e),
                metadata={"action": change.action},
            )
            continue

        # Check if file was in the approved plan (if provided)
        if planned_files and change.path not in planned_files:
            violation = f"File modified outside approved plan: {change.path}"
            violations.append(violation)
            record_permission_decision(
                operation="runtime_audit",
                target=change.path,
                decision=TraceDecision.DENY,
                rule_id="OUT_OF_PLAN_WRITE_DENIED",
                reason=violation,
                metadata={"action": change.action, "planned_files": list(planned_files)},
            )

    return RuntimeAuditResult(
        passed=len(violations) == 0,
        violations=violations,
        actual_changes=actual_changes,
    )


def validate_git_diff_consistency(
    workspace_root: Path,
    policy_set: PolicySet,
    planned_files: set[str] | None = None,
) -> bool:
    """Validate that Git diff is consistent with policies and planned changes.

    This is a convenience function that returns a boolean result.

    Args:
        workspace_root: Path to the workspace directory
        policy_set: The compiled PolicySet containing security policies
        planned_files: Optional set of planned file paths for validation

    Returns:
        True if the Git diff passes all validation checks, False otherwise
    """
    result = audit_git_diff(
        workspace_root=workspace_root,
        policy_set=policy_set,
        planned_files=planned_files,
    )
    return result.passed
