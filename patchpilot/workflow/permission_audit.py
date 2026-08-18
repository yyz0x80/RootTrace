"""Permission audit system for PatchPilot workflow.

This module provides a centralized permission checking and auditing system
that evaluates whether specific operations should be allowed, require user
approval, or be denied. The permission audit system enforces security boundaries
and provides structured decision records for trace events and audit logs.

The system evaluates permissions across multiple dimensions:
- Plan approval requirements
- Workspace read/write operations
- File path security (traversal, sensitive files, test files)
- Change plan scope compliance
- CI/CD configuration protection
- Command execution allowlist
- Git operation restrictions

Permission decisions are structured with clear rule identifiers and reasoning
to support audit trails and debugging.
"""

from enum import Enum
from pathlib import Path

from pydantic import BaseModel


class PermissionResult(str, Enum):
    """Permission decision result for operations.

    ALLOW: Operation is permitted without further approval
    ASK: Operation requires user confirmation before proceeding
    DENY: Operation is forbidden and will not be executed
    """

    ALLOW = "ALLOW"
    ASK = "ASK"
    DENY = "DENY"


class PermissionDecision(BaseModel):
    """Structured permission decision with detailed reasoning.

    Attributes:
        result: The permission decision (ALLOW, ASK, or DENY)
        reason: Human-readable explanation of the decision
        rule_id: Identifier of the security rule that triggered this decision
    """

    result: PermissionResult
    reason: str
    rule_id: str


# Rule identifiers for permission decisions
class RuleID(str, Enum):
    """Standardized rule identifiers for permission decisions.

    Each rule ID corresponds to a specific security check or policy
    enforcement point in the permission audit system.
    """

    # Planning and approval rules
    PLAN_APPROVAL_REQUIRED = "PLAN_APPROVAL_REQUIRED"

    # Workspace access rules
    WORKSPACE_READ_ALLOWED = "WORKSPACE_READ_ALLOWED"
    PLANNED_SOURCE_WRITE_ALLOWED = "PLANNED_SOURCE_WRITE_ALLOWED"

    # Path security rules
    PATH_TRAVERSAL_DENIED = "PATH_TRAVERSAL_DENIED"
    SENSITIVE_FILE_DENIED = "SENSITIVE_FILE_DENIED"
    TEST_WRITE_DENIED = "TEST_WRITE_DENIED"

    # Scope compliance rules
    OUT_OF_PLAN_WRITE_DENIED = "OUT_OF_PLAN_WRITE_DENIED"
    CI_WRITE_DENIED = "CI_WRITE_DENIED"

    # Command execution rules
    COMMAND_DENIED = "COMMAND_DENIED"
    GIT_PUSH_DENIED = "GIT_PUSH_DENIED"


class PermissionAuditor:
    """Centralized permission auditor for workflow operations.

    The PermissionAuditor evaluates permission requests against security
    policies and returns structured decisions. It integrates with the
    workspace policy, change plan scope, and command allowlist to provide
    consistent permission enforcement across the workflow.

    The auditor maintains no internal state - all decisions are based
    on the provided context and configuration.
    """

    def __init__(
        self,
        workspace_root: Path,
        planned_files: set[str] | None = None,
    ) -> None:
        """Initialize the permission auditor.

        Args:
            workspace_root: Root path of the target repository workspace
            planned_files: Set of file paths approved for modification in the change plan
        """
        self.workspace_root = workspace_root.resolve()
        self.planned_files = planned_files or set()

    def check_read_permission(self, relative_path: str) -> PermissionDecision:
        """Check if reading a file is allowed.

        Args:
            relative_path: Relative path to the file within the workspace

        Returns:
            PermissionDecision allowing or denying the read operation
        """
        # Resolve the path to check for traversal attempts
        try:
            resolved = self._resolve_path(relative_path)
        except ValueError as e:
            return PermissionDecision(
                result=PermissionResult.DENY,
                reason=str(e),
                rule_id=RuleID.PATH_TRAVERSAL_DENIED,
            )

        # Check for sensitive files
        if self._is_sensitive_file(resolved):
            return PermissionDecision(
                result=PermissionResult.DENY,
                reason=f"Reading sensitive file is not allowed: {relative_path}",
                rule_id=RuleID.SENSITIVE_FILE_DENIED,
            )

        # Read operations are generally allowed for non-sensitive files
        return PermissionDecision(
            result=PermissionResult.ALLOW,
            reason=f"Read operation allowed for: {relative_path}",
            rule_id=RuleID.WORKSPACE_READ_ALLOWED,
        )

    def check_write_permission(
        self,
        relative_path: str,
        action: str = "modify",
    ) -> PermissionDecision:
        """Check if writing to a file is allowed.

        Args:
            relative_path: Relative path to the file within the workspace
            action: Type of write operation (create, modify, delete)

        Returns:
            PermissionDecision allowing, asking for approval, or denying the write operation
        """
        # Resolve the path to check for traversal attempts
        try:
            resolved = self._resolve_path(relative_path)
        except ValueError as e:
            return PermissionDecision(
                result=PermissionResult.DENY,
                reason=str(e),
                rule_id=RuleID.PATH_TRAVERSAL_DENIED,
            )

        # Check for sensitive files
        if self._is_sensitive_file(resolved):
            return PermissionDecision(
                result=PermissionResult.DENY,
                reason=f"Writing sensitive file is not allowed: {relative_path}",
                rule_id=RuleID.SENSITIVE_FILE_DENIED,
            )

        # Check for test file modifications (Day 1 restriction)
        if self._is_test_file(relative_path):
            return PermissionDecision(
                result=PermissionResult.DENY,
                reason=f"Modifying test files is not allowed: {relative_path}. "
                "Test files must remain read-only during Day 1 implementation.",
                rule_id=RuleID.TEST_WRITE_DENIED,
            )

        # Check for CI/CD configuration files
        if self._is_ci_config(relative_path):
            return PermissionDecision(
                result=PermissionResult.DENY,
                reason=f"CI/CD configuration modification is not allowed: {relative_path}",
                rule_id=RuleID.CI_WRITE_DENIED,
            )

        # Check if file is in the approved change plan
        if self.planned_files and relative_path not in self.planned_files:
            return PermissionDecision(
                result=PermissionResult.DENY,
                reason=f"File modification outside approved plan: {relative_path}. "
                f"Approved files: {', '.join(sorted(self.planned_files))}",
                rule_id=RuleID.OUT_OF_PLAN_WRITE_DENIED,
            )

        # Write operation is allowed for planned source files
        return PermissionDecision(
            result=PermissionResult.ALLOW,
            reason=f"Write operation allowed for planned file: {relative_path}",
            rule_id=RuleID.PLANNED_SOURCE_WRITE_ALLOWED,
        )

    def check_command_permission(self, command: str) -> PermissionDecision:
        """Check if executing a command is allowed.

        Args:
            command: The command string to execute

        Returns:
            PermissionDecision allowing or denying the command execution
        """
        # Check for git push (always forbidden)
        if command.strip().startswith("git push"):
            return PermissionDecision(
                result=PermissionResult.DENY,
                reason="Git push operations are not allowed",
                rule_id=RuleID.GIT_PUSH_DENIED,
            )

        # Parse the base command
        parts = command.strip().split()
        if not parts:
            return PermissionDecision(
                result=PermissionResult.DENY,
                reason="Empty command is not allowed",
                rule_id=RuleID.COMMAND_DENIED,
            )

        base_command = parts[0]

        # Day 1 allowed commands
        allowed_commands = {"pytest", "python", "ruff", "git"}

        if base_command not in allowed_commands:
            return PermissionDecision(
                result=PermissionResult.DENY,
                reason=f"Command '{base_command}' is not allowed. "
                f"Allowed commands: {', '.join(sorted(allowed_commands))}",
                rule_id=RuleID.COMMAND_DENIED,
            )

        # Additional restrictions for python command
        if base_command == "python":
            # Only allow python -m pytest
            if len(parts) >= 2 and parts[1] == "-m" and len(parts) >= 3 and parts[2] == "pytest":
                return PermissionDecision(
                    result=PermissionResult.ALLOW,
                    reason="Python pytest command is allowed",
                    rule_id=RuleID.COMMAND_DENIED,
                )
            return PermissionDecision(
                result=PermissionResult.DENY,
                reason="Only 'python -m pytest' is allowed for python command",
                rule_id=RuleID.COMMAND_DENIED,
            )

        # Additional restrictions for git command
        if base_command == "git":
            # Only allow git diff and git status
            if len(parts) >= 2 and parts[1] in ("diff", "status"):
                return PermissionDecision(
                    result=PermissionResult.ALLOW,
                    reason=f"Git {parts[1]} command is allowed",
                    rule_id=RuleID.COMMAND_DENIED,
                )
            return PermissionDecision(
                result=PermissionResult.DENY,
                reason="Only 'git diff' and 'git status' are allowed for git command",
                rule_id=RuleID.COMMAND_DENIED,
            )

        # Additional restrictions for ruff command
        if base_command == "ruff":
            if len(parts) >= 2 and parts[1] == "check":
                return PermissionDecision(
                    result=PermissionResult.ALLOW,
                    reason="Ruff check command is allowed",
                    rule_id=RuleID.COMMAND_DENIED,
                )
            return PermissionDecision(
                result=PermissionResult.DENY,
                reason="Only 'ruff check' is allowed for ruff command",
                rule_id=RuleID.COMMAND_DENIED,
            )

        # Pytest is always allowed
        if base_command == "pytest":
            return PermissionDecision(
                result=PermissionResult.ALLOW,
                reason="Pytest command is allowed",
                rule_id=RuleID.COMMAND_DENIED,
            )

        return PermissionDecision(
            result=PermissionResult.ALLOW,
            reason=f"Command '{base_command}' is allowed",
            rule_id=RuleID.COMMAND_DENIED,
        )

    def check_plan_approval_required(self, has_plan: bool) -> PermissionDecision:
        """Check if plan approval is required for execution.

        Args:
            has_plan: Whether a change plan has been approved

        Returns:
            PermissionDecision indicating if plan approval is required
        """
        if not has_plan:
            return PermissionDecision(
                result=PermissionResult.ASK,
                reason="No change plan approved. User approval required to proceed.",
                rule_id=RuleID.PLAN_APPROVAL_REQUIRED,
            )

        return PermissionDecision(
            result=PermissionResult.ALLOW,
            reason="Change plan has been approved",
            rule_id=RuleID.PLAN_APPROVAL_REQUIRED,
        )

    def _resolve_path(self, relative_path: str) -> Path:
        """Resolve a relative path against the workspace root.

        Args:
            relative_path: Relative path to resolve

        Returns:
            Resolved absolute path

        Raises:
            ValueError: If path is absolute or attempts traversal
        """
        # Reject absolute paths
        if Path(relative_path).is_absolute():
            raise ValueError(f"Absolute path rejected: {relative_path}")

        # Resolve against workspace root
        resolved = (self.workspace_root / relative_path).resolve()

        # Check for path traversal
        try:
            resolved.relative_to(self.workspace_root)
        except ValueError:
            raise ValueError(f"Path escapes repository: {relative_path}")

        return resolved

    def _is_sensitive_file(self, resolved_path: Path) -> bool:
        """Check if a path refers to a sensitive file.

        Args:
            resolved_path: Resolved absolute path to check

        Returns:
            True if the file is sensitive, False otherwise
        """
        # Check for .env files
        if resolved_path.name == ".env":
            return True

        # Check for .git directory
        return resolved_path.name == ".git" or ".git" in resolved_path.parts

    def _is_test_file(self, relative_path: str) -> bool:
        """Check if a path refers to a test file.

        Args:
            relative_path: Relative path to check

        Returns:
            True if the path is a test file, False otherwise
        """
        # Check if path contains tests/ directory
        if "tests" in relative_path.split("/"):
            return True

        # Check if filename starts with test_
        parts = relative_path.split("/")
        return bool(parts and parts[-1].startswith("test_"))

    def _is_ci_config(self, relative_path: str) -> bool:
        """Check if a path refers to a CI/CD configuration file.

        Args:
            relative_path: Relative path to check

        Returns:
            True if the path is a CI/CD configuration, False otherwise
        """
        # Check for GitHub workflows
        return relative_path.startswith(".github/workflows/")


def audit_tool_permission(
    tool_name: str,
    tool_arguments: dict[str, str],
    auditor: PermissionAuditor,
) -> PermissionDecision:
    """Audit permission for a tool call.

    This function provides a convenient interface for auditing tool
    permissions by dispatching to the appropriate auditor method based
    on the tool name and arguments.

    Args:
        tool_name: Name of the tool being called
        tool_arguments: Arguments passed to the tool
        auditor: PermissionAuditor instance to use for the check

    Returns:
        PermissionDecision for the tool call
    """
    if tool_name == "read_file":
        path = tool_arguments.get("path", "")
        return auditor.check_read_permission(path)

    if tool_name in ("edit_file", "edit_file_by_line", "insert_text", "write_file"):
        path = tool_arguments.get("path", "")
        return auditor.check_write_permission(path)

    if tool_name == "run_command":
        command = tool_arguments.get("command", "")
        return auditor.check_command_permission(command)

    # search_code and other read-only tools are generally allowed
    return PermissionDecision(
        result=PermissionResult.ALLOW,
        reason=f"Tool '{tool_name}' is allowed",
        rule_id=RuleID.WORKSPACE_READ_ALLOWED,
    )
