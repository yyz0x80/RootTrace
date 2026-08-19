"""Permission audit system for PatchPilot workflow.

This module provides a centralized permission checking and auditing system
that evaluates whether specific operations should be allowed, require user
approval, or be denied. The permission audit system enforces security boundaries
and provides structured decision records for trace events and audit logs.

The system evaluates permissions across multiple dimensions:
- Plan approval requirements
- Workspace read/write operations using PolicySet
- Change plan scope compliance
- Command execution using PolicySet
- Git operation restrictions

Permission decisions are structured with clear rule identifiers and reasoning
to support audit trails and debugging.
"""

from enum import Enum
from pathlib import Path

from pydantic import BaseModel

from patchpilot.policy.evaluator import PolicyEvaluator
from patchpilot.policy.schema import PolicySet
from patchpilot.policy.tracing import TraceDecision, record_permission_decision


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
        constraint_id: Optional constraint ID from the policy set
    """

    result: PermissionResult
    reason: str
    rule_id: str
    constraint_id: str | None = None


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

    # Scope compliance rules
    OUT_OF_PLAN_WRITE_DENIED = "OUT_OF_PLAN_WRITE_DENIED"

    # Command execution rules
    COMMAND_DENIED = "COMMAND_DENIED"
    GIT_PUSH_DENIED = "GIT_PUSH_DENIED"


class PermissionAuditor:
    """Centralized permission auditor for workflow operations.

    The PermissionAuditor evaluates permission requests against security
    policies and returns structured decisions. It integrates with the
    PolicySet system for consistent permission enforcement across the workflow.

    The auditor maintains no internal state - all decisions are based
    on the provided context and configuration.
    """

    def __init__(
        self,
        workspace_root: Path,
        policy_set: PolicySet,
        planned_files: set[str] | None = None,
    ) -> None:
        """Initialize the permission auditor.

        Args:
            workspace_root: Root path of the target repository workspace
            policy_set: Compiled PolicySet containing all security policies
            planned_files: Set of file paths approved for modification in the change plan
        """
        self.workspace_root = workspace_root.resolve()
        self.policy_set = policy_set
        self.planned_files = planned_files or set()
        self.policy_evaluator = PolicyEvaluator(policy_set)

    def check_read_permission(self, relative_path: str) -> PermissionDecision:
        """Check if reading a file is allowed.

        Args:
            relative_path: Relative path to the file within the workspace

        Returns:
            PermissionDecision allowing or denying the read operation
        """
        # Resolve the path to check for traversal attempts
        try:
            self._resolve_path(relative_path)
        except ValueError as e:
            decision = PermissionDecision(
                result=PermissionResult.DENY,
                reason=str(e),
                rule_id=RuleID.PATH_TRAVERSAL_DENIED,
            )
            record_permission_decision(
                operation="read",
                target=relative_path,
                decision=TraceDecision.DENY,
                rule_id=RuleID.PATH_TRAVERSAL_DENIED,
                reason=str(e),
            )
            return decision

        # Use PolicyEvaluator to check against policy set
        try:
            self.policy_evaluator.assert_read_allowed(relative_path)
            decision = PermissionDecision(
                result=PermissionResult.ALLOW,
                reason=f"Read operation allowed for: {relative_path}",
                rule_id=RuleID.WORKSPACE_READ_ALLOWED,
            )
            record_permission_decision(
                operation="read",
                target=relative_path,
                decision=TraceDecision.ALLOW,
                rule_id=RuleID.WORKSPACE_READ_ALLOWED,
                reason=f"Read operation allowed for: {relative_path}",
            )
            return decision
        except PermissionError as e:
            decision = PermissionDecision(
                result=PermissionResult.DENY,
                reason=str(e),
                rule_id=RuleID.SENSITIVE_FILE_DENIED,
            )
            record_permission_decision(
                operation="read",
                target=relative_path,
                decision=TraceDecision.DENY,
                rule_id=RuleID.SENSITIVE_FILE_DENIED,
                reason=str(e),
            )
            return decision

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
            self._resolve_path(relative_path)
        except ValueError as e:
            decision = PermissionDecision(
                result=PermissionResult.DENY,
                reason=str(e),
                rule_id=RuleID.PATH_TRAVERSAL_DENIED,
            )
            record_permission_decision(
                operation="write",
                target=relative_path,
                decision=TraceDecision.DENY,
                rule_id=RuleID.PATH_TRAVERSAL_DENIED,
                reason=str(e),
                metadata={"action": action},
            )
            return decision

        # Use PolicyEvaluator to check against policy set
        try:
            self.policy_evaluator.assert_write_allowed(relative_path)
        except PermissionError as e:
            decision = PermissionDecision(
                result=PermissionResult.DENY,
                reason=str(e),
                rule_id=RuleID.SENSITIVE_FILE_DENIED,
            )
            record_permission_decision(
                operation="write",
                target=relative_path,
                decision=TraceDecision.DENY,
                rule_id=RuleID.SENSITIVE_FILE_DENIED,
                reason=str(e),
                metadata={"action": action},
            )
            return decision

        # Check if file is in the approved change plan
        if self.planned_files and relative_path not in self.planned_files:
            decision = PermissionDecision(
                result=PermissionResult.DENY,
                reason=f"File modification outside approved plan: {relative_path}. "
                f"Approved files: {', '.join(sorted(self.planned_files))}",
                rule_id=RuleID.OUT_OF_PLAN_WRITE_DENIED,
            )
            record_permission_decision(
                operation="write",
                target=relative_path,
                decision=TraceDecision.DENY,
                rule_id=RuleID.OUT_OF_PLAN_WRITE_DENIED,
                reason=decision.reason,
                metadata={"action": action, "planned_files": list(self.planned_files)},
            )
            return decision

        # Write operation is allowed for planned source files
        decision = PermissionDecision(
            result=PermissionResult.ALLOW,
            reason=f"Write operation allowed for planned file: {relative_path}",
            rule_id=RuleID.PLANNED_SOURCE_WRITE_ALLOWED,
        )
        record_permission_decision(
            operation="write",
            target=relative_path,
            decision=TraceDecision.ALLOW,
            rule_id=RuleID.PLANNED_SOURCE_WRITE_ALLOWED,
            reason=decision.reason,
            metadata={"action": action},
        )
        return decision

    def check_command_permission(self, command: str) -> PermissionDecision:
        """Check if executing a command is allowed.

        Args:
            command: The command string to execute

        Returns:
            PermissionDecision allowing or denying the command execution
        """
        # Check for git push (always forbidden)
        if command.strip().startswith("git push"):
            decision = PermissionDecision(
                result=PermissionResult.DENY,
                reason="Git push operations are not allowed",
                rule_id=RuleID.GIT_PUSH_DENIED,
            )
            record_permission_decision(
                operation="command",
                target=command,
                decision=TraceDecision.DENY,
                rule_id=RuleID.GIT_PUSH_DENIED,
                reason="Git push operations are not allowed",
            )
            return decision

        # Use PolicyEvaluator to check against policy set
        try:
            self.policy_evaluator.assert_command_allowed(command)
            decision = PermissionDecision(
                result=PermissionResult.ALLOW,
                reason=f"Command '{command}' is allowed",
                rule_id=RuleID.COMMAND_DENIED,
            )
            record_permission_decision(
                operation="command",
                target=command,
                decision=TraceDecision.ALLOW,
                rule_id=RuleID.COMMAND_DENIED,
                reason=f"Command '{command}' is allowed",
            )
            return decision
        except PermissionError as e:
            decision = PermissionDecision(
                result=PermissionResult.DENY,
                reason=str(e),
                rule_id=RuleID.COMMAND_DENIED,
            )
            record_permission_decision(
                operation="command",
                target=command,
                decision=TraceDecision.DENY,
                rule_id=RuleID.COMMAND_DENIED,
                reason=str(e),
            )
            return decision

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
