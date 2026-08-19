"""Structured trace logging for permission decisions.

This module provides functionality for logging permission decisions in a
structured format that can be used for audit trails, debugging, and
compliance verification.
"""

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel


class TraceDecision(str, Enum):
    """Permission decision result for operations.

    ALLOW: Operation is permitted without further approval
    ASK: Operation requires user confirmation before proceeding
    DENY: Operation is forbidden and will not be executed
    """

    ALLOW = "ALLOW"
    ASK = "ASK"
    DENY = "DENY"


class PermissionTrace(BaseModel):
    """Structured trace record for a permission decision.

    Attributes:
        timestamp: When the decision was made
        constraint_id: Optional constraint ID from the policy set
        operation: Type of operation (read, write, command, etc.)
        target: Target of the operation (file path, command, etc.)
        decision: The permission decision (ALLOW, ASK, or DENY)
        rule_id: Identifier of the security rule that triggered this decision
        reason: Human-readable explanation of the decision
        metadata: Additional context about the decision
    """

    timestamp: datetime
    constraint_id: str | None = None
    operation: str
    target: str
    decision: TraceDecision
    rule_id: str
    reason: str
    metadata: dict[str, Any] = {}


class PermissionTracer:
    """Tracer for recording permission decisions.

    The PermissionTracer maintains a history of permission decisions
    for audit trails and debugging. Each decision is recorded with
    structured metadata for traceability.
    """

    def __init__(self):
        """Initialize the permission tracer."""
        self.traces: list[PermissionTrace] = []

    def record_decision(
        self,
        operation: str,
        target: str,
        decision: TraceDecision,
        rule_id: str,
        reason: str,
        constraint_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a permission decision.

        Args:
            operation: Type of operation (read, write, command, etc.)
            target: Target of the operation (file path, command, etc.)
            decision: The permission decision (ALLOW, ASK, or DENY)
            rule_id: Identifier of the security rule that triggered this decision
            reason: Human-readable explanation of the decision
            constraint_id: Optional constraint ID from the policy set
            metadata: Additional context about the decision
        """
        trace = PermissionTrace(
            timestamp=datetime.now(UTC),
            constraint_id=constraint_id,
            operation=operation,
            target=target,
            decision=decision,
            rule_id=rule_id,
            reason=reason,
            metadata=metadata or {},
        )
        self.traces.append(trace)

    def get_denied_decisions(self) -> list[PermissionTrace]:
        """Get all denied permission decisions.

        Returns:
            List of denied permission traces
        """
        return [trace for trace in self.traces if trace.decision == TraceDecision.DENY]

    def get_decisions_by_target(self, target: str) -> list[PermissionTrace]:
        """Get all permission decisions for a specific target.

        Args:
            target: The target to filter by (file path, command, etc.)

        Returns:
            List of permission traces for the target
        """
        return [trace for trace in self.traces if trace.target == target]

    def get_decisions_by_rule(self, rule_id: str) -> list[PermissionTrace]:
        """Get all permission decisions triggered by a specific rule.

        Args:
            rule_id: The rule ID to filter by

        Returns:
            List of permission traces for the rule
        """
        return [trace for trace in self.traces if trace.rule_id == rule_id]

    def clear(self) -> None:
        """Clear all recorded traces."""
        self.traces.clear()

    def get_summary(self) -> dict[str, int]:
        """Get a summary of recorded decisions.

        Returns:
            Dictionary with counts of each decision type
        """
        summary = {
            "total": len(self.traces),
            "allowed": 0,
            "ask": 0,
            "denied": 0,
        }

        for trace in self.traces:
            if trace.decision == TraceDecision.ALLOW:
                summary["allowed"] += 1
            elif trace.decision == TraceDecision.ASK:
                summary["ask"] += 1
            elif trace.decision == TraceDecision.DENY:
                summary["denied"] += 1

        return summary


# Global tracer instance for use across the system
_global_tracer = PermissionTracer()


def get_global_tracer() -> PermissionTracer:
    """Get the global permission tracer instance.

    Returns:
        The global PermissionTracer instance
    """
    return _global_tracer


def record_permission_decision(
    operation: str,
    target: str,
    decision: TraceDecision,
    rule_id: str,
    reason: str,
    constraint_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Record a permission decision using the global tracer.

    Args:
        operation: Type of operation (read, write, command, etc.)
        target: Target of the operation (file path, command, etc.)
        decision: The permission decision (ALLOW, ASK, or DENY)
        rule_id: Identifier of the security rule that triggered this decision
        reason: Human-readable explanation of the decision
        constraint_id: Optional constraint ID from the policy set
        metadata: Additional context about the decision
    """
    _global_tracer.record_decision(
        operation=operation,
        target=target,
        decision=decision,
        rule_id=rule_id,
        reason=reason,
        constraint_id=constraint_id,
        metadata=metadata,
    )
