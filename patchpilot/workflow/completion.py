"""Completion state determination for workflow results.

This module provides the logic for determining the overall completion state
of a task based on execution results, verification status, and acceptance
evidence. The completion state follows a fixed priority hierarchy to ensure
consistent evaluation of task success or failure.

Completion states are evaluated in this priority order:
1. NEEDS_CLARIFICATION - Issue has unresolved ambiguities
2. BLOCKED - Environment, permission, or scope issues prevent execution
3. FAILED - Any AC is FAIL or unrecoverable code failure occurred
4. VERIFIED - Deterministic verification passed
5. PARTIALLY_VERIFIED - Deterministic verification did not fully pass
"""

from patchpilot.evidence.schema import (
    AcceptanceEvidence,
    CompletionState,
    ConstraintSeverity,
    ConstraintStatus,
    EvidenceStatus,
    FailureType,
)


class CompletionDecision:
    """Structured result of completion state determination.

    Provides both the completion state and supporting metrics for
    monitoring and evaluation.
    """

    def __init__(
        self,
        state: CompletionState,
        failure_type: FailureType | None = None,
        criterion_pass_count: int = 0,
        criterion_unverified_count: int = 0,
        constraint_violation_count: int = 0,
        evidence_precision_hint: str = "",
    ):
        self.state = state
        self.failure_type = failure_type
        self.criterion_pass_count = criterion_pass_count
        self.criterion_unverified_count = criterion_unverified_count
        self.constraint_violation_count = constraint_violation_count
        self.evidence_precision_hint = evidence_precision_hint


def determine_completion_state(
    *,
    has_ambiguity: bool,
    blocked: bool,
    execution_failed: bool,
    evidence: list[AcceptanceEvidence],
    environment_can_verify: bool = True,
) -> CompletionDecision:
    """Determine the overall completion state based on execution and verification results.

    This function implements a pure function that evaluates completion state
    using a fixed priority hierarchy based on the following rules:

    1. Security or scope cannot be satisfied before execution → BLOCKED
    2. Genuine product behavior ambiguity → NEEDS_CLARIFICATION
    3. Agent execution violates hard constraint → FAILED (CONSTRAINT_VIOLATION)
    4. Any required Criterion is FAIL → FAILED
    5. Environment cannot execute necessary verification → BLOCKED
    6. All required Criteria are PASS/ALREADY_SATISFIED and all hard Constraints are COMPLIANT → VERIFIED
    7. No FAIL, but at least one required Criterion is UNVERIFIED → PARTIALLY_VERIFIED
    8. Required Constraint is UNSUPPORTED → BLOCKED (critical) or PARTIALLY_VERIFIED (non-critical)

    Args:
        has_ambiguity: Whether the issue has unresolved ambiguities that
            prevent clear understanding of requirements.
        blocked: Whether environment, permission, or scope issues prevent
            execution of the task.
        execution_failed: Whether an unrecoverable code execution failure occurred
            (e.g., syntax errors, runtime exceptions).
        evidence: List of AcceptanceEvidence objects representing the verification
            status of each acceptance criterion.
        environment_can_verify: Whether the environment can execute necessary
            verification commands.

    Returns:
        CompletionDecision object containing the completion state and supporting metrics.
    """
    # Rule 1: Security or scope cannot be satisfied before execution
    if blocked:
        return CompletionDecision(
            state=CompletionState.BLOCKED,
            evidence_precision_hint="blocked by security or scope restrictions",
        )

    # Rule 2: Genuine product behavior ambiguity
    if has_ambiguity:
        return CompletionDecision(
            state=CompletionState.NEEDS_CLARIFICATION,
            evidence_precision_hint="requires clarification of requirements",
        )

    # Handle execution failure
    if execution_failed:
        return CompletionDecision(
            state=CompletionState.FAILED,
            failure_type=FailureType.EXECUTION_FAILURE,
            evidence_precision_hint="execution failed",
        )

    # Rule 5: Environment cannot execute necessary verification
    if not environment_can_verify:
        return CompletionDecision(
            state=CompletionState.BLOCKED,
            evidence_precision_hint="environment cannot execute necessary verification",
        )

    # Rule 3: Agent execution violates hard constraint
    has_hard_constraint_violation = False
    for item in evidence:
        if item.constraint and item.constraint.status == ConstraintStatus.VIOLATED:
            has_hard_constraint_violation = True
            break

    if has_hard_constraint_violation:
        return CompletionDecision(
            state=CompletionState.FAILED,
            failure_type=FailureType.CONSTRAINT_VIOLATION,
            constraint_violation_count=1,
            evidence_precision_hint="hard constraint violation detected",
        )

    # Rule 4: Any required Criterion is FAIL
    required_criteria = [item for item in evidence if item.required]
    required_fail_count = sum(
        1 for item in required_criteria if item.status == EvidenceStatus.FAIL
    )

    if required_fail_count > 0:
        return CompletionDecision(
            state=CompletionState.FAILED,
            failure_type=FailureType.ACCEPTANCE_CRITERION_FAILURE,
            criterion_pass_count=0,
            criterion_unverified_count=0,
            evidence_precision_hint=f"{required_fail_count} required criteria failed",
        )

    # Rule 8: Required Constraint is UNSUPPORTED
    unsupported_critical = False
    unsupported_non_critical = False
    for item in evidence:
        if item.required and item.constraint and item.constraint.status == ConstraintStatus.UNSUPPORTED:
            if item.constraint.severity == ConstraintSeverity.CRITICAL:
                unsupported_critical = True
            else:
                unsupported_non_critical = True

    if unsupported_critical:
        return CompletionDecision(
            state=CompletionState.BLOCKED,
            evidence_precision_hint="critical constraint cannot be verified",
        )

    if unsupported_non_critical:
        return CompletionDecision(
            state=CompletionState.PARTIALLY_VERIFIED,
            evidence_precision_hint="non-critical constraint cannot be verified",
        )

    # Calculate metrics for remaining rules
    required_pass_count = sum(
        1 for item in required_criteria if item.status == EvidenceStatus.PASS
    )
    required_unverified_count = sum(
        1 for item in required_criteria if item.status == EvidenceStatus.UNVERIFIED
    )

    # Default to VERIFIED if no criteria but verification passed (backward compatibility)
    if not required_criteria:
        return CompletionDecision(
            state=CompletionState.VERIFIED,
            evidence_precision_hint="no required criteria defined, verification passed",
        )

    # Rule 6: All required Criteria are PASS or ALREADY_SATISFIED and all hard Constraints are COMPLIANT
    all_required_pass = all(
        item.status in (EvidenceStatus.PASS, EvidenceStatus.UNVERIFIED)
        for item in required_criteria
    )
    all_constraints_compliant = all(
        item.constraint is None or item.constraint.status == ConstraintStatus.COMPLIANT
        for item in evidence
    )

    if all_required_pass and all_constraints_compliant and required_unverified_count == 0:
        return CompletionDecision(
            state=CompletionState.VERIFIED,
            criterion_pass_count=required_pass_count,
            criterion_unverified_count=0,
            evidence_precision_hint="all required criteria verified",
        )

    # Rule 7: No FAIL, but at least one required Criterion is UNVERIFIED
    if required_unverified_count > 0:
        return CompletionDecision(
            state=CompletionState.PARTIALLY_VERIFIED,
            criterion_pass_count=required_pass_count,
            criterion_unverified_count=required_unverified_count,
            evidence_precision_hint=f"{required_unverified_count} required criteria unverified",
        )

    # Fallback to PARTIALLY_VERIFIED for other cases
    return CompletionDecision(
        state=CompletionState.PARTIALLY_VERIFIED,
        criterion_pass_count=required_pass_count,
        criterion_unverified_count=required_unverified_count,
        evidence_precision_hint="verification incomplete",
    )
