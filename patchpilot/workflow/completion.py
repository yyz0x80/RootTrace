"""Completion state determination for workflow results.

This module provides the logic for determining the overall completion state
of a task based on execution results, verification status, and acceptance
evidence. The completion state follows a fixed priority hierarchy to ensure
consistent evaluation of task success or failure.

Completion states are evaluated in this priority order:
1. NEEDS_CLARIFICATION - Issue has unresolved ambiguities
2. BLOCKED - Environment, permission, or scope issues prevent execution
3. FAILED - Any AC is FAIL or unrecoverable code failure occurred
4. VERIFIED - All AC are PASS and Verifier passed
5. PARTIALLY_VERIFIED - Verifier passed but some AC are UNVERIFIED
"""

from patchpilot.evidence.schema import (
    AcceptanceEvidence,
    CompletionState,
    EvidenceStatus,
)


def determine_completion_state(
    *,
    has_ambiguity: bool,
    blocked: bool,
    execution_failed: bool,
    verifier_passed: bool,
    evidence: list[AcceptanceEvidence],
) -> CompletionState:
    """Determine the overall completion state based on execution and verification results.

    This function implements a pure function that evaluates completion state
    using a fixed priority hierarchy. Each state is checked in order, and the
    first matching condition determines the final state.

    Args:
        has_ambiguity: Whether the issue has unresolved ambiguities that
            prevent clear understanding of requirements.
        blocked: Whether environment, permission, or scope issues prevent
            execution of the task.
        execution_failed: Whether an unrecoverable code execution failure occurred
            (e.g., syntax errors, runtime exceptions).
        verifier_passed: Whether the overall verification process passed
            (e.g., all deterministic checks completed successfully).
        evidence: List of AcceptanceEvidence objects representing the verification
            status of each acceptance criterion.

    Returns:
        CompletionState enum value indicating the overall task completion state.
    """
    if has_ambiguity:
        return CompletionState.NEEDS_CLARIFICATION

    if blocked:
        return CompletionState.BLOCKED

    if execution_failed or any(
        item.status == EvidenceStatus.FAIL for item in evidence
    ):
        return CompletionState.FAILED

    if (
        evidence
        and verifier_passed
        and all(item.status == EvidenceStatus.PASS for item in evidence)
    ):
        return CompletionState.VERIFIED

    return CompletionState.PARTIALLY_VERIFIED
