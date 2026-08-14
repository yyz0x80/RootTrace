"""Tests for completion state determination."""

from patchpilot.evidence.schema import (
    AcceptanceEvidence,
    CompletionState,
    EvidenceStatus,
)
from patchpilot.workflow.completion import determine_completion_state


def test_needs_clarification_state():
    """Test that ambiguity takes highest priority."""
    evidence = [
        AcceptanceEvidence(
            criterion_id="ac1",
            description="Test criterion",
            status=EvidenceStatus.PASS,
            explanation="Test passed",
        )
    ]

    result = determine_completion_state(
        has_ambiguity=True,
        blocked=False,
        execution_failed=False,
        verifier_passed=True,
        evidence=evidence,
    )

    assert result == CompletionState.NEEDS_CLARIFICATION


def test_blocked_state():
    """Test that blocked state takes second priority."""
    evidence = [
        AcceptanceEvidence(
            criterion_id="ac1",
            description="Test criterion",
            status=EvidenceStatus.PASS,
            explanation="Test passed",
        )
    ]

    result = determine_completion_state(
        has_ambiguity=False,
        blocked=True,
        execution_failed=False,
        verifier_passed=True,
        evidence=evidence,
    )

    assert result == CompletionState.BLOCKED


def test_failed_state_execution_failure():
    """Test that execution failure results in FAILED state."""
    evidence = [
        AcceptanceEvidence(
            criterion_id="ac1",
            description="Test criterion",
            status=EvidenceStatus.PASS,
            explanation="Test passed",
        )
    ]

    result = determine_completion_state(
        has_ambiguity=False,
        blocked=False,
        execution_failed=True,
        verifier_passed=True,
        evidence=evidence,
    )

    assert result == CompletionState.FAILED


def test_failed_state_evidence_failure():
    """Test that evidence failure results in FAILED state."""
    evidence = [
        AcceptanceEvidence(
            criterion_id="ac1",
            description="Test criterion",
            status=EvidenceStatus.FAIL,
            explanation="Test failed",
        )
    ]

    result = determine_completion_state(
        has_ambiguity=False,
        blocked=False,
        execution_failed=False,
        verifier_passed=True,
        evidence=evidence,
    )

    assert result == CompletionState.FAILED


def test_verified_state():
    """Test that VERIFIED state requires non-empty evidence and all PASS."""
    evidence = [
        AcceptanceEvidence(
            criterion_id="ac1",
            description="Test criterion",
            status=EvidenceStatus.PASS,
            explanation="Test passed",
        ),
        AcceptanceEvidence(
            criterion_id="ac2",
            description="Another criterion",
            status=EvidenceStatus.PASS,
            explanation="Test passed",
        ),
    ]

    result = determine_completion_state(
        has_ambiguity=False,
        blocked=False,
        execution_failed=False,
        verifier_passed=True,
        evidence=evidence,
    )

    assert result == CompletionState.VERIFIED


def test_partially_verified_state_empty_evidence():
    """Test that empty evidence results in PARTIALLY_VERIFIED, not VERIFIED."""
    result = determine_completion_state(
        has_ambiguity=False,
        blocked=False,
        execution_failed=False,
        verifier_passed=True,
        evidence=[],
    )

    assert result == CompletionState.PARTIALLY_VERIFIED


def test_partially_verified_state_mixed_evidence():
    """Test that mixed evidence results in PARTIALLY_VERIFIED."""
    evidence = [
        AcceptanceEvidence(
            criterion_id="ac1",
            description="Test criterion",
            status=EvidenceStatus.PASS,
            explanation="Test passed",
        ),
        AcceptanceEvidence(
            criterion_id="ac2",
            description="Another criterion",
            status=EvidenceStatus.UNVERIFIED,
            explanation="Not verified",
        ),
    ]

    result = determine_completion_state(
        has_ambiguity=False,
        blocked=False,
        execution_failed=False,
        verifier_passed=True,
        evidence=evidence,
    )

    assert result == CompletionState.PARTIALLY_VERIFIED


def test_partially_verified_state_verifier_failed():
    """Test that verifier failure results in PARTIALLY_VERIFIED."""
    evidence = [
        AcceptanceEvidence(
            criterion_id="ac1",
            description="Test criterion",
            status=EvidenceStatus.PASS,
            explanation="Test passed",
        )
    ]

    result = determine_completion_state(
        has_ambiguity=False,
        blocked=False,
        execution_failed=False,
        verifier_passed=False,
        evidence=evidence,
    )

    assert result == CompletionState.PARTIALLY_VERIFIED


def test_partially_verified_state_all_unverified():
    """Test that all UNVERIFIED evidence results in PARTIALLY_VERIFIED."""
    evidence = [
        AcceptanceEvidence(
            criterion_id="ac1",
            description="Test criterion",
            status=EvidenceStatus.UNVERIFIED,
            explanation="Not verified",
        ),
        AcceptanceEvidence(
            criterion_id="ac2",
            description="Another criterion",
            status=EvidenceStatus.UNVERIFIED,
            explanation="Not verified",
        ),
    ]

    result = determine_completion_state(
        has_ambiguity=False,
        blocked=False,
        execution_failed=False,
        verifier_passed=True,
        evidence=evidence,
    )

    assert result == CompletionState.PARTIALLY_VERIFIED


def test_partially_verified_state_empty_evidence_with_verifier_failed():
    """Test that empty evidence with verifier failure results in PARTIALLY_VERIFIED."""
    result = determine_completion_state(
        has_ambiguity=False,
        blocked=False,
        execution_failed=False,
        verifier_passed=False,
        evidence=[],
    )

    assert result == CompletionState.PARTIALLY_VERIFIED
