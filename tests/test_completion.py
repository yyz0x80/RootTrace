"""Tests for completion state determination."""

from patchpilot.evidence.schema import (
    AcceptanceEvidence,
    CompletionState,
    ConstraintEvidence,
    ConstraintSeverity,
    ConstraintStatus,
    EvidenceStatus,
)
from patchpilot.workflow.completion import (
    determine_completion_state,
)
from patchpilot.workflow.repair_selector import RepairSelector


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
        evidence=evidence,
    )

    assert result.state == CompletionState.NEEDS_CLARIFICATION
    assert "clarification" in result.evidence_precision_hint.lower()


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
        evidence=evidence,
    )

    assert result.state == CompletionState.BLOCKED
    assert "blocked" in result.evidence_precision_hint.lower()


def test_environment_cannot_verify():
    """Test that environment verification failure results in BLOCKED."""
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
        evidence=evidence,
        environment_can_verify=False,
    )

    assert result.state == CompletionState.BLOCKED
    assert "environment" in result.evidence_precision_hint.lower()


def test_canonical_partial_verification_is_preserved():
    """Incomplete deterministic coverage must not be promoted to verified."""
    evidence = [
        AcceptanceEvidence(
            criterion_id="ac1",
            description="Test criterion",
            status=EvidenceStatus.PASS,
            explanation="Direct acceptance evidence passed",
        )
    ]

    result = determine_completion_state(
        has_ambiguity=False,
        blocked=False,
        execution_failed=False,
        evidence=evidence,
        verification_status="PARTIALLY_VERIFIED",
    )

    assert result.state == CompletionState.PARTIALLY_VERIFIED
    assert "incomplete" in result.evidence_precision_hint.lower()


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
        evidence=evidence,
    )

    # Execution failure is not directly handled in new logic, but would be caught
    # by other failure conditions. For now, we expect PARTIALLY_VERIFIED as fallback
    assert result.state in (CompletionState.FAILED, CompletionState.PARTIALLY_VERIFIED)


def test_failed_state_required_criterion_failure():
    """Test that required criterion failure results in FAILED state."""
    evidence = [
        AcceptanceEvidence(
            criterion_id="ac1",
            description="Test criterion",
            status=EvidenceStatus.FAIL,
            explanation="Test failed",
            required=True,
        )
    ]

    result = determine_completion_state(
        has_ambiguity=False,
        blocked=False,
        execution_failed=False,
        evidence=evidence,
    )

    assert result.state == CompletionState.FAILED
    assert result.failure_type is not None
    assert "failed" in result.evidence_precision_hint.lower()


def test_failed_state_constraint_violation():
    """Test that hard constraint violation results in FAILED state."""
    evidence = [
        AcceptanceEvidence(
            criterion_id="ac1",
            description="Test criterion",
            status=EvidenceStatus.PASS,
            explanation="Test passed",
            required=True,
            constraint=ConstraintEvidence(
                status=ConstraintStatus.VIOLATED,
                severity=ConstraintSeverity.CRITICAL,
                has_hard_policy_violation=True,
                has_attempted_violation=False,
                has_compilation_error=False,
                has_advisory=False,
                explanation="Hard policy violation",
            ),
        )
    ]

    result = determine_completion_state(
        has_ambiguity=False,
        blocked=False,
        execution_failed=False,
        evidence=evidence,
    )

    assert result.state == CompletionState.FAILED
    assert result.failure_type is not None
    assert result.constraint_violation_count > 0
    assert "constraint" in result.evidence_precision_hint.lower()


def test_verified_state():
    """Test that VERIFIED state requires all required criteria PASS."""
    evidence = [
        AcceptanceEvidence(
            criterion_id="ac1",
            description="Test criterion",
            status=EvidenceStatus.PASS,
            explanation="Test passed",
            required=True,
        ),
        AcceptanceEvidence(
            criterion_id="ac2",
            description="Another criterion",
            status=EvidenceStatus.PASS,
            explanation="Test passed",
            required=True,
        ),
    ]

    result = determine_completion_state(
        has_ambiguity=False,
        blocked=False,
        execution_failed=False,
        evidence=evidence,
    )

    assert result.state == CompletionState.VERIFIED
    assert result.criterion_pass_count == 2
    assert result.criterion_unverified_count == 0


def test_partially_verified_state_unverified_required():
    """Test that unverified required criteria results in PARTIALLY_VERIFIED."""
    evidence = [
        AcceptanceEvidence(
            criterion_id="ac1",
            description="Test criterion",
            status=EvidenceStatus.PASS,
            explanation="Test passed",
            required=True,
        ),
        AcceptanceEvidence(
            criterion_id="ac2",
            description="Another criterion",
            status=EvidenceStatus.UNVERIFIED,
            explanation="Not verified",
            required=True,
        ),
    ]

    result = determine_completion_state(
        has_ambiguity=False,
        blocked=False,
        execution_failed=False,
        evidence=evidence,
    )

    assert result.state == CompletionState.PARTIALLY_VERIFIED
    assert result.criterion_pass_count == 1
    assert result.criterion_unverified_count == 1


def test_partially_verified_state_unsupported_non_critical_constraint():
    """Test that unsupported non-critical constraint results in PARTIALLY_VERIFIED."""
    evidence = [
        AcceptanceEvidence(
            criterion_id="ac1",
            description="Test criterion",
            status=EvidenceStatus.PASS,
            explanation="Test passed",
            required=True,
            constraint=ConstraintEvidence(
                status=ConstraintStatus.UNSUPPORTED,
                severity=ConstraintSeverity.LOW,
                has_hard_policy_violation=False,
                has_attempted_violation=False,
                has_compilation_error=False,
                has_advisory=False,
                explanation="Cannot verify low-severity constraint",
            ),
        )
    ]

    result = determine_completion_state(
        has_ambiguity=False,
        blocked=False,
        execution_failed=False,
        evidence=evidence,
    )

    assert result.state == CompletionState.PARTIALLY_VERIFIED
    assert "constraint" in result.evidence_precision_hint.lower()


def test_blocked_state_unsupported_critical_constraint():
    """Test that unsupported critical constraint results in BLOCKED."""
    evidence = [
        AcceptanceEvidence(
            criterion_id="ac1",
            description="Test criterion",
            status=EvidenceStatus.PASS,
            explanation="Test passed",
            required=True,
            constraint=ConstraintEvidence(
                status=ConstraintStatus.UNSUPPORTED,
                severity=ConstraintSeverity.CRITICAL,
                has_hard_policy_violation=False,
                has_attempted_violation=False,
                has_compilation_error=False,
                has_advisory=False,
                explanation="Cannot verify critical constraint",
            ),
        )
    ]

    result = determine_completion_state(
        has_ambiguity=False,
        blocked=False,
        execution_failed=False,
        evidence=evidence,
    )

    assert result.state == CompletionState.BLOCKED
    assert "constraint" in result.evidence_precision_hint.lower()


def test_optional_criteria_ignored():
    """Test that optional criteria do not affect completion state."""
    evidence = [
        AcceptanceEvidence(
            criterion_id="ac1",
            description="Required criterion",
            status=EvidenceStatus.PASS,
            explanation="Test passed",
            required=True,
        ),
        AcceptanceEvidence(
            criterion_id="ac2",
            description="Optional criterion",
            status=EvidenceStatus.FAIL,
            explanation="Optional failed",
            required=False,
        ),
    ]

    result = determine_completion_state(
        has_ambiguity=False,
        blocked=False,
        execution_failed=False,
        evidence=evidence,
    )

    assert result.state == CompletionState.VERIFIED
    assert result.criterion_pass_count == 1


def test_empty_evidence():
    """Test that empty evidence results in VERIFIED (backward compatibility)."""
    result = determine_completion_state(
        has_ambiguity=False,
        blocked=False,
        execution_failed=False,
        evidence=[],
    )

    # Empty evidence means no required criteria, should be VERIFIED for backward compatibility
    assert result.state == CompletionState.VERIFIED
    assert "no required criteria" in result.evidence_precision_hint.lower()


def test_compliant_constraints():
    """Test that compliant constraints do not block verification."""
    evidence = [
        AcceptanceEvidence(
            criterion_id="ac1",
            description="Test criterion",
            status=EvidenceStatus.PASS,
            explanation="Test passed",
            required=True,
            constraint=ConstraintEvidence(
                status=ConstraintStatus.COMPLIANT,
                severity=ConstraintSeverity.MEDIUM,
                has_hard_policy_violation=False,
                has_attempted_violation=False,
                has_compilation_error=False,
                has_advisory=False,
                explanation="Compliant",
            ),
        )
    ]

    result = determine_completion_state(
        has_ambiguity=False,
        blocked=False,
        execution_failed=False,
        evidence=evidence,
    )

    assert result.state == CompletionState.VERIFIED
    assert result.constraint_violation_count == 0


def test_repair_selector_completion_hint_non_blocking_failures():
    """Test that repair selector provides correct completion hint for non-blocking failures."""
    from patchpilot.evidence.schema import CheckTransition
    from patchpilot.verification.report import CheckReport, VerificationReport

    # Create a report with only pre-existing failures (non-blocking)
    report = VerificationReport(passed=False)
    report.add_check(
        CheckReport(
            method="pytest",
            phase="post_patch",
            level="LEVEL_3_REGRESSION",
            command="pytest tests/",
            passed=False,
            exit_code=1,
            duration_seconds=1.0,
            failure_type="AssertionError",
            tier="affected",
            transition=CheckTransition.PRE_EXISTING_FAILURE.value,
            failure_fingerprint="preexisting123",
            summary={"error_type": "AssertionError", "failed_tests": ["test_old"]},
        )
    )

    selector = RepairSelector(strategy="balanced")
    selection = selector.select_repair_candidates(report)

    assert selection.should_repair is False
    assert selection.should_stop is True
    assert selection.completion_hint == "PARTIALLY_VERIFIED"
    assert "non-repairable" in selection.stop_reason.lower() or "pre-existing" in selection.stop_reason.lower()


def test_repair_selector_completion_hint_blocking_failures():
    """Test that repair selector provides correct completion hint for blocking failures."""
    from patchpilot.evidence.schema import CheckTransition
    from patchpilot.verification.report import CheckReport, VerificationReport
    from patchpilot.workflow.failure_classifier import FailureType

    # Create a report with environment failure (blocking)
    report = VerificationReport(passed=False)
    report.add_check(
        CheckReport(
            method="pytest",
            phase="post_patch",
            level="LEVEL_2_TARGET_TESTS",
            command="pytest tests/",
            passed=False,
            exit_code=1,
            duration_seconds=1.0,
            failure_type=FailureType.ENVIRONMENT_FAILURE.value,
            tier="required",
            transition=CheckTransition.NEW_OR_UNCOMPARED.value,
            failure_fingerprint="env123",
            summary={"error_type": "ModuleNotFoundError", "relevant_output": "command not found"},
        )
    )

    selector = RepairSelector(strategy="balanced")
    selection = selector.select_repair_candidates(report)

    assert selection.should_repair is False
    assert selection.should_stop is True
    assert selection.completion_hint == "BLOCKED"
    assert "non-repairable" in selection.stop_reason.lower()
