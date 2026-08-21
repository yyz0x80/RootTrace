"""Evidence aggregator for deterministic acceptance criteria verification.

This module implements the core aggregation logic that determines verification
status based on concrete evidence from baseline and post-patch verification.
It follows fixed deterministic rules rather than allowing models to decide PASS/FAIL.

The aggregator evaluates evidence across four dimensions:
1. Behavior Change - Did the fix change behavior from FAIL to PASS?
2. Behavior Preservation - Did the change preserve existing behavior?
3. Structural Contract - Do AST/mock checks verify the implementation?
4. Constraint - Did the change comply with policy constraints?

Each dimension has deterministic rules that compute status from verification results.
"""

from __future__ import annotations

from patchpilot.evidence.schema import (
    AcceptanceEvidence,
    BehaviorChangeEvidence,
    BehaviorChangeStatus,
    BehaviorPreservationEvidence,
    BehaviorPreservationStatus,
    ConstraintEvidence,
    ConstraintSeverity,
    ConstraintStatus,
    EvidenceStatus,
    StructuralContractEvidence,
    StructuralContractStatus,
)
from patchpilot.verification.report import VerificationReport


def aggregate_evidence(
    criterion_id: str,
    description: str,
    changed_files: list[str],
    tests: list[str],
    command_results: list[str],
    report: VerificationReport,
) -> AcceptanceEvidence:
    """Aggregate evidence for a single acceptance criterion using deterministic rules.

    Evaluates evidence across four dimensions and computes an overall status.
    The overall status is determined by the most severe evidence across all dimensions.

    Args:
        criterion_id: ID of the acceptance criterion
        description: Human-readable description of the criterion
        changed_files: List of files that were actually changed
        tests: List of test names/paths for this criterion
        command_results: List of verification command results
        report: Verification report with baseline and post-patch checks

    Returns:
        AcceptanceEvidence with computed status and detailed evidence per dimension
    """
    # Aggregate evidence for each dimension
    behavior_change = _aggregate_behavior_change(criterion_id, report)
    behavior_preservation = _aggregate_behavior_preservation(criterion_id, report)
    structural_contract = _aggregate_structural_contract(criterion_id, report)
    constraint = _aggregate_constraint(criterion_id, report)

    # Determine overall status based on all dimensions
    status = _determine_overall_status(
        behavior_change,
        behavior_preservation,
        structural_contract,
        constraint,
        report,
        criterion_id,
    )

    # Generate explanation based on the determining factor
    explanation = _generate_explanation(
        status,
        behavior_change,
        behavior_preservation,
        structural_contract,
        constraint,
        report,
        criterion_id,
    )

    return AcceptanceEvidence(
        criterion_id=criterion_id,
        description=description,
        status=status,
        changed_files=changed_files,
        tests=tests,
        command_results=command_results,
        explanation=explanation,
        behavior_change=behavior_change,
        behavior_preservation=behavior_preservation,
        structural_contract=structural_contract,
        constraint=constraint,
    )


def _aggregate_behavior_change(
    criterion_id: str,
    report: VerificationReport,
) -> BehaviorChangeEvidence | None:
    """Aggregate behavior change evidence based on baseline and post-patch results.

    Implements deterministic rules:
    - FAIL -> PASS: PASS (strongest repair evidence)
    - PASS -> PASS: ALREADY_SATISFIED
    - FAIL -> FAIL: FAIL
    - PASS -> FAIL: FAIL (regression)
    - No direct check: None (no evidence)

    Args:
        criterion_id: ID of the acceptance criterion
        report: Verification report with baseline and post-patch checks

    Returns:
        BehaviorChangeEvidence with computed status, or None if no direct checks exist
    """
    baseline_checks = [
        check
        for check in report.get_baseline_checks()
        if criterion_id in check.subject_ids and check.direct
    ]
    post_patch_checks = [
        check
        for check in report.get_post_patch_checks()
        if criterion_id in check.subject_ids and check.direct
    ]

    # If no direct checks, return None (no evidence)
    if not baseline_checks and not post_patch_checks:
        return None

    # Determine baseline and post-patch status
    baseline_passed = all(check.passed for check in baseline_checks) if baseline_checks else False
    post_patch_passed = all(check.passed for check in post_patch_checks) if post_patch_checks else False

    # Apply deterministic rules
    if not baseline_passed and post_patch_passed:
        status = BehaviorChangeStatus.PASS
        explanation = "Behavior changed from FAIL to PASS (strongest repair evidence)."
    elif baseline_passed and post_patch_passed:
        status = BehaviorChangeStatus.ALREADY_SATISFIED
        explanation = "Behavior was already satisfied in baseline (no change needed)."
    elif not baseline_passed and not post_patch_passed:
        status = BehaviorChangeStatus.FAIL
        explanation = "Behavior remains FAIL after patch (fix ineffective)."
    elif baseline_passed and not post_patch_passed:
        status = BehaviorChangeStatus.FAIL
        explanation = "Behavior changed from PASS to FAIL (regression detected)."
    else:
        status = BehaviorChangeStatus.UNVERIFIED
        explanation = "Insufficient evidence to determine behavior change."

    return BehaviorChangeEvidence(
        status=status,
        baseline_passed=baseline_passed,
        post_patch_passed=post_patch_passed,
        explanation=explanation,
    )


def _aggregate_behavior_preservation(
    criterion_id: str,
    report: VerificationReport,
) -> BehaviorPreservationEvidence | None:
    """Aggregate behavior preservation evidence.

    Implements deterministic rules:
    - PASS -> PASS: PASS
    - PASS -> FAIL: FAIL
    - FAIL -> FAIL/PASS: UNVERIFIED or separate baseline defect
    - No direct check: None (no evidence)

    Args:
        criterion_id: ID of the acceptance criterion
        report: Verification report with baseline and post-patch checks

    Returns:
        BehaviorPreservationEvidence with computed status, or None if no direct checks exist
    """
    baseline_checks = [
        check
        for check in report.get_baseline_checks()
        if criterion_id in check.subject_ids and check.direct
    ]
    post_patch_checks = [
        check
        for check in report.get_post_patch_checks()
        if criterion_id in check.subject_ids and check.direct
    ]

    # If no direct checks, return None (no evidence)
    if not baseline_checks and not post_patch_checks:
        return None

    # Determine baseline and post-patch status
    baseline_passed = all(check.passed for check in baseline_checks) if baseline_checks else False
    post_patch_passed = all(check.passed for check in post_patch_checks) if post_patch_checks else False

    # Apply deterministic rules
    if baseline_passed and post_patch_passed:
        status = BehaviorPreservationStatus.PASS
        explanation = "Behavior preserved (baseline PASS maintained)."
    elif baseline_passed and not post_patch_passed:
        status = BehaviorPreservationStatus.FAIL
        explanation = "Behavior not preserved (baseline PASS became FAIL)."
    elif not baseline_passed:
        status = BehaviorPreservationStatus.UNVERIFIED
        explanation = "Baseline has defect; preservation cannot be verified from broken baseline."
    else:
        status = BehaviorPreservationStatus.UNVERIFIED
        explanation = "Insufficient evidence to determine behavior preservation."

    return BehaviorPreservationEvidence(
        status=status,
        baseline_passed=baseline_passed,
        post_patch_passed=post_patch_passed,
        explanation=explanation,
    )


def _aggregate_structural_contract(
    criterion_id: str,
    report: VerificationReport,
) -> StructuralContractEvidence | None:
    """Aggregate structural contract evidence from AST/mock checks.

    Implements deterministic rules:
    - Specialized AST/mock check passed: PASS
    - Specialized check failed: FAIL
    - Only pytest: UNVERIFIED
    - No structural checks: None (no evidence)

    Args:
        criterion_id: ID of the acceptance criterion
        report: Verification report with structural checks

    Returns:
        StructuralContractEvidence with computed status, or None if no structural checks exist
    """
    # Find specialized structural checks (acceptance_probe, structural_check, ast_check, mock_check)
    specialized_checks = [
        check
        for check in report.checks
        if criterion_id in check.subject_ids
        and check.method in ("acceptance_probe", "structural_check", "ast_check", "mock_check")
    ]

    # Find pytest checks
    pytest_checks = [
        check
        for check in report.checks
        if criterion_id in check.subject_ids and check.method == "pytest"
    ]

    has_specialized_check = len(specialized_checks) > 0
    has_pytest_only = len(pytest_checks) > 0 and not has_specialized_check

    # If no structural checks at all, return None (no evidence)
    if not has_specialized_check and not has_pytest_only:
        return None

    if has_specialized_check:
        check_passed = all(check.passed for check in specialized_checks)
        if check_passed:
            status = StructuralContractStatus.PASS
            explanation = "Specialized acceptance probe and/or structural checks passed."
        else:
            status = StructuralContractStatus.FAIL
            explanation = "Specialized acceptance probe and/or structural checks failed."
    elif has_pytest_only:
        status = StructuralContractStatus.UNVERIFIED
        check_passed = False
        explanation = "Only pytest available; structural contract requires specialized acceptance probe or structural checks."
    else:
        status = StructuralContractStatus.UNVERIFIED
        check_passed = False
        explanation = "No structural contract checks available."

    return StructuralContractEvidence(
        status=status,
        has_specialized_check=has_specialized_check,
        check_passed=check_passed,
        has_pytest_only=has_pytest_only,
        explanation=explanation,
    )


def _aggregate_constraint(
    criterion_id: str,
    report: VerificationReport,
) -> ConstraintEvidence | None:
    """Aggregate constraint evidence for policy compliance.

    Implements deterministic rules:
    - Hard policy no violations: COMPLIANT
    - Attempted violation but no real change: COMPLIANT (separate attempted violation record)
    - Final diff violation: VIOLATED
    - Cannot compile: UNSUPPORTED
    - Advisory: ADVISORY
    - No constraint checks: None (no evidence)

    Args:
        criterion_id: ID of the acceptance criterion
        report: Verification report with constraint checks

    Returns:
        ConstraintEvidence with computed status, or None if no constraint checks exist
    """
    # Find constraint audit checks (now apply to all criteria since constraint audit is global)
    constraint_checks = [
        check
        for check in report.checks
        if check.phase == "constraint_audit"
    ]

    # If no constraint checks, return None (no evidence)
    if not constraint_checks:
        return None

    # Determine constraint status from check results (global constraint audit)
    has_hard_policy_violation = False
    has_attempted_violation = False
    has_compilation_error = False
    has_advisory = False

    for check in constraint_checks:
        if not check.passed:
            summary = check.summary or {}
            if summary.get("violation_type") == "hard_policy":
                has_hard_policy_violation = True
            elif summary.get("violation_type") == "attempted":
                has_attempted_violation = True
            elif summary.get("error_type") == "compilation":
                has_compilation_error = True
            elif summary.get("violation_type") == "advisory":
                has_advisory = True

    # Apply deterministic rules
    if has_compilation_error:
        status = ConstraintStatus.UNSUPPORTED
        severity = ConstraintSeverity.CRITICAL
        explanation = "Cannot compile (unsupported change)."
    elif has_hard_policy_violation:
        status = ConstraintStatus.VIOLATED
        severity = ConstraintSeverity.CRITICAL
        explanation = "Hard policy violation detected in final diff."
    elif has_advisory:
        status = ConstraintStatus.ADVISORY
        severity = ConstraintSeverity.LOW
        explanation = "Advisory issues detected (not blocking)."
    elif has_attempted_violation:
        status = ConstraintStatus.COMPLIANT
        severity = ConstraintSeverity.MEDIUM
        explanation = "Compliant (violation attempted but rejected, no real change made)."
    else:
        status = ConstraintStatus.COMPLIANT
        severity = ConstraintSeverity.MEDIUM
        explanation = "Compliant (no hard policy violations)."

    return ConstraintEvidence(
        status=status,
        severity=severity,
        has_hard_policy_violation=has_hard_policy_violation,
        has_attempted_violation=has_attempted_violation,
        has_compilation_error=has_compilation_error,
        has_advisory=has_advisory,
        explanation=explanation,
    )


def _determine_overall_status(
    behavior_change: BehaviorChangeEvidence | None,
    behavior_preservation: BehaviorPreservationEvidence | None,
    structural_contract: StructuralContractEvidence | None,
    constraint: ConstraintEvidence | None,
    report: VerificationReport | None = None,
    criterion_id: str = "",
) -> EvidenceStatus:
    """Determine overall evidence status from all dimensions.

    The overall status is determined by the most severe evidence:
    - FAIL if any dimension is FAIL or if post-patch checks mapped to criterion failed
    - PASS if at least one dimension is PASS and none are FAIL and post-patch checks pass
    - UNVERIFIED otherwise

    Args:
        behavior_change: Behavior change evidence
        behavior_preservation: Behavior preservation evidence
        structural_contract: Structural contract evidence
        constraint: Constraint evidence
        report: Verification report with post-patch check results
        criterion_id: ID of the acceptance criterion for post-patch failure check

    Returns:
        Overall EvidenceStatus
    """
    evidence_items = [
        behavior_change,
        behavior_preservation,
        structural_contract,
        constraint,
    ]

    # Check for FAIL status in any dimension
    for evidence in evidence_items:
        if evidence is None:
            continue
        if evidence.status in (BehaviorChangeStatus.FAIL, BehaviorPreservationStatus.FAIL,
                              StructuralContractStatus.FAIL, ConstraintStatus.VIOLATED):
            return EvidenceStatus.FAIL

    # Check if post-patch checks mapped to this criterion failed
    # This prevents marking AC as PASS when code cannot execute (e.g., import errors)
    if report and criterion_id:
        post_patch_checks = [
            check
            for check in report.get_post_patch_checks()
            if criterion_id in check.subject_ids
        ]
        if post_patch_checks and not all(check.passed for check in post_patch_checks):
            return EvidenceStatus.FAIL

    # Check for PASS/COMPLIANT in any dimension, but only if we have actual evidence
    has_pass = False
    has_any_evidence = False
    for evidence in evidence_items:
        if evidence is None:
            continue
        has_any_evidence = True
        if evidence.status in (
            BehaviorChangeStatus.PASS,
            BehaviorChangeStatus.ALREADY_SATISFIED,
            BehaviorPreservationStatus.PASS,
            StructuralContractStatus.PASS,
        ):
            has_pass = True

    # Only return PASS if we have actual evidence and at least one dimension passed
    if has_any_evidence and has_pass:
        return EvidenceStatus.PASS

    return EvidenceStatus.UNVERIFIED


def _generate_explanation(
    status: EvidenceStatus,
    behavior_change: BehaviorChangeEvidence | None,
    behavior_preservation: BehaviorPreservationEvidence | None,
    structural_contract: StructuralContractEvidence | None,
    constraint: ConstraintEvidence | None,
    report: VerificationReport | None = None,
    criterion_id: str = "",
) -> str:
    """Generate explanation based on the determining evidence dimension.

    Args:
        status: Overall evidence status
        behavior_change: Behavior change evidence
        behavior_preservation: Behavior preservation evidence
        structural_contract: Structural contract evidence
        constraint: Constraint evidence
        report: Verification report with post-patch check results
        criterion_id: ID of the acceptance criterion for post-patch failure check

    Returns:
        Human-readable explanation
    """
    if status == EvidenceStatus.FAIL:
        # Check if post-patch checks mapped to this criterion failed
        if report and criterion_id:
            post_patch_checks = [
                check
                for check in report.get_post_patch_checks()
                if criterion_id in check.subject_ids
            ]
            failed_post_patch_checks = [check for check in post_patch_checks if not check.passed]
            if failed_post_patch_checks:
                return f"FAIL: Post-patch verification failed for criterion {criterion_id}. Code cannot execute or tests fail."

        # Find the FAILing dimension
        if constraint and constraint.status == ConstraintStatus.VIOLATED:
            return f"FAIL: {constraint.explanation}"
        if behavior_change and behavior_change.status == BehaviorChangeStatus.FAIL:
            return f"FAIL: {behavior_change.explanation}"
        if behavior_preservation and behavior_preservation.status == BehaviorPreservationStatus.FAIL:
            return f"FAIL: {behavior_preservation.explanation}"
        if structural_contract and structural_contract.status == StructuralContractStatus.FAIL:
            return f"FAIL: {structural_contract.explanation}"
        return "FAIL: One or more verification dimensions failed."

    if status == EvidenceStatus.PASS:
        # Find the PASSing dimension
        if behavior_change and behavior_change.status == BehaviorChangeStatus.PASS:
            return f"PASS: {behavior_change.explanation}"
        if behavior_preservation and behavior_preservation.status == BehaviorPreservationStatus.PASS:
            return f"PASS: {behavior_preservation.explanation}"
        if structural_contract and structural_contract.status == StructuralContractStatus.PASS:
            return f"PASS: {structural_contract.explanation}"
        if constraint and constraint.status == ConstraintStatus.COMPLIANT:
            return f"PASS: {constraint.explanation}"
        return "PASS: At least one verification dimension passed."

    return "UNVERIFIED: Insufficient evidence to determine criterion satisfaction."
