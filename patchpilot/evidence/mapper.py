"""Evidence mapper for linking acceptance criteria to verification results.

This module provides the mapping logic that connects acceptance criteria
to actual code changes and verification results. It implements the core
algorithm for determining whether each acceptance criterion has been satisfied
based on concrete evidence from the workspace and verification reports.

The mapper follows a structured 4-step process:
1. Identify planned files for each acceptance criterion
2. Calculate intersection with actual workspace changes
3. Find verification checks mapped to the criterion
4. Determine evidence status based on fixed rules

This ensures that acceptance criteria are only marked as PASS when there is
concrete evidence of both code changes and successful verification.
"""

from __future__ import annotations

from patchpilot.evidence.schema import AcceptanceEvidence, EvidenceStatus
from patchpilot.issue.schema import NormalizedIssue
from patchpilot.planning.schema import ChangePlan
from patchpilot.tools import WorkspaceChange
from patchpilot.verification.report import CheckReport, VerificationReport


def map_acceptance_evidence(
    issue: NormalizedIssue,
    plan: ChangePlan,
    actual_changes: list[WorkspaceChange],
    report: VerificationReport,
) -> list[AcceptanceEvidence]:
    """Map acceptance criteria to concrete evidence from changes and verification.

    For each acceptance criterion in the issue, this function determines its
    verification status by analyzing:
    - Which files were planned to change for this criterion
    - Which files actually changed in the workspace
    - Which verification checks are associated with this criterion
    - Whether those checks passed or failed

    The status determination follows fixed rules:
    - FAIL: Any mapped verification check failed
    - PASS: Planned files changed AND a direct behavioral check passed
    - UNVERIFIED: No code change or no passing verification

    Args:
        issue: Normalized issue containing acceptance criteria
        plan: Change plan with planned changes and their AC mappings
        actual_changes: List of actual workspace changes from git status
        report: Verification report with check results and AC mappings

    Returns:
        List of AcceptanceEvidence objects, one per acceptance criterion,
        with status, changed files, mapped tests, and explanations
    """
    evidence_list: list[AcceptanceEvidence] = []

    # Get actual changed file paths for efficient lookup
    actual_paths = {change.path for change in actual_changes}

    for criterion in issue.acceptance_criteria:
        # Step 5.1: Find planned files for this acceptance criterion
        planned_files = {
            change.path
            for change in plan.planned_changes
            if criterion.id in change.acceptance_criteria
        }

        # Step 5.2: Calculate intersection with actual changes
        changed_files = sorted(planned_files & actual_paths)

        # Step 5.3: Find verification checks mapped to this criterion
        mapped_checks = [
            check
            for check in report.checks
            if criterion.id in check.acceptance_criteria
        ]
        direct_checks = [
            check
            for check in mapped_checks
            if criterion.id in check.direct_acceptance_criteria
        ]

        # Step 5.4: Determine status based on fixed rules
        status, explanation = _determine_evidence_status(
            changed_files=changed_files,
            mapped_checks=mapped_checks,
            direct_checks=direct_checks,
        )

        # Extract test names from mapped checks for evidence
        tests = [
            check.command
            for check in mapped_checks
            if "pytest" in check.command
        ]

        # Extract command results for evidence
        command_results = [
            f"{check.level}: {'PASSED' if check.passed else 'FAILED'}"
            for check in mapped_checks
        ]

        evidence = AcceptanceEvidence(
            criterion_id=criterion.id,
            description=criterion.description,
            status=status,
            changed_files=changed_files,
            tests=tests,
            command_results=command_results,
            explanation=explanation,
        )

        evidence_list.append(evidence)

    return evidence_list


def _determine_evidence_status(
    changed_files: list[str],
    mapped_checks: list[CheckReport],
    direct_checks: list[CheckReport],
) -> tuple[EvidenceStatus, str]:
    """Determine evidence status based on changed files and check results.

    Implements the fixed rule hierarchy for status determination:
    1. FAIL if any mapped check failed
    2. PASS if files changed and at least one direct behavioral check passed
    3. UNVERIFIED otherwise

    Args:
        changed_files: List of files that actually changed for this criterion
        mapped_checks: List of verification checks associated with this criterion
        direct_checks: Mapped checks that directly exercise the criterion.

    Returns:
        Tuple of (EvidenceStatus, explanation string)
    """
    # Rule 1: FAIL if any mapped check failed
    if any(not check.passed for check in mapped_checks):
        return (
            EvidenceStatus.FAIL,
            "A mapped deterministic verification check failed.",
        )

    # Rule 2: broad plan mappings are not sufficient proof of behavior. Only
    # precise test targets explicitly marked as direct evidence may pass an AC.
    if changed_files and any(check.passed for check in direct_checks):
        return (
            EvidenceStatus.PASS,
            "A planned source file changed and a direct behavioral check passed.",
        )

    # Rule 3: UNVERIFIED if no code change or no passing verification
    return (
        EvidenceStatus.UNVERIFIED,
        "The criterion lacks an actual code change or passing direct behavioral evidence.",
    )
