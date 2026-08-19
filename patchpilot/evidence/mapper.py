"""Evidence mapper for linking acceptance criteria to verification results.

This module provides the mapping logic that connects acceptance criteria
to actual code changes and verification results. It collects the evidence
and delegates status determination to the aggregator module.

The mapper follows a structured process:
1. Identify planned files for each acceptance criterion
2. Calculate intersection with actual workspace changes
3. Find verification checks mapped to the criterion
4. Delegate evidence aggregation to the aggregator module

This ensures that acceptance criteria are only marked as PASS when there is
concrete evidence determined by the aggregator's deterministic rules.
"""

from __future__ import annotations

from patchpilot.evidence.aggregator import aggregate_evidence
from patchpilot.evidence.schema import AcceptanceEvidence
from patchpilot.issue.schema import NormalizedIssue
from patchpilot.planning.schema import ChangePlan
from patchpilot.tools import WorkspaceChange
from patchpilot.verification.report import VerificationReport


def map_acceptance_evidence(
    issue: NormalizedIssue,
    plan: ChangePlan,
    actual_changes: list[WorkspaceChange],
    report: VerificationReport,
) -> list[AcceptanceEvidence]:
    """Map acceptance criteria to concrete evidence from changes and verification.

    For each acceptance criterion in the issue, this function collects evidence
    and delegates status determination to the aggregator module. The aggregator
    applies deterministic rules across multiple evidence dimensions.

    Args:
        issue: Normalized issue containing acceptance criteria
        plan: Change plan with planned changes and their AC mappings
        actual_changes: List of actual workspace changes from git status
        report: Verification report with check results and AC mappings

    Returns:
        List of AcceptanceEvidence objects, one per acceptance criterion,
        with status computed by the aggregator's deterministic rules
    """
    evidence_list: list[AcceptanceEvidence] = []

    # Get actual changed file paths for efficient lookup
    actual_paths = {change.path for change in actual_changes}

    for criterion in issue.acceptance_criteria:
        # Step 1: Find planned files for this acceptance criterion
        planned_files = {
            change.path
            for change in plan.planned_changes
            if criterion.id in change.acceptance_criteria
        }

        # Step 2: Calculate intersection with actual changes
        changed_files = sorted(planned_files & actual_paths)

        # Step 3: Find verification checks mapped to this criterion
        mapped_checks = [
            check
            for check in report.checks
            if criterion.id in check.subject_ids
        ]

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

        # Step 4: Delegate evidence aggregation to compute status
        evidence = aggregate_evidence(
            criterion_id=criterion.id,
            description=criterion.description,
            changed_files=changed_files,
            tests=tests,
            command_results=command_results,
            report=report,
        )

        evidence_list.append(evidence)

    return evidence_list
