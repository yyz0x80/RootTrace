"""Markdown renderer for acceptance coverage reports.

This module provides functionality to render acceptance evidence and completion
states into human-readable markdown format. It formats the verification results
in a structured way that clearly shows the status of each acceptance criterion,
the files that were changed, and the verification commands that were run.

The renderer handles empty lists gracefully by outputting "None" instead of
leaving empty sections that could confuse users.
"""

from patchpilot.evidence.schema import AcceptanceCoverageReport, AcceptanceEvidence


def render_acceptance_coverage(
    evidence: list[AcceptanceEvidence],
    final_status: str,
) -> str:
    """Render acceptance evidence coverage as a markdown report.

    Formats the acceptance criteria verification results into a structured
    markdown document that shows the final completion status and detailed
    information for each acceptance criterion.

    Args:
        evidence: List of AcceptanceEvidence objects containing verification
                  results for each acceptance criterion
        final_status: The overall completion state as a string value

    Returns:
        A formatted markdown string containing the acceptance coverage report
    """
    lines = [
        "# Acceptance Coverage",
        "",
        f"Final status: **{final_status}**",
        "",
    ]

    if not evidence:
        lines.extend([
            "No acceptance criteria evidence available.",
            "",
        ])
        return "\n".join(lines)

    for item in evidence:
        lines.extend([
            f"## {item.criterion_id}: {item.status.value}",
            "",
            item.description,
            "",
            "Changed files:",
            _format_list(item.changed_files),
            "",
            "Tests:",
            _format_list(item.tests),
            "",
            "Verification:",
            _format_list(item.command_results),
            "",
            f"Explanation: {item.explanation}",
            "",
        ])

        # Add detailed evidence categories if available
        if item.behavior_change:
            lines.extend([
                "### Behavior Change",
                f"Status: {item.behavior_change.status.value}",
                f"Baseline: {'PASS' if item.behavior_change.baseline_passed else 'FAIL'}",
                f"Post-patch: {'PASS' if item.behavior_change.post_patch_passed else 'FAIL'}",
                f"Explanation: {item.behavior_change.explanation}",
                "",
            ])

        if item.behavior_preservation:
            lines.extend([
                "### Behavior Preservation",
                f"Status: {item.behavior_preservation.status.value}",
                f"Baseline: {'PASS' if item.behavior_preservation.baseline_passed else 'FAIL'}",
                f"Post-patch: {'PASS' if item.behavior_preservation.post_patch_passed else 'FAIL'}",
                f"Explanation: {item.behavior_preservation.explanation}",
                "",
            ])

        if item.structural_contract:
            lines.extend([
                "### Structural Contract",
                f"Status: {item.structural_contract.status.value}",
                f"Specialized check: {'Yes' if item.structural_contract.has_specialized_check else 'No'}",
                f"Check passed: {'Yes' if item.structural_contract.check_passed else 'No'}",
                f"Pytest only: {'Yes' if item.structural_contract.has_pytest_only else 'No'}",
                f"Explanation: {item.structural_contract.explanation}",
                "",
            ])

        if item.constraint:
            lines.extend([
                "### Constraint",
                f"Status: {item.constraint.status.value}",
                f"Hard policy violation: {'Yes' if item.constraint.has_hard_policy_violation else 'No'}",
                f"Attempted violation: {'Yes' if item.constraint.has_attempted_violation else 'No'}",
                f"Compilation error: {'Yes' if item.constraint.has_compilation_error else 'No'}",
                f"Advisory: {'Yes' if item.constraint.has_advisory else 'No'}",
                f"Explanation: {item.constraint.explanation}",
                "",
            ])

    return "\n".join(lines)


def render_coverage_report(report: AcceptanceCoverageReport) -> str:
    """Render a complete acceptance coverage report.

    Formats the full AcceptanceCoverageReport into markdown, including
    the summary and all acceptance evidence details.

    Args:
        report: AcceptanceCoverageReport containing evidence and completion state

    Returns:
        A formatted markdown string containing the complete coverage report
    """
    lines = [
        "# Acceptance Coverage",
        "",
        f"Final status: **{report.completion_state.value}**",
        "",
    ]

    if report.summary:
        lines.extend([
            "## Summary",
            "",
            report.summary,
            "",
        ])

    if not report.acceptance_evidence:
        lines.extend([
            "No acceptance criteria evidence available.",
            "",
        ])
        return "\n".join(lines)

    for item in report.acceptance_evidence:
        lines.extend([
            f"## {item.criterion_id}: {item.status.value}",
            "",
            item.description,
            "",
            "Changed files:",
            _format_list(item.changed_files),
            "",
            "Tests:",
            _format_list(item.tests),
            "",
            "Verification:",
            _format_list(item.command_results),
            "",
            f"Explanation: {item.explanation}",
            "",
        ])

        # Add detailed evidence categories if available
        if item.behavior_change:
            lines.extend([
                "### Behavior Change",
                f"Status: {item.behavior_change.status.value}",
                f"Baseline: {'PASS' if item.behavior_change.baseline_passed else 'FAIL'}",
                f"Post-patch: {'PASS' if item.behavior_change.post_patch_passed else 'FAIL'}",
                f"Explanation: {item.behavior_change.explanation}",
                "",
            ])

        if item.behavior_preservation:
            lines.extend([
                "### Behavior Preservation",
                f"Status: {item.behavior_preservation.status.value}",
                f"Baseline: {'PASS' if item.behavior_preservation.baseline_passed else 'FAIL'}",
                f"Post-patch: {'PASS' if item.behavior_preservation.post_patch_passed else 'FAIL'}",
                f"Explanation: {item.behavior_preservation.explanation}",
                "",
            ])

        if item.structural_contract:
            lines.extend([
                "### Structural Contract",
                f"Status: {item.structural_contract.status.value}",
                f"Specialized check: {'Yes' if item.structural_contract.has_specialized_check else 'No'}",
                f"Check passed: {'Yes' if item.structural_contract.check_passed else 'No'}",
                f"Pytest only: {'Yes' if item.structural_contract.has_pytest_only else 'No'}",
                f"Explanation: {item.structural_contract.explanation}",
                "",
            ])

        if item.constraint:
            lines.extend([
                "### Constraint",
                f"Status: {item.constraint.status.value}",
                f"Hard policy violation: {'Yes' if item.constraint.has_hard_policy_violation else 'No'}",
                f"Attempted violation: {'Yes' if item.constraint.has_attempted_violation else 'No'}",
                f"Compilation error: {'Yes' if item.constraint.has_compilation_error else 'No'}",
                f"Advisory: {'Yes' if item.constraint.has_advisory else 'No'}",
                f"Explanation: {item.constraint.explanation}",
                "",
            ])

    return "\n".join(lines)


def _format_list(items: list[str]) -> str:
    """Format a list of items as markdown bullet points or None if empty.

    Args:
        items: List of strings to format

    Returns:
        A string containing either bullet-pointed items or "None"
    """
    if not items:
        return "None"
    return "\n".join(f"- `{item}`" for item in items)
