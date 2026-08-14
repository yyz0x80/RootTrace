"""Tests for acceptance evidence markdown renderer."""

from patchpilot.evidence.renderer import (
    render_acceptance_coverage,
    render_coverage_report,
)
from patchpilot.evidence.schema import (
    AcceptanceCoverageReport,
    AcceptanceEvidence,
    CompletionState,
    EvidenceStatus,
)


def test_render_acceptance_coverage_empty():
    """Test rendering with empty evidence list."""
    result = render_acceptance_coverage([], "VERIFIED")
    assert "# Acceptance Coverage" in result
    assert "Final status: **VERIFIED**" in result
    assert "No acceptance criteria evidence available" in result


def test_render_acceptance_coverage_basic():
    """Test rendering with basic evidence."""
    evidence = [
        AcceptanceEvidence(
            criterion_id="AC1",
            description="First criterion",
            status=EvidenceStatus.PASS,
            changed_files=["file1.py", "file2.py"],
            tests=["test_file1.py"],
            command_results=["pytest: PASSED"],
            explanation="All checks passed",
        )
    ]
    result = render_acceptance_coverage(evidence, "VERIFIED")
    assert "# Acceptance Coverage" in result
    assert "Final status: **VERIFIED**" in result
    assert "## AC1: PASS" in result
    assert "First criterion" in result
    assert "- `file1.py`" in result
    assert "- `file2.py`" in result
    assert "- `test_file1.py`" in result
    assert "- `pytest: PASSED`" in result
    assert "Explanation: All checks passed" in result


def test_render_acceptance_coverage_empty_lists():
    """Test rendering with empty lists in evidence."""
    evidence = [
        AcceptanceEvidence(
            criterion_id="AC1",
            description="Criterion with no changes",
            status=EvidenceStatus.UNVERIFIED,
            changed_files=[],
            tests=[],
            command_results=[],
            explanation="No changes made",
        )
    ]
    result = render_acceptance_coverage(evidence, "PARTIALLY_VERIFIED")
    assert "Changed files:" in result
    assert "None" in result
    assert "Tests:" in result
    assert "Verification:" in result


def test_render_acceptance_coverage_multiple():
    """Test rendering with multiple evidence items."""
    evidence = [
        AcceptanceEvidence(
            criterion_id="AC1",
            description="First criterion",
            status=EvidenceStatus.PASS,
            changed_files=["file1.py"],
            tests=["test_file1.py"],
            command_results=["pytest: PASSED"],
            explanation="Passed",
        ),
        AcceptanceEvidence(
            criterion_id="AC2",
            description="Second criterion",
            status=EvidenceStatus.FAIL,
            changed_files=["file2.py"],
            tests=["test_file2.py"],
            command_results=["pytest: FAILED"],
            explanation="Failed",
        ),
    ]
    result = render_acceptance_coverage(evidence, "FAILED")
    assert "## AC1: PASS" in result
    assert "## AC2: FAIL" in result
    assert "Final status: **FAILED**" in result


def test_render_coverage_report_empty():
    """Test rendering empty coverage report."""
    report = AcceptanceCoverageReport(
        acceptance_evidence=[],
        completion_state=CompletionState.NEEDS_CLARIFICATION,
        summary="",
    )
    result = render_coverage_report(report)
    assert "# Acceptance Coverage" in result
    assert "Final status: **NEEDS_CLARIFICATION**" in result
    assert "No acceptance criteria evidence available" in result


def test_render_coverage_report_with_summary():
    """Test rendering coverage report with summary."""
    report = AcceptanceCoverageReport(
        acceptance_evidence=[],
        completion_state=CompletionState.VERIFIED,
        summary="All acceptance criteria passed verification.",
    )
    result = render_coverage_report(report)
    assert "# Acceptance Coverage" in result
    assert "## Summary" in result
    assert "All acceptance criteria passed verification." in result


def test_render_coverage_report_full():
    """Test rendering full coverage report."""
    evidence = [
        AcceptanceEvidence(
            criterion_id="AC1",
            description="Implement feature",
            status=EvidenceStatus.PASS,
            changed_files=["src/feature.py"],
            tests=["tests/test_feature.py"],
            command_results=["pytest: PASSED", "ruff: PASSED"],
            explanation="Feature implemented and verified",
        )
    ]
    report = AcceptanceCoverageReport(
        acceptance_evidence=evidence,
        completion_state=CompletionState.VERIFIED,
        summary="Task completed successfully.",
    )
    result = render_coverage_report(report)
    assert "## Summary" in result
    assert "Task completed successfully." in result
    assert "## AC1: PASS" in result
    assert "Implement feature" in result
    assert "- `src/feature.py`" in result
    assert "- `tests/test_feature.py`" in result
