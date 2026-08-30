"""Tests for structured RootTrace pipeline diagnostics."""

from roottrace.diagnostics import (
    DiagnosticSeverity,
    PipelineDiagnostic,
    merge_diagnostics,
    project_diagnostics,
)


def test_diagnostics_are_stably_deduplicated_and_projected() -> None:
    informational = PipelineDiagnostic(
        code="tools.search_empty",
        stage="tools",
        severity=DiagnosticSeverity.INFO,
        message="Search returned no matches",
    )
    recoverable = PipelineDiagnostic(
        code="specialist.failed",
        stage="specialist",
        severity=DiagnosticSeverity.RECOVERABLE,
        message="Code specialist failed",
    )

    diagnostics = merge_diagnostics(
        [informational, recoverable],
        [recoverable],
    )

    assert diagnostics == [informational, recoverable]
    assert project_diagnostics(diagnostics) == ["Code specialist failed"]
