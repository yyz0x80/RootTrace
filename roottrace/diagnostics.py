"""Structured, bounded diagnostics for RootTrace pipeline execution."""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum

from pydantic import BaseModel, Field

from roottrace.evidence.schema import AgentRole

MAX_DIAGNOSTICS = 20
MAX_DIAGNOSTIC_CODE_CHARS = 80
MAX_DIAGNOSTIC_STAGE_CHARS = 80
MAX_DIAGNOSTIC_TOOL_CHARS = 80
MAX_DIAGNOSTIC_MESSAGE_CHARS = 500


class DiagnosticSeverity(StrEnum):
    """Impact level of a pipeline diagnostic."""

    INFO = "info"
    WARNING = "warning"
    RECOVERABLE = "recoverable"
    FATAL = "fatal"


class PipelineDiagnostic(BaseModel):
    """One bounded, auditable event emitted by the RCA pipeline."""

    code: str = Field(
        min_length=1,
        max_length=MAX_DIAGNOSTIC_CODE_CHARS,
    )
    stage: str = Field(
        min_length=1,
        max_length=MAX_DIAGNOSTIC_STAGE_CHARS,
    )
    severity: DiagnosticSeverity
    message: str = Field(
        min_length=1,
        max_length=MAX_DIAGNOSTIC_MESSAGE_CHARS,
    )
    agent: AgentRole | None = None
    tool: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_DIAGNOSTIC_TOOL_CHARS,
    )


def _diagnostic_key(diagnostic: PipelineDiagnostic) -> tuple[str, ...]:
    """Return the stable identity key used for diagnostic de-duplication."""
    return (
        diagnostic.code,
        diagnostic.stage,
        diagnostic.severity.value,
        diagnostic.message,
        diagnostic.agent.value if diagnostic.agent is not None else "",
        diagnostic.tool or "",
    )


def deduplicate_diagnostics(
    diagnostics: Iterable[PipelineDiagnostic],
) -> list[PipelineDiagnostic]:
    """Keep the first occurrence of each diagnostic in stable order."""
    retained: list[PipelineDiagnostic] = []
    seen: set[tuple[str, ...]] = set()
    for diagnostic in diagnostics:
        key = _diagnostic_key(diagnostic)
        if key in seen:
            continue
        seen.add(key)
        retained.append(diagnostic)
        if len(retained) >= MAX_DIAGNOSTICS:
            break
    return retained


def merge_diagnostics(
    *diagnostic_groups: Iterable[PipelineDiagnostic],
) -> list[PipelineDiagnostic]:
    """Merge diagnostic groups with stable ordering and bounded de-duplication."""
    return deduplicate_diagnostics(
        diagnostic
        for group in diagnostic_groups
        for diagnostic in group
    )


def project_diagnostics(
    diagnostics: Iterable[PipelineDiagnostic],
) -> list[str]:
    """Project actionable diagnostics to the legacy string error field."""
    messages: list[str] = []
    seen: set[str] = set()
    for diagnostic in deduplicate_diagnostics(diagnostics):
        if diagnostic.severity is DiagnosticSeverity.INFO:
            continue
        if diagnostic.message in seen:
            continue
        seen.add(diagnostic.message)
        messages.append(diagnostic.message)
        if len(messages) >= MAX_DIAGNOSTICS:
            break
    return messages


def diagnostics_from_legacy(
    messages: Iterable[str],
    *,
    stage: str = "legacy",
    code: str = "legacy.error",
    severity: DiagnosticSeverity = DiagnosticSeverity.RECOVERABLE,
) -> list[PipelineDiagnostic]:
    """Convert legacy string errors into structured diagnostics for compatibility."""
    diagnostics: list[PipelineDiagnostic] = []
    for message in messages:
        bounded = message[:MAX_DIAGNOSTIC_MESSAGE_CHARS]
        if not bounded:
            continue
        diagnostics.append(
            PipelineDiagnostic(
                code=code,
                stage=stage,
                severity=severity,
                message=bounded,
            )
        )
    return deduplicate_diagnostics(diagnostics)
