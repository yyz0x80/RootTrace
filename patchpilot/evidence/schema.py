"""Data schemas for acceptance evidence mapping and completion states."""

from enum import Enum

from pydantic import BaseModel, Field


class EvidenceStatus(str, Enum):
    """Verification status for a single acceptance criterion."""

    PASS = "PASS"
    FAIL = "FAIL"
    UNVERIFIED = "UNVERIFIED"


class CompletionState(str, Enum):
    """Overall completion state for a task based on acceptance evidence.

    States are evaluated by Workflow in fixed priority order:
    1. NEEDS_CLARIFICATION - Issue has unresolved ambiguities
    2. BLOCKED - Environment, permission, or scope issues prevent execution
    3. FAILED - Any AC is FAIL or unrecoverable code failure occurred
    4. VERIFIED - Deterministic verification passed
    5. PARTIALLY_VERIFIED - Deterministic verification did not fully pass
    """

    VERIFIED = "VERIFIED"
    PARTIALLY_VERIFIED = "PARTIALLY_VERIFIED"
    BLOCKED = "BLOCKED"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    FAILED = "FAILED"


class AcceptanceEvidence(BaseModel):
    """Evidence mapping for a single acceptance criterion.

    Represents the verification status and supporting evidence for one
    acceptance criterion, linking it to actual code changes, test results,
    and command outputs.

    Attributes:
        criterion_id: ID reference to the original AcceptanceCriterion.
        description: Human-readable description of the acceptance criterion.
        status: Verification status based on concrete evidence.
        changed_files: List of files that were modified to satisfy this criterion.
        tests: List of test names/paths that verify this criterion.
        command_results: List of verification command outputs supporting this criterion.
        explanation: Human-readable explanation of why this status was assigned.
    """

    criterion_id: str
    description: str
    status: EvidenceStatus
    changed_files: list[str] = Field(default_factory=list)
    tests: list[str] = Field(default_factory=list)
    command_results: list[str] = Field(default_factory=list)
    explanation: str


class AcceptanceCoverageReport(BaseModel):
    """Complete acceptance criteria coverage report.

    Aggregates evidence for all acceptance criteria and provides the overall
    completion state for the task.

    Attributes:
        acceptance_evidence: List of evidence for each acceptance criterion.
        completion_state: Overall task completion state.
        summary: Human-readable summary of the verification results.
    """

    acceptance_evidence: list[AcceptanceEvidence] = Field(default_factory=list)
    completion_state: CompletionState = CompletionState.NEEDS_CLARIFICATION
    summary: str = ""
