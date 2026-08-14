"""Workflow result data structure for final execution outcomes.

This module provides the WorkflowResult class which aggregates the complete
results of a workflow execution, including completion state, changed files,
acceptance evidence, verification reports, and the final patch.

The WorkflowResult serves as the comprehensive output of the PatchPilot workflow,
combining execution metadata with verification results and acceptance criteria
coverage into a single structured result.
"""

from typing import Any

from pydantic import BaseModel, Field

from patchpilot.evidence.schema import AcceptanceEvidence, CompletionState


class WorkflowResult(BaseModel):
    """Comprehensive result of a workflow execution.

    Aggregates all relevant information from a complete PatchPilot workflow run,
    including the completion state, file changes, acceptance evidence verification,
    deterministic verification results, and the final code patch.

    Attributes:
        run_id: Unique identifier for this workflow execution run.
        final_status: Overall completion state based on acceptance evidence
            and verification results (e.g., VERIFIED, FAILED, BLOCKED).
        changed_files: List of file paths that were modified during the workflow.
        acceptance_evidence: List of AcceptanceEvidence objects mapping each
            acceptance criterion to its verification status and supporting evidence.
        verification_report: Dictionary containing the complete verification
            report data from deterministic checks (ruff, pytest, etc.).
        patch: Git diff patch string containing all code changes made during
            the workflow execution.
    """

    run_id: str
    final_status: CompletionState
    changed_files: list[str] = Field(default_factory=list)
    acceptance_evidence: list[AcceptanceEvidence] = Field(default_factory=list)
    verification_report: dict[str, Any] = Field(default_factory=dict)
    patch: str = ""
