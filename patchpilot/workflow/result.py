"""Workflow result data structure for final execution outcomes.

This module provides the WorkflowResult class which aggregates the complete
results of a workflow execution, including completion state, changed files,
acceptance evidence, verification reports, and the final patch.

The WorkflowResult serves as the comprehensive output of the PatchPilot workflow,
combining execution metadata with verification results and acceptance criteria
coverage into a single structured result.
"""

import time
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
        duration_seconds: Total execution time in seconds.
        retry_count: Number of repair attempts made.
        max_rounds: Maximum number of agent rounds allowed.
        max_repairs: Maximum number of repair attempts allowed.
        prompt_tokens: Total prompt tokens used (null if not available).
        completion_tokens: Total completion tokens used (null if not available).
        total_cost: Total cost of the run (null if not available).
    """

    run_id: str
    final_status: CompletionState
    changed_files: list[str] = Field(default_factory=list)
    acceptance_evidence: list[AcceptanceEvidence] = Field(default_factory=list)
    verification_report: dict[str, Any] = Field(default_factory=dict)
    patch: str = ""
    duration_seconds: float = 0.0
    retry_count: int = 0
    max_rounds: int = 16
    max_repairs: int = 2
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_cost: float | None = None

    def to_run_summary(
        self,
        task_id: str = "",
        base_commit: str = "",
        model: str = "",
        output_dir: str = "artifacts",
    ) -> dict[str, Any]:
        """Convert workflow result to run summary format.

        Args:
            task_id: Optional task identifier for the run.
            base_commit: Base commit SHA for the repository.
            model: Model identifier used for the run.
            output_dir: Directory where artifacts are saved.

        Returns:
            Dictionary in run summary format suitable for JSON serialization.
        """
        # Determine exit code based on final status
        exit_code = 0 if self.final_status == CompletionState.VERIFIED else 1

        # Build artifacts dictionary
        artifacts = {
            "patch": f"{output_dir}/patch.diff",
            "verification_report": f"{output_dir}/verification_report.json",
            "acceptance_coverage": f"{output_dir}/acceptance_coverage.md",
            "execution_trace": f"{output_dir}/execution_trace.jsonl",
        }

        return {
            "run_id": self.run_id,
            "task_id": task_id,
            "phase": "execute",
            "base_commit": base_commit,
            "model": model,
            "max_rounds": self.max_rounds,
            "max_repairs": self.max_repairs,
            "retry_count": self.retry_count,
            "final_status": self.final_status.value,
            "exit_code": exit_code,
            "duration_seconds": self.duration_seconds,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_cost": self.total_cost,
            "artifacts": artifacts,
        }
