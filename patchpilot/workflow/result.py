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
from patchpilot.workflow.completion import CompletionDecision


class PrepareSummary(BaseModel):
    """Machine-readable outcome for one prepare phase."""

    phase: str = "prepare"
    outcome_code: str
    final_status: str | None = None
    exit_code: int
    reasons: list[str] = Field(default_factory=list)
    model: str = ""
    llm_call_count: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class RunSummary(BaseModel):
    """Summary of a workflow execution run.

    Aggregates key execution metadata and results for monitoring and evaluation.
    Includes run identification, configuration, timing, token usage, and artifact locations.

    Attributes:
        run_id: Unique identifier for this workflow execution run.
        task_id: Stable task identifier for evaluation tracking.
        phase: Execution phase (e.g., "execute").
        base_commit: Base commit SHA for the repository.
        model: Model identifier used for the run.
        max_rounds: Maximum number of agent rounds allowed.
        max_repairs: Maximum number of repair attempts allowed.
        retry_count: Number of repair attempts actually made.
        final_status: Overall completion state (e.g., VERIFIED, FAILED, BLOCKED).
        exit_code: CLI exit code based on final status.
        duration_seconds: Total execution time in seconds.
        llm_call_count: Number of successful model completions.
        prompt_tokens: Total prompt tokens used (null if not available).
        completion_tokens: Total completion tokens used (null if not available).
        total_cost: Total cost of the run (null if not available).
        outcome_code: Detailed outcome code from completion decision.
        criterion_pass_count: Number of required criteria that passed.
        criterion_unverified_count: Number of required criteria that are unverified.
        constraint_violation_count: Number of constraint violations detected.
        evidence_precision_hint: Human-readable hint about evidence precision.
        failure_type: Type of failure if final_status is FAILED.
        error_message: Error message if execution failed.
        artifacts: Dictionary mapping artifact names to their file paths.
    """

    run_id: str
    task_id: str
    phase: str
    base_commit: str
    model: str
    max_rounds: int
    max_repairs: int
    retry_count: int
    final_status: str
    exit_code: int
    duration_seconds: float
    llm_call_count: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_cost: float | None = None
    outcome_code: str = ""
    criterion_pass_count: int = 0
    criterion_unverified_count: int = 0
    constraint_violation_count: int = 0
    evidence_precision_hint: str = ""
    failure_type: str | None = None
    error_message: str | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)


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
        llm_call_count: Number of successful model completions.
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
    llm_call_count: int = 0
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
        decision: CompletionDecision | None = None,
    ) -> RunSummary:
        """Convert workflow result to run summary format.

        Args:
            task_id: Optional task identifier for the run.
            base_commit: Base commit SHA for the repository.
            model: Model identifier used for the run.
            output_dir: Directory where artifacts are saved.
            decision: Optional CompletionDecision object with detailed metrics.

        Returns:
            RunSummary object with execution metadata and results.
        """
        # Determine exit code based on final status
        # VERIFIED → 0, all other states → 1
        # PARTIALLY_VERIFIED is valuable but should not be treated as complete success for CI
        exit_code = 0 if self.final_status == CompletionState.VERIFIED else 1

        # Build artifacts dictionary
        artifacts = {
            "patch": f"{output_dir}/patch.diff",
            "verification_report": f"{output_dir}/verification_report.json",
            "acceptance_coverage": f"{output_dir}/acceptance_coverage.md",
            "acceptance_evidence": f"{output_dir}/acceptance_evidence.json",
            "execution_trace": f"{output_dir}/execution_trace.jsonl",
        }

        # Extract metrics from decision if available
        outcome_code = ""
        criterion_pass_count = 0
        criterion_unverified_count = 0
        constraint_violation_count = 0
        evidence_precision_hint = ""
        failure_type = None

        if decision:
            outcome_code = decision.state.value
            criterion_pass_count = decision.criterion_pass_count
            criterion_unverified_count = decision.criterion_unverified_count
            constraint_violation_count = decision.constraint_violation_count
            evidence_precision_hint = decision.evidence_precision_hint
            failure_type = decision.failure_type.value if decision.failure_type else None

        return RunSummary(
            run_id=self.run_id,
            task_id=task_id,
            phase="execute",
            base_commit=base_commit,
            model=model,
            max_rounds=self.max_rounds,
            max_repairs=self.max_repairs,
            retry_count=self.retry_count,
            final_status=self.final_status.value,
            exit_code=exit_code,
            duration_seconds=self.duration_seconds,
            llm_call_count=self.llm_call_count,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_cost=self.total_cost,
            outcome_code=outcome_code,
            criterion_pass_count=criterion_pass_count,
            criterion_unverified_count=criterion_unverified_count,
            constraint_violation_count=constraint_violation_count,
            evidence_precision_hint=evidence_precision_hint,
            failure_type=failure_type,
            artifacts=artifacts,
        )
