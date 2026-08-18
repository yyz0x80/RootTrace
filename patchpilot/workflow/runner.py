"""Workflow Runner for orchestrating the complete PatchPilot execution.

This module provides the WorkflowRunner class which orchestrates the entire
workflow from issue to verified patch, including:
- Repository workspace setup
- Docker sandbox initialization
- Agent execution for code modifications
- Verification with deterministic checks
- Repair loop with intelligent failure handling
- Scope gate validation for repair changes
- Early stopping for unrecoverable errors and repeated failures

The WorkflowRunner integrates all PatchPilot components into a unified
execution flow following the architecture:
CLI -> WorkflowRunner -> AgentLoop + Tools + Verifier -> Target Repository
"""

from __future__ import annotations

import logging
import platform
import shutil
import subprocess
import tarfile
import tempfile
import time
import uuid
from collections.abc import Callable
from fnmatch import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from patchpilot.agent_loop import (
    AgentLoop,
    AgentLoopError,
    AgentLoopLimitError,
    ExecuteLogCallback,
)
from patchpilot.evidence.schema import CompletionState
from patchpilot.issue.schema import NormalizedIssue
from patchpilot.planning.schema import ChangePlan, PlannedChange
from patchpilot.planning.scope_gate import (
    ScopeGateResult,
    _should_ignore_file,
    check_scope,
    validate_actual_changes,
)
from patchpilot.prompts import REPAIR_PROMPT, REPAIR_SYSTEM_PROMPT
from patchpilot.sandbox.docker_runner import DockerSandbox
from patchpilot.tools import (
    WorkspaceChange,
    _get_workspace_changes,
    generate_patch,
)
from patchpilot.verification.report import VerificationReport, failure_fingerprint
from patchpilot.verification.targets import select_target_tests
from patchpilot.workflow.completion import determine_completion_state
from patchpilot.workflow.execute_logger import ExecuteLogger
from patchpilot.workflow.failure_classifier import FailureType
from patchpilot.workflow.result import WorkflowResult
from patchpilot.workflow.trace import TraceEvent, TraceWriter
from patchpilot.workspace import Workspace

if TYPE_CHECKING:
    from patchpilot.evidence.schema import AcceptanceEvidence
    from patchpilot.models import ToolResult
    from patchpilot.verification.verifier import Verifier

logger = logging.getLogger(__name__)

MAX_REPAIR_GOAL_CHARS = 1_200
MAX_REPAIR_INTENT_CHARS = 2_000
MAX_REPAIR_CONSTRAINT_CHARS = 1_600
MAX_REPAIR_PATCH_CHARS = 6_000
MAX_REPAIR_FAILURE_OUTPUT_CHARS = 2_000


def _truncate_repair_text(text: str, limit: int) -> str:
    """Bound repair context while retaining evidence from both ends."""
    normalized = text.strip()
    if not normalized:
        return "(none)"
    if len(normalized) <= limit:
        return normalized

    marker = "\n... repair context truncated ...\n"
    remaining = limit - len(marker)
    head_size = (remaining * 2) // 3
    tail_size = remaining - head_size
    return normalized[:head_size] + marker + normalized[-tail_size:]


def _map_acceptance_evidence(**kwargs: Any) -> list[AcceptanceEvidence]:
    """Map acceptance evidence without introducing an import cycle."""
    from patchpilot.evidence.mapper import map_acceptance_evidence

    return map_acceptance_evidence(**kwargs)

# Standard ignore patterns for temporary and compiled files
# These patterns are applied regardless of the target repository's .gitignore
DEFAULT_IGNORE_PATTERNS = [
    # Python cache files
    "__pycache__/",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "*.egg-info/",
    ".pytest_cache/",
    ".ruff_cache/",
    ".mypy_cache/",
    # macOS files
    ".DS_Store",
    ".AppleDouble",
    ".LSOverride",
    # Editor temporary files
    "*.swp",
    "*.swo",
    "*.swn",
    "*.bak",
    "*~",
    # Build artifacts
    "dist/",
    "build/",
    "*.egg",
]


def _should_ignore_file_in_workspace(file_path: str) -> bool:
    """Check if a file should be ignored based on standard patterns.

    This function implements a multi-layer ignore system that does not depend
    on the target repository's .gitignore configuration, ensuring consistent
    behavior across different projects.

    Args:
        file_path: The file path to check (relative to workspace root)

    Returns:
        True if the file should be ignored, False otherwise
    """
    from pathlib import PurePosixPath

    normalized_path = str(PurePosixPath(file_path))

    for pattern in DEFAULT_IGNORE_PATTERNS:
        # Handle directory patterns (ending with /)
        if pattern.endswith("/"):
            dir_name = pattern[:-1]
            # Check if directory name appears in path
            if dir_name in normalized_path.split("/"):
                return True
        # Handle file patterns with wildcards
        elif "*" in pattern:
            # Check if filename matches the pattern
            filename = normalized_path.split("/")[-1]
            if fnmatch(filename, pattern):
                return True
        # Handle exact matches
        elif normalized_path == pattern or normalized_path.startswith(pattern + "/"):
            return True

    return False


def _clean_temporary_files(workspace_path: Path) -> None:
    """Remove common temporary and compiled files from workspace.

    This function actively cleans the workspace after extracting the git archive,
    ensuring that cache files, build artifacts, and other temporary files are removed
    regardless of the target repository's .gitignore configuration.

    Args:
        workspace_path: Path to the workspace directory to clean
    """
    # Remove Python cache directories
    for pycache_dir in workspace_path.rglob("__pycache__"):
        if pycache_dir.is_dir():
            shutil.rmtree(pycache_dir)
            logger.info("Removed __pycache__ directory: %s", pycache_dir)

    # Remove Python compiled files
    for pyc_file in workspace_path.rglob("*.pyc"):
        if pyc_file.is_file():
            pyc_file.unlink()
            logger.debug("Removed .pyc file: %s", pyc_file)

    for pyo_file in workspace_path.rglob("*.pyo"):
        if pyo_file.is_file():
            pyo_file.unlink()
            logger.debug("Removed .pyo file: %s", pyo_file)

    for pyd_file in workspace_path.rglob("*.pyd"):
        if pyd_file.is_file():
            pyd_file.unlink()
            logger.debug("Removed .pyd file: %s", pyd_file)

    # Remove macOS metadata files
    for ds_store in workspace_path.rglob(".DS_Store"):
        if ds_store.is_file():
            ds_store.unlink()
            logger.debug("Removed .DS_Store file: %s", ds_store)

    # Remove editor temporary files
    for swp_file in workspace_path.rglob("*.swp"):
        if swp_file.is_file():
            swp_file.unlink()
            logger.debug("Removed .swp file: %s", swp_file)

    for swo_file in workspace_path.rglob("*.swo"):
        if swo_file.is_file():
            swo_file.unlink()
            logger.debug("Removed .swo file: %s", swo_file)

    for bak_file in workspace_path.rglob("*.bak"):
        if bak_file.is_file():
            bak_file.unlink()
            logger.debug("Removed .bak file: %s", bak_file)


# Maximum number of repair attempts as specified in requirements
MAX_REPAIR_ATTEMPTS = 2

# Failure types that are considered unrecoverable
UNRECOVERABLE_FAILURE_TYPES = {
    FailureType.ENVIRONMENT_FAILURE,
    FailureType.PERMISSION_FAILURE,
    FailureType.TIMEOUT,
    FailureType.REQUIREMENT_AMBIGUITY,
    FailureType.SCOPE_VIOLATION,
}

BASELINE_BLOCKING_FAILURE_TYPES = {
    FailureType.ENVIRONMENT_FAILURE.value,
    FailureType.PERMISSION_FAILURE.value,
    FailureType.TIMEOUT.value,
}


class WorkflowExecuteLogCallback(ExecuteLogCallback):
    """Record structured logs and tool trace events for agent execution.

    Bridges the AgentLoop's callback interface with the structured
    ExecuteLogger and the persistent workflow trace.
    """

    _MUTATING_TOOLS: ClassVar[set[str]] = {
        "edit_file",
        "edit_file_by_line",
        "apply_patch",
    }

    def __init__(self, trace_writer: TraceWriter, run_id: str) -> None:
        """Initialize logging and trace state for one workflow run."""
        self._coding_section_logged = False
        self._trace_writer = trace_writer
        self._run_id = run_id
        self._workflow_stage = "CODING"
        self._retry_count = 0

    def set_trace_context(
        self,
        *,
        workflow_stage: str,
        retry_count: int,
    ) -> None:
        """Update the stage metadata used by subsequent tool events."""
        self._workflow_stage = workflow_stage
        self._retry_count = retry_count

    def on_round_start(self, round_number: int) -> None:
        """Called at the start of each agent round.

        Args:
            round_number: Current round number
        """
        # Log the CODING section header only once on first round
        if not self._coding_section_logged:
            ExecuteLogger.log_section("CODING")
            self._coding_section_logged = True

    def on_tool_call(self, round_number: int, tool_name: str, args: dict[str, Any]) -> None:
        """Called when the agent makes a tool call.

        Args:
            round_number: Current round number
            tool_name: Name of the tool being called
            args: Tool arguments
        """
        ExecuteLogger.log_coding_round(round_number, tool_name, args)

    def on_tool_result(
        self,
        round_number: int,
        tool_name: str,
        args: dict[str, Any],
        result: ToolResult,
        duration_seconds: float,
    ) -> None:
        """Record one completed tool execution in the workflow trace."""
        modified_files = []
        path = args.get("path")
        if result.ok and tool_name in self._MUTATING_TOOLS and isinstance(path, str):
            modified_files.append(path)

        self._trace_writer.write(
            TraceEvent(
                run_id=self._run_id,
                event_type="tool_call",
                workflow_stage=self._workflow_stage,
                tool_name=tool_name,
                tool_arguments=args,
                tool_duration=duration_seconds,
                modified_files=modified_files,
                round_number=round_number,
                retry_count=self._retry_count,
                final_status="SUCCESS" if result.ok else "FAILURE",
            )
        )

    def on_round_complete(self, round_number: int) -> None:
        """Called when the agent completes a round with a final answer.

        Args:
            round_number: Current round number
        """
        ExecuteLogger.log_coding_complete(round_number)


def _run(command: list[str], cwd: Path) -> None:
    """Run a command with subprocess, checking for success.

    Args:
        command: List of command arguments.
        cwd: Working directory for command execution.

    Raises:
        subprocess.CalledProcessError: If command fails.
    """
    subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


class WorkflowRunnerError(RuntimeError):
    """Base exception for Workflow Runner failures."""


class WorkflowRunnerSetupError(WorkflowRunnerError):
    """Raised when workflow setup fails (e.g., repository copy, sandbox start)."""


class WorkflowRunnerExecutionError(WorkflowRunnerError):
    """Raised when workflow execution fails (e.g., agent errors, verification failures)."""

    def __init__(
        self,
        message: str,
        *,
        verification_report: dict[str, Any] | None = None,
        failure_type: str = "WORKFLOW_ERROR",
    ) -> None:
        """Initialize an execution error with optional partial artifacts."""
        super().__init__(message)
        self.verification_report = verification_report
        self.failure_type = failure_type


class WorkflowRunner:
    """Orchestrate the complete PatchPilot workflow from issue to verified patch.

    The WorkflowRunner manages the entire execution lifecycle:
    1. Creates a temporary workspace copy of the target repository
    2. Initializes the Docker sandbox for isolated execution
    3. Runs the agent to implement the initial code changes
    4. Executes verification with deterministic checks
    5. Attempts repairs when verification fails, with intelligent stopping:
       - Stops immediately for unrecoverable failures
       - Stops when the same failure repeats
       - Limits to MAX_REPAIR_ATTEMPTS repair attempts
       - Validates repair changes against scope gate using git diff --name-only
    6. Returns the final verification report

    Attributes:
        agent_loop: The AgentLoop instance for running the coding agent
        verifier: Function to run verification and return a VerificationReport
        workspace: The Workspace instance for path resolution and security
        sandbox: The DockerSandbox instance for isolated execution
        _temp_dir: TemporaryDirectory context manager for repository copy
    """

    def __init__(
        self,
        agent_loop: AgentLoop,
        verifier: Callable[[], VerificationReport] | None,
        workspace: Workspace,
        sandbox: DockerSandbox | None = None,
        max_repair_attempts: int = MAX_REPAIR_ATTEMPTS,
    ) -> None:
        """Initialize the WorkflowRunner with required components.

        Args:
            agent_loop: AgentLoop instance for running the coding agent
            verifier: Optional function that runs verification and returns a VerificationReport.
                      If None, WorkflowRunner uses the built-in Verifier.
            workspace: Workspace instance for path resolution and security
            sandbox: Optional DockerSandbox instance (created if None)
            max_repair_attempts: Maximum number of repair attempts (default: MAX_REPAIR_ATTEMPTS)
        """
        if max_repair_attempts < 0:
            raise ValueError("max_repair_attempts must be non-negative")

        self.agent_loop = agent_loop
        self.verifier = verifier
        self.workspace = workspace
        self.sandbox = sandbox
        self.max_repair_attempts = max_repair_attempts
        self._temp_dir: tempfile.TemporaryDirectory | None = None
        self._sandbox_verifier: Verifier | None = None

    @property
    def temp_dir(self) -> Path | None:
        """Get the temporary directory path as a Path object."""
        if self._temp_dir is None:
            return None
        return Path(self._temp_dir.name)

    def _run_verification(
        self,
        *,
        run_id: str,
        change_plan: ChangePlan | None,
        retry_count: int,
    ) -> VerificationReport:
        """Run an injected verifier or the built-in sandbox verifier.

        Args:
            run_id: Unique identifier for this workflow run
            change_plan: Optional ChangePlan for target test selection
            retry_count: Current retry attempt number

        Returns:
            VerificationReport containing verification results

        Raises:
            WorkflowRunnerSetupError: If sandbox is not started when using built-in verifier
        """
        if self.verifier is not None:
            report = self.verifier()
            report.retry_count = retry_count
            return report

        if self.sandbox is None:
            raise WorkflowRunnerSetupError(
                "Cannot run verification before the sandbox is started"
            )

        selection = select_target_tests(change_plan)

        verifier = self._get_sandbox_verifier()
        return verifier.verify(
            run_id=run_id,
            target_tests=selection.tests,
            target_acceptance_criteria=(
                selection.acceptance_criteria
            ),
            target_direct_acceptance_criteria=(
                selection.direct_acceptance_criteria
            ),
            retry_count=retry_count,
        )

    def _get_sandbox_verifier(self) -> Verifier:
        """Return the Verifier shared by all sandbox verification phases."""
        if self.sandbox is None:
            raise WorkflowRunnerSetupError(
                "Cannot create a verifier before the sandbox is started"
            )

        if self._sandbox_verifier is None:
            # Import here to avoid a circular dependency during module loading.
            from patchpilot.verification.verifier import Verifier

            self._sandbox_verifier = Verifier(self.sandbox)

        return self._sandbox_verifier

    def _run_traced_verification(
        self,
        *,
        trace_writer: TraceWriter,
        workflow_stage: str,
        run_id: str,
        change_plan: ChangePlan | None,
        retry_count: int,
    ) -> VerificationReport:
        """Run verification and append its complete report to the trace."""
        report = self._run_verification(
            run_id=run_id,
            change_plan=change_plan,
            retry_count=retry_count,
        )
        trace_writer.write(
            TraceEvent(
                run_id=run_id,
                event_type="verification",
                workflow_stage=workflow_stage,
                verification_result=report.to_dict(),
                retry_count=report.retry_count,
                final_status="SUCCESS" if report.passed else "FAILURE",
            )
        )
        return report

    def execute(
        self,
        issue: str,
        plan: str,
        change_plan: ChangePlan | None = None,
        normalized_issue: NormalizedIssue | None = None,
        trace_path: Path | None = None,
    ) -> WorkflowResult:
        """Execute the complete workflow from issue to verified patch.

        This method implements the core workflow logic following the proper execution order:
        1. Preflight validation (handled in CLI before this method)
        2. Approved-plan validation (handled in CLI before this method)
        3. Create temporary workspace copy
        4. Update workspace to use temporary path
        5. Update agent loop tools to use temporary workspace
        6. Start Docker Sandbox
        7. Run baseline verification before model execution
        8. Run Coding Agent for initial modification
        9. Check actual changes via _get_workspace_changes
        10. Validate agent made changes if required
        11. Runtime scope validation against approved plan
        12. Run final verification
        13. Repair loop if verification failed:
            - Run Repair Agent
            - Get workspace changes via _get_workspace_changes
            - Scope gate validation
            - Run verifier
        14. Generate patch with all changes
        15. Generate acceptance evidence before workspace cleanup
        16. Determine final completion state
        17. Write final trace event before workspace cleanup
        18. Create and return WorkflowResult

        Args:
            issue: The original issue description
            plan: The approved change plan for the agent to follow
            change_plan: Optional ChangePlan object for runtime scope validation
            normalized_issue: Optional normalized issue used for acceptance
                evidence mapping. JSON issue input is parsed as a compatibility
                fallback when this value is not provided.
            trace_path: Optional persistent path for the execution trace. When
                omitted, the trace is written beside the temporary workspace.

        Returns:
            WorkflowResult containing the final execution results including completion state,
            changed files, acceptance evidence, verification report, and patch

        Raises:
            WorkflowRunnerSetupError: If workspace or sandbox setup fails
            WorkflowRunnerExecutionError: If agent execution fails critically
        """
        # Generate a stable run_id for this workflow execution
        run_id = str(uuid.uuid4())
        execution_start_time = time.time()

        if normalized_issue is None:
            try:
                normalized_issue = NormalizedIssue.model_validate_json(issue)
            except ValueError:
                # Preserve compatibility with legacy callers that pass plain text.
                normalized_issue = NormalizedIssue(
                    title="Task from issue",
                    task_type="other",
                    problem_statement=issue,
                    acceptance_criteria=[],
                    constraints=[],
                    ambiguous_points=[],
                    expected_test_areas=[],
                    implementation_notes=[],
                )

        # Log issue and plan
        ExecuteLogger.log_issue(issue)
        if change_plan:
            ExecuteLogger.log_plan(
                base_commit=change_plan.base_commit,
                planned_changes_count=len(change_plan.planned_changes),
            )

        # Step 3: Create temporary workspace copy
        workspace_path = self._create_temporary_workspace()

        # Step 4: Update workspace to use temporary path
        self.workspace = Workspace(workspace_path)

        # Initialize one persistent trace writer for the complete workflow.
        output_trace_path = trace_path or workspace_path.parent / "trace.jsonl"
        trace_writer = TraceWriter(output_trace_path)
        trace_writer.start_run()

        # Step 5: Update agent loop tools to use temporary workspace
        self.agent_loop.update_workspace(self.workspace)

        # Set up execute log callback for structured logging
        execute_callback = WorkflowExecuteLogCallback(trace_writer, run_id)
        self.agent_loop.execute_log_callback = execute_callback

        # Step 6: Start Docker Sandbox
        self._start_sandbox(workspace_path)

        report: VerificationReport | None = None
        try:
            # Step 7: Verify the sandbox baseline before model execution.
            # The injected callback is retained as a legacy test/custom seam.
            # Production execution uses the built-in sandbox verifier.
            if self.verifier is None:
                logger.info("Running sandbox baseline verification")
                baseline_report = self._run_traced_verification(
                    trace_writer=trace_writer,
                    workflow_stage="BASELINE_VERIFY",
                    run_id=run_id,
                    change_plan=change_plan,
                    retry_count=0,
                )
                baseline_results = {
                    check.command or check.level: check.passed
                    for check in baseline_report.checks
                }
                ExecuteLogger.log_verification(
                    baseline_results,
                    section_title="BASELINE VERIFY",
                )

                if baseline_report.failure_type in BASELINE_BLOCKING_FAILURE_TYPES:
                    logger.warning(
                        "Baseline verification blocked execution: %s. "
                        "The coding agent was not called.",
                        baseline_report.failure_type,
                    )
                    from patchpilot.evidence.schema import (
                        AcceptanceEvidence,
                        EvidenceStatus,
                    )

                    reason = (
                        "Baseline verification was blocked before model execution: "
                        f"{baseline_report.failure_type}."
                    )
                    evidence = [
                        AcceptanceEvidence(
                            criterion_id=criterion.id,
                            description=criterion.description,
                            status=EvidenceStatus.UNVERIFIED,
                            explanation=reason,
                        )
                        for criterion in normalized_issue.acceptance_criteria
                    ]
                    final_status = determine_completion_state(
                        has_ambiguity=False,
                        blocked=True,
                        execution_failed=False,
                        verifier_passed=False,
                        evidence=evidence,
                    )
                    trace_writer.write(
                        TraceEvent(
                            run_id=run_id,
                            event_type="workflow_completed",
                            workflow_stage="BASELINE_VERIFY",
                            verification_result=baseline_report.to_dict(),
                            final_status=final_status.value,
                        )
                    )
                    artifacts = {
                        "verification_report.json": (
                            "artifacts/verification_report.json"
                        )
                    }
                    if trace_path is not None:
                        artifacts["execution_trace.jsonl"] = str(trace_path)
                    ExecuteLogger.log_result(
                        passed=False,
                        artifacts=artifacts,
                        final_status=final_status.value,
                    )
                    return WorkflowResult(
                        run_id=run_id,
                        final_status=final_status,
                        changed_files=[],
                        acceptance_evidence=evidence,
                        verification_report=baseline_report.to_dict(),
                        patch="",
                    )

                if not baseline_report.passed:
                    logger.info(
                        "Baseline verification found a repairable failure: %s. "
                        "Continuing with the coding agent.",
                        baseline_report.failure_type,
                    )

            # Step 8: Coding Agent initial modification
            logger.info("Running coding agent for initial implementation")
            initial_prompt = f"Implement the following plan:\n\n{plan}"
            try:
                self.agent_loop.run(
                    issue=initial_prompt,
                    reset_state=True,
                )
            except AgentLoopError as error:
                partial_changes = _get_workspace_changes(workspace_path)
                if not partial_changes:
                    raise
                logger.warning(
                    "Initial coding agent stopped after producing a partial "
                    "patch; continuing with scope validation and deterministic "
                    "verification: %s",
                    error,
                )

            # Step 9: Check actual changes via _get_workspace_changes
            actual_changes = _get_workspace_changes(workspace_path)

            # Step 10: Validate agent made changes if required
            if not actual_changes:
                # Only enforce this check for tasks that require code changes
                if change_plan is not None and change_plan.planned_changes:
                    raise WorkflowRunnerExecutionError(
                        "Coding agent finished without modifying any repository files. "
                        "Tasks requiring code changes (feature, bugfix, refactor) must modify at least one file."
                    )
                logger.info("Agent completed without file changes (allowed for non-code-change tasks)")

            # Log changes section
            modified = [c.path for c in actual_changes if c.action == "modify"]
            created = [c.path for c in actual_changes if c.action == "create"]
            deleted = [c.path for c in actual_changes if c.action == "delete"]
            ExecuteLogger.log_changes(modified, created, deleted)

            # Step 11: Runtime scope validation against approved plan
            if change_plan is not None:
                logger.info("Running runtime scope validation")
                try:
                    validate_actual_changes(change_plan, actual_changes)
                    logger.info("Runtime scope validation passed")
                    ExecuteLogger.log_scope_validation(allowed=True)
                except RuntimeError as e:
                    logger.error("Runtime scope validation failed: %s", e)
                    ExecuteLogger.log_scope_validation(allowed=False, violations=[str(e)])
                    raise WorkflowRunnerExecutionError(
                        f"Runtime scope validation failed: {e}"
                    ) from e

            # Step 12: Final verification after the initial implementation
            logger.info("Running post-change verification")
            retry_count = 0

            report = self._run_traced_verification(
                trace_writer=trace_writer,
                workflow_stage="VERIFY",
                run_id=run_id,
                change_plan=change_plan,
                retry_count=retry_count,
            )

            # Log verification results
            verification_results = {}
            for check in report.checks:
                # Use command as the check name for logging
                check_name = check.command if check.command else check.level
                verification_results[check_name] = check.passed
            ExecuteLogger.log_verification(verification_results)

            # Step 13: Repair loop if verification failed
            previous_failure = None

            while not report.passed:
                # Check for repeated failure (fingerprint check)
                current_failure = failure_fingerprint(report)

                if current_failure == previous_failure:
                    logger.warning(
                        "Same failure repeated. Stopping repair loop to avoid futile attempts."
                    )
                    ExecuteLogger.log_repair_stopped("Same failure repeated")
                    break

                previous_failure = current_failure

                # Check for unrecoverable errors
                if report.failure_type in UNRECOVERABLE_FAILURE_TYPES:
                    logger.warning(
                        "Unrecoverable failure detected: %s. Stopping repair loop.",
                        report.failure_type,
                    )
                    ExecuteLogger.log_repair_stopped(f"Unrecoverable failure: {report.failure_type}")
                    break

                # Check repair attempt limit
                if retry_count >= self.max_repair_attempts:
                    logger.warning(
                        "Maximum repair attempts (%d) reached. Stopping repair loop.",
                        self.max_repair_attempts,
                    )
                    ExecuteLogger.log_repair_stopped("Maximum repair attempts reached")
                    break

                retry_count += 1
                report.retry_count = retry_count
                ExecuteLogger.log_repair_attempt(retry_count, self.max_repair_attempts)
                logger.info(
                    "Starting repair attempt %d/%d",
                    retry_count,
                    self.max_repair_attempts,
                )

                # Build a bounded failure-diff context instead of restarting
                # the generic repository-discovery workflow.
                current_changes = _get_workspace_changes(workspace_path)
                current_patch = generate_patch(
                    workspace_path,
                    current_changes,
                )
                repair_prompt = self._build_repair_prompt(
                    issue=issue,
                    plan=plan,
                    failure_report=report,
                    current_patch=current_patch,
                    current_changes=current_changes,
                    change_plan=change_plan,
                    normalized_issue=normalized_issue,
                )

                # Run agent with repair prompt
                execute_callback.set_trace_context(
                    workflow_stage="REPAIR",
                    retry_count=retry_count,
                )
                repair_agent_error: AgentLoopError | None = None
                try:
                    self.agent_loop.run(
                        issue=repair_prompt,
                        system_prompt=REPAIR_SYSTEM_PROMPT,
                        reset_state=True,
                    )
                except AgentLoopError as error:
                    repair_agent_error = error

                # Get workspace changes after repair
                repair_changes = _get_workspace_changes(workspace_path)
                repaired_patch = generate_patch(
                    workspace_path,
                    repair_changes,
                )
                if repaired_patch == current_patch:
                    logger.warning(
                        "Repair agent did not change the patch. Stopping repair loop."
                    )
                    ExecuteLogger.log_repair_stopped(
                        "No patch delta during repair"
                    )
                    if repair_agent_error is not None:
                        logger.warning(
                            "Repair agent also stopped with an error: %s",
                            repair_agent_error,
                        )
                        raise repair_agent_error
                    break

                if repair_agent_error is not None:
                    logger.warning(
                        "Repair agent stopped after updating the patch; "
                        "continuing with scope validation and deterministic "
                        "verification: %s",
                        repair_agent_error,
                    )

                # Scope gate validation after repair
                scope_result = self._check_repair_scope()
                if not scope_result.allowed:
                    logger.warning(
                        "Scope gate rejected repair changes: %s. Stopping repair loop.",
                        scope_result.violations,
                    )
                    ExecuteLogger.log_repair_stopped(f"Scope gate rejected: {scope_result.violations}")
                    # Create a failure report indicating scope rejection
                    report.passed = False
                    report.failure_type = "SCOPE_VIOLATION"
                    # Add scope violations to the report for visibility
                    if report.checks:
                        report.checks[-1].summary = report.checks[-1].summary or {}
                        report.checks[-1].summary["scope_violations"] = scope_result.violations
                        report.checks[-1].summary["scope_warnings"] = scope_result.warnings
                    break

                # Re-run verification
                report = self._run_traced_verification(
                    trace_writer=trace_writer,
                    workflow_stage="REPAIR_VERIFY",
                    run_id=run_id,
                    change_plan=change_plan,
                    retry_count=retry_count,
                )

            # Step 14: Generate patch with all changes
            logger.info("Generating patch with all changes")
            final_changes = _get_workspace_changes(workspace_path)
            report.patch = generate_patch(workspace_path, final_changes)
            logger.info("Patch generated successfully")

            # Step 15: Generate acceptance evidence before workspace cleanup
            # This must happen before the temporary workspace is destroyed
            if change_plan is not None:
                evidence = _map_acceptance_evidence(
                    issue=normalized_issue,
                    plan=change_plan,
                    actual_changes=final_changes,
                    report=report,
                )
            else:
                evidence = []

            # Step 16: Determine final completion state
            final_status = determine_completion_state(
                has_ambiguity=bool(normalized_issue.ambiguous_points),
                blocked=report.failure_type in {
                    "ENVIRONMENT_FAILURE",
                    "PERMISSION_FAILURE",
                    "SCOPE_VIOLATION",
                    "TIMEOUT",
                },
                execution_failed=not report.passed,
                verifier_passed=report.passed,
                evidence=evidence,
            )

            # Step 17: Write final trace event before workspace cleanup
            # Use the shared writer so this remains the final event in the trace.
            trace_writer.write(
                TraceEvent(
                    run_id=run_id,
                    event_type="workflow_completed",
                    workflow_stage="RESULT",
                    modified_files=[change.path for change in final_changes],
                    verification_result=report.to_dict(),
                    retry_count=report.retry_count,
                    final_status=final_status.value,
                )
            )

            # Step 18: Create and return WorkflowResult
            duration_seconds = time.time() - execution_start_time
            result = WorkflowResult(
                run_id=run_id,
                final_status=final_status,
                changed_files=[change.path for change in final_changes],
                acceptance_evidence=evidence,
                verification_report=report.to_dict(),
                patch=report.patch or "",
                duration_seconds=duration_seconds,
                retry_count=report.retry_count,
                max_repairs=self.max_repair_attempts,
            )

            # Log final result section
            artifacts = {}
            if report.patch:
                artifacts["patch.diff"] = "artifacts/patch.diff"
            artifacts["verification_report.json"] = "artifacts/verification_report.json"
            if trace_path is not None:
                artifacts["execution_trace.jsonl"] = str(trace_path)

            ExecuteLogger.log_result(
                passed=report.passed,
                artifacts=artifacts,
                final_status=final_status.value,
            )

            # Return the final workflow result.
            return result

        except AgentLoopError as error:
            if report is None:
                raise
            raise WorkflowRunnerExecutionError(
                str(error),
                verification_report=report.to_dict(),
                failure_type=(
                    "AGENT_ROUND_LIMIT"
                    if isinstance(error, AgentLoopLimitError)
                    else "AGENT_ERROR"
                ),
            ) from error
        finally:
            # Cleanup: Stop sandbox and remove temporary directory
            self._cleanup()

    def execute_baseline(
        self,
        issue: str,
        trace_path: Path | None = None,
    ) -> WorkflowResult:
        """Run the raw issue through one agent pass and one verification pass.

        This path intentionally omits normalization, planning, acceptance
        evidence mapping, scope-plan validation, and repair attempts. It keeps
        the same workspace protections, tool allowlist, Docker sandbox, and
        deterministic verifier as the full PatchPilot workflow.
        """
        run_id = str(uuid.uuid4())
        execution_start_time = time.time()
        workspace_path = self._create_temporary_workspace()
        self.workspace = Workspace(workspace_path)

        output_trace_path = trace_path or workspace_path.parent / "trace.jsonl"
        trace_writer = TraceWriter(output_trace_path)
        trace_writer.start_run()

        self.agent_loop.update_workspace(self.workspace)
        execute_callback = WorkflowExecuteLogCallback(trace_writer, run_id)
        self.agent_loop.execute_log_callback = execute_callback
        self._start_sandbox(workspace_path)

        agent_error: AgentLoopError | None = None
        try:
            try:
                self.agent_loop.run(
                    issue=issue,
                    reset_state=True,
                )
            except AgentLoopError as error:
                agent_error = error
                logger.warning("Baseline agent execution failed: %s", error)

            final_changes = _get_workspace_changes(workspace_path)
            report = self._run_traced_verification(
                trace_writer=trace_writer,
                workflow_stage="VERIFY",
                run_id=run_id,
                change_plan=None,
                retry_count=0,
            )
            report.patch = generate_patch(workspace_path, final_changes)

            if report.failure_type in BASELINE_BLOCKING_FAILURE_TYPES:
                final_status = CompletionState.BLOCKED
            elif agent_error is not None or not report.passed:
                final_status = CompletionState.FAILED
            else:
                final_status = CompletionState.VERIFIED

            trace_writer.write(
                TraceEvent(
                    run_id=run_id,
                    event_type="workflow_completed",
                    workflow_stage="RESULT",
                    modified_files=[change.path for change in final_changes],
                    verification_result=report.to_dict(),
                    retry_count=0,
                    final_status=final_status.value,
                )
            )

            return WorkflowResult(
                run_id=run_id,
                final_status=final_status,
                changed_files=[change.path for change in final_changes],
                acceptance_evidence=[],
                verification_report=report.to_dict(),
                patch=report.patch or "",
                duration_seconds=time.time() - execution_start_time,
                retry_count=0,
                max_rounds=self.agent_loop.max_rounds,
                max_repairs=0,
            )
        finally:
            self._cleanup()

    def _create_temporary_workspace(self) -> Path:
        """Create a temporary copy of the repository workspace using git archive.

        Creates a temporary directory and exports the repository HEAD commit
        using git archive to provide an isolated working environment for the agent.
        Removes sensitive environment files and initializes a new git repository
        for baseline tracking.

        On macOS, uses a Docker-friendly path in the user's home directory
        to avoid Docker Desktop file sharing restrictions.

        Returns:
            Path to the temporary workspace directory

        Raises:
            WorkflowRunnerSetupError: If temporary workspace creation fails
        """
        try:
            # Create temporary directory
            # On macOS, use a path in home directory for better Docker compatibility
            # Docker Desktop on macOS has restrictions on /tmp and /var/folders paths
            if platform.system() == "Darwin":
                # macOS: use home directory which is usually shared with Docker
                home_temp = Path.home() / ".patchpilot_temp"
                home_temp.mkdir(exist_ok=True)
                self._temp_dir = tempfile.TemporaryDirectory(
                    prefix="patchpilot-",
                    dir=home_temp
                )
            else:
                # Linux: use standard system temp directory
                self._temp_dir = tempfile.TemporaryDirectory(prefix="patchpilot-")
            
            temp_root = Path(self._temp_dir.name)
            logger.info("Created temporary workspace: %s", temp_root)

            # Create workspace directory
            workspace_path = temp_root / "repo"
            workspace_path.mkdir(parents=True, exist_ok=True)
            archive_path = temp_root / "source.tar"

            # Export HEAD commit using git archive
            logger.info("Exporting repository HEAD using git archive")
            subprocess.run(
                [
                    "git",
                    "archive",
                    "--format=tar",
                    "-o",
                    str(archive_path),
                    "HEAD",
                ],
                cwd=self.workspace.root,
                check=True,
                capture_output=True,
                text=True,
            )

            # Extract archive to workspace
            logger.info("Extracting archive to workspace")
            with tarfile.open(archive_path) as archive:
                archive.extractall(workspace_path, filter="data")

            # Clean temporary and compiled files
            logger.info("Cleaning temporary and compiled files from workspace")
            _clean_temporary_files(workspace_path)

            # Remove sensitive environment files
            sensitive_files = [
                ".env",
                ".env.local",
                ".env.production",
            ]

            for name in sensitive_files:
                path = workspace_path / name
                if path.exists():
                    path.unlink()
                    logger.info("Removed sensitive file: %s", name)

            # Initialize new git repository for baseline tracking
            logger.info("Initializing git repository in workspace")
            _run(["git", "init", "-q"], workspace_path)
            _run(
                ["git", "config", "user.email", "patchpilot@local"],
                workspace_path,
            )
            _run(
                ["git", "config", "user.name", "PatchPilot"],
                workspace_path,
            )
            _run(["git", "add", "-A"], workspace_path)
            _run(
                ["git", "commit", "-q", "-m", "PatchPilot baseline"],
                workspace_path,
            )

            logger.info("Temporary workspace setup complete")

            # Log workspace section with structured output
            ExecuteLogger.log_workspace_setup(str(workspace_path))

            return workspace_path

        except (OSError, subprocess.CalledProcessError, tarfile.TarError) as e:
            raise WorkflowRunnerSetupError(
                f"Failed to create temporary workspace: {e}"
            ) from e

    def _start_sandbox(self, workspace_path: Path) -> None:
        """Start the Docker sandbox for isolated execution.

        Initializes the DockerSandbox instance if not provided,
        then starts the container for secure command execution.

        Args:
            workspace_path: Path to the temporary workspace directory

        Raises:
            WorkflowRunnerSetupError: If sandbox startup fails
        """
        try:
            # Validate workspace path exists before starting sandbox
            if not workspace_path.exists():
                raise RuntimeError(
                    f"Workspace path does not exist: {workspace_path}. "
                    f"Cannot start Docker sandbox with non-existent directory."
                )
            
            if not workspace_path.is_dir():
                raise RuntimeError(
                    f"Workspace path is not a directory: {workspace_path}"
                )
            
            if self.sandbox is None:
                self.sandbox = DockerSandbox(workspace=workspace_path)

            self.sandbox.start()
            self.agent_loop.tools.update_command_runner(self.sandbox)
            logger.info("Docker sandbox started successfully")

            # Log sandbox section with structured output
            ExecuteLogger.log_sandbox_start()

        except Exception as e:
            raise WorkflowRunnerSetupError(
                f"Failed to start Docker sandbox: {e}"
            ) from e

    def _build_repair_prompt(
        self,
        issue: str,
        plan: str,
        failure_report: VerificationReport,
        current_patch: str = "",
        current_changes: list[WorkspaceChange] | None = None,
        change_plan: ChangePlan | None = None,
        normalized_issue: NormalizedIssue | None = None,
    ) -> str:
        """Build a bounded repair prompt from the current failure differential.

        The prompt keeps the task goal and all programmatic scope constraints,
        but replaces repeated issue and plan payloads with the current patch,
        latest deterministic failure, approved files, and relevant acceptance
        criteria.

        Args:
            issue: Original issue used as a fallback task goal.
            plan: Serialized plan used only when no structured plan is present.
            failure_report: Deterministic verification failure details.
            current_patch: Current workspace diff before the repair attempt.
            current_changes: Files changed by the initial implementation.
            change_plan: Structured approved change plan when available.
            normalized_issue: Structured issue and acceptance criteria.

        Returns:
            Compact, structured repair prompt string.
        """
        failed_checks = failure_report.get_failed_checks()
        if not failed_checks:
            failure_summary = "No specific failure details available"
            relevant_criterion_ids: list[str] = []
        else:
            latest_failure = failed_checks[-1]
            relevant_criterion_ids = latest_failure.acceptance_criteria
            failure_summary = (
                f"Command: {latest_failure.command}\n"
                f"Failure Type: {latest_failure.failure_type}\n"
                f"Exit Code: {latest_failure.exit_code}"
            )
            if latest_failure.summary:
                summary_dict = latest_failure.summary
                if summary_dict.get("failed_tests"):
                    failure_summary += (
                        f"\nFailed Tests: {', '.join(summary_dict['failed_tests'])}"
                    )
                if summary_dict.get("error_type"):
                    failure_summary += (
                        f"\nError Type: {summary_dict['error_type']}"
                    )
                if "relevant_output" in summary_dict:
                    failure_summary += (
                        "\nRelevant Output:\n"
                        + _truncate_repair_text(
                            str(summary_dict["relevant_output"]),
                            MAX_REPAIR_FAILURE_OUTPUT_CHARS,
                        )
                    )

        if normalized_issue is not None:
            task_goal = (
                f"{normalized_issue.title}: "
                f"{normalized_issue.problem_statement}"
            )
            criteria = normalized_issue.acceptance_criteria
            if relevant_criterion_ids:
                relevant_ids = set(relevant_criterion_ids)
                criteria = [
                    criterion
                    for criterion in criteria
                    if criterion.id in relevant_ids
                ]
            acceptance_criteria = "\n".join(
                f"- {criterion.id}: {criterion.description}"
                for criterion in criteria
            )
            task_constraints = "\n".join(
                f"- {constraint}"
                for constraint in normalized_issue.constraints
            )
        else:
            task_goal = issue
            acceptance_criteria = "\n".join(
                f"- {criterion_id}"
                for criterion_id in relevant_criterion_ids
            )
            task_constraints = ""

        if change_plan is not None:
            change_intent = "\n".join(
                (
                    f"- {change.action.value} {change.path}: "
                    f"{change.description}"
                )
                for change in change_plan.planned_changes
            )
            allowed_paths = [
                change.path
                for change in change_plan.planned_changes
            ]
        else:
            change_intent = plan
            allowed_paths = [
                change.path
                for change in (current_changes or [])
            ]

        if change_plan is not None and change_plan.out_of_scope:
            out_of_scope = "\n".join(
                f"- Out of scope: {item}"
                for item in change_plan.out_of_scope
            )
            task_constraints = "\n".join(
                part
                for part in (task_constraints, out_of_scope)
                if part
            )

        allowed_files = "\n".join(
            f"- {path}"
            for path in dict.fromkeys(allowed_paths)
        )

        return REPAIR_PROMPT.format(
            task_goal=_truncate_repair_text(
                task_goal,
                MAX_REPAIR_GOAL_CHARS,
            ),
            change_intent=_truncate_repair_text(
                change_intent,
                MAX_REPAIR_INTENT_CHARS,
            ),
            allowed_files=allowed_files or "(no writable source file identified)",
            task_constraints=_truncate_repair_text(
                task_constraints,
                MAX_REPAIR_CONSTRAINT_CHARS,
            ),
            acceptance_criteria=(
                acceptance_criteria
                or "(use the approved change intent and failed verification)"
            ),
            current_patch=_truncate_repair_text(
                current_patch,
                MAX_REPAIR_PATCH_CHARS,
            ),
            failure=failure_summary,
        )

    def _get_modified_files(self) -> list[str]:
        """Get list of modified files using git status --porcelain.

        Runs git status --porcelain in the workspace to identify which files
        have been modified, created, or deleted by the agent.

        Returns:
            List of modified file paths relative to repository root

        Raises:
            WorkflowRunnerExecutionError: If git command fails
        """
        try:
            changes = _get_workspace_changes(self.workspace.root)

            # Extract file paths from workspace changes
            modified_files = [change.path for change in changes]

            logger.info("Workspace changes detected: %s", modified_files)
            return modified_files

        except RuntimeError as e:
            raise WorkflowRunnerExecutionError(
                f"Failed to get workspace changes: {e}"
            ) from e
        except subprocess.CalledProcessError as e:
            raise WorkflowRunnerExecutionError(
                f"git status --porcelain failed: {e}"
            ) from e

    def _check_repair_scope(self) -> ScopeGateResult:
        """Check if repair changes pass scope gate validation.

        After the repair agent makes changes, this method:
        1. Gets the list of modified files via git diff --name-only
        2. Builds a minimal ChangePlan from the modified files
        3. Validates the plan against scope restrictions

        Returns:
            ScopeGateResult indicating whether changes are allowed
        """
        modified_files = self._get_modified_files()

        # Filter out ignored files (cache files, build artifacts, etc.)
        # to prevent false positives in scope validation
        modified_files = [
            file for file in modified_files
            if not _should_ignore_file(file)
        ]

        # Build a minimal ChangePlan for scope validation
        # Since this is post-repair validation, we use the actual modified files
        # as the planned changes with minimal metadata
        planned_changes = [
            PlannedChange(
                path=file,
                action="modify",
                description="Modified during repair attempt",
                acceptance_criteria=[],
            )
            for file in modified_files
        ]

        # Create a minimal ChangePlan for validation
        # Use low risk level since repairs are scoped to fix specific failures
        change_plan = ChangePlan(
            relevant_files=modified_files,
            planned_changes=planned_changes,
            planned_tests=[],
            out_of_scope=[],
            risk_level="low",
        )

        # Run scope gate validation
        scope_result = check_scope(change_plan)

        if not scope_result.allowed:
            logger.warning(
                "Scope gate rejected repair changes: %s",
                scope_result.violations,
            )
        if scope_result.warnings:
            logger.info(
                "Scope gate warnings: %s",
                scope_result.warnings,
            )

        return scope_result

    def _cleanup(self) -> None:
        """Clean up resources by stopping sandbox and removing temporary directory."""
        # Stop sandbox if running
        if self.sandbox:
            try:
                self.sandbox.stop()
                logger.info("Docker sandbox stopped")
            except (OSError, RuntimeError) as e:
                logger.warning("Failed to stop Docker sandbox: %s", e)

        # Clean up temporary directory using context manager
        if self._temp_dir is not None:
            try:
                self._temp_dir.cleanup()
                logger.info("Removed temporary workspace: %s", self._temp_dir.name)
            except (OSError, RuntimeError) as e:
                logger.warning("Failed to remove temporary workspace: %s", e)


def run_workflow(
    agent_loop: AgentLoop,
    verifier: Callable[[], VerificationReport] | None,
    workspace: Workspace,
    issue: str,
    plan: str,
    sandbox: DockerSandbox | None = None,
    change_plan: ChangePlan | None = None,
    normalized_issue: NormalizedIssue | None = None,
    trace_path: Path | None = None,
) -> WorkflowResult:
    """Convenience function to run the complete workflow with default configuration.

    This function provides a simple interface for executing the complete PatchPilot
    workflow without explicitly managing a WorkflowRunner instance.

    Args:
        agent_loop: AgentLoop instance for running the coding agent
        verifier: Optional function that runs verification and returns a VerificationReport.
                  If None, WorkflowRunner uses the built-in Verifier.
        workspace: Workspace instance for path resolution and security
        issue: The original issue description
        plan: The approved change plan for the agent to follow
        sandbox: Optional DockerSandbox instance (created if None)
        change_plan: Optional ChangePlan object for runtime scope validation
        normalized_issue: Optional normalized issue used for evidence mapping
        trace_path: Optional persistent path for the execution trace

    Returns:
        WorkflowResult containing the final execution results including completion state,
        changed files, acceptance evidence, verification report, and patch

    Raises:
        WorkflowRunnerSetupError: If workspace or sandbox setup fails
        WorkflowRunnerExecutionError: If agent execution fails critically
    """
    runner = WorkflowRunner(
        agent_loop=agent_loop,
        verifier=verifier,
        workspace=workspace,
        sandbox=sandbox,
    )

    execute_kwargs: dict[str, Any] = {
        "issue": issue,
        "plan": plan,
        "change_plan": change_plan,
    }
    if normalized_issue is not None:
        execute_kwargs["normalized_issue"] = normalized_issue
    if trace_path is not None:
        execute_kwargs["trace_path"] = trace_path

    return runner.execute(
        **execute_kwargs,
    )
