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
import subprocess
import tarfile
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from patchpilot.agent_loop import AgentLoop, ExecuteLogCallback
from patchpilot.planning.schema import ChangePlan, PlannedChange
from patchpilot.planning.scope_gate import (
    ScopeGateResult,
    check_scope,
    validate_actual_changes,
)
from patchpilot.prompts import REPAIR_PROMPT
from patchpilot.sandbox.docker_runner import DockerSandbox
from patchpilot.tools import _get_workspace_changes, generate_patch
from patchpilot.verification.report import VerificationReport, failure_fingerprint
from patchpilot.workflow.execute_logger import ExecuteLogger
from patchpilot.workflow.failure_classifier import FailureType
from patchpilot.workspace import Workspace

logger = logging.getLogger(__name__)

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


class WorkflowExecuteLogCallback(ExecuteLogCallback):
    """Execute log callback for workflow runner using ExecuteLogger.

    Bridges the AgentLoop's callback interface with the structured
    ExecuteLogger to produce clean section-based logging output.
    """

    def __init__(self) -> None:
        """Initialize the callback with a section flag."""
        self._coding_section_logged = False

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
        verifier: Callable[[], VerificationReport],
        workspace: Workspace,
        sandbox: DockerSandbox | None = None,
    ) -> None:
        """Initialize the WorkflowRunner with required components.

        Args:
            agent_loop: AgentLoop instance for running the coding agent
            verifier: Function that runs verification and returns a VerificationReport
            workspace: Workspace instance for path resolution and security
            sandbox: Optional DockerSandbox instance (created if None)
        """
        self.agent_loop = agent_loop
        self.verifier = verifier
        self.workspace = workspace
        self.sandbox = sandbox
        self._temp_dir: tempfile.TemporaryDirectory | None = None

    @property
    def temp_dir(self) -> Path | None:
        """Get the temporary directory path as a Path object."""
        if self._temp_dir is None:
            return None
        return Path(self._temp_dir.name)

    def execute(
        self,
        issue: str,
        plan: str,
        change_plan: ChangePlan | None = None,
    ) -> VerificationReport:
        """Execute the complete workflow from issue to verified patch.

        This method implements the core workflow logic following the proper execution order:
        1. Preflight validation (handled in CLI before this method)
        2. Baseline validation (handled in CLI before this method)
        3. Create temporary workspace copy
        4. Update workspace to use temporary path
        5. Update agent loop tools to use temporary workspace
        6. Start Docker Sandbox
        7. Run Coding Agent for initial modification
        8. Check actual changes via _get_workspace_changes
        9. Validate agent made changes if required
        10. Runtime scope validation against approved plan
        11. Run Verifier
        12. Repair loop if verification failed:
            - Run Repair Agent
            - Get workspace changes via _get_workspace_changes
            - Scope gate validation
            - Run verifier
        13. Generate patch with all changes
        14. Return final verification report

        Args:
            issue: The original issue description
            plan: The approved change plan for the agent to follow
            change_plan: Optional ChangePlan object for runtime scope validation

        Returns:
            VerificationReport containing the final verification results

        Raises:
            WorkflowRunnerSetupError: If workspace or sandbox setup fails
            WorkflowRunnerExecutionError: If agent execution fails critically
        """
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

        # Step 5: Update agent loop tools to use temporary workspace
        self.agent_loop.update_workspace(self.workspace)

        # Set up execute log callback for structured logging
        execute_callback = WorkflowExecuteLogCallback()
        self.agent_loop.execute_log_callback = execute_callback

        # Step 6: Start Docker Sandbox
        self._start_sandbox(workspace_path)

        try:
            # Step 7: Coding Agent initial modification
            logger.info("Running coding agent for initial implementation")
            initial_prompt = f"Implement the following plan:\n\n{plan}"
            self.agent_loop.run(issue=initial_prompt)

            # Step 8: Check actual changes via _get_workspace_changes
            actual_changes = _get_workspace_changes(workspace_path)

            # Step 9: Validate agent made changes if required
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

            # Step 10: Runtime scope validation against approved plan
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

            # Step 11: Initial verification
            logger.info("Running initial verification")
            report = self.verifier()

            # Log verification results
            verification_results = {}
            for check in report.checks:
                # Use command as the check name for logging
                check_name = check.command if check.command else check.level
                verification_results[check_name] = check.passed
            ExecuteLogger.log_verification(verification_results)

            # Step 12: Repair loop if verification failed
            retry_count = 0
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
                if retry_count >= MAX_REPAIR_ATTEMPTS:
                    logger.warning(
                        "Maximum repair attempts (%d) reached. Stopping repair loop.",
                        MAX_REPAIR_ATTEMPTS,
                    )
                    ExecuteLogger.log_repair_stopped("Maximum repair attempts reached")
                    break

                retry_count += 1
                ExecuteLogger.log_repair_attempt(retry_count, MAX_REPAIR_ATTEMPTS)
                logger.info(
                    "Starting repair attempt %d/%d",
                    retry_count,
                    MAX_REPAIR_ATTEMPTS,
                )

                # Build repair prompt with failure feedback
                repair_prompt = self._build_repair_prompt(
                    issue=issue,
                    plan=plan,
                    failure_report=report,
                )

                # Run agent with repair prompt
                self.agent_loop.run(issue=repair_prompt)

                # Get workspace changes after repair
                repair_changes = _get_workspace_changes(workspace_path)
                if not repair_changes:
                    logger.warning(
                        "Repair agent finished without modifying any files. Stopping repair loop."
                    )
                    ExecuteLogger.log_repair_stopped("No changes made during repair")
                    break

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
                report = self.verifier()
                report.retry_count = retry_count

            # Step 13: Generate patch with all changes
            logger.info("Generating patch with all changes")
            final_changes = _get_workspace_changes(workspace_path)
            report.patch = generate_patch(workspace_path, final_changes)
            logger.info("Patch generated successfully")

            # Log final result section
            artifacts = {}
            if report.patch:
                artifacts["patch.diff"] = "artifacts/patch.diff"
            artifacts["verification_report.json"] = "artifacts/verification_report.json"

            ExecuteLogger.log_result(passed=report.passed, artifacts=artifacts)

            # Step 14: Return final verification report
            return report

        finally:
            # Cleanup: Stop sandbox and remove temporary directory
            self._cleanup()

    def _create_temporary_workspace(self) -> Path:
        """Create a temporary copy of the repository workspace using git archive.

        Creates a temporary directory and exports the repository HEAD commit
        using git archive to provide an isolated working environment for the agent.
        Removes sensitive environment files and initializes a new git repository
        for baseline tracking.

        Returns:
            Path to the temporary workspace directory

        Raises:
            WorkflowRunnerSetupError: If temporary workspace creation fails
        """
        try:
            # Create temporary directory
            self._temp_dir = tempfile.TemporaryDirectory(prefix="patchpilot-")
            temp_root = Path(self._temp_dir.name)
            logger.info("Created temporary workspace: %s", temp_root)

            # Create workspace directory
            workspace_path = temp_root / "repo"
            workspace_path.mkdir()
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
            if self.sandbox is None:
                self.sandbox = DockerSandbox(workspace=workspace_path)

            self.sandbox.start()
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
    ) -> str:
        """Build a repair prompt based on the failure report.

        Constructs a prompt that includes the original issue, approved plan,
        and condensed failure information to guide the agent's repair attempt.

        Args:
            issue: The original issue description
            plan: The approved change plan
            failure_report: VerificationReport containing failure details

        Returns:
            Formatted repair prompt string
        """
        # Extract condensed failure information
        failed_checks = failure_report.get_failed_checks()
        if not failed_checks:
            failure_summary = "No specific failure details available"
        else:
            latest_failure = failed_checks[-1]
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
                if "relevant_output" in summary_dict:
                    failure_summary += (
                        f"\nRelevant Output:\n{summary_dict['relevant_output'][:1000]}"
                    )

        # Use the REPAIR_PROMPT template from prompts.py
        return REPAIR_PROMPT.format(
            issue=issue,
            plan=plan,
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
    verifier: Callable[[], VerificationReport],
    workspace: Workspace,
    issue: str,
    plan: str,
    sandbox: DockerSandbox | None = None,
    change_plan: ChangePlan | None = None,
) -> VerificationReport:
    """Convenience function to run the complete workflow with default configuration.

    This function provides a simple interface for executing the complete PatchPilot
    workflow without explicitly managing a WorkflowRunner instance.

    Args:
        agent_loop: AgentLoop instance for running the coding agent
        verifier: Function that runs verification and returns a VerificationReport
        workspace: Workspace instance for path resolution and security
        issue: The original issue description
        plan: The approved change plan for the agent to follow
        sandbox: Optional DockerSandbox instance (created if None)
        change_plan: Optional ChangePlan object for runtime scope validation

    Returns:
        VerificationReport containing the final verification results

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

    return runner.execute(
        issue=issue,
        plan=plan,
        change_plan=change_plan,
    )
