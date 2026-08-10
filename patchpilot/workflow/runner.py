"""Workflow Runner for orchestrating the complete PatchPilot execution.

This module provides the WorkflowRunner class which orchestrates the entire
workflow from issue to verified patch, including:
- Repository workspace setup
- Docker sandbox initialization
- Agent execution for code modifications
- Verification with deterministic checks
- Repair loop with intelligent failure handling
- Early stopping for unrecoverable errors and repeated failures

The WorkflowRunner integrates all PatchPilot components into a unified
execution flow following the architecture:
CLI -> WorkflowRunner -> AgentLoop + Tools + Verifier -> Target Repository
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

from patchpilot.agent_loop import AgentLoop
from patchpilot.prompts import REPAIR_PROMPT
from patchpilot.sandbox.docker_runner import DockerSandbox
from patchpilot.verification.report import VerificationReport, failure_fingerprint
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
}


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
    ) -> VerificationReport:
        """Execute the complete workflow from issue to verified patch.

        This method implements the core workflow logic:
        1. Create temporary repository copy
        2. Start Docker Sandbox
        3. Run Coding Agent for initial modification
        4. Run Verifier
        5. Enter repair loop if verification fails
        6. Return final verification report

        Args:
            issue: The original issue description
            plan: The approved change plan for the agent to follow

        Returns:
            VerificationReport containing the final verification results

        Raises:
            WorkflowRunnerSetupError: If workspace or sandbox setup fails
            WorkflowRunnerExecutionError: If agent execution fails critically
        """
        # Step 1: Create temporary repository copy
        self._create_temporary_workspace()

        # Step 2: Start Docker Sandbox
        self._start_sandbox()

        try:
            # Step 3: Coding Agent initial modification
            logger.info("Running coding agent for initial implementation")
            initial_prompt = f"Implement the following plan:\n\n{plan}"
            self.agent_loop.run(issue=initial_prompt)

            # Step 4: Initial verification
            logger.info("Running initial verification")
            report = self.verifier()

            # Step 5: Repair loop if verification failed
            retry_count = 0
            previous_failure = None

            while not report.passed:
                # Step 5a: Check for repeated failure (fingerprint check)
                current_failure = failure_fingerprint(report)

                if current_failure == previous_failure:
                    logger.warning(
                        "Same failure repeated. Stopping repair loop to avoid futile attempts."
                    )
                    break

                previous_failure = current_failure

                # Step 5b: Check for unrecoverable errors
                if report.failure_type in UNRECOVERABLE_FAILURE_TYPES:
                    logger.warning(
                        "Unrecoverable failure detected: %s. Stopping repair loop.",
                        report.failure_type,
                    )
                    break

                # Step 5c: Check repair attempt limit
                if retry_count >= MAX_REPAIR_ATTEMPTS:
                    logger.warning(
                        "Maximum repair attempts (%d) reached. Stopping repair loop.",
                        MAX_REPAIR_ATTEMPTS,
                    )
                    break

                retry_count += 1
                logger.info(
                    "Starting repair attempt %d/%d",
                    retry_count,
                    MAX_REPAIR_ATTEMPTS,
                )

                # Step 5d: Build repair prompt with failure feedback
                repair_prompt = self._build_repair_prompt(
                    issue=issue,
                    plan=plan,
                    failure_report=report,
                )

                # Step 5e: Run agent with repair prompt
                self.agent_loop.run(issue=repair_prompt)

                # Step 5f: Re-run verification
                report = self.verifier()
                report.retry_count = retry_count

            return report

        finally:
            # Cleanup: Stop sandbox and remove temporary directory
            self._cleanup()

    def _create_temporary_workspace(self) -> None:
        """Create a temporary copy of the repository workspace.

        Creates a temporary directory and copies the repository contents
        to provide an isolated working environment for the agent.
        Uses TemporaryDirectory context manager for automatic cleanup.

        Raises:
            WorkflowRunnerSetupError: If temporary workspace creation fails
        """
        try:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="patchpilot-")
            temp_path = Path(self._temp_dir.name)
            logger.info("Created temporary workspace: %s", temp_path)

            # Copy repository contents to temporary directory
            if self.workspace.root.exists():
                shutil.copytree(
                    self.workspace.root,
                    temp_path / "repo",
                    ignore=shutil.ignore_patterns(
                        ".git",
                        "__pycache__",
                        "*.pyc",
                        ".pytest_cache",
                        ".mypy_cache",
                        "*.egg-info",
                        "build",
                        "dist",
                    ),
                )
                logger.info("Copied repository to temporary workspace")
            else:
                raise WorkflowRunnerSetupError(
                    f"Source repository does not exist: {self.workspace.root}"
                )

        except (OSError, shutil.Error) as e:
            raise WorkflowRunnerSetupError(
                f"Failed to create temporary workspace: {e}"
            ) from e

    def _start_sandbox(self) -> None:
        """Start the Docker sandbox for isolated execution.

        Initializes the DockerSandbox instance if not provided,
        then starts the container for secure command execution.

        Raises:
            WorkflowRunnerSetupError: If sandbox startup fails
        """
        try:
            if self.sandbox is None:
                # Use the temporary workspace for sandbox
                workspace_path = self.temp_dir / "repo" if self.temp_dir else self.workspace.root
                self.sandbox = DockerSandbox(workspace=workspace_path)

            self.sandbox.start()
            logger.info("Docker sandbox started successfully")

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
    )
