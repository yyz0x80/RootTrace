"""Tests for the Workflow Runner orchestration component."""

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from patchpilot.agent_loop import AgentLoop
from patchpilot.planning.scope_gate import ScopeGateResult
from patchpilot.verification.report import CheckReport, VerificationReport
from patchpilot.workflow.failure_classifier import FailureType
from patchpilot.workflow.runner import (
    MAX_REPAIR_ATTEMPTS,
    WorkflowRunner,
    WorkflowRunnerSetupError,
    WorkflowRunnerExecutionError,
    run_workflow,
)
from patchpilot.workspace import Workspace


class TestWorkflowRunnerInit:
    """Tests for WorkflowRunner initialization."""

    def test_init_with_valid_parameters(self):
        """Test initialization with valid parameters."""
        mock_agent_loop = Mock(spec=AgentLoop)
        mock_verifier = Mock()
        mock_workspace = Mock(spec=Workspace)
        mock_sandbox = Mock()

        runner = WorkflowRunner(
            agent_loop=mock_agent_loop,
            verifier=mock_verifier,
            workspace=mock_workspace,
            sandbox=mock_sandbox,
        )

        assert runner.agent_loop == mock_agent_loop
        assert runner.verifier == mock_verifier
        assert runner.workspace == mock_workspace
        assert runner.sandbox == mock_sandbox
        assert runner.temp_dir is None

    def test_init_without_sandbox(self):
        """Test initialization without sandbox (should be created later)."""
        mock_agent_loop = Mock(spec=AgentLoop)
        mock_verifier = Mock()
        mock_workspace = Mock(spec=Workspace)

        runner = WorkflowRunner(
            agent_loop=mock_agent_loop,
            verifier=mock_verifier,
            workspace=mock_workspace,
        )

        assert runner.agent_loop == mock_agent_loop
        assert runner.verifier == mock_verifier
        assert runner.workspace == mock_workspace
        assert runner.sandbox is None


class TestWorkflowRunnerExecute:
    """Tests for WorkflowRunner.execute method."""

    def test_execute_immediate_success(self):
        """Test successful execution on first attempt."""
        mock_agent_loop = Mock(spec=AgentLoop)
        mock_verifier = Mock()
        mock_workspace = Mock(spec=Workspace)
        mock_sandbox = Mock()

        # Mock successful verification
        mock_report = VerificationReport(passed=True)
        mock_verifier.return_value = mock_report

        runner = WorkflowRunner(
            agent_loop=mock_agent_loop,
            verifier=mock_verifier,
            workspace=mock_workspace,
            sandbox=mock_sandbox,
        )

        # Mock internal setup methods and workspace changes
        from patchpilot.tools import WorkspaceChange
        mock_changes = [WorkspaceChange(path="src/file.py", action="modify")]
        
        with patch.object(runner, '_create_temporary_workspace'), \
             patch.object(runner, '_start_sandbox'), \
             patch.object(runner, '_cleanup'), \
             patch('patchpilot.workflow.runner._get_workspace_changes') as mock_get_changes:

            mock_get_changes.return_value = mock_changes

            result = runner.execute(
                issue="Fix the bug",
                plan="Implement the fix",
                change_plan=None,
            )

        assert result.passed is True
        assert mock_agent_loop.run.call_count == 1
        assert mock_verifier.call_count == 1

    def test_execute_with_repair_success(self):
        """Test execution that requires one repair attempt."""
        mock_agent_loop = Mock(spec=AgentLoop)
        mock_verifier = Mock()
        mock_workspace = Mock(spec=Workspace)
        mock_sandbox = Mock()

        # First verification fails, second succeeds
        failed_report = VerificationReport(passed=False)
        failed_report.failure_type = FailureType.CODE_FAILURE
        failed_report.add_check(
            CheckReport(
                level="standard",
                command="pytest tests/",
                passed=False,
                exit_code=1,
                duration_seconds=1.0,
                failure_type="AssertionError",
                summary={"error_type": "AssertionError", "failed_tests": ["test_1"]},
            )
        )

        success_report = VerificationReport(passed=True)
        mock_verifier.side_effect = [failed_report, success_report]

        runner = WorkflowRunner(
            agent_loop=mock_agent_loop,
            verifier=mock_verifier,
            workspace=mock_workspace,
            sandbox=mock_sandbox,
        )

        # Mock internal setup methods and scope gate
        from patchpilot.tools import WorkspaceChange
        mock_changes = [WorkspaceChange(path="src/file.py", action="modify")]
        
        with patch.object(runner, '_create_temporary_workspace'), \
             patch.object(runner, '_start_sandbox'), \
             patch.object(runner, '_cleanup'), \
             patch.object(runner, '_check_repair_scope') as mock_scope_check, \
             patch('patchpilot.workflow.runner._get_workspace_changes') as mock_get_changes:

            # Mock scope gate to allow changes
            mock_scope_check.return_value = ScopeGateResult(allowed=True)
            mock_get_changes.return_value = mock_changes

            result = runner.execute(
                issue="Fix the bug",
                plan="Implement the fix",
                change_plan=None,
            )

        assert result.passed is True
        assert mock_agent_loop.run.call_count == 2  # Initial + 1 repair
        assert mock_verifier.call_count == 2

    def test_execute_unrecoverable_failure_stops_after_detection(self):
        """Test that unrecoverable failures stop the repair loop after detection."""
        mock_agent_loop = Mock(spec=AgentLoop)
        mock_verifier = Mock()
        mock_workspace = Mock(spec=Workspace)
        mock_sandbox = Mock()

        # First verification fails with recoverable error
        first_report = VerificationReport(passed=False)
        first_report.failure_type = FailureType.CODE_FAILURE
        first_report.add_check(
            CheckReport(
                level="standard",
                command="pytest tests/",
                passed=False,
                exit_code=1,
                duration_seconds=1.0,
                failure_type="AssertionError",
                summary={"error_type": "AssertionError", "failed_tests": ["test_1"], "relevant_output": "assertion failed"},
            )
        )

        # Second verification fails with unrecoverable error (same fingerprint to trigger stall detection first)
        unrecoverable_report = VerificationReport(passed=False)
        unrecoverable_report.failure_type = FailureType.ENVIRONMENT_FAILURE
        unrecoverable_report.add_check(
            CheckReport(
                level="standard",
                command="pytest tests/",
                passed=False,
                exit_code=1,
                duration_seconds=1.0,
                failure_type="AssertionError",
                summary={"error_type": "AssertionError", "failed_tests": ["test_1"], "relevant_output": "assertion failed"},
            )
        )

        # Provide the unrecoverable report for any additional verifier calls
        mock_verifier.side_effect = [first_report, unrecoverable_report]

        runner = WorkflowRunner(
            agent_loop=mock_agent_loop,
            verifier=mock_verifier,
            workspace=mock_workspace,
            sandbox=mock_sandbox,
        )

        # Mock internal setup methods and scope gate
        from patchpilot.tools import WorkspaceChange
        mock_changes = [WorkspaceChange(path="src/file.py", action="modify")]
        
        with patch.object(runner, '_create_temporary_workspace'), \
             patch.object(runner, '_start_sandbox'), \
             patch.object(runner, '_cleanup'), \
             patch.object(runner, '_check_repair_scope') as mock_scope_check, \
             patch('patchpilot.workflow.runner._get_workspace_changes') as mock_get_changes:

            # Mock scope gate to allow changes
            mock_scope_check.return_value = ScopeGateResult(allowed=True)
            mock_get_changes.return_value = mock_changes

            result = runner.execute(
                issue="Fix the bug",
                plan="Implement the fix",
                change_plan=None,
            )

        assert result.passed is False
        # Should attempt initial + 1 repair (then stop due to same failure)
        assert mock_agent_loop.run.call_count == 2
        assert mock_verifier.call_count == 2

    def test_execute_repeated_failure_stops_early(self):
        """Test that repeated failures stop the repair loop early."""
        mock_agent_loop = Mock(spec=AgentLoop)
        mock_verifier = Mock()
        mock_workspace = Mock(spec=Workspace)
        mock_sandbox = Mock()

        # Same failure repeats
        failed_report = VerificationReport(passed=False)
        failed_report.failure_type = FailureType.CODE_FAILURE
        failed_report.add_check(
            CheckReport(
                level="standard",
                command="pytest tests/",
                passed=False,
                exit_code=1,
                duration_seconds=1.0,
                failure_type="AssertionError",
                summary={
                    "failed_tests": ["test_example"],
                    "error_type": "AssertionError",
                    "relevant_output": "expected 'high', got None",
                },
            )
        )
        mock_verifier.return_value = failed_report

        runner = WorkflowRunner(
            agent_loop=mock_agent_loop,
            verifier=mock_verifier,
            workspace=mock_workspace,
            sandbox=mock_sandbox,
        )

        # Mock internal setup methods and scope gate
        from patchpilot.tools import WorkspaceChange
        mock_changes = [WorkspaceChange(path="src/file.py", action="modify")]
        
        with patch.object(runner, '_create_temporary_workspace'), \
             patch.object(runner, '_start_sandbox'), \
             patch.object(runner, '_cleanup'), \
             patch.object(runner, '_check_repair_scope') as mock_scope_check, \
             patch('patchpilot.workflow.runner._get_workspace_changes') as mock_get_changes:

            # Mock scope gate to allow changes
            mock_scope_check.return_value = ScopeGateResult(allowed=True)
            mock_get_changes.return_value = mock_changes

            result = runner.execute(
                issue="Fix the bug",
                plan="Implement the fix",
                change_plan=None,
            )

        assert result.passed is False
        # Should stop after 2 attempts (initial + 1 repair that repeats)
        assert mock_agent_loop.run.call_count == 2
        assert mock_verifier.call_count == 2

    def test_execute_max_repair_attempts(self):
        """Test that repair loop respects MAX_REPAIR_ATTEMPTS."""
        mock_agent_loop = Mock(spec=AgentLoop)
        mock_verifier = Mock()
        mock_workspace = Mock(spec=Workspace)
        mock_sandbox = Mock()

        # Different failures to avoid stall detection
        report1 = VerificationReport(passed=False)
        report1.failure_type = FailureType.CODE_FAILURE
        report1.add_check(
            CheckReport(
                level="standard",
                command="pytest tests/",
                passed=False,
                exit_code=1,
                duration_seconds=1.0,
                failure_type="AssertionError",
                summary={"error_type": "AssertionError", "failed_tests": ["test_1"]},
            )
        )

        report2 = VerificationReport(passed=False)
        report2.failure_type = FailureType.CODE_FAILURE
        report2.add_check(
            CheckReport(
                level="standard",
                command="pytest tests/",
                passed=False,
                exit_code=1,
                duration_seconds=1.0,
                failure_type="TypeError",
                summary={"error_type": "TypeError", "failed_tests": ["test_2"]},
            )
        )

        report3 = VerificationReport(passed=False)
        report3.failure_type = FailureType.CODE_FAILURE
        report3.add_check(
            CheckReport(
                level="standard",
                command="pytest tests/",
                passed=False,
                exit_code=1,
                duration_seconds=1.0,
                failure_type="ValueError",
                summary={"error_type": "ValueError", "failed_tests": ["test_3"]},
            )
        )

        mock_verifier.side_effect = [report1, report2, report3]

        runner = WorkflowRunner(
            agent_loop=mock_agent_loop,
            verifier=mock_verifier,
            workspace=mock_workspace,
            sandbox=mock_sandbox,
        )

        # Mock internal setup methods and scope gate
        from patchpilot.tools import WorkspaceChange
        mock_changes = [WorkspaceChange(path="src/file.py", action="modify")]
        
        with patch.object(runner, '_create_temporary_workspace'), \
             patch.object(runner, '_start_sandbox'), \
             patch.object(runner, '_cleanup'), \
             patch.object(runner, '_check_repair_scope') as mock_scope_check, \
             patch('patchpilot.workflow.runner._get_workspace_changes') as mock_get_changes:

            # Mock scope gate to allow changes
            mock_scope_check.return_value = ScopeGateResult(allowed=True)
            mock_get_changes.return_value = mock_changes

            result = runner.execute(
                issue="Fix the bug",
                plan="Implement the fix",
                change_plan=None,
            )

        assert result.passed is False
        # Initial + MAX_REPAIR_ATTEMPTS (2) = 3 total
        assert mock_agent_loop.run.call_count == 3
        assert mock_verifier.call_count == 3


class TestWorkflowRunnerSetup:
    """Tests for WorkflowRunner setup and cleanup methods."""

    def test_create_temporary_workspace(self):
        """Test temporary workspace creation using git archive with real git repo."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a real git repository
            source_repo = Path(tmp_dir) / "source"
            source_repo.mkdir()
            
            # Initialize git repo
            subprocess.run(["git", "init"], cwd=source_repo, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=source_repo, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.name", "Test User"], cwd=source_repo, check=True, capture_output=True)
            
            # Create a test file and commit
            (source_repo / "test.py").write_text("print('hello')")
            subprocess.run(["git", "add", "."], cwd=source_repo, check=True, capture_output=True)
            subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=source_repo, check=True, capture_output=True)
            
            # Create a sensitive .env file
            (source_repo / ".env").write_text("SECRET_KEY=test")
            subprocess.run(["git", "add", "."], cwd=source_repo, check=True, capture_output=True)
            subprocess.run(["git", "commit", "-m", "Add env file"], cwd=source_repo, check=True, capture_output=True)

            # Setup runner with real repository
            mock_agent_loop = Mock(spec=AgentLoop)
            mock_verifier = Mock()
            mock_workspace = Mock(spec=Workspace)
            mock_workspace.root = source_repo

            runner = WorkflowRunner(
                agent_loop=mock_agent_loop,
                verifier=mock_verifier,
                workspace=mock_workspace,
            )

            # Run the workspace creation
            runner._create_temporary_workspace()

            # Verify temporary directory was created
            assert runner.temp_dir is not None
            workspace_path = runner.temp_dir / "repo"
            
            # Verify workspace exists and has files
            assert workspace_path.exists()
            assert (workspace_path / "test.py").exists()
            
            # Verify .env was removed
            assert not (workspace_path / ".env").exists()
            
            # Verify git repository was initialized in workspace
            git_config = workspace_path / ".git"
            assert git_config.exists()
            
            # Verify we can get git log (means repo is properly initialized)
            result = subprocess.run(["git", "log", "--oneline"], cwd=workspace_path, capture_output=True, text=True, check=False)
            assert result.returncode == 0
            assert "PatchPilot baseline" in result.stdout

    def test_create_temporary_workspace_missing_source(self):
        """Test that missing source repository raises error when git archive fails."""
        mock_agent_loop = Mock(spec=AgentLoop)
        mock_verifier = Mock()
        mock_workspace = Mock(spec=Workspace)
        mock_workspace.root = Path("/nonexistent/repo")

        runner = WorkflowRunner(
            agent_loop=mock_agent_loop,
            verifier=mock_verifier,
            workspace=mock_workspace,
        )

        with patch('patchpilot.workflow.runner.tempfile.TemporaryDirectory') as mock_temp_dir, \
             patch('patchpilot.workflow.runner.subprocess.run') as mock_subprocess_run, \
             pytest.raises(WorkflowRunnerSetupError, match="Failed to create temporary workspace"):

            mock_temp_instance = Mock()
            mock_temp_instance.name = "/tmp/patchpilot-test"
            mock_temp_dir.return_value = mock_temp_instance

            # Mock git archive to fail (simulating missing repository)
            mock_subprocess_run.side_effect = subprocess.CalledProcessError(
                1, "git", stderr="fatal: not a git repository"
            )

            runner._create_temporary_workspace()

    def test_start_sandbox_when_provided(self):
        """Test starting sandbox when already provided."""
        mock_agent_loop = Mock(spec=AgentLoop)
        mock_verifier = Mock()
        mock_workspace = Mock(spec=Workspace)
        mock_sandbox = Mock()

        runner = WorkflowRunner(
            agent_loop=mock_agent_loop,
            verifier=mock_verifier,
            workspace=mock_workspace,
            sandbox=mock_sandbox,
        )

        runner._start_sandbox(Path("/fake/workspace"))

        mock_sandbox.start.assert_called_once()

    def test_start_sandbox_creates_when_none(self):
        """Test that sandbox is created when not provided."""
        mock_agent_loop = Mock(spec=AgentLoop)
        mock_verifier = Mock()
        mock_workspace = Mock(spec=Workspace)
        mock_workspace.root = Path("/fake/repo")

        runner = WorkflowRunner(
            agent_loop=mock_agent_loop,
            verifier=mock_verifier,
            workspace=mock_workspace,
        )
        # Set a temp_dir so the sandbox can use it
        mock_temp_dir = Mock()
        mock_temp_dir.name = "/tmp/patchpilot-test"
        runner._temp_dir = mock_temp_dir

        with patch('patchpilot.workflow.runner.DockerSandbox') as mock_docker_sandbox:
            mock_sandbox_instance = Mock()
            mock_docker_sandbox.return_value = mock_sandbox_instance

            runner._start_sandbox(Path("/fake/workspace"))

            mock_docker_sandbox.assert_called_once()
            mock_sandbox_instance.start.assert_called_once()
            assert runner.sandbox == mock_sandbox_instance

    def test_cleanup_stops_sandbox(self):
        """Test that cleanup stops the sandbox."""
        mock_agent_loop = Mock(spec=AgentLoop)
        mock_verifier = Mock()
        mock_workspace = Mock(spec=Workspace)
        mock_sandbox = Mock()

        runner = WorkflowRunner(
            agent_loop=mock_agent_loop,
            verifier=mock_verifier,
            workspace=mock_workspace,
            sandbox=mock_sandbox,
        )

        runner._cleanup()

        mock_sandbox.stop.assert_called_once()

    def test_cleanup_removes_temp_dir(self):
        """Test that cleanup removes temporary directory."""
        mock_agent_loop = Mock(spec=AgentLoop)
        mock_verifier = Mock()
        mock_workspace = Mock(spec=Workspace)
        mock_sandbox = Mock()

        # Create a real temporary directory for this test
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="patchpilot-test-")
        temp_dir = Path(temp_dir_obj.name)
        (temp_dir / "test.txt").write_text("test")

        runner = WorkflowRunner(
            agent_loop=mock_agent_loop,
            verifier=mock_verifier,
            workspace=mock_workspace,
            sandbox=mock_sandbox,
        )
        runner._temp_dir = temp_dir_obj

        runner._cleanup()

        assert not temp_dir.exists()
        
        # Cleanup the temp_dir_obj since the test didn't use it in a context manager
        if temp_dir.exists():
            temp_dir_obj.cleanup()


class TestWorkflowRunnerBuildRepairPrompt:
    """Tests for _build_repair_prompt method."""

    def test_build_repair_prompt_with_failure(self):
        """Test building repair prompt with failure details."""
        mock_agent_loop = Mock(spec=AgentLoop)
        mock_verifier = Mock()
        mock_workspace = Mock(spec=Workspace)

        runner = WorkflowRunner(
            agent_loop=mock_agent_loop,
            verifier=mock_verifier,
            workspace=mock_workspace,
        )

        # Create a failure report
        failure_report = VerificationReport(passed=False)
        failure_report.add_check(
            CheckReport(
                level="standard",
                command="pytest tests/",
                passed=False,
                exit_code=1,
                duration_seconds=1.0,
                failure_type="AssertionError",
                summary={
                    "failed_tests": ["test_example"],
                    "error_type": "AssertionError",
                    "relevant_output": "expected 'high', got None",
                },
            )
        )

        prompt = runner._build_repair_prompt(
            issue="Fix the bug",
            plan="Implement the fix",
            failure_report=failure_report,
        )

        assert "Fix the bug" in prompt
        assert "Implement the fix" in prompt
        assert "pytest tests/" in prompt
        assert "AssertionError" in prompt
        assert "test_example" in prompt

    def test_build_repair_prompt_without_failed_checks(self):
        """Test building repair prompt when no failed checks exist."""
        mock_agent_loop = Mock(spec=AgentLoop)
        mock_verifier = Mock()
        mock_workspace = Mock(spec=Workspace)

        runner = WorkflowRunner(
            agent_loop=mock_agent_loop,
            verifier=mock_verifier,
            workspace=mock_workspace,
        )

        # Create a failure report without failed checks
        failure_report = VerificationReport(passed=False)

        prompt = runner._build_repair_prompt(
            issue="Fix the bug",
            plan="Implement the fix",
            failure_report=failure_report,
        )

        assert "Fix the bug" in prompt
        assert "Implement the fix" in prompt
        assert "No specific failure details available" in prompt


class TestRunWorkflow:
    """Tests for the convenience run_workflow function."""

    def test_run_workflow_convenience(self):
        """Test the convenience function with default parameters."""
        mock_agent_loop = Mock(spec=AgentLoop)
        mock_verifier = Mock()
        mock_workspace = Mock(spec=Workspace)
        mock_sandbox = Mock()

        # Mock successful verification
        mock_report = VerificationReport(passed=True)
        mock_verifier.return_value = mock_report

        with patch('patchpilot.workflow.runner.WorkflowRunner') as mock_runner_class:
            mock_runner_instance = Mock()
            mock_runner_instance.execute.return_value = mock_report
            mock_runner_class.return_value = mock_runner_instance

            result = run_workflow(
                agent_loop=mock_agent_loop,
                verifier=mock_verifier,
                workspace=mock_workspace,
                issue="Fix the bug",
                plan="Implement the fix",
                sandbox=mock_sandbox,
            )

        assert result.passed is True
        mock_runner_class.assert_called_once_with(
            agent_loop=mock_agent_loop,
            verifier=mock_verifier,
            workspace=mock_workspace,
            sandbox=mock_sandbox,
        )
        mock_runner_instance.execute.assert_called_once_with(
            issue="Fix the bug",
            plan="Implement the fix",
            change_plan=None,
        )


class TestWorkflowRunnerScopeGate:
    """Tests for scope gate integration in repair loop."""

    def test_scope_gate_allows_safe_changes(self):
        """Test that scope gate allows safe repair changes."""
        mock_agent_loop = Mock(spec=AgentLoop)
        mock_verifier = Mock()
        mock_workspace = Mock(spec=Workspace)
        mock_sandbox = Mock()

        # First verification fails, second succeeds
        failed_report = VerificationReport(passed=False)
        failed_report.failure_type = FailureType.CODE_FAILURE
        failed_report.add_check(
            CheckReport(
                level="standard",
                command="pytest tests/",
                passed=False,
                exit_code=1,
                duration_seconds=1.0,
                failure_type="AssertionError",
                summary={"error_type": "AssertionError", "failed_tests": ["test_1"]},
            )
        )

        success_report = VerificationReport(passed=True)
        mock_verifier.side_effect = [failed_report, success_report]

        runner = WorkflowRunner(
            agent_loop=mock_agent_loop,
            verifier=mock_verifier,
            workspace=mock_workspace,
            sandbox=mock_sandbox,
        )

        # Mock internal setup methods and git diff
        from patchpilot.tools import WorkspaceChange
        mock_changes = [WorkspaceChange(path="src/module.py", action="modify")]
        
        with patch.object(runner, '_create_temporary_workspace'), \
             patch.object(runner, '_start_sandbox'), \
             patch.object(runner, '_cleanup'), \
             patch.object(runner, '_get_modified_files') as mock_git_diff, \
             patch('patchpilot.workflow.runner._get_workspace_changes') as mock_get_changes:

            # Mock git diff to return safe files
            mock_git_diff.return_value = ["src/module.py", "README.md"]
            mock_get_changes.return_value = mock_changes

            result = runner.execute(
                issue="Fix the bug",
                plan="Implement the fix",
                change_plan=None,
            )

        assert result.passed is True
        assert mock_agent_loop.run.call_count == 2  # Initial + 1 repair
        assert mock_verifier.call_count == 2

    def test_scope_gate_blocks_forbidden_changes(self):
        """Test that scope gate blocks forbidden repair changes."""
        mock_agent_loop = Mock(spec=AgentLoop)
        mock_verifier = Mock()
        mock_workspace = Mock(spec=Workspace)
        mock_sandbox = Mock()

        # First verification fails
        failed_report = VerificationReport(passed=False)
        failed_report.failure_type = FailureType.CODE_FAILURE
        failed_report.add_check(
            CheckReport(
                level="standard",
                command="pytest tests/",
                passed=False,
                exit_code=1,
                duration_seconds=1.0,
                failure_type="AssertionError",
                summary={"error_type": "AssertionError", "failed_tests": ["test_1"]},
            )
        )

        mock_verifier.return_value = failed_report

        runner = WorkflowRunner(
            agent_loop=mock_agent_loop,
            verifier=mock_verifier,
            workspace=mock_workspace,
            sandbox=mock_sandbox,
        )

        # Mock internal setup methods and git diff
        from patchpilot.tools import WorkspaceChange
        mock_changes = [WorkspaceChange(path=".env", action="modify")]
        
        with patch.object(runner, '_create_temporary_workspace'), \
             patch.object(runner, '_start_sandbox'), \
             patch.object(runner, '_cleanup'), \
             patch.object(runner, '_get_modified_files') as mock_git_diff, \
             patch('patchpilot.workflow.runner._get_workspace_changes') as mock_get_changes:

            # Mock git diff to return forbidden file (.env)
            mock_git_diff.return_value = [".env"]
            mock_get_changes.return_value = mock_changes

            result = runner.execute(
                issue="Fix the bug",
                plan="Implement the fix",
                change_plan=None,
            )

        assert result.passed is False
        assert result.failure_type == "SCOPE_VIOLATION"
        # Should attempt initial + 1 repair (then stop due to scope violation)
        assert mock_agent_loop.run.call_count == 2
        assert mock_verifier.call_count == 1  # Only initial verification, scope gate blocks re-verification

    def test_execute_no_changes_with_change_plan_raises_error(self):
        """Test that agent with change plan must modify at least one file."""
        mock_agent_loop = Mock(spec=AgentLoop)
        mock_verifier = Mock()
        mock_workspace = Mock(spec=Workspace)
        mock_sandbox = Mock()

        # Mock successful verification
        mock_report = VerificationReport(passed=True)
        mock_verifier.return_value = mock_report

        # Create a change plan with planned changes
        from patchpilot.planning.schema import ChangePlan, PlannedChange, ChangeAction
        change_plan = ChangePlan(
            risk_level="low",
            planned_changes=[
                PlannedChange(
                    path="src/file.py",
                    action=ChangeAction.MODIFY,
                    description="Fix bug",
                )
            ]
        )

        runner = WorkflowRunner(
            agent_loop=mock_agent_loop,
            verifier=mock_verifier,
            workspace=mock_workspace,
            sandbox=mock_sandbox,
        )

        # Mock internal setup methods and empty changes
        with patch.object(runner, '_create_temporary_workspace'), \
             patch.object(runner, '_start_sandbox'), \
             patch.object(runner, '_cleanup'), \
             patch('patchpilot.workflow.runner._get_workspace_changes') as mock_get_changes:

            # Mock no changes
            mock_get_changes.return_value = []

            with pytest.raises(WorkflowRunnerExecutionError) as exc_info:
                runner.execute(
                    issue="Fix the bug",
                    plan="Implement the fix",
                    change_plan=change_plan,
                )

            assert "without modifying any repository files" in str(exc_info.value)

    def test_execute_no_changes_without_change_plan_allowed(self):
        """Test that agent without change plan can complete without file changes."""
        mock_agent_loop = Mock(spec=AgentLoop)
        mock_verifier = Mock()
        mock_workspace = Mock(spec=Workspace)
        mock_sandbox = Mock()

        # Mock successful verification
        mock_report = VerificationReport(passed=True)
        mock_verifier.return_value = mock_report

        runner = WorkflowRunner(
            agent_loop=mock_agent_loop,
            verifier=mock_verifier,
            workspace=mock_workspace,
            sandbox=mock_sandbox,
        )

        # Mock internal setup methods and empty changes
        with patch.object(runner, '_create_temporary_workspace'), \
             patch.object(runner, '_start_sandbox'), \
             patch.object(runner, '_cleanup'), \
             patch('patchpilot.workflow.runner._get_workspace_changes') as mock_get_changes:

            # Mock no changes
            mock_get_changes.return_value = []

            result = runner.execute(
                issue="Fix the bug",
                plan="Implement the fix",
                change_plan=None,
            )

        assert result.passed is True
        assert mock_agent_loop.run.call_count == 1
        assert mock_verifier.call_count == 1

    def test_scope_gate_blocks_cicd_changes(self):
        """Test that scope gate blocks CI/CD repair changes."""
        mock_agent_loop = Mock(spec=AgentLoop)
        mock_verifier = Mock()
        mock_workspace = Mock(spec=Workspace)
        mock_sandbox = Mock()

        # First verification fails
        failed_report = VerificationReport(passed=False)
        failed_report.failure_type = FailureType.CODE_FAILURE
        failed_report.add_check(
            CheckReport(
                level="standard",
                command="pytest tests/",
                passed=False,
                exit_code=1,
                duration_seconds=1.0,
                failure_type="AssertionError",
                summary={"error_type": "AssertionError", "failed_tests": ["test_1"]},
            )
        )

        mock_verifier.return_value = failed_report

        runner = WorkflowRunner(
            agent_loop=mock_agent_loop,
            verifier=mock_verifier,
            workspace=mock_workspace,
            sandbox=mock_sandbox,
        )

        # Mock internal setup methods and git diff
        from patchpilot.tools import WorkspaceChange
        mock_changes = [WorkspaceChange(path=".github/workflows/test.yml", action="modify")]
        
        with patch.object(runner, '_create_temporary_workspace'), \
             patch.object(runner, '_start_sandbox'), \
             patch.object(runner, '_cleanup'), \
             patch.object(runner, '_get_modified_files') as mock_git_diff, \
             patch('patchpilot.workflow.runner._get_workspace_changes') as mock_get_changes:

            # Mock git diff to return CI/CD file
            mock_git_diff.return_value = [".github/workflows/test.yml"]
            mock_get_changes.return_value = mock_changes

            result = runner.execute(
                issue="Fix the bug",
                plan="Implement the fix",
                change_plan=None,
            )

        assert result.passed is False
        assert result.failure_type == "SCOPE_VIOLATION"
        assert mock_agent_loop.run.call_count == 2
        assert mock_verifier.call_count == 1

    def test_get_modified_files_calls_git_status(self):
        """Test that _get_modified_files correctly calls git status --porcelain."""
        mock_agent_loop = Mock(spec=AgentLoop)
        mock_verifier = Mock()
        mock_workspace = Mock(spec=Workspace)
        mock_workspace.root = Path("/fake/repo")

        runner = WorkflowRunner(
            agent_loop=mock_agent_loop,
            verifier=mock_verifier,
            workspace=mock_workspace,
        )

        with patch('patchpilot.workflow.runner._get_workspace_changes') as mock_get_changes:
            from patchpilot.tools import WorkspaceChange
            mock_get_changes.return_value = [
                WorkspaceChange(path="src/file1.py", action="modify"),
                WorkspaceChange(path="src/file2.py", action="modify"),
            ]

            modified_files = runner._get_modified_files()

            assert modified_files == ["src/file1.py", "src/file2.py"]
            mock_get_changes.assert_called_once_with(Path("/fake/repo"))


class TestConstants:
    """Tests for module constants."""

    def test_max_repair_attempts(self):
        """Test that MAX_REPAIR_ATTEMPTS matches requirements."""
        assert MAX_REPAIR_ATTEMPTS == 2
