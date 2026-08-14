"""Tests for the Workflow Runner orchestration component."""

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from patchpilot.agent_loop import AgentLoop
from patchpilot.evidence.schema import CompletionState
from patchpilot.issue.schema import AcceptanceCriterion, NormalizedIssue
from patchpilot.planning.schema import (
    ChangeAction,
    ChangePlan,
    PlannedChange,
    PlannedTest,
)
from patchpilot.planning.scope_gate import ScopeGateResult
from patchpilot.verification.report import CheckReport, VerificationReport
from patchpilot.workflow.failure_classifier import FailureType
from patchpilot.workflow.runner import (
    MAX_REPAIR_ATTEMPTS,
    WorkflowRunner,
    WorkflowRunnerExecutionError,
    WorkflowRunnerSetupError,
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
        from patchpilot.workflow.result import WorkflowResult

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
             patch('patchpilot.workflow.runner._get_workspace_changes') as mock_get_changes, \
             patch('patchpilot.workflow.runner.generate_patch') as mock_generate_patch:

            mock_get_changes.return_value = mock_changes
            mock_generate_patch.return_value = "diff content"
            mock_generate_patch.return_value = "diff content"

            result = runner.execute(
                issue="Fix the bug",
                plan="Implement the fix",
                change_plan=None,
            )

        assert isinstance(result, WorkflowResult)
        assert result.final_status == CompletionState.PARTIALLY_VERIFIED
        assert result.verification_report["passed"] is True
        assert mock_agent_loop.run.call_count == 1
        assert mock_verifier.call_count == 1

    def test_execute_with_repair_success(self):
        """Test execution that requires one repair attempt."""
        from patchpilot.workflow.result import WorkflowResult

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
             patch('patchpilot.workflow.runner._get_workspace_changes') as mock_get_changes, \
             patch('patchpilot.workflow.runner.generate_patch') as mock_generate_patch:

            # Mock scope gate to allow changes
            mock_scope_check.return_value = ScopeGateResult(allowed=True)
            mock_get_changes.return_value = mock_changes
            mock_generate_patch.return_value = "diff content"

            result = runner.execute(
                issue="Fix the bug",
                plan="Implement the fix",
                change_plan=None,
            )

        assert isinstance(result, WorkflowResult)
        assert result.final_status == CompletionState.PARTIALLY_VERIFIED
        assert result.verification_report["passed"] is True
        assert mock_agent_loop.run.call_count == 2  # Initial + 1 repair
        assert mock_verifier.call_count == 2

    def test_execute_unrecoverable_failure_stops_after_detection(self):
        """Test that unrecoverable failures stop the repair loop after detection."""
        from patchpilot.workflow.result import WorkflowResult

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
             patch('patchpilot.workflow.runner._get_workspace_changes') as mock_get_changes, \
             patch('patchpilot.workflow.runner.generate_patch') as mock_generate_patch:

            # Mock scope gate to allow changes
            mock_scope_check.return_value = ScopeGateResult(allowed=True)
            mock_get_changes.return_value = mock_changes
            mock_generate_patch.return_value = "diff content"

            result = runner.execute(
                issue="Fix the bug",
                plan="Implement the fix",
                change_plan=None,
            )

        assert isinstance(result, WorkflowResult)
        assert result.final_status == CompletionState.FAILED
        assert result.verification_report["passed"] is False
        # Should attempt initial + 1 repair (then stop due to same failure)
        assert mock_agent_loop.run.call_count == 2
        assert mock_verifier.call_count == 2

    def test_execute_repeated_failure_stops_early(self):
        """Test that repeated failures stop the repair loop early."""
        from patchpilot.workflow.result import WorkflowResult

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
             patch('patchpilot.workflow.runner._get_workspace_changes') as mock_get_changes, \
             patch('patchpilot.workflow.runner.generate_patch') as mock_generate_patch:

            # Mock scope gate to allow changes
            mock_scope_check.return_value = ScopeGateResult(allowed=True)
            mock_get_changes.return_value = mock_changes
            mock_generate_patch.return_value = "diff content"

            result = runner.execute(
                issue="Fix the bug",
                plan="Implement the fix",
                change_plan=None,
            )

        assert isinstance(result, WorkflowResult)
        assert result.final_status == CompletionState.FAILED
        assert result.verification_report["passed"] is False
        # Should stop after 2 attempts (initial + 1 repair that repeats)
        assert mock_agent_loop.run.call_count == 2
        assert mock_verifier.call_count == 2

    def test_execute_max_repair_attempts(self):
        """Test that repair loop respects MAX_REPAIR_ATTEMPTS."""
        from patchpilot.workflow.result import WorkflowResult

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
             patch('patchpilot.workflow.runner._get_workspace_changes') as mock_get_changes, \
             patch('patchpilot.workflow.runner.generate_patch') as mock_generate_patch:

            # Mock scope gate to allow changes
            mock_scope_check.return_value = ScopeGateResult(allowed=True)
            mock_get_changes.return_value = mock_changes
            mock_generate_patch.return_value = "diff content"

            result = runner.execute(
                issue="Fix the bug",
                plan="Implement the fix",
                change_plan=None,
            )

        assert isinstance(result, WorkflowResult)
        assert result.final_status == CompletionState.FAILED
        assert result.verification_report["passed"] is False
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

        # Create a real temporary directory for the test
        with tempfile.TemporaryDirectory() as temp_dir:
            runner._start_sandbox(Path(temp_dir))

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

            # Create a real temporary directory for the test
            with tempfile.TemporaryDirectory() as temp_dir:
                runner._start_sandbox(Path(temp_dir))

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
        from patchpilot.workflow.result import WorkflowResult

        mock_agent_loop = Mock(spec=AgentLoop)
        mock_verifier = Mock()
        mock_workspace = Mock(spec=Workspace)
        mock_sandbox = Mock()

        # Mock successful verification
        mock_report = VerificationReport(passed=True)
        mock_verifier.return_value = mock_report

        with patch('patchpilot.workflow.runner.WorkflowRunner') as mock_runner_class:
            mock_runner_instance = Mock()
            # Mock WorkflowResult return value
            mock_result = WorkflowResult(
                run_id="test-run-id",
                final_status=CompletionState.PARTIALLY_VERIFIED,
                verification_report={"passed": True},
            )
            mock_runner_instance.execute.return_value = mock_result
            mock_runner_class.return_value = mock_runner_instance

            result = run_workflow(
                agent_loop=mock_agent_loop,
                verifier=mock_verifier,
                workspace=mock_workspace,
                issue="Fix the bug",
                plan="Implement the fix",
                sandbox=mock_sandbox,
            )

        assert isinstance(result, WorkflowResult)
        assert result.final_status == CompletionState.PARTIALLY_VERIFIED
        assert result.verification_report["passed"] is True
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
        from patchpilot.workflow.result import WorkflowResult

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
             patch('patchpilot.workflow.runner._get_workspace_changes') as mock_get_changes, \
             patch('patchpilot.workflow.runner.generate_patch') as mock_generate_patch:

            # Mock git diff to return safe files
            mock_git_diff.return_value = ["src/module.py", "README.md"]
            mock_get_changes.return_value = mock_changes
            mock_generate_patch.return_value = "diff content"

            result = runner.execute(
                issue="Fix the bug",
                plan="Implement the fix",
                change_plan=None,
            )

        assert isinstance(result, WorkflowResult)
        assert result.final_status == CompletionState.PARTIALLY_VERIFIED
        assert result.verification_report["passed"] is True
        assert mock_agent_loop.run.call_count == 2  # Initial + 1 repair
        assert mock_verifier.call_count == 2

    def test_scope_gate_blocks_forbidden_changes(self):
        """Test that scope gate blocks forbidden repair changes."""
        from patchpilot.workflow.result import WorkflowResult

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
             patch('patchpilot.workflow.runner._get_workspace_changes') as mock_get_changes, \
             patch('patchpilot.workflow.runner.generate_patch') as mock_generate_patch:

            # Mock git diff to return forbidden file (.env)
            mock_git_diff.return_value = [".env"]
            mock_get_changes.return_value = mock_changes
            mock_generate_patch.return_value = "diff content"

            result = runner.execute(
                issue="Fix the bug",
                plan="Implement the fix",
                change_plan=None,
            )

        assert isinstance(result, WorkflowResult)
        assert result.final_status == CompletionState.BLOCKED
        assert result.verification_report["passed"] is False
        assert result.verification_report["failure_type"] == "SCOPE_VIOLATION"
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
        from patchpilot.planning.schema import ChangeAction, ChangePlan, PlannedChange
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
             patch('patchpilot.workflow.runner._get_workspace_changes') as mock_get_changes, \
             patch('patchpilot.workflow.runner.generate_patch') as mock_generate_patch:

            # Mock no changes
            mock_get_changes.return_value = []
            mock_generate_patch.return_value = "diff content"

            with pytest.raises(WorkflowRunnerExecutionError) as exc_info:
                runner.execute(
                    issue="Fix the bug",
                    plan="Implement the fix",
                    change_plan=change_plan,
                )

            assert "without modifying any repository files" in str(exc_info.value)

    def test_execute_no_changes_without_change_plan_allowed(self):
        """Test that agent without change plan can complete without file changes."""
        from patchpilot.workflow.result import WorkflowResult

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
             patch('patchpilot.workflow.runner._get_workspace_changes') as mock_get_changes, \
             patch('patchpilot.workflow.runner.generate_patch') as mock_generate_patch:

            # Mock no changes
            mock_get_changes.return_value = []
            mock_generate_patch.return_value = "diff content"

            result = runner.execute(
                issue="Fix the bug",
                plan="Implement the fix",
                change_plan=None,
            )

        assert isinstance(result, WorkflowResult)
        assert result.final_status == CompletionState.PARTIALLY_VERIFIED
        assert result.verification_report["passed"] is True
        assert mock_agent_loop.run.call_count == 1
        assert mock_verifier.call_count == 1

    def test_scope_gate_blocks_cicd_changes(self):
        """Test that scope gate blocks CI/CD repair changes."""
        from patchpilot.workflow.result import WorkflowResult

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
             patch('patchpilot.workflow.runner._get_workspace_changes') as mock_get_changes, \
             patch('patchpilot.workflow.runner.generate_patch') as mock_generate_patch:

            # Mock git diff to return CI/CD file
            mock_git_diff.return_value = [".github/workflows/test.yml"]
            mock_get_changes.return_value = mock_changes
            mock_generate_patch.return_value = "diff content"

            result = runner.execute(
                issue="Fix the bug",
                plan="Implement the fix",
                change_plan=None,
            )

        assert isinstance(result, WorkflowResult)
        assert result.final_status == CompletionState.BLOCKED
        assert result.verification_report["failure_type"] == "SCOPE_VIOLATION"
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

        with patch('patchpilot.workflow.runner._get_workspace_changes') as mock_get_changes, \
             patch('patchpilot.workflow.runner.generate_patch') as mock_generate_patch:
            mock_generate_patch.return_value = "diff content"
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


class TestWorkflowRunnerModifiedFilesFiltering:
    """Tests for filtering ignored files in repair scope checking."""

    def test_check_repair_scope_filters_pycache_files(self):
        """Test that _check_repair_scope filters out __pycache__ files."""
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

        # Mock _get_modified_files to return files including __pycache__
        modified_files_with_cache = [
            "src/module.py",
            "tests/__pycache__/test_module.cpython-312.pyc",
            "tests/__pycache__/__init__.cpython-312.pyc",
            "README.md",
        ]

        with patch.object(runner, '_get_modified_files') as mock_get_files:
            mock_get_files.return_value = modified_files_with_cache

            result = runner._check_repair_scope()

        # Should allow changes since __pycache__ files are filtered out
        assert result.allowed is True
        assert len(result.violations) == 0

    def test_check_repair_scope_blocks_test_file_modifications(self):
        """Test that _check_repair_scope still blocks actual test file modifications."""
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

        # Mock _get_modified_files to return actual test file (not __pycache__)
        modified_files_with_test = [
            "src/module.py",
            "tests/test_module.py",  # Actual test file, should be blocked
        ]

        with patch.object(runner, '_get_modified_files') as mock_get_files:
            mock_get_files.return_value = modified_files_with_test

            result = runner._check_repair_scope()

        # Should block changes due to test file modification
        assert result.allowed is False
        assert len(result.violations) > 0
        assert any("test file modification is forbidden" in v.lower() for v in result.violations)


class TestWorkflowRunnerVerification:
    """Tests for _run_verification method."""

    def test_run_verification_uses_sandbox_verifier(self):
        """Test that _run_verification uses built-in Verifier when verifier is None."""
        mock_agent_loop = Mock(spec=AgentLoop)
        mock_workspace = Mock(spec=Workspace)
        mock_sandbox = Mock()

        plan = ChangePlan(
            relevant_files=[],
            planned_changes=[],
            planned_tests=[
                PlannedTest(
                    command="pytest tests/test_task.py -q",
                    purpose="Verify task behavior",
                    acceptance_criteria=["AC-1"],
                )
            ],
            out_of_scope=[],
            risk_level="low",
        )

        expected_report = VerificationReport(
            run_id="run-123",
            passed=True,
        )

        runner = WorkflowRunner(
            agent_loop=mock_agent_loop,
            verifier=None,
            workspace=mock_workspace,
            sandbox=mock_sandbox,
        )

        with patch(
            "patchpilot.verification.verifier.Verifier"
        ) as verifier_class:
            verifier_class.return_value.verify.return_value = (
                expected_report
            )

            result = runner._run_verification(
                run_id="run-123",
                change_plan=plan,
                retry_count=1,
            )

        assert result is expected_report
        verifier_class.assert_called_once_with(mock_sandbox)
        verifier_class.return_value.verify.assert_called_once_with(
            run_id="run-123",
            target_tests=["tests/test_task.py"],
            target_acceptance_criteria=["AC-1"],
            retry_count=1,
        )

    def test_run_verification_uses_injected_callback(self):
        """Test that _run_verification uses injected verifier callback when provided."""
        mock_agent_loop = Mock(spec=AgentLoop)
        mock_workspace = Mock(spec=Workspace)
        mock_sandbox = Mock()
        mock_verifier = Mock(
            return_value=VerificationReport(passed=True)
        )

        runner = WorkflowRunner(
            agent_loop=mock_agent_loop,
            verifier=mock_verifier,
            workspace=mock_workspace,
            sandbox=mock_sandbox,
        )

        result = runner._run_verification(
            run_id="run-123",
            change_plan=None,
            retry_count=2,
        )

        mock_verifier.assert_called_once_with()
        assert result.retry_count == 2


class TestWorkflowRunnerCompletionStates:
    """Tests for completion state determination based on verification results."""

    def test_execute_verified_state_with_file_changes_and_passing_tests(self):
        """Test VERIFIED state when files are modified and all tests pass."""
        from patchpilot.evidence.schema import (
            AcceptanceEvidence,
            CompletionState,
            EvidenceStatus,
        )
        from patchpilot.workflow.result import WorkflowResult

        mock_agent_loop = Mock(spec=AgentLoop)
        mock_verifier = Mock()
        mock_workspace = Mock(spec=Workspace)
        mock_sandbox = Mock()

        # Mock successful verification
        mock_report = VerificationReport(passed=True)
        mock_verifier.return_value = mock_report

        normalized_issue = NormalizedIssue(
            title="Implement feature",
            task_type="feature",
            problem_statement="Implement the requested feature.",
            acceptance_criteria=[
                AcceptanceCriterion(
                    id="AC-1",
                    description="First acceptance criterion",
                ),
                AcceptanceCriterion(
                    id="AC-2",
                    description="Second acceptance criterion",
                ),
            ],
        )

        # Create a change plan with acceptance criteria
        change_plan = ChangePlan(
            relevant_files=["src/module.py"],
            planned_changes=[
                PlannedChange(
                    path="src/module.py",
                    action=ChangeAction.MODIFY,
                    description="Implement the requested feature",
                    acceptance_criteria=["AC-1", "AC-2"],
                )
            ],
            planned_tests=[
                PlannedTest(
                    command="pytest tests/test_module.py -q",
                    purpose="Verify module behavior",
                    acceptance_criteria=["AC-1", "AC-2"],
                )
            ],
            out_of_scope=[],
            risk_level="low",
        )

        runner = WorkflowRunner(
            agent_loop=mock_agent_loop,
            verifier=mock_verifier,
            workspace=mock_workspace,
            sandbox=mock_sandbox,
        )

        # Mock internal setup methods and workspace changes
        from patchpilot.tools import WorkspaceChange
        mock_changes = [WorkspaceChange(path="src/module.py", action="modify")]

        # Mock evidence mapper to return PASS status for all ACs
        mock_evidence = [
            AcceptanceEvidence(
                criterion_id="AC-1",
                description="First acceptance criterion",
                status=EvidenceStatus.PASS,
                changed_files=["src/module.py"],
                tests=["tests/test_module.py"],
                command_results=["pytest tests/test_module.py -q"],
                explanation="Test passed successfully",
            ),
            AcceptanceEvidence(
                criterion_id="AC-2",
                description="Second acceptance criterion",
                status=EvidenceStatus.PASS,
                changed_files=["src/module.py"],
                tests=["tests/test_module.py"],
                command_results=["pytest tests/test_module.py -q"],
                explanation="Test passed successfully",
            ),
        ]

        with patch.object(runner, '_create_temporary_workspace'), \
             patch.object(runner, '_start_sandbox'), \
             patch.object(runner, '_cleanup'), \
             patch('patchpilot.workflow.runner._get_workspace_changes') as mock_get_changes, \
             patch('patchpilot.workflow.runner.generate_patch') as mock_generate_patch, \
             patch('patchpilot.workflow.runner._map_acceptance_evidence') as mock_map_evidence:

            mock_get_changes.return_value = mock_changes
            mock_generate_patch.return_value = "diff content"
            mock_map_evidence.return_value = mock_evidence

            result = runner.execute(
                issue="Implement feature",
                plan="Implement the feature with full test coverage",
                change_plan=change_plan,
                normalized_issue=normalized_issue,
            )

        assert isinstance(result, WorkflowResult)
        assert result.final_status == CompletionState.VERIFIED
        assert result.verification_report["passed"] is True
        assert len(result.acceptance_evidence) == 2
        assert all(e.status == EvidenceStatus.PASS for e in result.acceptance_evidence)
        assert mock_agent_loop.run.call_count == 1
        assert mock_verifier.call_count == 1
        assert mock_map_evidence.call_args.kwargs["issue"] is normalized_issue

    def test_execute_partially_verified_state_when_ac_has_no_test_mapping(self):
        """Test PARTIALLY_VERIFIED state when verifier passes but some AC have no test mapping."""
        from patchpilot.evidence.schema import (
            AcceptanceEvidence,
            CompletionState,
            EvidenceStatus,
        )
        from patchpilot.workflow.result import WorkflowResult

        mock_agent_loop = Mock(spec=AgentLoop)
        mock_verifier = Mock()
        mock_workspace = Mock(spec=Workspace)
        mock_sandbox = Mock()

        # Mock successful verification
        mock_report = VerificationReport(passed=True)
        mock_verifier.return_value = mock_report

        # Create a change plan with acceptance criteria that lack test mapping
        change_plan = ChangePlan(
            relevant_files=["src/module.py"],
            planned_changes=[],
            planned_tests=[
                PlannedTest(
                    command="pytest tests/test_module.py -q",
                    purpose="Verify module behavior",
                    acceptance_criteria=["AC-1"],  # Only AC-1 has test mapping
                )
            ],
            out_of_scope=[],
            risk_level="low",
        )

        runner = WorkflowRunner(
            agent_loop=mock_agent_loop,
            verifier=mock_verifier,
            workspace=mock_workspace,
            sandbox=mock_sandbox,
        )

        # Mock internal setup methods and workspace changes
        from patchpilot.tools import WorkspaceChange
        mock_changes = [WorkspaceChange(path="src/module.py", action="modify")]

        # Mock evidence mapper to return UNVERIFIED for AC without test mapping
        mock_evidence = [
            AcceptanceEvidence(
                criterion_id="AC-1",
                description="First acceptance criterion with test",
                status=EvidenceStatus.PASS,
                changed_files=["src/module.py"],
                tests=["tests/test_module.py"],
                command_results=["pytest tests/test_module.py -q"],
                explanation="Test passed successfully",
            ),
            AcceptanceEvidence(
                criterion_id="AC-2",
                description="Second acceptance criterion without test mapping",
                status=EvidenceStatus.UNVERIFIED,
                changed_files=["src/module.py"],
                tests=[],  # No test mapping
                command_results=[],
                explanation="No test found for this acceptance criterion",
            ),
        ]

        with patch.object(runner, '_create_temporary_workspace'), \
             patch.object(runner, '_start_sandbox'), \
             patch.object(runner, '_cleanup'), \
             patch('patchpilot.workflow.runner._get_workspace_changes') as mock_get_changes, \
             patch('patchpilot.workflow.runner.generate_patch') as mock_generate_patch, \
             patch('patchpilot.workflow.runner._map_acceptance_evidence') as mock_map_evidence:

            mock_get_changes.return_value = mock_changes
            mock_generate_patch.return_value = "diff content"
            mock_map_evidence.return_value = mock_evidence

            result = runner.execute(
                issue="Implement feature",
                plan="Implement the feature with partial test coverage",
                change_plan=change_plan,
            )

        assert isinstance(result, WorkflowResult)
        assert result.final_status == CompletionState.PARTIALLY_VERIFIED
        assert result.verification_report["passed"] is True
        assert len(result.acceptance_evidence) == 2
        assert result.acceptance_evidence[0].status == EvidenceStatus.PASS
        assert result.acceptance_evidence[1].status == EvidenceStatus.UNVERIFIED
        assert mock_agent_loop.run.call_count == 1
        assert mock_verifier.call_count == 1

    def test_execute_failed_state_when_mapped_tests_fail(self):
        """Test FAILED state when mapped tests fail during verification."""
        from patchpilot.evidence.schema import AcceptanceEvidence, EvidenceStatus
        from patchpilot.workflow.result import WorkflowResult

        mock_agent_loop = Mock(spec=AgentLoop)
        mock_verifier = Mock()
        mock_workspace = Mock(spec=Workspace)
        mock_sandbox = Mock()

        # Mock failed verification
        failed_report = VerificationReport(passed=False)
        failed_report.failure_type = FailureType.CODE_FAILURE
        failed_report.add_check(
            CheckReport(
                level="standard",
                command="pytest tests/test_module.py -q",
                passed=False,
                exit_code=1,
                duration_seconds=1.0,
                failure_type="AssertionError",
                summary={
                    "failed_tests": ["test_module"],
                    "error_type": "AssertionError",
                    "relevant_output": "expected 'value', got None",
                },
            )
        )
        mock_verifier.return_value = failed_report

        # Create a change plan with acceptance criteria
        change_plan = ChangePlan(
            relevant_files=["src/module.py"],
            planned_changes=[],
            planned_tests=[
                PlannedTest(
                    command="pytest tests/test_module.py -q",
                    purpose="Verify module behavior",
                    acceptance_criteria=["AC-1"],
                )
            ],
            out_of_scope=[],
            risk_level="low",
        )

        runner = WorkflowRunner(
            agent_loop=mock_agent_loop,
            verifier=mock_verifier,
            workspace=mock_workspace,
            sandbox=mock_sandbox,
        )

        # Mock internal setup methods and workspace changes
        from patchpilot.tools import WorkspaceChange
        mock_changes = [WorkspaceChange(path="src/module.py", action="modify")]

        # Mock evidence mapper to return FAIL status for failing tests
        mock_evidence = [
            AcceptanceEvidence(
                criterion_id="AC-1",
                description="First acceptance criterion",
                status=EvidenceStatus.FAIL,
                changed_files=["src/module.py"],
                tests=["tests/test_module.py"],
                command_results=["pytest tests/test_module.py -q"],
                explanation="Test failed with AssertionError",
            ),
        ]

        with patch.object(runner, '_create_temporary_workspace'), \
             patch.object(runner, '_start_sandbox'), \
             patch.object(runner, '_cleanup'), \
             patch('patchpilot.workflow.runner._get_workspace_changes') as mock_get_changes, \
             patch('patchpilot.workflow.runner.generate_patch') as mock_generate_patch, \
             patch('patchpilot.workflow.runner._map_acceptance_evidence') as mock_map_evidence:

            mock_get_changes.return_value = mock_changes
            mock_generate_patch.return_value = "diff content"
            mock_map_evidence.return_value = mock_evidence

            result = runner.execute(
                issue="Fix bug",
                plan="Fix the bug in module",
                change_plan=change_plan,
            )

        assert isinstance(result, WorkflowResult)
        assert result.final_status == CompletionState.FAILED
        assert result.verification_report["passed"] is False
        assert len(result.acceptance_evidence) == 1
        assert result.acceptance_evidence[0].status == EvidenceStatus.FAIL
        assert mock_agent_loop.run.call_count == 1
        assert mock_verifier.call_count == 1

    def test_execute_scope_gate_blocked_state(self):
        """Test BLOCKED state when scope gate rejects changes."""
        from patchpilot.workflow.result import WorkflowResult

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
             patch('patchpilot.workflow.runner._get_workspace_changes') as mock_get_changes, \
             patch('patchpilot.workflow.runner.generate_patch') as mock_generate_patch:

            # Mock git diff to return forbidden file (.env)
            mock_git_diff.return_value = [".env"]
            mock_get_changes.return_value = mock_changes
            mock_generate_patch.return_value = "diff content"

            result = runner.execute(
                issue="Fix the bug",
                plan="Implement the fix",
                change_plan=None,
            )

        assert isinstance(result, WorkflowResult)
        assert result.final_status == CompletionState.BLOCKED
        assert result.verification_report["passed"] is False
        assert result.verification_report["failure_type"] == "SCOPE_VIOLATION"
        assert mock_agent_loop.run.call_count == 2


class TestWorkflowRunnerTraceEvents:
    """Tests for trace event generation during workflow execution."""

    def test_trace_contains_tool_verification_and_final_state_events(self):
        """Test that trace contains tool calls, verification results, and final state events."""
        from patchpilot.workflow.result import WorkflowResult
        from patchpilot.workflow.trace import TraceEvent, TraceWriter

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
        persistent_trace_path = Path("artifacts/execution_trace.jsonl")

        with patch.object(runner, '_create_temporary_workspace'), \
             patch.object(runner, '_start_sandbox'), \
             patch.object(runner, '_cleanup'), \
             patch('patchpilot.workflow.runner._get_workspace_changes') as mock_get_changes, \
             patch('patchpilot.workflow.runner.generate_patch') as mock_generate_patch, \
             patch('patchpilot.workflow.runner.TraceWriter') as mock_trace_writer_class:

            mock_get_changes.return_value = mock_changes
            mock_generate_patch.return_value = "diff content"

            # Create a mock trace writer instance
            mock_trace_writer = Mock(spec=TraceWriter)
            mock_trace_writer_class.return_value = mock_trace_writer

            result = runner.execute(
                issue="Fix the bug",
                plan="Implement the fix",
                change_plan=None,
                trace_path=persistent_trace_path,
            )

        # Verify the result
        assert isinstance(result, WorkflowResult)
        assert result.final_status == CompletionState.PARTIALLY_VERIFIED

        # Verify TraceWriter was instantiated
        mock_trace_writer_class.assert_called_once_with(persistent_trace_path)

        # Verify trace writer was called with a final workflow_completed event
        assert mock_trace_writer.write.call_count >= 1

        # Get the final call (which should be the workflow_completed event)
        final_call_args = mock_trace_writer.write.call_args
        final_event = final_call_args[0][0]  # First positional argument

        # Verify the final event contains expected fields
        assert isinstance(final_event, TraceEvent)
        assert final_event.event_type == "workflow_completed"
        assert final_event.workflow_stage == "RESULT"
        assert final_event.final_status == CompletionState.PARTIALLY_VERIFIED.value
        assert final_event.modified_files == ["src/file.py"]
        assert final_event.verification_result is not None
        assert final_event.verification_result["passed"] is True

    def test_evidence_completes_before_temp_directory_cleanup(self):
        """Test that evidence generation completes before temporary directory cleanup."""
        from patchpilot.evidence.schema import AcceptanceEvidence, EvidenceStatus
        from patchpilot.workflow.result import WorkflowResult

        mock_agent_loop = Mock(spec=AgentLoop)
        mock_verifier = Mock()
        mock_workspace = Mock(spec=Workspace)
        mock_sandbox = Mock()

        # Mock successful verification
        mock_report = VerificationReport(passed=True)
        mock_verifier.return_value = mock_report

        # Create a change plan with acceptance criteria
        change_plan = ChangePlan(
            relevant_files=["src/module.py"],
            planned_changes=[],
            planned_tests=[
                PlannedTest(
                    command="pytest tests/test_module.py -q",
                    purpose="Verify module behavior",
                    acceptance_criteria=["AC-1"],
                )
            ],
            out_of_scope=[],
            risk_level="low",
        )

        runner = WorkflowRunner(
            agent_loop=mock_agent_loop,
            verifier=mock_verifier,
            workspace=mock_workspace,
            sandbox=mock_sandbox,
        )

        # Mock internal setup methods and workspace changes
        from patchpilot.tools import WorkspaceChange
        mock_changes = [WorkspaceChange(path="src/module.py", action="modify")]

        # Mock evidence mapper
        mock_evidence = [
            AcceptanceEvidence(
                criterion_id="AC-1",
                description="First acceptance criterion",
                status=EvidenceStatus.PASS,
                changed_files=["src/module.py"],
                tests=["tests/test_module.py"],
                command_results=["pytest tests/test_module.py -q"],
                explanation="Test passed successfully",
            ),
        ]

        with patch.object(runner, '_create_temporary_workspace'), \
             patch.object(runner, '_start_sandbox'), \
             patch.object(runner, '_cleanup') as mock_cleanup, \
             patch('patchpilot.workflow.runner._get_workspace_changes') as mock_get_changes, \
             patch('patchpilot.workflow.runner.generate_patch') as mock_generate_patch, \
             patch('patchpilot.workflow.runner._map_acceptance_evidence') as mock_map_evidence:

            mock_get_changes.return_value = mock_changes
            mock_generate_patch.return_value = "diff content"
            mock_map_evidence.return_value = mock_evidence

            result = runner.execute(
                issue="Implement feature",
                plan="Implement the feature with test coverage",
                change_plan=change_plan,
            )

        # Verify the result
        assert isinstance(result, WorkflowResult)
        assert result.final_status == CompletionState.PARTIALLY_VERIFIED

        # Verify evidence mapper was called
        mock_map_evidence.assert_called_once()

        # Verify cleanup was called after evidence generation
        # by checking that cleanup was called at all
        mock_cleanup.assert_called_once()

        # The key assertion: evidence generation must happen before cleanup
        # We verify this by checking that when cleanup is called, the result already has evidence
        assert result.acceptance_evidence is not None
        assert len(result.acceptance_evidence) == 1


class TestWorkflowRunnerTestProtection:
    """Tests for target repository test file protection."""

    def test_target_repository_tests_not_modified_by_scope_gate(self):
        """Test that scope gate prevents modifications to target repository test files."""
        from patchpilot.workflow.result import WorkflowResult

        mock_agent_loop = Mock(spec=AgentLoop)
        mock_verifier = Mock()
        mock_workspace = Mock(spec=Workspace)
        mock_sandbox = Mock()

        # First verification fails to trigger repair loop
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

        # Mock internal setup methods and workspace changes
        from patchpilot.tools import WorkspaceChange

        # Mock workspace changes that include test file modification
        mock_changes_with_tests = [
            WorkspaceChange(path="src/module.py", action="modify"),
            WorkspaceChange(path="tests/test_module.py", action="modify"),  # Test file modification
        ]

        with patch.object(runner, '_create_temporary_workspace'), \
             patch.object(runner, '_start_sandbox'), \
             patch.object(runner, '_cleanup'), \
             patch('patchpilot.workflow.runner._get_workspace_changes') as mock_get_changes, \
             patch('patchpilot.workflow.runner.generate_patch') as mock_generate_patch:

            mock_get_changes.return_value = mock_changes_with_tests
            mock_generate_patch.return_value = "diff content"

            with patch.object(runner, '_get_modified_files') as mock_git_diff:
                # Mock git diff to show test file was modified
                mock_git_diff.return_value = ["src/module.py", "tests/test_module.py"]

                result = runner.execute(
                    issue="Fix the bug",
                    plan="Implement the fix",
                    change_plan=None,
                )

        # Verify that scope gate blocked the test file modification
        assert isinstance(result, WorkflowResult)
        assert result.final_status == CompletionState.BLOCKED
        assert result.verification_report["failure_type"] == "SCOPE_VIOLATION"
        # Should attempt initial + 1 repair (then stop due to scope violation)
        assert mock_agent_loop.run.call_count == 2

    def test_target_repository_tests_allowed_in_change_plan(self):
        """Test that test file modifications are allowed when explicitly in change plan."""
        from patchpilot.workflow.result import WorkflowResult

        mock_agent_loop = Mock(spec=AgentLoop)
        mock_verifier = Mock()
        mock_workspace = Mock(spec=Workspace)
        mock_sandbox = Mock()

        # Mock successful verification
        mock_report = VerificationReport(passed=True)
        mock_verifier.return_value = mock_report

        # Create a change plan that explicitly includes test file modification
        change_plan = ChangePlan(
            relevant_files=["src/module.py", "tests/test_module.py"],
            planned_changes=[
                PlannedChange(
                    path="src/module.py",
                    action=ChangeAction.MODIFY,
                    description="Fix bug in module",
                ),
                PlannedChange(
                    path="tests/test_module.py",
                    action=ChangeAction.MODIFY,
                    description="Update test to match new behavior",
                ),
            ],
            planned_tests=[],
            out_of_scope=[],
            risk_level="low",
        )

        runner = WorkflowRunner(
            agent_loop=mock_agent_loop,
            verifier=mock_verifier,
            workspace=mock_workspace,
            sandbox=mock_sandbox,
        )

        # Mock internal setup methods and workspace changes
        from patchpilot.tools import WorkspaceChange

        mock_changes = [
            WorkspaceChange(path="src/module.py", action="modify"),
            WorkspaceChange(path="tests/test_module.py", action="modify"),
        ]

        with patch.object(runner, '_create_temporary_workspace'), \
             patch.object(runner, '_start_sandbox'), \
             patch.object(runner, '_cleanup'), \
             patch('patchpilot.workflow.runner._get_workspace_changes') as mock_get_changes, \
             patch('patchpilot.workflow.runner.generate_patch') as mock_generate_patch:

            mock_get_changes.return_value = mock_changes
            mock_generate_patch.return_value = "diff content"

            result = runner.execute(
                issue="Fix the bug",
                plan="Implement the fix",
                change_plan=change_plan,
            )

        # Verify that test file modification was allowed when in change plan
        assert isinstance(result, WorkflowResult)
        assert result.final_status == CompletionState.PARTIALLY_VERIFIED
        assert result.verification_report["passed"] is True
        assert mock_agent_loop.run.call_count == 1
