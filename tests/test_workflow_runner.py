"""Tests for the Workflow Runner orchestration component."""

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from patchpilot.agent_loop import AgentLoop, AgentLoopError
from patchpilot.evidence.schema import CompletionState
from patchpilot.issue.schema import AcceptanceCriterion, NormalizedIssue
from patchpilot.planning.schema import (
    ChangeAction,
    ChangePlan,
    PlannedChange,
)
from patchpilot.planning.scope_gate import ScopeGateResult
from patchpilot.prompts import REPAIR_SYSTEM_PROMPT
from patchpilot.tools import WorkspaceChange
from patchpilot.verification.report import CheckReport, VerificationReport
from patchpilot.workflow.failure_classifier import FailureType
from patchpilot.workflow.runner import (
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
        mock_agent_loop.force_tool_selection = False
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
        mock_agent_loop.force_tool_selection = False
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
        mock_agent_loop.force_tool_selection = False
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

            result = runner.execute(
                issue="Fix the bug",
                plan="Implement the fix",
                change_plan=None,
            )

        assert isinstance(result, WorkflowResult)
        assert result.final_status == CompletionState.VERIFIED
        assert result.verification_report["passed"] is True
        assert mock_agent_loop.run.call_count == 1
        assert mock_verifier.call_count == 1

    def test_execute_with_repair_success(self):
        """Test execution that requires one repair attempt."""
        from patchpilot.workflow.result import WorkflowResult

        mock_agent_loop = Mock(spec=AgentLoop)
        mock_agent_loop.force_tool_selection = False
        mock_verifier = Mock()
        mock_workspace = Mock(spec=Workspace)
        mock_sandbox = Mock()

        # First verification fails, second succeeds
        failed_report = VerificationReport(passed=False)
        failed_report.failure_type = FailureType.CODE_FAILURE
        failed_report.add_check(
            CheckReport(
                method="pytest",
                phase="post_patch",
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
            mock_generate_patch.side_effect = [
                "diff before repair",
                "diff after repair",
                "diff after repair",
            ]

            result = runner.execute(
                issue="Fix the bug",
                plan="Implement the fix",
                change_plan=None,
            )

        assert isinstance(result, WorkflowResult)
        assert result.final_status == CompletionState.VERIFIED
        assert result.verification_report["passed"] is True
        assert mock_agent_loop.run.call_count == 2  # Initial + 1 repair
        assert mock_verifier.call_count == 2
        repair_call = mock_agent_loop.run.call_args_list[1]
        assert repair_call.kwargs["system_prompt"] == REPAIR_SYSTEM_PROMPT
        assert repair_call.kwargs["reset_state"] is True
        assert (
            "<current_patch>\ndiff before repair"
            in repair_call.kwargs["issue"]
        )
        assert "<latest_verification_failure>" in repair_call.kwargs["issue"]

    def test_repair_without_patch_delta_stops_before_verification(self):
        """A no-op repair should not spend another verifier attempt."""
        mock_agent_loop = Mock(spec=AgentLoop)
        mock_agent_loop.force_tool_selection = False
        failed_report = VerificationReport(passed=False)
        failed_report.failure_type = FailureType.CODE_FAILURE
        failed_report.add_check(
            CheckReport(
                method="pytest",
                phase="post_patch",
                level="LEVEL_2_TARGET_TESTS",
                command="python -m pytest tests/test_example.py",
                passed=False,
                exit_code=1,
                duration_seconds=0.1,
                failure_type="TEST_FAILURE",
            ),
        )
        mock_verifier = Mock(return_value=failed_report)
        runner = WorkflowRunner(
            agent_loop=mock_agent_loop,
            verifier=mock_verifier,
            workspace=Mock(spec=Workspace),
            sandbox=Mock(),
        )
        changes = [WorkspaceChange(path="src/file.py", action="modify")]

        with (
            patch.object(runner, "_create_temporary_workspace"),
            patch.object(runner, "_start_sandbox"),
            patch.object(runner, "_cleanup"),
            patch(
                "patchpilot.workflow.runner._get_workspace_changes",
                return_value=changes,
            ),
            patch(
                "patchpilot.workflow.runner.generate_patch",
                return_value="unchanged patch",
            ),
        ):
            result = runner.execute(
                issue="Fix the bug",
                plan="Implement the fix",
                change_plan=None,
            )

        assert result.final_status == CompletionState.FAILED
        assert mock_agent_loop.run.call_count == 2
        assert mock_verifier.call_count == 1

    def test_initial_agent_error_with_partial_patch_enters_repair(self):
        """Verify and repair a partial patch left by an AgentLoop error."""
        mock_agent_loop = Mock(spec=AgentLoop)
        mock_agent_loop.force_tool_selection = False
        mock_agent_loop.run.side_effect = [
            AgentLoopError("Agent stopped after tool failures"),
            "Repair complete",
        ]
        failed_report = VerificationReport(passed=False)
        failed_report.failure_type = FailureType.CODE_FAILURE
        failed_report.add_check(
            CheckReport(
                method="pytest",
                phase="post_patch",
                level="LEVEL_2_TARGET_TESTS",
                command="python -m pytest tests/test_example.py",
                passed=False,
                exit_code=1,
                duration_seconds=1.0,
                failure_type="TEST_FAILURE",
                summary={"relevant_output": "one assertion failed"},
            )
        )
        passed_report = VerificationReport(passed=True)
        mock_verifier = Mock(side_effect=[failed_report, passed_report])
        runner = WorkflowRunner(
            agent_loop=mock_agent_loop,
            verifier=mock_verifier,
            workspace=Mock(spec=Workspace),
            sandbox=Mock(),
        )
        changes = [WorkspaceChange(path="src/file.py", action="modify")]

        with (
            patch.object(runner, "_create_temporary_workspace"),
            patch.object(runner, "_start_sandbox"),
            patch.object(runner, "_cleanup"),
            patch.object(
                runner,
                "_check_repair_scope",
                return_value=ScopeGateResult(allowed=True),
            ),
            patch(
                "patchpilot.workflow.runner._get_workspace_changes",
                return_value=changes,
            ),
            patch(
                "patchpilot.workflow.runner.generate_patch",
                side_effect=[
                    "diff before repair",
                    "diff after repair",
                    "diff after repair",
                ],
            ),
        ):
            result = runner.execute(
                issue="Fix the bug",
                plan="Implement the fix",
                change_plan=None,
            )

        assert result.verification_report["passed"] is True
        assert mock_verifier.call_count == 2
        assert mock_agent_loop.run.call_count == 2
        assert (
            mock_agent_loop.run.call_args_list[1].kwargs["system_prompt"]
            == REPAIR_SYSTEM_PROMPT
        )

    def test_repair_agent_error_preserves_last_verification_report(self):
        """Test that repair failures expose the last deterministic report."""
        mock_agent_loop = Mock(spec=AgentLoop)
        mock_agent_loop.force_tool_selection = False
        mock_agent_loop.run.side_effect = [
            "Initial implementation complete",
            AgentLoopError("Repair agent stopped"),
        ]
        failed_report = VerificationReport(passed=False)
        failed_report.failure_type = FailureType.CODE_FAILURE
        failed_report.add_check(
            CheckReport(
                method="pytest",
                phase="post_patch",
                level="LEVEL_2_TARGET_TESTS",
                command="python -m pytest tests/test_example.py",
                passed=False,
                exit_code=1,
                duration_seconds=1.0,
                failure_type="TEST_FAILURE",
            )
        )
        mock_verifier = Mock(return_value=failed_report)
        runner = WorkflowRunner(
            agent_loop=mock_agent_loop,
            verifier=mock_verifier,
            workspace=Mock(spec=Workspace),
            sandbox=Mock(),
        )

        from patchpilot.tools import WorkspaceChange

        changes = [WorkspaceChange(path="src/file.py", action="modify")]
        with (
            patch.object(runner, "_create_temporary_workspace"),
            patch.object(runner, "_start_sandbox"),
            patch.object(runner, "_cleanup"),
            patch(
                "patchpilot.workflow.runner._get_workspace_changes",
                return_value=changes,
            ),
            patch(
                "patchpilot.workflow.runner.generate_patch",
                return_value="diff content",
            ),
            pytest.raises(WorkflowRunnerExecutionError) as exc_info,
        ):
            runner.execute(
                issue="Fix the bug",
                plan="Implement the fix",
                change_plan=None,
            )

        error = exc_info.value
        assert error.failure_type == "AGENT_ERROR"
        assert error.verification_report == failed_report.to_dict()
        assert error.verification_report["retry_count"] == 1

    def test_execute_unrecoverable_failure_stops_after_detection(self):
        """Test that unrecoverable failures stop the repair loop after detection."""
        from patchpilot.workflow.result import WorkflowResult

        mock_agent_loop = Mock(spec=AgentLoop)
        mock_agent_loop.force_tool_selection = False
        mock_verifier = Mock()
        mock_workspace = Mock(spec=Workspace)
        mock_sandbox = Mock()

        # First verification fails with recoverable error
        first_report = VerificationReport(passed=False)
        first_report.failure_type = FailureType.CODE_FAILURE
        first_report.add_check(
            CheckReport(
                method="pytest",
                phase="post_patch",
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
                method="pytest",
                phase="post_patch",
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
            mock_generate_patch.side_effect = [
                "diff before repair",
                "diff after repair",
                "diff after repair",
            ]

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
        mock_agent_loop.force_tool_selection = False
        mock_verifier = Mock()
        mock_workspace = Mock(spec=Workspace)
        mock_sandbox = Mock()

        # Same failure repeats
        failed_report = VerificationReport(passed=False)
        failed_report.failure_type = FailureType.CODE_FAILURE
        failed_report.add_check(
            CheckReport(
                method="pytest",
                phase="post_patch",
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
            mock_generate_patch.side_effect = [
                "diff before repair",
                "diff after repair",
                "diff after repair",
            ]

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
        mock_agent_loop.force_tool_selection = False
        mock_verifier = Mock()
        mock_workspace = Mock(spec=Workspace)
        mock_sandbox = Mock()

        # Different failures to avoid stall detection
        report1 = VerificationReport(passed=False)
        report1.failure_type = FailureType.CODE_FAILURE
        report1.add_check(
            CheckReport(
                method="pytest",
                phase="post_patch",
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
                method="pytest",
                phase="post_patch",
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
                method="pytest",
                phase="post_patch",
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
            mock_generate_patch.side_effect = [
                "diff before first repair",
                "diff after first repair",
                "diff after first repair",
                "diff after second repair",
                "diff after second repair",
            ]

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
            mock_agent_loop.force_tool_selection = False
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
        mock_agent_loop.force_tool_selection = False
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
        mock_agent_loop.force_tool_selection = False
        mock_agent_loop.tools = Mock()
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
            mock_agent_loop.tools.update_command_runner.assert_called_once_with(
                mock_sandbox
            )

    def test_start_sandbox_creates_when_none(self):
        """Test that sandbox is created when not provided."""
        mock_agent_loop = Mock(spec=AgentLoop)
        mock_agent_loop.force_tool_selection = False
        mock_agent_loop.tools = Mock()
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
                mock_agent_loop.tools.update_command_runner.assert_called_once_with(
                    mock_sandbox_instance
                )

    def test_execute_baseline_runs_raw_issue_and_verifies_once(self):
        """Test that the baseline omits planning and repair behavior."""
        mock_agent_loop = Mock(spec=AgentLoop)
        mock_agent_loop.force_tool_selection = False
        mock_agent_loop.max_rounds = 8
        mock_agent_loop.tools = Mock()
        mock_verifier = Mock(return_value=VerificationReport(passed=True))
        mock_workspace = Mock(spec=Workspace)
        runner = WorkflowRunner(
            agent_loop=mock_agent_loop,
            verifier=mock_verifier,
            workspace=mock_workspace,
            max_repair_attempts=0,
        )

        from patchpilot.tools import WorkspaceChange

        changes = [WorkspaceChange(path="module.py", action="modify")]
        with tempfile.TemporaryDirectory() as temp_dir, \
             patch.object(
                 runner,
                 "_create_temporary_workspace",
                 return_value=Path(temp_dir),
             ), \
             patch.object(runner, "_start_sandbox"), \
             patch.object(runner, "_cleanup"), \
             patch(
                 "patchpilot.workflow.runner._get_workspace_changes",
                 return_value=changes,
             ), \
             patch(
                 "patchpilot.workflow.runner.generate_patch",
                 return_value="diff content",
             ):
            result = runner.execute_baseline(
                issue="Fix the raw issue.",
                trace_path=Path(temp_dir) / "trace.jsonl",
            )

        mock_agent_loop.run.assert_called_once_with(
            issue="Fix the raw issue.",
            reset_state=True,
        )
        mock_verifier.assert_called_once()
        assert result.final_status == CompletionState.VERIFIED
        assert result.max_repairs == 0
        assert result.patch == "diff content"

    def test_execute_baseline_verifies_after_agent_limit(self):
        """Test that an agent limit still produces a deterministic result."""
        from patchpilot.agent_loop import AgentLoopLimitError

        mock_agent_loop = Mock(spec=AgentLoop)
        mock_agent_loop.force_tool_selection = False
        mock_agent_loop.max_rounds = 4
        mock_agent_loop.tools = Mock()
        mock_agent_loop.run.side_effect = AgentLoopLimitError("round limit")
        mock_verifier = Mock(return_value=VerificationReport(passed=True))
        runner = WorkflowRunner(
            agent_loop=mock_agent_loop,
            verifier=mock_verifier,
            workspace=Mock(spec=Workspace),
            max_repair_attempts=0,
        )

        with tempfile.TemporaryDirectory() as temp_dir, \
             patch.object(
                 runner,
                 "_create_temporary_workspace",
                 return_value=Path(temp_dir),
             ), \
             patch.object(runner, "_start_sandbox"), \
             patch.object(runner, "_cleanup"), \
             patch(
                 "patchpilot.workflow.runner._get_workspace_changes",
                 return_value=[],
             ), \
             patch(
                 "patchpilot.workflow.runner.generate_patch",
                 return_value="",
             ):
            result = runner.execute_baseline(
                issue="Fix the raw issue.",
                trace_path=Path(temp_dir) / "trace.jsonl",
            )

        mock_verifier.assert_called_once()
        assert result.final_status == CompletionState.FAILED

    def test_cleanup_stops_sandbox(self):
        """Test that cleanup stops the sandbox."""
        mock_agent_loop = Mock(spec=AgentLoop)
        mock_agent_loop.force_tool_selection = False
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
        mock_agent_loop.force_tool_selection = False
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
        mock_agent_loop.force_tool_selection = False
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
                method="pytest",
                phase="post_patch",
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
            current_patch="diff --git a/src/file.py b/src/file.py",
            current_changes=[
                WorkspaceChange(path="src/file.py", action="modify"),
            ],
        )

        assert "Fix the bug" in prompt
        assert "Implement the fix" in prompt
        assert "pytest tests/" in prompt
        assert "AssertionError" in prompt
        assert "test_example" in prompt
        assert "diff --git a/src/file.py" in prompt
        assert "src/file.py" in prompt

    def test_build_repair_prompt_without_failed_checks(self):
        """Test building repair prompt when no failed checks exist."""
        mock_agent_loop = Mock(spec=AgentLoop)
        mock_agent_loop.force_tool_selection = False
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

    def test_build_repair_prompt_uses_structured_failure_diff(self):
        """Structured repair context should omit repeated generic payloads."""
        runner = WorkflowRunner(
            agent_loop=Mock(spec=AgentLoop),
            verifier=Mock(),
            workspace=Mock(spec=Workspace),
        )
        from patchpilot.issue.schema import TaskConstraint

        issue = NormalizedIssue(
            title="Correct median",
            task_type="bug",
            problem_statement="Preserve odd and even median behavior.",
            acceptance_criteria=[
                AcceptanceCriterion(
                    id="AC-1",
                    description="Even inputs use both middle values.",
                ),
                AcceptanceCriterion(
                    id="AC-2",
                    description="Odd inputs preserve the middle value.",
                ),
            ],
            constraints=[
                TaskConstraint(
                    id="C-1",
                    description="Change only benchmark/statistics.py.",
                    kind="WRITE_SCOPE",
                ),
            ],
            ambiguous_points=[],
            expected_test_areas=[],
            implementation_notes=[],
        )
        change_plan = ChangePlan(
            planned_changes=[
                PlannedChange(
                    path="benchmark/statistics.py",
                    action=ChangeAction.MODIFY,
                    description="Handle odd and even input lengths.",
                    acceptance_criteria=["AC-1", "AC-2"],
                ),
            ],
            planned_tests=[],
            out_of_scope=["Do not modify tests."],
            risk_level="low",
        )
        failure_report = VerificationReport(passed=False)
        failure_report.add_check(
            CheckReport(
                method="pytest",
                phase="post_patch",
                level="LEVEL_2_TARGET_TESTS",
                command="python -m pytest tests/test_statistics.py -q",
                passed=False,
                exit_code=1,
                duration_seconds=0.1,
                failure_type="TEST_FAILURE",
                summary={
                    "failed_tests": ["test_median_for_odd_length_input"],
                    "relevant_output": "assert 3.0 == 5.0",
                },
                subject_ids=["AC-2"],
            ),
        )

        prompt = runner._build_repair_prompt(
            issue="SHOULD_NOT_BE_REPEATED",
            plan="SHOULD_NOT_BE_REPEATED",
            failure_report=failure_report,
            current_patch=(
                "diff --git a/benchmark/statistics.py "
                "b/benchmark/statistics.py\n"
                "+return average"
            ),
            current_changes=[
                WorkspaceChange(
                    path="benchmark/statistics.py",
                    action="modify",
                ),
            ],
            change_plan=change_plan,
            normalized_issue=issue,
        )

        assert "SHOULD_NOT_BE_REPEATED" not in prompt
        assert "benchmark/statistics.py" in prompt
        assert "Change only benchmark/statistics.py." in prompt
        assert "Out of scope: Do not modify tests." in prompt
        assert "AC-2: Odd inputs preserve the middle value." in prompt
        assert "AC-1: Even inputs" not in prompt
        assert "assert 3.0 == 5.0" in prompt

    def test_build_repair_prompt_bounds_large_patch(self):
        """Large patches should be truncated before entering model context."""
        runner = WorkflowRunner(
            agent_loop=Mock(spec=AgentLoop),
            verifier=Mock(),
            workspace=Mock(spec=Workspace),
        )

        prompt = runner._build_repair_prompt(
            issue="Fix the bug",
            plan="Modify src/file.py",
            failure_report=VerificationReport(passed=False),
            current_patch="a" * 20_000,
        )

        patch_section = prompt.split("<current_patch>\n", 1)[1].split(
            "\n</current_patch>",
            1,
        )[0]
        assert "repair context truncated" in patch_section
        assert len(patch_section) == 6_000


class TestRunWorkflow:
    """Tests for the convenience run_workflow function."""

    def test_run_workflow_convenience(self):
        """Test the convenience function with default parameters."""
        from patchpilot.workflow.result import WorkflowResult

        mock_agent_loop = Mock(spec=AgentLoop)
        mock_agent_loop.force_tool_selection = False
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
        mock_agent_loop.force_tool_selection = False
        mock_verifier = Mock()
        mock_workspace = Mock(spec=Workspace)
        mock_sandbox = Mock()

        # First verification fails, second succeeds
        failed_report = VerificationReport(passed=False)
        failed_report.failure_type = FailureType.CODE_FAILURE
        failed_report.add_check(
            CheckReport(
                method="pytest",
                phase="post_patch",
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
            mock_generate_patch.side_effect = [
                "diff before repair",
                "diff after repair",
                "diff after repair",
            ]

            result = runner.execute(
                issue="Fix the bug",
                plan="Implement the fix",
                change_plan=None,
            )

        assert isinstance(result, WorkflowResult)
        assert result.final_status == CompletionState.VERIFIED
        assert result.verification_report["passed"] is True
        assert mock_agent_loop.run.call_count == 2  # Initial + 1 repair
        assert mock_verifier.call_count == 2

    def test_scope_gate_blocks_forbidden_changes(self):
        """Test that scope gate blocks forbidden repair changes."""
        from patchpilot.workflow.result import WorkflowResult

        mock_agent_loop = Mock(spec=AgentLoop)
        mock_agent_loop.force_tool_selection = False
        mock_verifier = Mock()
        mock_workspace = Mock(spec=Workspace)
        mock_sandbox = Mock()

        # First verification fails
        failed_report = VerificationReport(passed=False)
        failed_report.failure_type = FailureType.CODE_FAILURE
        failed_report.add_check(
            CheckReport(
                method="pytest",
                phase="post_patch",
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
            mock_generate_patch.side_effect = [
                "diff before repair",
                "diff after repair",
                "diff after repair",
            ]

            result = runner.execute(
                issue="Fix the bug",
                plan="Implement the fix",
                change_plan=None,
            )

        assert isinstance(result, WorkflowResult)
        assert result.final_status == CompletionState.BLOCKED
        assert result.verification_report["passed"] is False
        assert result.verification_report["failure_type"] == "SCOPE_VIOLATION"
        assert result.retry_count == 1
        # Should attempt initial + 1 repair (then stop due to scope violation)
        assert mock_agent_loop.run.call_count == 2
        assert mock_verifier.call_count == 1  # Only initial verification, scope gate blocks re-verification

    def test_execute_no_changes_with_change_plan_raises_error(self):
        """Test that agent with change plan must modify at least one file."""
        mock_agent_loop = Mock(spec=AgentLoop)
        mock_agent_loop.force_tool_selection = False
        mock_verifier = Mock()
        mock_workspace = Mock(spec=Workspace)
        mock_sandbox = Mock()

        # Mock successful verification
        mock_report = VerificationReport(passed=True)
        mock_verifier.return_value = mock_report

        # Create a change plan with planned changes
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
            mock_generate_patch.side_effect = [
                "diff before repair",
                "diff after repair",
                "diff after repair",
            ]

            with pytest.raises(WorkflowRunnerExecutionError) as exc_info:
                runner.execute(
                    issue="Fix the bug",
                    plan="Implement the fix",
                    change_plan=change_plan,
                )

            assert "without modifying any repository files" in str(exc_info.value)

    def test_execute_no_changes_without_change_plan_blocked(self):
        """Test that agent without change plan is blocked when no file changes are made."""
        from patchpilot.workflow.result import WorkflowResult

        mock_agent_loop = Mock(spec=AgentLoop)
        mock_agent_loop.force_tool_selection = False
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
        # Should be BLOCKED due to NO_SOURCE_CHANGES
        assert result.final_status == CompletionState.BLOCKED
        assert result.verification_report["failure_type"] == "NO_SOURCE_CHANGES"
        assert mock_agent_loop.run.call_count == 1
        assert mock_verifier.call_count == 1

    def test_scope_gate_blocks_cicd_changes(self):
        """Test that scope gate blocks CI/CD repair changes."""
        from patchpilot.workflow.result import WorkflowResult

        mock_agent_loop = Mock(spec=AgentLoop)
        mock_agent_loop.force_tool_selection = False
        mock_verifier = Mock()
        mock_workspace = Mock(spec=Workspace)
        mock_sandbox = Mock()

        # First verification fails
        failed_report = VerificationReport(passed=False)
        failed_report.failure_type = FailureType.CODE_FAILURE
        failed_report.add_check(
            CheckReport(
                method="pytest",
                phase="post_patch",
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
            mock_generate_patch.side_effect = [
                "diff before repair",
                "diff after repair",
                "diff after repair",
            ]

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
        mock_agent_loop.force_tool_selection = False
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
            mock_get_changes.return_value = []
            mock_generate_patch.return_value = "diff"

            result = runner._get_modified_files()
            assert result == []


class TestWorkflowRunnerConfigurableRepairLimit:
    """Tests for configurable repair limit functionality."""

    def test_workflow_runner_uses_configured_repair_limit(self):
        """Test that WorkflowRunner respects configured max_repairs parameter."""
        from patchpilot.workflow.result import WorkflowResult

        mock_agent_loop = Mock(spec=AgentLoop)
        mock_agent_loop.force_tool_selection = False
        mock_verifier = Mock()
        mock_workspace = Mock(spec=Workspace)
        mock_sandbox = Mock()

        # Create different failures to avoid stall detection
        failed_reports = []
        for i in range(5):  # Create 5 different failures
            report = VerificationReport(passed=False)
            report.failure_type = FailureType.CODE_FAILURE
            report.add_check(
                CheckReport(
                    method="pytest",
                    phase="post_patch",
                    level="standard",
                    command="pytest tests/",
                    passed=False,
                    exit_code=1,
                    duration_seconds=1.0,
                    failure_type=f"ErrorType{i}",
                    summary={"error_type": f"ErrorType{i}", "failed_tests": [f"test_{i}"]},
                )
            )
            failed_reports.append(report)

        mock_verifier.side_effect = failed_reports

        # Create runner with custom max_repair_attempts = 1
        runner = WorkflowRunner(
            agent_loop=mock_agent_loop,
            verifier=mock_verifier,
            workspace=mock_workspace,
            sandbox=mock_sandbox,
            max_repair_attempts=1,  # Custom limit
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
            mock_generate_patch.side_effect = [
                "diff before repair",
                "diff after repair",
                "diff after repair",
            ]

            result = runner.execute(
                issue="Fix the bug",
                plan="Implement the fix",
                change_plan=None,
            )

        assert isinstance(result, WorkflowResult)
        assert result.verification_report["passed"] is False
        # Should attempt initial + 1 repair (custom limit) = 2 total
        assert mock_agent_loop.run.call_count == 2
        assert mock_verifier.call_count == 2

    def test_workflow_runner_default_repair_limit(self):
        """Test that WorkflowRunner uses default MAX_REPAIR_ATTEMPTS when not configured."""
        from patchpilot.workflow.result import WorkflowResult

        mock_agent_loop = Mock(spec=AgentLoop)
        mock_agent_loop.force_tool_selection = False
        mock_verifier = Mock()
        mock_workspace = Mock(spec=Workspace)
        mock_sandbox = Mock()

        # Create different failures to test default limit
        failed_reports = []
        for i in range(4):  # Create 4 different failures
            report = VerificationReport(passed=False)
            report.failure_type = FailureType.CODE_FAILURE
            report.add_check(
                CheckReport(
                    method="pytest",
                    phase="post_patch",
                    level="standard",
                    command="pytest tests/",
                    passed=False,
                    exit_code=1,
                    duration_seconds=1.0,
                    failure_type=f"ErrorType{i}",
                    summary={"error_type": f"ErrorType{i}", "failed_tests": [f"test_{i}"]},
                )
            )
            failed_reports.append(report)

        mock_verifier.side_effect = failed_reports

        # Create runner without custom max_repair_attempts (should use default)
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
            mock_generate_patch.side_effect = [
                "diff before first repair",
                "diff after first repair",
                "diff after first repair",
                "diff after second repair",
                "diff after second repair",
            ]

            result = runner.execute(
                issue="Fix the bug",
                plan="Implement the fix",
                change_plan=None,
            )

        assert isinstance(result, WorkflowResult)
        assert result.verification_report["passed"] is False
        # Should attempt initial + MAX_REPAIR_ATTEMPTS (2) = 3 total
        assert mock_agent_loop.run.call_count == 3
        assert mock_verifier.call_count == 3


class TestWorkflowRunnerExitCodes:
    """Tests for exit code behavior based on completion status."""

    def test_partially_verified_returns_nonzero_exit_code(self):
        """Test that PARTIALLY_VERIFIED status results in non-zero exit code."""
        from patchpilot.evidence.schema import AcceptanceEvidence, EvidenceStatus
        from patchpilot.workflow.result import WorkflowResult

        # Create a workflow result with PARTIALLY_VERIFIED status
        result = WorkflowResult(
            run_id="test-run-id",
            final_status=CompletionState.PARTIALLY_VERIFIED,
            changed_files=["src/file.py"],
            acceptance_evidence=[
                AcceptanceEvidence(
                    criterion_id="AC-1",
                    description="Test criterion",
                    status=EvidenceStatus.PASS,
                    explanation="Test explanation",
                ),
                AcceptanceEvidence(
                    criterion_id="AC-2",
                    description="Unverified criterion",
                    status=EvidenceStatus.UNVERIFIED,
                    explanation="No test coverage",
                ),
            ],
            verification_report={"passed": True},
            patch="diff content",
        )

        # Convert to run summary and check exit code
        summary = result.to_run_summary(
            task_id="test-task",
            base_commit="abc123",
            model="test-model",
            output_dir="artifacts",
        )

        # PARTIALLY_VERIFIED should result in exit code 1
        assert summary.exit_code == 1
        assert summary.final_status == "PARTIALLY_VERIFIED"

    def test_verified_returns_zero_exit_code(self):
        """Test that VERIFIED status results in zero exit code."""
        from patchpilot.evidence.schema import AcceptanceEvidence, EvidenceStatus
        from patchpilot.workflow.result import WorkflowResult

        # Create a workflow result with VERIFIED status
        result = WorkflowResult(
            run_id="test-run-id",
            final_status=CompletionState.VERIFIED,
            changed_files=["src/file.py"],
            acceptance_evidence=[
                AcceptanceEvidence(
                    criterion_id="AC-1",
                    description="Test criterion",
                    status=EvidenceStatus.PASS,
                    explanation="Test explanation",
                ),
            ],
            verification_report={"passed": True},
            patch="diff content",
        )

        # Convert to run summary and check exit code
        summary = result.to_run_summary(
            task_id="test-task",
            base_commit="abc123",
            model="test-model",
            output_dir="artifacts",
        )

        # VERIFIED should result in exit code 0
        assert summary.exit_code == 0
        assert summary.final_status == "VERIFIED"

    def test_failed_returns_nonzero_exit_code(self):
        """Test that FAILED status results in non-zero exit code."""
        from patchpilot.evidence.schema import AcceptanceEvidence, EvidenceStatus
        from patchpilot.workflow.result import WorkflowResult

        # Create a workflow result with FAILED status
        result = WorkflowResult(
            run_id="test-run-id",
            final_status=CompletionState.FAILED,
            changed_files=["src/file.py"],
            acceptance_evidence=[
                AcceptanceEvidence(
                    criterion_id="AC-1",
                    description="Failed criterion",
                    status=EvidenceStatus.FAIL,
                    explanation="Test failed",
                ),
            ],
            verification_report={"passed": False},
            patch="diff content",
        )

        # Convert to run summary and check exit code
        summary = result.to_run_summary(
            task_id="test-task",
            base_commit="abc123",
            model="test-model",
            output_dir="artifacts",
        )

        # FAILED should result in exit code 1
        assert summary.exit_code == 1
        assert summary.final_status == "FAILED"

    def test_no_source_changes_rejects_final_completion(self):
        """Test that final completion is rejected when no source changes are made."""
        mock_agent_loop = Mock(spec=AgentLoop)
        mock_agent_loop.force_tool_selection = False
        mock_verifier = Mock()
        mock_workspace = Mock(spec=Workspace)
        mock_sandbox = Mock()

        # Mock a passing verification report
        passed_report = VerificationReport(passed=True)
        mock_verifier.return_value = passed_report

        runner = WorkflowRunner(
            agent_loop=mock_agent_loop,
            verifier=mock_verifier,
            workspace=mock_workspace,
            sandbox=mock_sandbox,
        )

        # Mock empty changes (no source modifications)
        with (
            patch.object(runner, "_create_temporary_workspace"),
            patch.object(runner, "_start_sandbox"),
            patch.object(runner, "_cleanup"),
            patch(
                "patchpilot.workflow.runner._get_workspace_changes",
                return_value=[],  # No changes
            ),
            patch(
                "patchpilot.workflow.runner.generate_patch",
                return_value="",
            ),
        ):
            result = runner.execute(
                issue="Fix the bug",
                plan="Implement the fix",
                change_plan=None,
            )

        # Should be BLOCKED due to NO_SOURCE_CHANGES
        assert result.final_status == CompletionState.BLOCKED
        assert result.verification_report["failure_type"] == "NO_SOURCE_CHANGES"

    def test_blocked_returns_nonzero_exit_code(self):
        """Test that BLOCKED status results in non-zero exit code."""
        from patchpilot.workflow.result import WorkflowResult

        # Create a workflow result with BLOCKED status
        result = WorkflowResult(
            run_id="test-run-id",
            final_status=CompletionState.BLOCKED,
            changed_files=[],
            acceptance_evidence=[],
            verification_report={"passed": False, "failure_type": "SCOPE_VIOLATION"},
            patch="",
        )

        # Convert to run summary and check exit code
        summary = result.to_run_summary(
            task_id="test-task",
            base_commit="abc123",
            model="test-model",
            output_dir="artifacts",
        )

        # BLOCKED should result in exit code 1
        assert summary.exit_code == 1
        assert summary.final_status == "BLOCKED"
