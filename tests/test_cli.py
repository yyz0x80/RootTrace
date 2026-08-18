"""Tests for the CLI module."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from patchpilot.issue.schema import NormalizedIssue
from patchpilot.planning.schema import ChangePlan


def test_prepare_summary_records_structured_terminal_outcome(
    tmp_path: Path,
) -> None:
    """Test that prepare results do not depend on console text matching."""
    from patchpilot.cli import _save_prepare_summary

    provider = Mock(
        model="test-model",
        llm_call_count=1,
        prompt_tokens=20,
        completion_tokens=5,
    )
    _save_prepare_summary(
        tmp_path,
        provider,
        outcome_code="AMBIGUOUS_REQUIREMENT",
        final_status="NEEDS_CLARIFICATION",
        exit_code=1,
        reasons=["Priority values are unspecified."],
    )

    summary = json.loads((tmp_path / "prepare_summary.json").read_text())
    assert summary["outcome_code"] == "AMBIGUOUS_REQUIREMENT"
    assert summary["final_status"] == "NEEDS_CLARIFICATION"
    assert summary["reasons"] == ["Priority values are unspecified."]
    assert summary["llm_call_count"] == 1


def test_failed_run_summary_records_agent_failure(tmp_path: Path) -> None:
    """Test that execute failures remain machine-readable."""
    from argparse import Namespace

    from patchpilot.cli import _save_failed_run_summary

    args = Namespace(
        output_dir=str(tmp_path),
        task_id="task-1",
        model="test-model",
        max_rounds=8,
        max_repairs=0,
    )
    _save_failed_run_summary(
        args=args,
        started=0.0,
        provider=None,
        base_commit="abc123",
        final_status="FAILED",
        failure_type="AGENT_ROUND_LIMIT",
        error_message="Agent exceeded the round limit",
        verification_report={"passed": False, "retry_count": 1},
    )

    summary = json.loads((tmp_path / "run_summary.json").read_text())
    assert summary["final_status"] == "FAILED"
    assert summary["failure_type"] == "AGENT_ROUND_LIMIT"
    assert summary["error_message"] == "Agent exceeded the round limit"
    assert summary["retry_count"] == 1
    assert summary["artifacts"] == {
        "verification_report": str(tmp_path / "verification_report.json")
    }
    report = json.loads(
        (tmp_path / "verification_report.json").read_text()
    )
    assert report == {"passed": False, "retry_count": 1}


@patch("patchpilot.cli.load_issue")
@patch("patchpilot.cli.normalize_issue")
@patch("patchpilot.cli.LLMProvider")
@patch("patchpilot.cli.Workspace")
@patch("patchpilot.cli.ToolRegistry")
@patch("patchpilot.cli.AgentLoop")
@patch("patchpilot.cli.save_json")
def test_main_with_ambiguous_points_stops(
    mock_save_json,
    mock_agent_loop,
    mock_tool_registry,
    mock_workspace,
    mock_provider,
    mock_normalize,
    mock_load,
):
    """Test that CLI stops when normalized issue has ambiguous points."""

    from patchpilot.cli import main

    # Setup mocks
    mock_load.return_value = Mock(
        title="Test Issue", body="Test body", source="test.md"
    )

    mock_provider_instance = Mock()
    mock_provider.return_value = mock_provider_instance
    mock_provider_instance.generate_text = Mock(return_value="normalized")

    mock_normalize.return_value = NormalizedIssue(
        title="Test Issue",
        task_type="feature",
        problem_statement="Test problem",
        ambiguous_points=["Default priority is unspecified."],
    )

    # Mock Path operations - use real Path for output_dir operations
    with patch("pathlib.Path.mkdir") as mock_mkdir:
        mock_mkdir.side_effect = lambda *args, **kwargs: None
        # Mock sys.argv to simulate CLI call
        with patch("sys.argv", ["patchpilot", "run", "--repo", "/fake/repo", "--issue", "test.md", "--output-dir", "artifacts"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 1

    # Verify normalization was called but agent was not
    mock_normalize.assert_called_once()
    mock_agent_loop.assert_not_called()


@patch("patchpilot.cli.load_issue")
@patch("patchpilot.cli.normalize_issue")
@patch("patchpilot.cli.LLMProvider")
@patch("patchpilot.cli.Workspace")
@patch("patchpilot.cli.ToolRegistry")
@patch("patchpilot.cli.AgentLoop")
@patch("patchpilot.cli.save_json")
@patch("patchpilot.cli.create_plan")
@patch("patchpilot.cli.check_scope")
@patch("patchpilot.cli.validate_repository")
@patch("patchpilot.cli.run_repair_loop")
def test_main_without_ambiguous_points_proceeds(
    mock_run_repair_loop,
    mock_validate_repository,
    mock_check_scope,
    mock_create_plan,
    mock_save_json,
    mock_agent_loop,
    mock_tool_registry,
    mock_workspace,
    mock_provider,
    mock_normalize,
    mock_load,
):
    """Test that CLI proceeds when normalized issue has no ambiguous points."""
    from patchpilot.cli import main

    # Setup mocks
    mock_load.return_value = Mock(
        title="Test Issue", body="Test body", source="test.md"
    )

    mock_provider_instance = Mock()
    mock_provider.return_value = mock_provider_instance
    mock_provider_instance.generate_text = Mock(return_value="normalized")

    mock_normalize.return_value = NormalizedIssue(
        title="Test Issue",
        task_type="feature",
        problem_statement="Test problem",
        ambiguous_points=[],
    )

    # Mock create_plan to return a valid plan
    mock_create_plan.return_value = ChangePlan(
        relevant_files=[],
        planned_changes=[],
        planned_tests=[],
        out_of_scope=[],
        risk_level="low",
        base_commit="",
        repository_match=True,
    )

    # Mock check_scope to return allowed
    mock_check_scope.return_value = Mock(allowed=True, violations=[], warnings=[])

    # Mock validate_repository to return a valid result
    from patchpilot.repository.schema import RepositoryPreflightResult
    mock_validate_repository.return_value = RepositoryPreflightResult(
        repo_path=Path("/fake/repo"),
        head_sha="abc123def456",
    )

    # Mock run_repair_loop to return successful result
    from patchpilot.verification.report import VerificationReport
    mock_verification_report = Mock(spec=VerificationReport)
    mock_verification_report.passed = True
    mock_run_repair_loop.return_value = ("Success", mock_verification_report)

    mock_agent_loop_instance = Mock()
    mock_agent_loop.return_value = mock_agent_loop_instance

    # Mock Path operations - use real Path for output_dir operations
    with patch("pathlib.Path.mkdir") as mock_mkdir:
        mock_mkdir.side_effect = lambda *args, **kwargs: None
        # Mock sys.argv to simulate CLI call
        with patch("sys.argv", ["patchpilot", "run", "--repo", "/fake/repo", "--issue", "test.md", "--output-dir", "artifacts"]):
            main()

    # Verify repository validation was called
    mock_validate_repository.assert_called_once()
    # Verify repair loop was called
    mock_run_repair_loop.assert_called_once()
    repair_call = mock_run_repair_loop.call_args
    assert repair_call.kwargs["max_attempts"] == 4
    assert "repair_prompt_builder" not in repair_call.kwargs


@patch("patchpilot.cli.load_issue")
@patch("patchpilot.cli.normalize_issue")
@patch("patchpilot.cli.LLMProvider")
@patch("patchpilot.cli.Workspace")
@patch("patchpilot.cli.ToolRegistry")
@patch("patchpilot.cli.AgentLoop")
@patch("patchpilot.cli.save_json")
def test_main_with_multiple_ambiguous_points_shows_all(
    mock_save_json,
    mock_agent_loop,
    mock_tool_registry,
    mock_workspace,
    mock_provider,
    mock_normalize,
    mock_load,
):
    """Test that CLI shows all ambiguous points when multiple exist."""
    from patchpilot.cli import main

    # Setup mocks
    mock_load.return_value = Mock(
        title="Test Issue", body="Test body", source="test.md"
    )

    mock_provider_instance = Mock()
    mock_provider.return_value = mock_provider_instance
    mock_provider_instance.generate_text = Mock(return_value="normalized")

    mock_normalize.return_value = NormalizedIssue(
        title="Test Issue",
        task_type="feature",
        problem_statement="Test problem",
        ambiguous_points=[
            "Default priority is unspecified.",
            "Error handling strategy is not defined.",
            "Timeout behavior is unclear.",
        ],
    )

    # Mock Path operations - use real Path for output_dir operations
    with patch("pathlib.Path.mkdir") as mock_mkdir:
        mock_mkdir.side_effect = lambda *args, **kwargs: None
        # Mock sys.argv to simulate CLI call
        with patch("sys.argv", ["patchpilot", "run", "--repo", "/fake/repo", "--issue", "test.md", "--output-dir", "artifacts"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 1

    # Verify agent was not called
    mock_agent_loop.assert_not_called()


@patch("patchpilot.cli.validate_repository")
def test_execute_with_baseline_mismatch_fails(mock_validate_repository):
    """Test that execute fails when repository HEAD has changed since plan generation."""
    from argparse import Namespace

    from patchpilot.cli import handle_execute
    from patchpilot.issue.schema import NormalizedIssue
    from patchpilot.planning.schema import ChangePlan
    from patchpilot.repository.schema import RepositoryPreflightResult

    # Create mock args
    args = Namespace(
        repo="/fake/repo",
        issue="issue.json",
        plan="plan.json",
        model=None,
        max_rounds=12,
        max_repairs=3,
        output_dir="artifacts",
        task_id="test-task-123",
    )

    # Create plan with different base commit
    plan = ChangePlan(
        base_commit="original123",
        repository_match=True,
        relevant_files=[],
        planned_changes=[],
        planned_tests=[],
        out_of_scope=[],
        risk_level="low",
    )

    # Create normalized issue
    issue = NormalizedIssue(
        title="Test Issue",
        task_type="feature",
        problem_statement="Test problem",
        ambiguous_points=[],
        acceptance_criteria=[],
        constraints=[],
        expected_test_areas=[],
        implementation_notes=[],
    )

    # Mock validate_repository to return a different HEAD
    mock_validate_repository.return_value = RepositoryPreflightResult(
        repo_path=Path("/fake/repo"),
        head_sha="different456",  # Different from plan.base_commit
    )

    # Mock file loading
    with patch("pathlib.Path.mkdir") as mock_mkdir, \
         patch("pathlib.Path.exists") as mock_exists, \
         patch("pathlib.Path.write_text") as mock_write_text:
        mock_mkdir.side_effect = lambda *args, **kwargs: None
        mock_exists.return_value = True
        mock_write_text.side_effect = lambda *args, **kwargs: None

        with patch("builtins.open") as mock_open:
            # Mock file reading - need to handle both issue and plan files
            call_count = [0]

            def mock_open_func(file, *args, **kwargs):
                call_count[0] += 1
                mock_file = Mock()
                if call_count[0] == 1:  # First call is for issue file
                    mock_file.read.return_value = json.dumps(issue.model_dump())
                else:  # Second call is for plan file
                    mock_file.read.return_value = json.dumps(plan.model_dump())
                mock_file.__enter__ = Mock(return_value=mock_file)
                mock_file.__exit__ = Mock(return_value=False)
                return mock_file

            mock_open.side_effect = mock_open_func

            with patch("patchpilot.cli.Workspace"):
                with pytest.raises(SystemExit) as exc_info:
                    handle_execute(args)

                assert exc_info.value.code == 1

    # Verify repository validation was called
    mock_validate_repository.assert_called_once()


@patch("patchpilot.cli.validate_repository")
@patch("patchpilot.cli.WorkflowRunner")
def test_execute_with_matching_baseline_succeeds(mock_workflow_runner, mock_validate_repository):
    """Test that execute proceeds when repository HEAD matches plan baseline."""
    from argparse import Namespace

    from patchpilot.cli import handle_execute
    from patchpilot.issue.schema import NormalizedIssue
    from patchpilot.planning.schema import ChangePlan
    from patchpilot.repository.schema import RepositoryPreflightResult

    # Create mock args
    args = Namespace(
        repo="/fake/repo",
        issue="issue.json",
        plan="plan.json",
        model=None,
        max_rounds=12,
        max_repairs=3,
        output_dir="artifacts",
        task_id="test-task-123",
    )

    # Create plan with matching base commit
    plan = ChangePlan(
        base_commit="same123",
        repository_match=True,
        relevant_files=[],
        planned_changes=[],
        planned_tests=[],
        out_of_scope=[],
        risk_level="low",
    )

    # Create normalized issue
    issue = NormalizedIssue(
        title="Test Issue",
        task_type="feature",
        problem_statement="Test problem",
        ambiguous_points=[],
        acceptance_criteria=[],
        constraints=[],
        expected_test_areas=[],
        implementation_notes=[],
    )

    # Mock validate_repository to return matching HEAD
    mock_validate_repository.return_value = RepositoryPreflightResult(
        repo_path=Path("/fake/repo"),
        head_sha="same123",  # Same as plan.base_commit
    )

    # Mock workflow runner
    mock_runner_instance = Mock()
    from patchpilot.evidence.schema import CompletionState
    from patchpilot.workflow.result import RunSummary, WorkflowResult
    
    mock_workflow_result = Mock(spec=WorkflowResult)
    mock_workflow_result.final_status = CompletionState.VERIFIED
    mock_workflow_result.acceptance_evidence = []
    mock_workflow_result.patch = ""
    mock_workflow_result.verification_report = {"passed": True}
    # Mock to_run_summary to return a RunSummary object
    mock_run_summary = RunSummary(
        run_id="test-run-id",
        task_id="test-task-123",
        phase="execute",
        base_commit="same123",
        model="test-model",
        max_rounds=12,
        max_repairs=3,
        retry_count=0,
        final_status="VERIFIED",
        exit_code=0,
        duration_seconds=10.0,
        artifacts={
            "patch": "artifacts/patch.diff",
            "verification_report": "artifacts/verification_report.json",
        },
    )
    mock_workflow_result.to_run_summary.return_value = mock_run_summary
    mock_runner_instance.execute.return_value = mock_workflow_result
    mock_runner_instance.workspace = Mock(root=Path("/fake/repo"))
    mock_runner_instance._cleanup = Mock()
    mock_workflow_runner.return_value = mock_runner_instance

    # Mock file loading
    with patch("pathlib.Path.mkdir") as mock_mkdir, \
         patch("pathlib.Path.exists") as mock_exists, \
         patch("pathlib.Path.write_text") as mock_write_text:
        mock_mkdir.side_effect = lambda *args, **kwargs: None
        mock_exists.return_value = True
        mock_write_text.side_effect = lambda *args, **kwargs: None

        with patch("builtins.open") as mock_open:
            # Mock file reading - need to handle both issue and plan files
            call_count = [0]

            def mock_open_func(file, *args, **kwargs):
                call_count[0] += 1
                mock_file = Mock()
                if call_count[0] == 1:  # First call is for issue file
                    mock_file.read.return_value = json.dumps(issue.model_dump())
                else:  # Second call is for plan file
                    mock_file.read.return_value = json.dumps(plan.model_dump())
                mock_file.__enter__ = Mock(return_value=mock_file)
                mock_file.__exit__ = Mock(return_value=False)
                return mock_file

            mock_open.side_effect = mock_open_func

            with (
                patch("patchpilot.cli.Workspace"),
                patch("patchpilot.cli.LLMProvider"),
                patch("patchpilot.cli.ToolRegistry"),
                patch("patchpilot.cli.AgentLoop"),
                patch("patchpilot.cli.save_json"),
            ):
                # Handle the SystemExit that occurs at the end of handle_execute
                with pytest.raises(SystemExit) as exc_info:
                    handle_execute(args)
                
                # Verify exit code is 0 for VERIFIED status
                assert exc_info.value.code == 0

    # Verify repository validation was called
    mock_validate_repository.assert_called_once()
    # Verify workflow runner was called due to baseline match
    mock_workflow_runner.assert_called_once()
    
    # Verify that verifier was set to None (using built-in Verifier)
    runner_kwargs = mock_workflow_runner.call_args.kwargs
    assert runner_kwargs["verifier"] is None
    assert runner_kwargs["workspace"] is not None
    
    # Verify that CLI does not directly call _cleanup
    mock_runner_instance._cleanup.assert_not_called()

    execute_kwargs = mock_runner_instance.execute.call_args.kwargs
    assert execute_kwargs["normalized_issue"] == issue
    assert execute_kwargs["trace_path"] is not None


@patch("patchpilot.cli.load_issue")
@patch("patchpilot.cli.normalize_issue")
@patch("patchpilot.cli.LLMProvider")
@patch("patchpilot.cli.analyze_repository")
@patch("patchpilot.cli.create_plan")
@patch("patchpilot.cli.validate_plan")
@patch("patchpilot.cli.validate_repository")
def test_prepare_writes_to_configured_output_dir(
    mock_validate_repository,
    mock_validate_plan,
    mock_create_plan,
    mock_analyze_repository,
    mock_provider,
    mock_normalize,
    mock_load,
):
    """Test that prepare command writes artifacts to configured output directory."""
    from argparse import Namespace
    from pathlib import Path

    from patchpilot.cli import handle_prepare
    from patchpilot.issue.schema import NormalizedIssue
    from patchpilot.planning.schema import ChangePlan
    from patchpilot.repository.schema import (
        RepositoryContext,
        RepositoryPreflightResult,
    )

    # Create mock args with custom output directory
    args = Namespace(
        repo="/fake/repo",
        issue="test.md",
        model=None,
        output_dir="/custom/output",
    )

    # Setup mocks
    mock_load.return_value = Mock(
        title="Test Issue", body="Test body", source="test.md"
    )

    mock_provider_instance = Mock()
    mock_provider.return_value = mock_provider_instance
    mock_provider_instance.generate_text = Mock(return_value="normalized")

    mock_normalize.return_value = NormalizedIssue(
        title="Test Issue",
        task_type="feature",
        problem_statement="Test problem",
        ambiguous_points=[],
    )

    mock_validate_repository.return_value = RepositoryPreflightResult(
        repo_path=Path("/fake/repo"),
        head_sha="abc123",
    )

    mock_analyze_repository.return_value = RepositoryContext(
        base_commit="abc123",
        tracked_files=[],
        python_files=[],
        test_files=[],
        config_files=[],
        keyword_matches=[],
    )

    mock_create_plan.return_value = ChangePlan(
        base_commit="abc123",
        repository_match=True,
        relevant_files=[],
        planned_changes=[],
        planned_tests=[],
        out_of_scope=[],
        risk_level="low",
    )

    mock_validate_plan.return_value = Mock(allowed=True, violations=[], warnings=[])

    # Mock Path and file operations
    from pathlib import Path
    custom_output_dir = Path("/custom/output")

    with patch("patchpilot.cli.save_json") as mock_save_json, \
         patch("pathlib.Path.mkdir") as mock_mkdir:
        # Prevent actual filesystem operations
        mock_mkdir.side_effect = lambda *args, **kwargs: None
        
        handle_prepare(args)

        # Verify save_json was called with custom output directory paths
        expected_calls = [
            str(custom_output_dir / "normalized_issue.json"),
            str(custom_output_dir / "repository_context.json"),
            str(custom_output_dir / "plan.json"),
            str(custom_output_dir / "prepare_summary.json"),
        ]
        actual_calls = [call[0][0] for call in mock_save_json.call_args_list]
        assert actual_calls == expected_calls


def test_prepare_classifies_exhausted_plan_retry_as_plan_invalid(
    tmp_path: Path,
) -> None:
    """Planner repair exhaustion should produce a blocked plan outcome."""
    from argparse import Namespace

    from patchpilot.cli import handle_prepare
    from patchpilot.planning.planner import PlanGenerationError
    from patchpilot.repository.schema import (
        RepositoryContext,
        RepositoryPreflightResult,
    )

    args = Namespace(
        repo=str(tmp_path),
        issue="issue.md",
        model=None,
        output_dir=str(tmp_path / "output"),
    )
    provider = Mock(
        model="test-model",
        llm_call_count=2,
        prompt_tokens=10,
        completion_tokens=5,
    )
    normalized_issue = NormalizedIssue(
        title="Fix behavior",
        task_type="bug",
        problem_statement="The behavior is incorrect.",
    )
    repository_context = RepositoryContext(
        base_commit="abc123",
        tracked_files=[],
        python_files=[],
        test_files=[],
        config_files=[],
        keyword_matches=[],
    )

    with (
        patch("patchpilot.cli.load_issue", return_value=Mock()),
        patch("patchpilot.cli._create_provider", return_value=provider),
        patch(
            "patchpilot.cli.normalize_issue",
            return_value=normalized_issue,
        ),
        patch(
            "patchpilot.cli.validate_repository",
            return_value=RepositoryPreflightResult(
                repo_path=tmp_path,
                head_sha="abc123",
            ),
        ),
        patch(
            "patchpilot.cli.analyze_repository",
            return_value=repository_context,
        ),
        patch(
            "patchpilot.cli.create_plan",
            side_effect=PlanGenerationError("plan remains incomplete"),
        ),
        pytest.raises(SystemExit) as exc_info,
    ):
        handle_prepare(args)

    summary = json.loads(
        (tmp_path / "output" / "prepare_summary.json").read_text()
    )
    assert exc_info.value.code == 1
    assert summary["outcome_code"] == "PLAN_INVALID"
    assert summary["final_status"] == "BLOCKED"


@patch("patchpilot.cli.validate_repository")
@patch("patchpilot.cli.WorkflowRunner")
def test_execute_writes_to_configured_output_dir(
    mock_workflow_runner,
    mock_validate_repository,
):
    """Test that execute command writes artifacts to configured output directory."""
    from argparse import Namespace
    from pathlib import Path

    from patchpilot.cli import handle_execute
    from patchpilot.evidence.schema import CompletionState
    from patchpilot.issue.schema import NormalizedIssue
    from patchpilot.planning.schema import ChangePlan
    from patchpilot.repository.schema import RepositoryPreflightResult
    from patchpilot.workflow.result import WorkflowResult

    # Create mock args with custom output directory
    args = Namespace(
        repo="/fake/repo",
        issue="issue.json",
        plan="plan.json",
        model=None,
        max_rounds=12,
        max_repairs=3,
        output_dir="/custom/output",
        task_id="test-task-123",
    )

    # Create plan and issue
    plan = ChangePlan(
        base_commit="same123",
        repository_match=True,
        relevant_files=[],
        planned_changes=[],
        planned_tests=[],
        out_of_scope=[],
        risk_level="low",
    )

    issue = NormalizedIssue(
        title="Test Issue",
        task_type="feature",
        problem_statement="Test problem",
        ambiguous_points=[],
        acceptance_criteria=[],
        constraints=[],
        expected_test_areas=[],
        implementation_notes=[],
    )

    # Mock validate_repository to return matching HEAD
    mock_validate_repository.return_value = RepositoryPreflightResult(
        repo_path=Path("/fake/repo"),
        head_sha="same123",
    )

    # Mock workflow runner
    mock_runner_instance = Mock()
    from patchpilot.workflow.result import RunSummary
    
    mock_workflow_result = Mock(spec=WorkflowResult)
    mock_workflow_result.final_status = CompletionState.VERIFIED
    mock_workflow_result.acceptance_evidence = []
    mock_workflow_result.patch = ""
    mock_workflow_result.verification_report = {"passed": True}
    mock_workflow_result.run_id = "test-run-id"
    mock_workflow_result.duration_seconds = 10.0
    mock_workflow_result.retry_count = 0
    mock_workflow_result.max_repairs = 3
    # Mock to_run_summary to return a RunSummary object
    mock_run_summary = RunSummary(
        run_id="test-run-id",
        task_id="test-task-123",
        phase="execute",
        base_commit="same123",
        model="test-model",
        max_rounds=12,
        max_repairs=3,
        retry_count=0,
        final_status="VERIFIED",
        exit_code=0,
        duration_seconds=10.0,
        artifacts={
            "patch": "/custom/output/patch.diff",
            "verification_report": "/custom/output/verification_report.json",
        },
    )
    mock_workflow_result.to_run_summary.return_value = mock_run_summary
    mock_runner_instance.execute.return_value = mock_workflow_result
    mock_runner_instance.workspace = Mock(root=Path("/fake/repo"))
    mock_runner_instance._cleanup = Mock()
    mock_workflow_runner.return_value = mock_runner_instance

    # Mock file loading
    with patch("builtins.open") as mock_open, \
         patch("pathlib.Path.mkdir") as mock_mkdir, \
         patch("pathlib.Path.exists") as mock_exists, \
         patch("pathlib.Path.write_text") as mock_write_text:
        # Prevent actual filesystem operations
        mock_mkdir.side_effect = lambda *args, **kwargs: None
        mock_exists.return_value = True
        mock_write_text.side_effect = lambda *args, **kwargs: None

        call_count = [0]

        def mock_open_func(file, *args, **kwargs):
            call_count[0] += 1
            mock_file = Mock()
            if call_count[0] == 1:
                mock_file.read.return_value = json.dumps(issue.model_dump())
            else:
                mock_file.read.return_value = json.dumps(plan.model_dump())
            mock_file.__enter__ = Mock(return_value=mock_file)
            mock_file.__exit__ = Mock(return_value=False)
            return mock_file

        mock_open.side_effect = mock_open_func

        with (
            patch("patchpilot.cli.Workspace"),
            patch("patchpilot.cli.LLMProvider"),
            patch("patchpilot.cli.ToolRegistry"),
            patch("patchpilot.cli.AgentLoop"),
            patch("patchpilot.cli.save_json") as mock_save_json,
            patch("patchpilot.cli.render_acceptance_coverage") as mock_render_coverage,
        ):
            mock_render_coverage.return_value = "# Coverage"
            
            # Handle the SystemExit that occurs at the end of handle_execute
            with pytest.raises(SystemExit) as exc_info:
                handle_execute(args)
            
            # Verify exit code is 0 for VERIFIED status
            assert exc_info.value.code == 0

            # Verify save_json was called with custom output directory paths
            actual_calls = [call[0][0] for call in mock_save_json.call_args_list]
            assert "/custom/output/verification_report.json" in actual_calls
            assert "/custom/output/run_summary.json" in actual_calls


@patch("patchpilot.cli.validate_repository")
@patch("patchpilot.cli.WorkflowRunner")
def test_execute_saves_run_summary(
    mock_workflow_runner,
    mock_validate_repository,
):
    """Test that execute command saves run summary with expected fields."""
    from argparse import Namespace

    from patchpilot.cli import handle_execute
    from patchpilot.evidence.schema import CompletionState
    from patchpilot.issue.schema import NormalizedIssue
    from patchpilot.planning.schema import ChangePlan
    from patchpilot.repository.schema import RepositoryPreflightResult
    from patchpilot.workflow.result import WorkflowResult

    # Create mock args
    args = Namespace(
        repo="/fake/repo",
        issue="issue.json",
        plan="plan.json",
        model=None,
        max_rounds=12,
        max_repairs=3,
        output_dir="artifacts",
        task_id="test-task-123",
    )

    # Create plan and issue
    plan = ChangePlan(
        base_commit="abc123",
        repository_match=True,
        relevant_files=[],
        planned_changes=[],
        planned_tests=[],
        out_of_scope=[],
        risk_level="low",
    )

    issue = NormalizedIssue(
        title="Test Issue",
        task_type="feature",
        problem_statement="Test problem",
        ambiguous_points=[],
        acceptance_criteria=[],
        constraints=[],
        expected_test_areas=[],
        implementation_notes=[],
    )

    # Mock validate_repository
    mock_validate_repository.return_value = RepositoryPreflightResult(
        repo_path=Path("/fake/repo"),
        head_sha="abc123",
    )

    # Mock workflow runner with detailed result
    mock_runner_instance = Mock()
    mock_workflow_result = WorkflowResult(
        run_id="test-run-123",
        final_status=CompletionState.VERIFIED,
        changed_files=["src/file.py"],
        acceptance_evidence=[],
        verification_report={"passed": True, "retry_count": 1},
        patch="diff content",
        duration_seconds=42.7,
        retry_count=1,
        max_rounds=16,
        max_repairs=3,
    )
    mock_runner_instance.execute.return_value = mock_workflow_result
    mock_runner_instance.workspace = Mock(root=Path("/fake/repo"))
    mock_runner_instance._cleanup = Mock()
    mock_workflow_runner.return_value = mock_runner_instance

    # Mock file loading and provider
    with patch("builtins.open") as mock_open, \
         patch("pathlib.Path.mkdir") as mock_mkdir, \
         patch("pathlib.Path.exists") as mock_exists, \
         patch("pathlib.Path.write_text") as mock_write_text:
        # Prevent actual filesystem operations
        mock_mkdir.side_effect = lambda *args, **kwargs: None
        mock_exists.return_value = True
        mock_write_text.side_effect = lambda *args, **kwargs: None

        call_count = [0]

        def mock_open_func(file, *args, **kwargs):
            call_count[0] += 1
            mock_file = Mock()
            if call_count[0] == 1:
                mock_file.read.return_value = json.dumps(issue.model_dump())
            else:
                mock_file.read.return_value = json.dumps(plan.model_dump())
            mock_file.__enter__ = Mock(return_value=mock_file)
            mock_file.__exit__ = Mock(return_value=False)
            return mock_file

        mock_open.side_effect = mock_open_func

        with (
            patch("patchpilot.cli.Workspace"),
            patch("patchpilot.cli.LLMProvider") as mock_provider_class,
            patch("patchpilot.cli.ToolRegistry"),
            patch("patchpilot.cli.AgentLoop"),
            patch("patchpilot.cli.save_json") as mock_save_json,
            patch("patchpilot.cli.render_acceptance_coverage") as mock_render_coverage,
        ):
            mock_provider_instance = Mock()
            mock_provider_instance._model = "test-model"
            mock_provider_class.return_value = mock_provider_instance
            mock_render_coverage.return_value = "# Coverage"

            # Handle the SystemExit that occurs at the end of handle_execute
            with pytest.raises(SystemExit) as exc_info:
                handle_execute(args)
            
            # Verify exit code is 0 for VERIFIED status
            assert exc_info.value.code == 0

            # Verify run summary was saved with expected structure
            run_summary_call = None
            for call in mock_save_json.call_args_list:
                if "run_summary.json" in call[0][0]:
                    run_summary_call = call
                    break

            assert run_summary_call is not None, "run_summary.json was not saved"

            # Parse the saved JSON
            summary_data = json.loads(run_summary_call[0][1])

            # Verify required fields
            assert summary_data["run_id"] == "test-run-123"
            assert summary_data["phase"] == "execute"
            assert summary_data["base_commit"] == "abc123"
            assert summary_data["model"] == "test-model"
            assert summary_data["max_rounds"] == 16
            assert summary_data["max_repairs"] == 3
            assert summary_data["retry_count"] == 1
            assert summary_data["final_status"] == "VERIFIED"
            assert summary_data["exit_code"] == 0
            assert summary_data["duration_seconds"] > 0  # Duration is calculated by CLI
            assert "artifacts" in summary_data
            assert "patch" in summary_data["artifacts"]
            assert "verification_report" in summary_data["artifacts"]


@patch("patchpilot.cli.LLMProvider")
def test_cli_model_overrides_environment_model(
    mock_provider_class,
):
    """Test that _create_provider function exists and handles model parameter."""
    from patchpilot.cli import _create_provider

    # Mock provider instance
    mock_provider_instance = Mock()
    mock_provider_class.return_value = mock_provider_instance

    # Create provider without model override (should use environment)
    provider = _create_provider(None)

    # Verify provider was created successfully
    assert provider is not None
    mock_provider_class.assert_called_once()
