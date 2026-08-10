"""Tests for the CLI module."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from patchpilot.issue.schema import NormalizedIssue
from patchpilot.planning.schema import ChangePlan


@patch("patchpilot.cli.Path")
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
    mock_path,
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

    # Mock Path.exists to return True
    mock_path.return_value.exists.return_value = True
    # Mock Path constructor to return a proper Path-like object
    mock_path_instance = Mock()
    mock_path_instance.exists.return_value = True
    mock_path.return_value = mock_path_instance

    # Mock sys.argv to simulate CLI call
    with patch("sys.argv", ["patchpilot", "run", "--repo", "/fake/repo", "--issue", "test.md"]):
        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1

    # Verify normalization was called but agent was not
    mock_normalize.assert_called_once()
    mock_agent_loop.assert_not_called()


@patch("patchpilot.cli.Path")
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
    mock_path,
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

    # Mock Path.exists to return True
    mock_path.return_value.exists.return_value = True
    # Mock Path constructor to return a proper Path-like object
    mock_path_instance = Mock()
    mock_path_instance.exists.return_value = True
    mock_path.return_value = mock_path_instance

    # Mock sys.argv to simulate CLI call
    with patch("sys.argv", ["patchpilot", "run", "--repo", "/fake/repo", "--issue", "test.md"]):
        main()

    # Verify repository validation was called
    mock_validate_repository.assert_called_once()
    # Verify repair loop was called
    mock_run_repair_loop.assert_called_once()


@patch("patchpilot.cli.Path")
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
    mock_path,
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

    # Mock Path.exists to return True
    mock_path.return_value.exists.return_value = True
    # Mock Path constructor to return a proper Path-like object
    mock_path_instance = Mock()
    mock_path_instance.exists.return_value = True
    mock_path.return_value = mock_path_instance

    # Mock sys.argv to simulate CLI call
    with patch("sys.argv", ["patchpilot", "run", "--repo", "/fake/repo", "--issue", "test.md"]):
        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1

    # Verify agent was not called
    mock_agent_loop.assert_not_called()
