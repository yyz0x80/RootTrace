"""Tests for the CLI module."""


from unittest.mock import Mock, patch

import pytest

from patchpilot.issue.schema import NormalizedIssue


@patch("patchpilot.cli.load_issue")
@patch("patchpilot.cli.normalize_issue")
@patch("patchpilot.cli.LLMProvider")
@patch("patchpilot.cli.Workspace")
@patch("patchpilot.cli.ToolRegistry")
@patch("patchpilot.cli.AgentLoop")
def test_main_with_ambiguous_points_stops(
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
    mock_provider_instance.complete_text = Mock(return_value="normalized")

    mock_normalize.return_value = NormalizedIssue(
        title="Test Issue",
        task_type="feature",
        problem_statement="Test problem",
        ambiguous_points=["Default priority is unspecified."],
    )

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
def test_main_without_ambiguous_points_proceeds(
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
    mock_provider_instance.complete_text = Mock(return_value="normalized")

    mock_normalize.return_value = NormalizedIssue(
        title="Test Issue",
        task_type="feature",
        problem_statement="Test problem",
        ambiguous_points=[],
    )

    mock_agent_loop_instance = Mock()
    mock_agent_loop.return_value = mock_agent_loop_instance
    mock_agent_loop_instance.run = Mock(return_value="Success")

    # Mock Path.exists to return True
    mock_path.return_value.exists.return_value = True

    # Mock sys.argv to simulate CLI call
    with patch("sys.argv", ["patchpilot", "run", "--repo", "/fake/repo", "--issue", "test.md"]):
        main()

    # Verify agent was called
    mock_agent_loop.assert_called_once()
    mock_agent_loop_instance.run.assert_called_once()


@patch("patchpilot.cli.load_issue")
@patch("patchpilot.cli.normalize_issue")
@patch("patchpilot.cli.LLMProvider")
@patch("patchpilot.cli.Workspace")
@patch("patchpilot.cli.ToolRegistry")
@patch("patchpilot.cli.AgentLoop")
def test_main_with_multiple_ambiguous_points_shows_all(
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
    mock_provider_instance.complete_text = Mock(return_value="normalized")

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

    # Mock sys.argv to simulate CLI call
    with patch("sys.argv", ["patchpilot", "run", "--repo", "/fake/repo", "--issue", "test.md"]):
        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1

    # Verify agent was not called
    mock_agent_loop.assert_not_called()
