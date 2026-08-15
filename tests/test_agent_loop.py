"""Tests for the AgentLoop module."""

from unittest.mock import Mock

import pytest

from patchpilot.agent_loop import (
    MAX_FAILURE_SUMMARY_CHARS,
    AgentLoop,
    AgentLoopError,
    AgentLoopLimitError,
    AgentState,
)
from patchpilot.models import (
    AssistantTurn,
    ToolCall,
    ToolFailureType,
    ToolResult,
)


class TestAgentLoopInit:
    """Tests for AgentLoop initialization."""

    def test_init_with_valid_parameters(self):
        """Test initialization with valid parameters."""
        mock_provider = Mock()
        mock_tools = Mock()
        mock_tools.get_tool_schemas.return_value = []

        agent_loop = AgentLoop(
            provider=mock_provider,
            tools=mock_tools,
            max_rounds=10,
            system_prompt="Test prompt",
        )

        assert agent_loop.provider == mock_provider
        assert agent_loop.tools == mock_tools
        assert agent_loop.max_rounds == 10
        assert agent_loop.system_prompt == "Test prompt"

    def test_init_with_default_max_rounds(self):
        """Test initialization with default max_rounds."""
        mock_provider = Mock()
        mock_tools = Mock()
        mock_tools.get_tool_schemas.return_value = []

        agent_loop = AgentLoop(
            provider=mock_provider,
            tools=mock_tools,
        )

        assert agent_loop.max_rounds == 16

    def test_init_with_invalid_max_rounds(self):
        """Test that max_rounds must be at least 1."""
        mock_provider = Mock()
        mock_tools = Mock()

        with pytest.raises(ValueError, match="max_rounds must be at least 1"):
            AgentLoop(
                provider=mock_provider,
                tools=mock_tools,
                max_rounds=0,
            )

        with pytest.raises(ValueError, match="max_rounds must be at least 1"):
            AgentLoop(
                provider=mock_provider,
                tools=mock_tools,
                max_rounds=-5,
            )


class TestAgentLoopRun:
    """Tests for AgentLoop.run method."""

    def test_run_with_empty_issue(self):
        """Test that empty issue raises ValueError."""
        mock_provider = Mock()
        mock_tools = Mock()
        mock_tools.get_tool_schemas.return_value = []

        agent_loop = AgentLoop(
            provider=mock_provider,
            tools=mock_tools,
        )

        with pytest.raises(ValueError, match="issue must not be empty"):
            agent_loop.run("")

        with pytest.raises(ValueError, match="issue must not be empty"):
            agent_loop.run("   ")

    def test_run_immediate_completion(self):
        """Test successful completion without tool calls."""
        mock_provider = Mock()
        mock_tools = Mock()
        mock_tools.get_tool_schemas.return_value = []

        mock_provider.complete.return_value = AssistantTurn(
            content="Task completed successfully",
            tool_calls=[],
        )

        agent_loop = AgentLoop(
            provider=mock_provider,
            tools=mock_tools,
        )

        result = agent_loop.run("Fix the bug")

        assert result == "Task completed successfully"
        assert mock_provider.complete.call_count == 1

    def test_run_with_single_tool_call(self):
        """Test execution of a single tool call."""
        mock_provider = Mock()
        mock_tools = Mock()
        mock_tools.get_tool_schemas.return_value = [
            {"type": "function", "function": {"name": "search_code", "parameters": {}}}
        ]
        mock_tools.get_available_tools.return_value = ["search_code"]
        mock_tools.execute.return_value = ToolResult(
            ok=True,
            content="Search results found",
        )

        # First call: request tool, second call: final response
        mock_provider.complete.side_effect = [
            AssistantTurn(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_123",
                        name="search_code",
                        arguments={"query": "test"},
                    )
                ],
            ),
            AssistantTurn(
                content="Fixed the bug based on search results",
                tool_calls=[],
            ),
        ]

        agent_loop = AgentLoop(
            provider=mock_provider,
            tools=mock_tools,
        )

        result = agent_loop.run("Search and fix")

        assert result == "Fixed the bug based on search results"
        assert mock_provider.complete.call_count == 2
        mock_tools.execute.assert_called_once_with(
            name="search_code",
            arguments={"query": "test"},
        )

    def test_run_with_multiple_tool_calls(self):
        """Test execution of multiple tool calls in one round."""
        mock_provider = Mock()
        mock_tools = Mock()
        mock_tools.get_tool_schemas.return_value = [
            {"type": "function", "function": {"name": "search_code", "parameters": {}}},
            {"type": "function", "function": {"name": "read_file", "parameters": {}}},
        ]
        mock_tools.get_available_tools.return_value = ["search_code", "read_file"]
        mock_tools.execute.side_effect = [
            ToolResult(ok=True, content="Search results"),
            ToolResult(ok=True, content="File content"),
        ]

        mock_provider.complete.side_effect = [
            AssistantTurn(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="search_code",
                        arguments={"query": "test"},
                    ),
                    ToolCall(
                        id="call_2",
                        name="read_file",
                        arguments={"path": "test.py"},
                    ),
                ],
            ),
            AssistantTurn(
                content="Task complete",
                tool_calls=[],
            ),
        ]

        agent_loop = AgentLoop(
            provider=mock_provider,
            tools=mock_tools,
        )

        result = agent_loop.run("Complex task")

        assert result == "Task complete"
        assert mock_tools.execute.call_count == 2

    def test_run_max_rounds_exceeded(self):
        """Test that exceeding max rounds raises AgentLoopLimitError."""
        mock_provider = Mock()
        mock_tools = Mock()
        mock_tools.get_tool_schemas.return_value = [
            {"type": "function", "function": {"name": "test", "parameters": {}}}
        ]
        mock_tools.get_available_tools.return_value = ["test"]
        mock_tools.execute.return_value = ToolResult(ok=True, content="result")

        # Always request tool calls, never complete
        mock_provider.complete.return_value = AssistantTurn(
            content=None,
            tool_calls=[
                ToolCall(
                    id="call_123",
                    name="test",
                    arguments={},
                )
            ],
        )

        agent_loop = AgentLoop(
            provider=mock_provider,
            tools=mock_tools,
            max_rounds=2,
        )

        with pytest.raises(AgentLoopLimitError, match="exceeded the maximum of 2 rounds"):
            agent_loop.run("Never-ending task")

    def test_run_model_returns_no_content_no_tools(self):
        """Test error when model returns neither content nor tool calls."""
        mock_provider = Mock()
        mock_tools = Mock()
        mock_tools.get_tool_schemas.return_value = []

        mock_provider.complete.return_value = AssistantTurn(
            content=None,
            tool_calls=[],
        )

        agent_loop = AgentLoop(
            provider=mock_provider,
            tools=mock_tools,
        )

        with pytest.raises(AgentLoopError, match="neither tool calls nor final content"):
            agent_loop.run("Test issue")


class TestExecuteTool:
    """Tests for _execute_tool method."""

    def test_execute_tool_success(self):
        """Test successful tool execution."""
        mock_provider = Mock()
        mock_tools = Mock()
        mock_tools.get_tool_schemas.return_value = []

        mock_tools.execute.return_value = ToolResult(
            ok=True,
            content="Tool executed successfully",
        )

        agent_loop = AgentLoop(
            provider=mock_provider,
            tools=mock_tools,
        )

        tool_call = ToolCall(
            id="call_123",
            name="search_code",
            arguments={"query": "test"},
        )

        result = agent_loop._execute_tool(tool_call)

        assert result.ok
        assert result.content == "Tool executed successfully"
        mock_tools.execute.assert_called_once_with(
            name="search_code",
            arguments={"query": "test"},
        )

    def test_execute_tool_unknown_tool(self):
        """Test handling of unknown tool name."""
        mock_provider = Mock()
        mock_tools = Mock()
        mock_tools.get_tool_schemas.return_value = []
        mock_tools.get_available_tools.return_value = ["search_code", "read_file"]
        mock_tools.execute.side_effect = KeyError("unknown_tool")

        agent_loop = AgentLoop(
            provider=mock_provider,
            tools=mock_tools,
        )

        tool_call = ToolCall(
            id="call_123",
            name="unknown_tool",
            arguments={},
        )

        result = agent_loop._execute_tool(tool_call)

        assert not result.ok
        assert "Unknown tool: unknown_tool" in result.content
        assert "search_code" in result.content

    def test_execute_tool_exception(self):
        """Test handling of tool execution exceptions."""
        mock_provider = Mock()
        mock_tools = Mock()
        mock_tools.get_tool_schemas.return_value = []
        mock_tools.execute.side_effect = ValueError("Invalid input")

        agent_loop = AgentLoop(
            provider=mock_provider,
            tools=mock_tools,
        )

        tool_call = ToolCall(
            id="call_123",
            name="search_code",
            arguments={},
        )

        result = agent_loop._execute_tool(tool_call)

        assert not result.ok
        assert "Tool execution failed" in result.content
        assert "ValueError" in result.content

    def test_execute_tool_invalid_result_type(self):
        """Test handling of invalid tool result type."""
        mock_provider = Mock()
        mock_tools = Mock()
        mock_tools.get_tool_schemas.return_value = []
        mock_tools.execute.return_value = "invalid string result"

        agent_loop = AgentLoop(
            provider=mock_provider,
            tools=mock_tools,
        )

        tool_call = ToolCall(
            id="call_123",
            name="search_code",
            arguments={},
        )

        result = agent_loop._execute_tool(tool_call)

        assert not result.ok
        assert "invalid result type" in result.content
        assert "str" in result.content


class TestBuildAssistantMessage:
    """Tests for _build_assistant_message static method."""

    def test_build_message_with_content_only(self):
        """Test building message with content but no tool calls."""
        turn = AssistantTurn(
            content="Hello, world!",
            tool_calls=[],
        )

        message = AgentLoop._build_assistant_message(turn)

        assert message["role"] == "assistant"
        assert message["content"] == "Hello, world!"
        assert "tool_calls" not in message

    def test_build_message_with_tool_calls(self):
        """Test building message with tool calls."""
        turn = AssistantTurn(
            content=None,
            tool_calls=[
                ToolCall(
                    id="call_1",
                    name="search_code",
                    arguments={"query": "test"},
                ),
                ToolCall(
                    id="call_2",
                    name="read_file",
                    arguments={"path": "test.py"},
                ),
            ],
        )

        message = AgentLoop._build_assistant_message(turn)

        assert message["role"] == "assistant"
        assert message["content"] is None
        assert "tool_calls" in message
        assert len(message["tool_calls"]) == 2

        # Check first tool call
        first_call = message["tool_calls"][0]
        assert first_call["id"] == "call_1"
        assert first_call["type"] == "function"
        assert first_call["function"]["name"] == "search_code"
        assert isinstance(first_call["function"]["arguments"], str)

        # Check second tool call
        second_call = message["tool_calls"][1]
        assert second_call["id"] == "call_2"
        assert second_call["type"] == "function"
        assert second_call["function"]["name"] == "read_file"

    def test_build_message_with_both_content_and_tools(self):
        """Test building message with both content and tool calls."""
        turn = AssistantTurn(
            content="I'll search for that",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    name="search_code",
                    arguments={"query": "test"},
                ),
            ],
        )

        message = AgentLoop._build_assistant_message(turn)

        assert message["role"] == "assistant"
        assert message["content"] == "I'll search for that"
        assert len(message["tool_calls"]) == 1


class TestFormatToolResult:
    """Tests for _format_tool_result static method."""

    def test_format_successful_result(self):
        """Test formatting a successful tool result."""
        result = ToolResult(ok=True, content="Operation successful")

        formatted = AgentLoop._format_tool_result(result)

        assert formatted == "SUCCESS\nOperation successful"

    def test_format_error_result(self):
        """Test formatting an error tool result."""
        result = ToolResult(ok=False, content="File not found")

        formatted = AgentLoop._format_tool_result(result)

        assert formatted == "ERROR [TOOL_FAILURE]\nFile not found"

    def test_format_verification_failure(self):
        """Test that verification failures are explicit for the model."""
        result = ToolResult(
            ok=False,
            content="1 test failed",
            failure_type=ToolFailureType.VERIFICATION_FAILURE,
        )

        formatted = AgentLoop._format_tool_result(result)

        assert formatted == (
            "ERROR [VERIFICATION_FAILURE]\n1 test failed"
        )

    def test_format_multiline_content(self):
        """Test formatting multiline content."""
        result = ToolResult(ok=True, content="Line 1\nLine 2\nLine 3")

        formatted = AgentLoop._format_tool_result(result)

        assert formatted == "SUCCESS\nLine 1\nLine 2\nLine 3"


class TestAgentState:
    """Tests for AgentState class."""

    def test_initial_state(self):
        """Test initial state values."""
        state = AgentState()

        assert len(state.files_modified) == 0
        assert len(state.tool_usage_count) == 0
        assert state.consecutive_failures == 0
        assert state.last_tool_success is True
        assert state.total_edits == 0
        assert len(state.unique_files_read) == 0

    def test_record_tool_call_success(self):
        """Test recording a successful tool call."""
        state = AgentState()

        state.record_tool_call("search_code", True)

        assert state.tool_usage_count["search_code"] == 1
        assert state.consecutive_failures == 0
        assert state.last_tool_success is True

    def test_record_tool_call_failure(self):
        """Test recording a failed tool call."""
        state = AgentState()

        state.record_tool_call("edit_file", False)

        assert state.tool_usage_count["edit_file"] == 1
        assert state.consecutive_failures == 1
        assert state.last_tool_success is False

    def test_consecutive_failures_tracking(self):
        """Test consecutive failure tracking."""
        state = AgentState()

        state.record_tool_call("edit_file", False)
        assert state.consecutive_failures == 1

        state.record_tool_call("edit_file", False)
        assert state.consecutive_failures == 2

        state.record_tool_call("edit_file", True)
        assert state.consecutive_failures == 0

    def test_verification_failure_does_not_count_as_tool_failure(self):
        """Test that a failed verification breaks the tool-failure sequence."""
        state = AgentState()
        state.record_tool_call("read_file", False)

        state.record_tool_call(
            "run_command",
            False,
            "pytest failed",
            ToolFailureType.VERIFICATION_FAILURE,
        )

        assert state.consecutive_failures == 0
        assert state.last_tool_success is True
        assert (
            state.last_failure_type
            == ToolFailureType.VERIFICATION_FAILURE
        )
        assert state.recent_failures == []

    def test_record_file_edit(self):
        """Test recording file edits."""
        state = AgentState()

        state.record_file_edit("test.py")
        state.record_file_edit("test.py")  # Same file again
        state.record_file_edit("other.py")

        assert "test.py" in state.files_modified
        assert "other.py" in state.files_modified
        assert len(state.files_modified) == 2
        assert state.total_edits == 3

    def test_record_file_read(self):
        """Test recording file reads."""
        state = AgentState()

        state.record_file_read("test.py")
        state.record_file_read("test.py")  # Same file again
        state.record_file_read("other.py")

        assert "test.py" in state.unique_files_read
        assert "other.py" in state.unique_files_read
        assert len(state.unique_files_read) == 2

    def test_get_progress_summary(self):
        """Test progress summary generation."""
        state = AgentState()

        state.record_file_edit("test.py")
        state.record_file_read("other.py")
        state.record_tool_call("search_code", True)
        state.record_tool_call("edit_file", False)

        summary = state.get_progress_summary()

        assert "Files modified: 1" in summary
        assert "Total edits: 1" in summary
        assert "Files read: 1" in summary
        assert "Consecutive failures: 1" in summary
        assert "Tool usage:" in summary

    def test_should_stop_early(self):
        """Test early stopping logic."""
        state = AgentState()

        # Should not stop with no failures
        assert not state.should_stop_early(max_consecutive_failures=3)

        # Should not stop with 1 failure
        state.record_tool_call("edit_file", False)
        assert not state.should_stop_early(max_consecutive_failures=3)

        # Should not stop with 2 failures
        state.record_tool_call("edit_file", False)
        assert not state.should_stop_early(max_consecutive_failures=3)

        # Should stop with 3 failures
        state.record_tool_call("edit_file", False)
        assert state.should_stop_early(max_consecutive_failures=3)

        # Should not stop after a success
        state.record_tool_call("edit_file", True)
        assert not state.should_stop_early(max_consecutive_failures=3)


class TestAgentLoopProgressTracking:
    """Tests for AgentLoop progress tracking features."""

    def test_init_with_progress_tracking_enabled(self):
        """Test initialization with progress tracking enabled."""
        mock_provider = Mock()
        mock_tools = Mock()
        mock_tools.get_tool_schemas.return_value = []

        agent_loop = AgentLoop(
            provider=mock_provider,
            tools=mock_tools,
            enable_progress_tracking=True,
        )

        assert agent_loop.enable_progress_tracking is True
        assert agent_loop.state is not None

    def test_init_with_progress_tracking_disabled(self):
        """Test initialization with progress tracking disabled."""
        mock_provider = Mock()
        mock_tools = Mock()
        mock_tools.get_tool_schemas.return_value = []

        agent_loop = AgentLoop(
            provider=mock_provider,
            tools=mock_tools,
            enable_progress_tracking=False,
        )

        assert agent_loop.enable_progress_tracking is False

    def test_init_with_early_stopping_enabled(self):
        """Test initialization with early stopping enabled."""
        mock_provider = Mock()
        mock_tools = Mock()
        mock_tools.get_tool_schemas.return_value = []

        agent_loop = AgentLoop(
            provider=mock_provider,
            tools=mock_tools,
            enable_early_stopping=True,
            max_consecutive_failures=5,
        )

        assert agent_loop.enable_early_stopping is True
        assert agent_loop.max_consecutive_failures == 5

    def test_early_stopping_trigger(self):
        """Test that early stopping is triggered after consecutive failures."""
        mock_provider = Mock()
        mock_tools = Mock()
        mock_tools.get_tool_schemas.return_value = [
            {"type": "function", "function": {"name": "test", "parameters": {}}}
        ]
        mock_tools.get_available_tools.return_value = ["test"]
        mock_tools.execute.return_value = ToolResult(ok=False, content="Failed")

        mock_provider.complete.return_value = AssistantTurn(
            content=None,
            tool_calls=[
                ToolCall(
                    id="call_123",
                    name="test",
                    arguments={},
                )
            ],
        )

        agent_loop = AgentLoop(
            provider=mock_provider,
            tools=mock_tools,
            max_rounds=10,
            enable_early_stopping=True,
            max_consecutive_failures=2,
            enable_progress_tracking=True,
        )

        with pytest.raises(AgentLoopError, match="consecutive tool failures"):
            agent_loop.run("Test issue")

    def test_pytest_failure_does_not_trigger_tool_failure_stop(self):
        """Test that a failed Pytest run does not consume the tool failure budget."""
        mock_provider = Mock()
        mock_tools = Mock()
        mock_tools.get_tool_schemas.return_value = [
            {
                "type": "function",
                "function": {"name": "run_command", "parameters": {}},
            },
            {
                "type": "function",
                "function": {"name": "read_file", "parameters": {}},
            },
        ]
        mock_tools.get_available_tools.return_value = [
            "run_command",
            "read_file",
        ]
        mock_tools.execute.side_effect = [
            ToolResult(
                ok=False,
                content="1 test failed",
                failure_type=ToolFailureType.VERIFICATION_FAILURE,
            ),
            ToolResult(ok=False, content="Absolute path rejected"),
            ToolResult(ok=False, content="Absolute path rejected"),
        ]
        mock_provider.complete.side_effect = [
            AssistantTurn(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="run_command",
                        arguments={"command": "pytest"},
                    )
                ],
            ),
            AssistantTurn(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_2",
                        name="read_file",
                        arguments={"path": "/tmp/repo"},
                    )
                ],
            ),
            AssistantTurn(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_3",
                        name="read_file",
                        arguments={"path": "/tmp/repo"},
                    )
                ],
            ),
            AssistantTurn(content="Recovered", tool_calls=[]),
        ]
        agent_loop = AgentLoop(
            provider=mock_provider,
            tools=mock_tools,
            max_rounds=4,
            max_consecutive_failures=3,
        )

        result = agent_loop.run("Fix the implementation")

        assert result == "Recovered"
        assert agent_loop.state.consecutive_failures == 2

    def test_terminal_failure_summary_is_redacted_and_bounded(self):
        """Test that terminal failure summaries hide secrets and stay bounded."""
        mock_tools = Mock()
        mock_tools.get_tool_schemas.return_value = []
        mock_tools.sanitize_workspace_paths.side_effect = (
            lambda content: content.replace("/tmp/workspace/", "")
        )
        agent_loop = AgentLoop(provider=Mock(), tools=mock_tools)
        content = (
            "/tmp/workspace/module.py ZHIPU_API_KEY=secret-value "
            "Authorization: Bearer bearer-value "
            + "x" * 1_000
        )

        summary = agent_loop._summarize_tool_failure(content)

        assert "/tmp/workspace" not in summary
        assert "secret-value" not in summary
        assert "bearer-value" not in summary
        assert "<redacted>" in summary
        assert len(summary) <= MAX_FAILURE_SUMMARY_CHARS

    def test_state_tracking_during_execution(self):
        """Test that state is tracked during tool execution."""
        mock_provider = Mock()
        mock_tools = Mock()
        mock_tools.get_tool_schemas.return_value = [
            {"type": "function", "function": {"name": "edit_file", "parameters": {}}},
            {"type": "function", "function": {"name": "read_file", "parameters": {}}},
        ]
        mock_tools.get_available_tools.return_value = ["edit_file", "read_file"]
        mock_tools.execute.side_effect = [
            ToolResult(ok=True, content="File edited"),
            ToolResult(ok=True, content="File read"),
        ]

        mock_provider.complete.side_effect = [
            AssistantTurn(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="edit_file",
                        arguments={"path": "test.py"},
                    ),
                    ToolCall(
                        id="call_2",
                        name="read_file",
                        arguments={"path": "other.py"},
                    ),
                ],
            ),
            AssistantTurn(
                content="Task complete",
                tool_calls=[],
            ),
        ]

        agent_loop = AgentLoop(
            provider=mock_provider,
            tools=mock_tools,
            enable_progress_tracking=True,
        )

        agent_loop.run("Test issue")

        # Verify state was tracked
        assert "test.py" in agent_loop.state.files_modified
        assert "other.py" in agent_loop.state.unique_files_read
        assert agent_loop.state.total_edits == 1
        assert agent_loop.state.tool_usage_count["edit_file"] == 1
        assert agent_loop.state.tool_usage_count["read_file"] == 1
        assert agent_loop.state.consecutive_failures == 0  # All tools succeeded


class TestAgentStateFailurePatternDetection:
    """Tests for AgentState failure pattern detection."""

    def test_record_tool_call_with_error_content(self):
        """Test recording tool calls with error content for pattern detection."""
        state = AgentState()

        state.record_tool_call("edit_file", False, "old_text not found")
        state.record_tool_call("edit_file", False, "old_text not found")

        assert state.consecutive_failures == 2
        assert len(state.recent_failures) == 2
        assert state.recent_failures[0] == state.recent_failures[1]

    def test_detect_repeated_failure_pattern(self):
        """Test detection of repeated failure patterns."""
        state = AgentState()

        # No failures yet
        is_repeated, desc = state.detect_repeated_failure_pattern()
        assert not is_repeated
        assert desc == ""

        # Single failure
        state.record_tool_call("edit_file", False, "old_text not found")
        is_repeated, desc = state.detect_repeated_failure_pattern()
        assert not is_repeated
        assert desc == ""

        # Repeated failure
        state.record_tool_call("edit_file", False, "old_text not found")
        is_repeated, desc = state.detect_repeated_failure_pattern()
        assert is_repeated
        assert "edit_file" in desc

    def test_failure_signature_generation(self):
        """Test failure signature generation."""
        state = AgentState()

        sig1 = state._generate_failure_signature("edit_file", "old_text not found")
        sig2 = state._generate_failure_signature("edit_file", "old_text not found")
        sig3 = state._generate_failure_signature("edit_file", "different error")

        assert sig1 == sig2
        assert sig1 != sig3

    def test_recent_failures_limit(self):
        """Test that recent failures list is limited to 5 entries."""
        state = AgentState()

        # Add 7 failures
        for i in range(7):
            state.record_tool_call("edit_file", False, f"error {i}")

        assert len(state.recent_failures) == 5
        # Should keep the most recent 5 (errors 2-6, with 0-1 being evicted)
        assert "error 0" not in state.recent_failures[0]
        assert "error 6" in state.recent_failures[-1]

    def test_consecutive_failures_reset_on_success(self):
        """Test that consecutive failures reset on success."""
        state = AgentState()

        state.record_tool_call("edit_file", False, "error")
        state.record_tool_call("edit_file", False, "error")
        assert state.consecutive_failures == 2

        state.record_tool_call("edit_file", True, "")
        assert state.consecutive_failures == 0

    def test_failure_pattern_different_tools(self):
        """Test that failures on different tools don't trigger pattern detection."""
        state = AgentState()

        state.record_tool_call("edit_file", False, "old_text not found")
        state.record_tool_call("read_file", False, "file not found")

        is_repeated, _desc = state.detect_repeated_failure_pattern()
        assert not is_repeated
