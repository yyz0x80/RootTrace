"""Tests for the AgentLoop module."""

from unittest.mock import Mock

import pytest

from patchpilot.agent_loop import (
    AgentLoop,
    AgentLoopError,
    AgentLoopLimitError,
)
from patchpilot.models import AssistantTurn, ToolCall, ToolResult


class TestAgentLoopInit:
    """Tests for AgentLoop initialization."""

    def test_init_with_valid_parameters(self):
        """Test initialization with valid parameters."""
        mock_provider = Mock()
        mock_tools = Mock()
        mock_tools.get_schemas.return_value = []

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
        mock_tools.get_schemas.return_value = []

        agent_loop = AgentLoop(
            provider=mock_provider,
            tools=mock_tools,
        )

        assert agent_loop.max_rounds == 12

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
        mock_tools.get_schemas.return_value = []

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
        mock_tools.get_schemas.return_value = []

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
        mock_tools.get_schemas.return_value = [
            {"name": "search_code", "parameters": {}}
        ]
        mock_tools.get_names.return_value = ["search_code"]
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
        mock_tools.get_schemas.return_value = [
            {"name": "search_code", "parameters": {}},
            {"name": "read_file", "parameters": {}},
        ]
        mock_tools.get_names.return_value = ["search_code", "read_file"]
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
        mock_tools.get_schemas.return_value = [{"name": "test", "parameters": {}}]
        mock_tools.get_names.return_value = ["test"]
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
        mock_tools.get_schemas.return_value = []

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
        mock_tools.get_schemas.return_value = []

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
        mock_tools.get_schemas.return_value = []
        mock_tools.get_names.return_value = ["search_code", "read_file"]
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
        mock_tools.get_schemas.return_value = []
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
        mock_tools.get_schemas.return_value = []
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

        assert formatted == "ERROR\nFile not found"

    def test_format_multiline_content(self):
        """Test formatting multiline content."""
        result = ToolResult(ok=True, content="Line 1\nLine 2\nLine 3")

        formatted = AgentLoop._format_tool_result(result)

        assert formatted == "SUCCESS\nLine 1\nLine 2\nLine 3"
