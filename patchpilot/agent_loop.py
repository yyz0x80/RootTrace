"""Agent Loop module for PatchPilot.

This module provides the AgentLoop class which coordinates:
- Model responses and tool execution
- Message history management
- Round limit enforcement
- Tool result formatting
- Structured logging callbacks for execute workflow
- Intelligent round management with progress tracking
- State tracking for file modifications and operations
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from typing import Any

from patchpilot.models import AssistantTurn, ToolCall, ToolResult
from patchpilot.prompts import SYSTEM_PROMPT
from patchpilot.provider import LLMProvider
from patchpilot.tools import ToolRegistry
from patchpilot.workspace import Workspace

logger = logging.getLogger(__name__)


class ExecuteLogCallback:
    """Callback interface for structured execute logging during agent execution."""

    def on_round_start(self, round_number: int) -> None:
        """Called at the start of each agent round.

        Args:
            round_number: Current round number
        """

    def on_tool_call(self, round_number: int, tool_name: str, args: dict[str, Any]) -> None:
        """Called when the agent makes a tool call.

        Args:
            round_number: Current round number
            tool_name: Name of the tool being called
            args: Tool arguments
        """

    def on_round_complete(self, round_number: int) -> None:
        """Called when the agent completes a round with a final answer.

        Args:
            round_number: Current round number
        """


@dataclass
class AgentState:
    """Track the state of agent execution for progress monitoring and early stopping.

    Attributes:
        files_modified: Set of file paths that have been modified
        tool_usage_count: Counter tracking how many times each tool was used
        consecutive_failures: Number of consecutive tool failures
        last_tool_success: Whether the last tool call succeeded
        total_edits: Total number of edit operations performed
        unique_files_read: Set of file paths that have been read
    """

    files_modified: set[str]
    tool_usage_count: Counter[str]
    consecutive_failures: int
    last_tool_success: bool
    total_edits: int
    unique_files_read: set[str]

    def __init__(self) -> None:
        self.files_modified = set()
        self.tool_usage_count = Counter()
        self.consecutive_failures = 0
        self.last_tool_success = True
        self.total_edits = 0
        self.unique_files_read = set()

    def record_tool_call(self, tool_name: str, success: bool) -> None:
        """Record a tool call and update failure tracking.

        Args:
            tool_name: Name of the tool that was called
            success: Whether the tool call succeeded
        """
        self.tool_usage_count[tool_name] += 1
        self.last_tool_success = success

        if success:
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1

    def record_file_edit(self, file_path: str) -> None:
        """Record that a file was edited.

        Args:
            file_path: Path to the file that was edited
        """
        self.files_modified.add(file_path)
        self.total_edits += 1

    def record_file_read(self, file_path: str) -> None:
        """Record that a file was read.

        Args:
            file_path: Path to the file that was read
        """
        self.unique_files_read.add(file_path)

    def get_progress_summary(self) -> str:
        """Generate a human-readable progress summary.

        Returns:
            String describing the current progress state
        """
        summary_parts = [
            f"Files modified: {len(self.files_modified)}",
            f"Total edits: {self.total_edits}",
            f"Files read: {len(self.unique_files_read)}",
            f"Consecutive failures: {self.consecutive_failures}",
        ]

        if self.tool_usage_count:
            summary_parts.append(
                f"Tool usage: {', '.join(f'{k}:{v}' for k, v in self.tool_usage_count.most_common(5))}"
            )

        return ", ".join(summary_parts)

    def should_stop_early(self, max_consecutive_failures: int = 3) -> bool:
        """Determine if execution should stop early due to repeated failures.

        Args:
            max_consecutive_failures: Maximum allowed consecutive failures

        Returns:
            True if early stopping is recommended
        """
        return self.consecutive_failures >= max_consecutive_failures


class AgentLoopError(RuntimeError):
    """Base exception for Agent Loop failures."""


class AgentLoopLimitError(AgentLoopError):
    """Raised when the Agent exceeds the configured round limit."""


class AgentLoop:
    """Coordinate model responses and tool execution with intelligent round management."""

    def __init__(
        self,
        provider: LLMProvider,
        tools: ToolRegistry,
        max_rounds: int = 16,
        system_prompt: str = SYSTEM_PROMPT,
        execute_log_callback: ExecuteLogCallback | None = None,
        enable_early_stopping: bool = True,
        max_consecutive_failures: int = 3,
        enable_progress_tracking: bool = True,
    ) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")

        self.provider = provider
        self.tools = tools
        self.max_rounds = max_rounds
        self.system_prompt = system_prompt
        self.execute_log_callback = execute_log_callback
        self.enable_early_stopping = enable_early_stopping
        self.max_consecutive_failures = max_consecutive_failures
        self.enable_progress_tracking = enable_progress_tracking
        self.state = AgentState()

    def update_workspace(self, workspace: Workspace) -> None:
        """Update the workspace used by the tool registry.

        Args:
            workspace: New Workspace instance to use for path resolution
        """
        self.tools.update_workspace(workspace)

    def run(self, issue: str) -> str:
        """Run the Agent Loop until the model returns a final answer.

        Args:
            issue: The repository task or issue description.

        Returns:
            The model's final text response.

        Raises:
            ValueError: If the issue is empty.
            AgentLoopError: If the model finishes without a valid response.
            AgentLoopLimitError: If the maximum number of rounds is reached.
        """
        if not issue.strip():
            raise ValueError("issue must not be empty")

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {
                "role": "user",
                "content": issue.strip(),
            },
        ]

        tool_schemas = self.tools.get_tool_schemas()

        for round_number in range(1, self.max_rounds + 1):
            logger.info(
                "Starting Agent round %d/%d",
                round_number,
                self.max_rounds,
            )

            # Display progress if tracking is enabled
            if self.enable_progress_tracking and round_number > 1:
                progress_summary = self.state.get_progress_summary()
                print(f"[Progress] {progress_summary}")

            # Check for early stopping due to repeated failures
            if self.enable_early_stopping and self.state.should_stop_early(self.max_consecutive_failures):
                logger.warning(
                    "Early stopping triggered after %d consecutive failures",
                    self.state.consecutive_failures,
                )
                raise AgentLoopError(
                    f"Agent stopped early after {self.state.consecutive_failures} "
                    f"consecutive tool failures"
                )

            # Notify callback of round start
            if self.execute_log_callback:
                self.execute_log_callback.on_round_start(round_number)

            # Inject state summary into system prompt for context
            enhanced_messages = self._inject_state_context(messages, round_number)

            assistant_turn = self.provider.complete(
                messages=enhanced_messages,
                tools=tool_schemas,
            )

            # Print round number and tool calls for user visibility
            if assistant_turn.tool_calls:
                for tool_call in assistant_turn.tool_calls:
                    # Format arguments for display
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    print(f"[Round {round_number}] {tool_call.name}({args_str})")

                    # Notify callback of tool call
                    if self.execute_log_callback:
                        self.execute_log_callback.on_tool_call(
                            round_number, tool_call.name, tool_call.arguments
                        )
            else:
                print(f"[Round {round_number}] final answer")

                # Notify callback of round completion
                if self.execute_log_callback:
                    self.execute_log_callback.on_round_complete(round_number)

            # The assistant message must be stored before its tool results.
            messages.append(
                self._build_assistant_message(assistant_turn)
            )

            # No tool calls means the model has finished the task.
            if not assistant_turn.tool_calls:
                if assistant_turn.content:
                    logger.info(
                        "Agent completed after %d round(s)",
                        round_number,
                    )
                    return assistant_turn.content

                raise AgentLoopError(
                    "Model returned neither tool calls nor final content"
                )

            for tool_call in assistant_turn.tool_calls:
                logger.info(
                    "Executing tool: %s, arguments=%s",
                    tool_call.name,
                    tool_call.arguments,
                )

                tool_result = self._execute_tool(tool_call)

                # Update state tracking
                if self.enable_progress_tracking:
                    self.state.record_tool_call(tool_call.name, tool_result.ok)

                    # Track file operations
                    if tool_call.name in ["edit_file", "edit_file_by_line", "apply_patch"] and tool_result.ok and "path" in tool_call.arguments:
                        self.state.record_file_edit(tool_call.arguments["path"])
                    elif tool_call.name == "read_file" and tool_result.ok and "path" in tool_call.arguments:
                        self.state.record_file_read(tool_call.arguments["path"])

                logger.info(
                    "Tool completed: %s, success=%s",
                    tool_call.name,
                    tool_result.ok,
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": self._format_tool_result(tool_result),
                    }
                )

        raise AgentLoopLimitError(
            f"Agent exceeded the maximum of "
            f"{self.max_rounds} rounds"
        )

    def _inject_state_context(self, messages: list[dict[str, Any]], round_number: int) -> list[dict[str, Any]]:
        """Inject state summary into the system message for agent context.

        This provides the agent with information about its progress and operations
        to help it make better decisions about next steps.

        Args:
            messages: Current message history
            round_number: Current round number

        Returns:
            Enhanced messages with state context injected
        """
        if not self.enable_progress_tracking:
            return messages

        # Only inject state context periodically to avoid token bloat
        if round_number % 3 != 0 and round_number != 1:
            return messages

        state_summary = self.state.get_progress_summary()
        state_context = f"\n\n[Current Progress]\n{state_summary}\n"

        # Inject into the system message
        enhanced_messages = []
        for msg in messages:
            if msg["role"] == "system":
                enhanced_msg = msg.copy()
                enhanced_msg["content"] = msg["content"] + state_context
                enhanced_messages.append(enhanced_msg)
            else:
                enhanced_messages.append(msg)

        return enhanced_messages

    def _execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute one tool call through the registry.

        Tool failures are converted into observations so that the model
        can inspect the error and decide whether to retry.
        """
        try:
            result = self.tools.execute(
                name=tool_call.name,
                arguments=tool_call.arguments,
            )
        except KeyError:
            return ToolResult(
                ok=False,
                content=(
                    f"Unknown tool: {tool_call.name}. "
                    f"Available tools: "
                    f"{', '.join(self.tools.get_available_tools())}"
                ),
            )
        except Exception as exc:
            logger.exception(
                "Tool execution failed: %s",
                tool_call.name,
            )

            return ToolResult(
                ok=False,
                content=(
                    f"Tool execution failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )

        if not isinstance(result, ToolResult):
            return ToolResult(
                ok=False,
                content=(
                    f"Tool {tool_call.name} returned an invalid "
                    f"result type: {type(result).__name__}"
                ),
            )

        return result

    @staticmethod
    def _build_assistant_message(
        turn: AssistantTurn,
    ) -> dict[str, Any]:
        """Convert an internal AssistantTurn into a chat message."""
        message: dict[str, Any] = {
            "role": "assistant",
            "content": turn.content,
        }

        if turn.tool_calls:
            message["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        # OpenAI-compatible APIs expect JSON text here.
                        "arguments": json.dumps(
                            tool_call.arguments,
                            ensure_ascii=False,
                        ),
                    },
                }
                for tool_call in turn.tool_calls
            ]

        return message

    @staticmethod
    def _format_tool_result(result: ToolResult) -> str:
        """Format a ToolResult as model-readable text."""
        status = "SUCCESS" if result.ok else "ERROR"
        return f"{status}\n{result.content}"


__all__ = [
    "AgentLoop",
    "AgentLoopError",
    "AgentLoopLimitError",
    "AgentState",
    "ExecuteLogCallback",
]