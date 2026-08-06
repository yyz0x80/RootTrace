from __future__ import annotations

import json
import logging
from typing import Any

from patchpilot.models import AssistantTurn, ToolCall, ToolResult
from patchpilot.prompts import SYSTEM_PROMPT
from patchpilot.provider import LLMProvider
from patchpilot.tools import ToolRegistry

logger = logging.getLogger(__name__)


class AgentLoopError(RuntimeError):
    """Base exception for Agent Loop failures."""


class AgentLoopLimitError(AgentLoopError):
    """Raised when the Agent exceeds the configured round limit."""


class AgentLoop:
    """Coordinate model responses and tool execution."""

    def __init__(
        self,
        provider: LLMProvider,
        tools: ToolRegistry,
        max_rounds: int = 12,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")

        self.provider = provider
        self.tools = tools
        self.max_rounds = max_rounds
        self.system_prompt = system_prompt

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

        tool_schemas = self.tools.get_schemas()

        for round_number in range(1, self.max_rounds + 1):
            logger.info(
                "Starting Agent round %d/%d",
                round_number,
                self.max_rounds,
            )

            assistant_turn = self.provider.complete(
                messages=messages,
                tools=tool_schemas,
            )

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
                    f"{', '.join(self.tools.get_names())}"
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