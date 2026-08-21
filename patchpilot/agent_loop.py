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
import re
import shlex
from collections import Counter
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from patchpilot.models import (
    AssistantTurn,
    ToolCall,
    ToolFailureType,
    ToolResult,
)
from patchpilot.prompts import SYSTEM_PROMPT
from patchpilot.provider import LLMProvider
from patchpilot.tools import ToolRegistry
from patchpilot.workspace import Workspace

logger = logging.getLogger(__name__)

MAX_FAILURE_SUMMARY_CHARS = 500
DEFAULT_EMPTY_RESPONSE_RETRIES = 2
EDIT_TOOL_NAMES = frozenset({"edit_file", "write_file"})
SENSITIVE_VALUE_PATTERN = re.compile(
    r"(?i)\b([a-z0-9_]*(?:api[_-]?key|token|password|secret))\b"
    r"(\s*[:=]\s*)([^\s,;]+)"
)
AUTHORIZATION_PATTERN = re.compile(
    r"(?i)(authorization\s*:\s*(?:bearer\s+)?)([^\s,;]+)"
)


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

    def on_tool_result(
        self,
        round_number: int,
        tool_name: str,
        args: dict[str, Any],
        result: ToolResult,
        duration_seconds: float,
    ) -> None:
        """Called when a tool execution completes with timing information.

        Args:
            round_number: Current round number
            tool_name: Name of the tool that was executed
            args: Tool arguments
            result: Result of the tool execution
            duration_seconds: Time taken to execute the tool in seconds
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
        consecutive_failures: Number of consecutive failures with the same
            normalized fingerprint.
        last_tool_success: Whether the last tool call succeeded
        total_edits: Total number of edit operations performed
        unique_files_read: Set of file paths that have been read
        recent_failures: List of recent failure signatures for pattern detection
        last_failure_type: Classification of the most recent failure.
        edit_revision: Monotonic revision incremented after each effective edit.
        verified_edit_revision: Revision covered by the latest passing Pytest run.
        linted_edit_revision: Revision covered by the latest passing Ruff run.
        consecutive_completions: Number of consecutive completion attempts without edits.
        last_completion_edit_revision: Edit revision at the last completion attempt.
    """

    files_modified: set[str]
    tool_usage_count: Counter[str]
    consecutive_failures: int
    last_tool_success: bool
    total_edits: int
    unique_files_read: set[str]
    recent_failures: list[str]
    last_failure_type: ToolFailureType | None
    edit_revision: int
    verified_edit_revision: int | None
    linted_edit_revision: int | None
    consecutive_completions: int
    last_completion_edit_revision: int

    def __init__(self) -> None:
        self.files_modified = set()
        self.tool_usage_count = Counter()
        self.consecutive_failures = 0
        self.last_tool_success = True
        self.total_edits = 0
        self.unique_files_read = set()
        self.recent_failures = []
        self.last_failure_type = None
        self.edit_revision = 0
        self.verified_edit_revision = None
        self.linted_edit_revision = None
        self.consecutive_completions = 0
        self.last_completion_edit_revision = 0

    def record_tool_call(
        self,
        tool_name: str,
        success: bool,
        error_content: str = "",
        failure_type: ToolFailureType | None = None,
    ) -> None:
        """Record a tool call and update failure tracking.

        Args:
            tool_name: Name of the tool that was called
            success: Whether the tool call succeeded
            error_content: Error message content for failure pattern detection
            failure_type: Classification for an unsuccessful tool result.
        """
        self.tool_usage_count[tool_name] += 1
        effective_failure_type = failure_type
        if not success and effective_failure_type is None:
            effective_failure_type = ToolFailureType.TOOL_FAILURE
        self.last_failure_type = effective_failure_type
        self.last_tool_success = (
            success
            or effective_failure_type
            == ToolFailureType.VERIFICATION_FAILURE
        )

        if success or effective_failure_type == ToolFailureType.VERIFICATION_FAILURE:
            self.consecutive_failures = 0
            self.recent_failures.clear()
        else:
            failure_signature = self._generate_failure_signature(tool_name, error_content)
            if (
                self.recent_failures
                and self.recent_failures[-1] == failure_signature
            ):
                self.consecutive_failures += 1
            else:
                self.consecutive_failures = 1
            self.recent_failures.append(failure_signature)
            # Keep only the last 5 failures to avoid unbounded growth
            if len(self.recent_failures) > 5:
                self.recent_failures.pop(0)

    def record_file_edit(self, file_path: str) -> None:
        """Record that a file was edited.

        Args:
            file_path: Path to the file that was edited
        """
        self.files_modified.add(file_path)
        self.total_edits += 1
        self.edit_revision += 1
        self.reset_completion_tracking()

    def record_file_read(self, file_path: str) -> None:
        """Record that a file was read.

        Args:
            file_path: Path to the file that was read
        """
        self.unique_files_read.add(file_path)

    def record_pytest_passed(self) -> None:
        """Record a passing Pytest run for the current edit revision."""
        self.verified_edit_revision = self.edit_revision
        self.reset_completion_tracking()

    def record_ruff_passed(self) -> None:
        """Record a passing Ruff run for the current edit revision."""
        self.linted_edit_revision = self.edit_revision
        self.reset_completion_tracking()

    def completion_blocker(self, *, require_edit: bool = True) -> str | None:
        """Return the deterministic reason that completion is not yet allowed."""
        if require_edit and self.edit_revision == 0:
            return "make at least one effective source edit"
        if self.edit_revision == 0:
            return None
        if self.verified_edit_revision != self.edit_revision:
            return "run Pytest successfully after the latest source edit"
        if self.linted_edit_revision != self.edit_revision:
            return "run Ruff successfully after the latest source edit"
        return None

    def record_completion_attempt(self) -> None:
        """Record a completion attempt for NO_PROGRESS detection."""
        self.consecutive_completions += 1
        # Only update the reference revision on the first completion in a sequence
        if self.consecutive_completions == 1:
            self.last_completion_edit_revision = self.edit_revision

    def reset_completion_tracking(self) -> None:
        """Reset completion tracking after an effective edit."""
        self.consecutive_completions = 0
        self.last_completion_edit_revision = self.edit_revision

    def detect_no_progress(self) -> bool:
        """Detect if consecutive completions occurred without patch delta.

        Returns:
            True if there were 2+ consecutive completions with no edit changes
        """
        return (
            self.consecutive_completions >= 2
            and self.last_completion_edit_revision == self.edit_revision
        )

    def _generate_failure_signature(self, tool_name: str, error_content: str) -> str:
        """Generate a signature for failure pattern detection.

        Args:
            tool_name: Name of the tool that failed
            error_content: Error message content

        Returns:
            A string signature representing the failure pattern
        """
        # Normalize error content for pattern matching
        normalized_error = error_content.strip().lower()[:100]  # Limit to 100 chars
        return f"{tool_name}:{normalized_error}"

    def detect_repeated_failure_pattern(self) -> tuple[bool, str]:
        """Detect if the same failure is being repeated.

        Returns:
            Tuple of (is_repeated, failure_description)
        """
        if len(self.recent_failures) < 2:
            return False, ""

        # Check if the last 2 failures are identical
        if self.recent_failures[-1] == self.recent_failures[-2]:
            tool_name = self.recent_failures[-1].split(":", 1)[0]
            return True, f"Repeated {tool_name} failure detected"

        return False, ""

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

        if self.recent_failures:
            summary_parts.append(f"Recent failures: {len(self.recent_failures)}")

        if self.last_failure_type is not None:
            summary_parts.append(
                f"Last failure: {self.last_failure_type.value}"
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


class AgentNoProgressError(AgentLoopError):
    """Raised when repeated completion attempts make no observable progress."""


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
        max_empty_response_retries: int = DEFAULT_EMPTY_RESPONSE_RETRIES,
        force_tool_selection: bool = False,
        enforce_completion_gate: bool = False,
        require_edit_for_completion: bool = True,
    ) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")
        if max_empty_response_retries < 0:
            raise ValueError("max_empty_response_retries must be non-negative")

        self.provider = provider
        self.tools = tools
        self.max_rounds = max_rounds
        self.system_prompt = system_prompt
        self.execute_log_callback = execute_log_callback
        self.enable_early_stopping = enable_early_stopping
        self.max_consecutive_failures = max_consecutive_failures
        self.enable_progress_tracking = enable_progress_tracking
        self.force_tool_selection = force_tool_selection
        self.enforce_completion_gate = enforce_completion_gate
        self.require_edit_for_completion = require_edit_for_completion
        self.max_empty_response_retries = max_empty_response_retries
        self.state = AgentState()

    def update_workspace(self, workspace: Workspace) -> None:
        """Update the workspace used by the tool registry.

        Args:
            workspace: New Workspace instance to use for path resolution
        """
        self.tools.update_workspace(workspace)

    def configure_test_writes(self, allowed_new_files: set[str]) -> None:
        """Apply harness-compiled test artifact permissions to editing tools."""
        self.tools.configure_test_writes(allowed_new_files)

    def run(
        self,
        issue: str,
        *,
        system_prompt: str | None = None,
        reset_state: bool = False,
    ) -> str:
        """Run the Agent Loop until the model returns a final answer.

        Args:
            issue: The repository task or issue description.
            system_prompt: Optional prompt override for a focused execution
                mode such as verification repair.
            reset_state: Whether to discard progress counters from a previous
                independent Agent run.

        Returns:
            The model's final text response.

        Raises:
            ValueError: If the issue is empty.
            AgentLoopError: If the model finishes without a valid response.
            AgentLoopLimitError: If the maximum number of rounds is reached.
        """
        if not issue.strip():
            raise ValueError("issue must not be empty")

        if reset_state:
            self.state = AgentState()

        active_system_prompt = (
            self.system_prompt
            if system_prompt is None
            else system_prompt
        )

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": active_system_prompt,
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

            # Stop only when the same normalized failure repeats. Different
            # tool errors may represent useful recovery attempts.
            if self.enable_early_stopping and self.state.should_stop_early(self.max_consecutive_failures):
                logger.warning(
                    "Early stopping triggered after %d repeated failures",
                    self.state.consecutive_failures,
                )
                raise AgentLoopError(
                    f"Agent stopped early after {self.state.consecutive_failures} "
                    f"repeated tool failures"
                )

            # Notify callback of round start
            if self.execute_log_callback:
                self.execute_log_callback.on_round_start(round_number)

            # Inject state summary into system prompt for context
            enhanced_messages = self._inject_state_context(messages, round_number)

            # Apply forced tool selection for repair rounds
            active_tool_schemas = (
                self._filter_repair_tools(tool_schemas)
                if self.force_tool_selection
                else tool_schemas
            )

            # Require the first repair action to use a tool. Later rounds must
            # allow a text completion after the deterministic gate is satisfied.
            tool_choice = (
                "required"
                if self.force_tool_selection and not self.state.tool_usage_count
                else None
            )

            assistant_turn = self._complete_with_empty_response_retry(
                messages=enhanced_messages,
                tools=active_tool_schemas,
                tool_choice=tool_choice,
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
                print(f"[Round {round_number}] completion candidate")

            # The assistant message must be stored before its tool results.
            messages.append(
                self._build_assistant_message(assistant_turn)
            )

            # No tool calls means the model has finished the task.
            if not assistant_turn.tool_calls:
                content = (assistant_turn.content or "").strip()
                if not content:
                    raise AgentLoopError(
                        "Model returned neither tool calls nor final content"
                    )

                blocker = (
                    self.state.completion_blocker(
                        require_edit=self.require_edit_for_completion,
                    )
                    if self.enforce_completion_gate
                    else None
                )
                self.state.record_completion_attempt()
                if blocker is not None:
                    if self.state.detect_no_progress():
                        logger.warning(
                            "NO_PROGRESS detected after repeated blocked "
                            "completion attempts: %s",
                            blocker,
                        )
                        raise AgentNoProgressError(
                            "Agent stopped after repeated completion attempts "
                            f"without progress: {blocker}"
                        )
                    logger.warning("Completion rejected: %s", blocker)
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Completion rejected by the deterministic gate: "
                                f"{blocker}. Continue with the required tool calls."
                            ),
                        }
                    )
                    continue

                # Check for NO_PROGRESS: consecutive completions without patch delta
                if self.state.detect_no_progress():
                    logger.warning(
                        "NO_PROGRESS detected: %d consecutive completions without patch delta",
                        self.state.consecutive_completions,
                    )
                    raise AgentLoopError(
                        f"Agent stopped after {self.state.consecutive_completions} "
                        f"consecutive completion attempts without making any edits"
                    )

                logger.info(
                    "Agent completed after %d round(s)",
                    round_number,
                )
                if self.execute_log_callback:
                    self.execute_log_callback.on_round_complete(
                        round_number
                    )
                return content

            for tool_call in assistant_turn.tool_calls:
                logger.info(
                    "Executing tool: %s, arguments=%s",
                    tool_call.name,
                    tool_call.arguments,
                )

                started_at = perf_counter()
                tool_result = self._execute_tool(tool_call)
                duration_seconds = perf_counter() - started_at
                if not tool_result.ok and tool_result.failure_type is None:
                    tool_result.failure_type = ToolFailureType.TOOL_FAILURE

                # Notify callback of tool result with timing
                if self.execute_log_callback:
                    self.execute_log_callback.on_tool_result(
                        round_number,
                        tool_call.name,
                        tool_call.arguments,
                        tool_result,
                        duration_seconds,
                    )

                # State is also the completion gate's evidence store, so it
                # must be maintained even when progress display is disabled.
                error_content = tool_result.content if not tool_result.ok else ""
                self.state.record_tool_call(
                    tool_call.name,
                    tool_result.ok,
                    error_content,
                    tool_result.failure_type,
                )

                if self._is_effective_edit(tool_call, tool_result):
                    self.state.record_file_edit(tool_call.arguments["path"])
                elif (
                    tool_call.name == "read_file"
                    and tool_result.ok
                    and "path" in tool_call.arguments
                ):
                    self.state.record_file_read(tool_call.arguments["path"])
                elif (
                    tool_call.name == "run_command"
                    and tool_result.ok
                    and "command" in tool_call.arguments
                ):
                    if self._is_pytest_tool_call(tool_call):
                        self.state.record_pytest_passed()
                    elif self._is_ruff_tool_call(tool_call):
                        self.state.record_ruff_passed()

                logger.info(
                    "Tool completed: %s, success=%s",
                    tool_call.name,
                    tool_result.ok,
                )
                if not tool_result.ok:
                    logger.warning(
                        "Tool failure: name=%s, type=%s, summary=%s",
                        tool_call.name,
                        tool_result.failure_type.value,
                        self._summarize_tool_failure(tool_result.content),
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

    def _complete_with_empty_response_retry(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str | None = None,
    ) -> AssistantTurn:
        """Retry bounded empty model responses without consuming a round."""
        for retry_number in range(self.max_empty_response_retries + 1):
            assistant_turn = self.provider.complete(
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
            )
            if assistant_turn.tool_calls or (
                assistant_turn.content
                and assistant_turn.content.strip()
            ):
                return assistant_turn
            if retry_number < self.max_empty_response_retries:
                logger.warning(
                    "Model returned an empty response; retrying (%d/%d)",
                    retry_number + 1,
                    self.max_empty_response_retries,
                )

        raise AgentLoopError(
            "Model returned neither tool calls nor final content after "
            f"{self.max_empty_response_retries} retries"
        )

    @staticmethod
    def _filter_repair_tools(tool_schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Filter tool schemas to allow repair tools including command execution.

        Repair rounds need read, edit, and command tools to fix code and verify fixes.
        The run_command tool is already secured by command whitelist and workspace policy.

        Args:
            tool_schemas: Full list of available tool schemas

        Returns:
            Filtered list containing the bounded repair tools.
        """
        repair_allowed_tools = {
            "read_file",
            "edit_file",
            "write_file",
            "write_scratch_test",
            "run_command",
        }
        return [
            schema
            for schema in tool_schemas
            if schema.get("function", {}).get("name") in repair_allowed_tools
        ]

    @staticmethod
    def _is_effective_edit(
        tool_call: ToolCall,
        tool_result: ToolResult,
    ) -> bool:
        """Return whether a tool result represents a persisted file change."""
        if (
            tool_call.name not in EDIT_TOOL_NAMES
            or not tool_result.ok
            or "path" not in tool_call.arguments
        ):
            return False
        normalized_content = tool_result.content.strip()
        return (
            normalized_content != "(no diff)"
            and not normalized_content.startswith("PREVIEW MODE")
        )

    @staticmethod
    def _is_pytest_tool_call(tool_call: ToolCall) -> bool:
        """Return whether a tool call invokes an allowed Pytest command."""
        if tool_call.name != "run_command":
            return False
        command = tool_call.arguments.get("command")
        if not isinstance(command, str):
            return False
        try:
            args = shlex.split(command)
        except ValueError:
            return False
        return bool(
            args
            and (
                args[0] == "pytest"
                or args[:3] == ["python", "-m", "pytest"]
            )
        )

    @staticmethod
    def _is_ruff_tool_call(tool_call: ToolCall) -> bool:
        """Return whether a tool call invokes an allowed Ruff check."""
        if tool_call.name != "run_command":
            return False
        command = tool_call.arguments.get("command")
        if not isinstance(command, str):
            return False
        try:
            args = shlex.split(command)
        except ValueError:
            return False
        return args[:2] == ["ruff", "check"]



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

        # Check for repeated failure patterns and add recovery guidance
        if self.enable_early_stopping:
            is_repeated, failure_desc = self.state.detect_repeated_failure_pattern()
            if is_repeated:
                state_context += f"\n[WARNING] {failure_desc}. You MUST re-read the relevant file(s) before retrying. Do not repeat the same operation.\n"

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

    def _summarize_tool_failure(self, content: str) -> str:
        """Return a redacted and bounded terminal summary for a tool failure."""
        sanitizer = getattr(self.tools, "sanitize_workspace_paths", None)
        if callable(sanitizer):
            sanitized_content = sanitizer(content)
            if isinstance(sanitized_content, str):
                content = sanitized_content

        content = SENSITIVE_VALUE_PATTERN.sub(
            lambda match: f"{match.group(1)}{match.group(2)}<redacted>",
            content,
        )
        content = AUTHORIZATION_PATTERN.sub(
            lambda match: f"{match.group(1)}<redacted>",
            content,
        )
        summary = " ".join(content.split())
        if len(summary) <= MAX_FAILURE_SUMMARY_CHARS:
            return summary
        return f"{summary[: MAX_FAILURE_SUMMARY_CHARS - 3]}..."

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
        if result.ok:
            return f"SUCCESS\n{result.content}"

        failure_type = result.failure_type or ToolFailureType.TOOL_FAILURE
        status = f"ERROR [{failure_type.value}]"
        return f"{status}\n{result.content}"


__all__ = [
    "AgentLoop",
    "AgentLoopError",
    "AgentLoopLimitError",
    "AgentState",
    "ExecuteLogCallback",
]
