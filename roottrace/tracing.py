"""Structured execution tracing for RootTrace runs.

This module provides structured tracing functionality that records detailed
execution events as JSONL (JSON Lines) for workflow analysis, debugging,
and audit purposes. The trace captures the complete execution lifecycle
including tool calls, model interactions, and verification results.

The trace module maintains minimal dependencies and focuses on reliable
event recording without affecting workflow performance.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

SENSITIVE_ARGUMENTS = {
    "old_text",
    "new_text",
    "content",
    "prompt",
    "api_key",
    "token",
    "password",
    "secret",
}


def summarize_tool_arguments(
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Redact sensitive information from tool arguments.

    Replaces sensitive argument values with "<redacted>" and truncates
    long string values to prevent excessive trace file size.

    Args:
        arguments: Dictionary of tool argument names to values.

    Returns:
        Dictionary with sensitive values redacted and long strings truncated.
    """
    summary: dict[str, Any] = {}

    for key, value in arguments.items():
        if key.lower() in SENSITIVE_ARGUMENTS:
            summary[key] = "<redacted>"
        elif isinstance(value, str) and len(value) > 200:
            summary[key] = f"<{len(value)} chars>"
        else:
            summary[key] = value

    return summary


class TraceEvent(BaseModel):
    """Single execution event in the workflow trace.

    Represents a discrete event during workflow execution, capturing
    contextual information about the workflow stage, model interactions,
    tool usage, and verification results.

    Attributes:
        run_id: Unique identifier for the workflow execution run.
        event_type: Type of event (e.g., "tool_call", "verification", "completion").
        timestamp: ISO 8601 timestamp when the event occurred.
        workflow_stage: Current RCA workflow stage (e.g., "PLANNING", "VERIFY").
        model: Model identifier if the event involves model interaction.
        tool_name: Name of the tool being called (for tool_call events).
        tool_arguments: Arguments passed to the tool.
        tool_duration: Duration of tool execution in seconds.
        permission_result: Result of permission check for tool execution.
        modified_files: List of files modified by this event.
        round_number: Agent round associated with a tool event.
        verification_result: Detailed verification command results.
        degradation: Structured metadata when a bounded fallback was used.
        git_verification_policy: Deterministic policy for bounded Git access.
        retry_count: Number of retry attempts for this operation.
        final_status: Final status of the operation (e.g., "SUCCESS", "FAILURE").
        prompt_tokens: Number of prompt tokens used (for model events).
        completion_tokens: Number of completion tokens used (for model events).
        total_cost: Total cost in USD for the operation (for model events).
    """

    run_id: str
    event_type: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )
    workflow_stage: str
    model: str | None = None
    tool_name: str | None = None
    tool_arguments: dict[str, Any] | None = None
    tool_duration: float | None = None
    permission_result: str | None = None
    modified_files: list[str] = Field(default_factory=list)
    round_number: int | None = None
    verification_result: dict[str, Any] | None = None
    degradation: dict[str, Any] | None = None
    git_verification_policy: dict[str, Any] | None = None
    retry_count: int = 0
    final_status: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_cost: float | None = None


class TraceWriter:
    """Writer for execution trace events in JSONL format.

    Provides thread-safe appending of trace events to a JSONL file,
    where each line is a self-contained JSON object. This format allows
    for efficient streaming writes and easy line-by-line parsing.

    The writer automatically creates parent directories as needed and
    handles UTF-8 encoding for international character support.

    Example:
        writer = TraceWriter(Path("/tmp/trace.jsonl"))
        event = TraceEvent(
            run_id="abc123",
            event_type="tool_call",
            workflow_stage="EVIDENCE",
            tool_name="read_file",
            tool_arguments={"path": "src/main.py"}
        )
        writer.write(event)
    """

    def __init__(self, path: Path) -> None:
        """Initialize the trace writer.

        Args:
            path: File path where trace events will be written.
                Parent directories will be created automatically if they don't exist.
        """
        self.path = path

    def write(self, event: TraceEvent) -> None:
        """Write a trace event to the JSONL file.

        The event is serialized to JSON and appended as a single line
        followed by a newline character. Parent directories are created
        if they don't exist. Sensitive tool arguments are redacted before
        writing.

        Args:
            event: TraceEvent instance to write to the trace file.
        """
        if event.tool_arguments is not None:
            event.tool_arguments = summarize_tool_arguments(event.tool_arguments)
        if event.degradation is not None:
            event.degradation = summarize_tool_arguments(event.degradation)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(event.model_dump_json())
            stream.write("\n")

    def start_run(self) -> None:
        """Initialize an empty trace file for a new workflow run.

        A workflow artifact represents one run. Truncating it at the start
        prevents events from previous runs from being mixed with the current
        run while preserving append behavior for individual event writes.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")
