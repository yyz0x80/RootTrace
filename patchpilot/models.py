from dataclasses import dataclass
from enum import StrEnum
from typing import Any


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class AssistantTurn:
    content: str | None
    tool_calls: list[ToolCall]
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None


class ToolFailureType(StrEnum):
    """Classify failures returned by agent tools."""

    VERIFICATION_FAILURE = "VERIFICATION_FAILURE"
    TOOL_FAILURE = "TOOL_FAILURE"


@dataclass
class ToolResult:
    ok: bool
    content: str
    failure_type: ToolFailureType | None = None
