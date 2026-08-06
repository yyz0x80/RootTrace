from dataclasses import dataclass
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


@dataclass
class ToolResult:
    ok: bool
    content: str