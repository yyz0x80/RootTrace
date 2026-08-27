from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field


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


class Usage(BaseModel):
    """Exact or null token usage; null means the provider did not return it."""

    llm_calls: int = Field(default=0, ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)

