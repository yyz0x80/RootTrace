"""Shared result contracts for read-only RootTrace tools."""

from dataclasses import dataclass
from enum import StrEnum


class ToolFailureType(StrEnum):
    """Classify failures returned by agent tools."""

    VERIFICATION_FAILURE = "VERIFICATION_FAILURE"
    TOOL_FAILURE = "TOOL_FAILURE"


@dataclass
class ToolResult:
    ok: bool
    content: str
    failure_type: ToolFailureType | None = None
