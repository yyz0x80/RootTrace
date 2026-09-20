"""Language-model provider, configuration, schema, and usage contracts."""

from roottrace.llm.config import ModelConfig, ModelConfigManager
from roottrace.llm.errors import (
    is_llm_error,
    is_non_retryable_llm_error,
    is_quota_exhausted,
    is_retryable_llm_error,
    llm_retry_delay_seconds,
)
from roottrace.llm.provider import (
    LLMProvider,
    ToolCallParseError,
    create_provider_from_config,
)
from roottrace.llm.schema import AssistantTurn, ToolCall, Usage
from roottrace.llm.usage import UsageTracker

__all__ = [
    "AssistantTurn",
    "LLMProvider",
    "ModelConfig",
    "ModelConfigManager",
    "ToolCall",
    "ToolCallParseError",
    "Usage",
    "UsageTracker",
    "create_provider_from_config",
    "is_llm_error",
    "is_non_retryable_llm_error",
    "is_quota_exhausted",
    "is_retryable_llm_error",
    "llm_retry_delay_seconds",
]
