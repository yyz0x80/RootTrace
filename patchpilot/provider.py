"""Provider module for calling OpenAI-compatible APIs.

This module handles communication with OpenAI-compatible API providers,
including request conversion, response parsing, error handling, and retries.
"""

import json
import os
import random
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from time import sleep
from typing import Any

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)

from patchpilot.models import AssistantTurn, ToolCall

# Load environment variables from .env file if it exists
load_dotenv()

DEFAULT_MAX_RETRIES = 4
BACKOFF_BASE_SECONDS = 1.0
BACKOFF_CAP_SECONDS = 8.0
MAX_RETRY_AFTER_SECONDS = 30.0


def _retry_after_seconds(error: OpenAIError) -> float | None:
    """Return the server-requested retry delay in seconds, when advertised.

    OpenAI-compatible providers may include a ``Retry-After`` header (delta
    seconds or an HTTP date) on 429 or 5xx responses. Honoring it avoids
    retrying too early; a missing or unparsable header returns ``None`` so the
    caller falls back to jittered exponential backoff.
    """
    response = getattr(error, "response", None)
    headers = getattr(error, "headers", None) or getattr(response, "headers", None)
    if not headers:
        return None
    value = headers.get("Retry-After") or headers.get("retry-after")
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    try:
        retry_at = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if retry_at.tzinfo is None:
        now = datetime.now(UTC).replace(tzinfo=None)
    else:
        now = datetime.now(UTC)
    return max((retry_at - now).total_seconds(), 0.0)


def _retry_delay_seconds(
    attempt: int,
    *,
    retry_after_seconds: float | None = None,
    rng: random.Random | None = None,
) -> float:
    """Compute a bounded, jittered delay before the next transient retry.

    A server-provided ``Retry-After`` value takes precedence and is capped to
    keep total retry time bounded. Otherwise the delay uses capped exponential
    backoff with full jitter so concurrent specialists do not retry in lockstep
    and re-trigger the same limit together.
    """
    if retry_after_seconds is not None:
        return min(max(retry_after_seconds, 0.0), MAX_RETRY_AFTER_SECONDS)
    backoff = min(BACKOFF_CAP_SECONDS, BACKOFF_BASE_SECONDS * (2**attempt))
    rng = rng or random
    return rng.uniform(0.0, backoff)


def _is_retryable(error: OpenAIError) -> bool:
    """Return whether a provider error is transient and safe to retry.

    Rate limits (429), connection failures, request timeouts, and server-side
    (5xx) errors may succeed on a later attempt. Client errors (4xx) and other
    terminal failures must surface immediately instead of burning retry budget.
    """
    if isinstance(error, (RateLimitError, APIConnectionError, APITimeoutError)):
        return True
    if isinstance(error, APIStatusError):
        return error.status_code >= 500
    return False


class ToolCallParseError(Exception):
    """Raised when tool call arguments cannot be parsed as JSON."""



class LLMProvider:
    """Provider for OpenAI-compatible API interactions.

    This class handles:
    - API authentication and configuration
    - Message and tool conversion
    - Response parsing and tool call extraction
    - Error handling and retries for transient failures (rate limits,
      connection errors, timeouts, and server errors)
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        """Initialize the provider with explicit or environment configuration.

        Args:
            model: Optional model identifier.
            api_key: Optional API key. If not provided, reads from ZHIPU_API_KEY.
            base_url: Optional API base URL. If not provided, reads from PATCHPILOT_BASE_URL.
            max_retries: Total completion attempts for retryable transient
                failures (rate limits, connection errors, timeouts, 5xx).

        Raises:
            ValueError: If API key or base URL cannot be determined.
        """
        if not 1 <= max_retries <= 10:
            raise ValueError("max_retries must be between 1 and 10")
        if api_key is None:
            api_key = os.getenv("ZHIPU_API_KEY")
        if not api_key:
            raise ValueError("ZHIPU_API_KEY environment variable is not set")

        if base_url is None:
            base_url = os.getenv("PATCHPILOT_BASE_URL")
        if not base_url:
            raise ValueError("PATCHPILOT_BASE_URL environment variable is not set")

        configured_model = model or os.getenv("PATCHPILOT_MODEL")
        if not configured_model:
            raise ValueError(
                "Model is required through --model or PATCHPILOT_MODEL"
            )

        # The provider owns the retry policy, so SDK-internal retries are
        # disabled to keep attempt counts and timing deterministic/auditable.
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
        )
        self._model = configured_model
        self._max_retries = max_retries
        self._llm_call_count = 0
        self._prompt_tokens: int | None = 0
        self._completion_tokens: int | None = 0
        self._reasoning_tokens: int | None = 0

    @property
    def model(self) -> str:
        """Get the configured model identifier.

        Returns:
            The model name being used for API calls.
        """
        return self._model

    @property
    def llm_call_count(self) -> int:
        """Return the number of successful model completions."""
        return self._llm_call_count

    @property
    def prompt_tokens(self) -> int | None:
        """Return exact accumulated prompt tokens when fully available."""
        return self._prompt_tokens

    @property
    def completion_tokens(self) -> int | None:
        """Return exact accumulated completion tokens when fully available."""
        return self._completion_tokens

    @property
    def reasoning_tokens(self) -> int | None:
        """Return exact accumulated reasoning tokens when fully available."""
        return self._reasoning_tokens

    def _record_usage(
        self,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        reasoning_tokens: int | None = None,
    ) -> None:
        """Record one successful completion without estimating missing usage."""
        self._llm_call_count += 1

        if prompt_tokens is None:
            self._prompt_tokens = None
        elif self._prompt_tokens is not None:
            self._prompt_tokens += prompt_tokens

        if completion_tokens is None:
            self._completion_tokens = None
        elif self._completion_tokens is not None:
            self._completion_tokens += completion_tokens

        if reasoning_tokens is None:
            self._reasoning_tokens = None
        elif self._reasoning_tokens is not None:
            self._reasoning_tokens += reasoning_tokens

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str | None = None,
    ) -> AssistantTurn:
        """Complete a conversation turn with the LLM.

        Args:
            messages: List of message dictionaries with role and content
            tools: List of tool schemas available for the model to call
            tool_choice: Optional OpenAI-compatible tool selection mode.

        Returns:
            AssistantTurn containing the response content and any tool calls

        Raises:
            ToolCallParseError: If tool call arguments cannot be parsed as JSON
            OpenAIError: For retryable API errors after all attempts are
                exhausted; non-retryable errors raise immediately
        """
        last_error = None

        for attempt in range(self._max_retries):
            try:
                api_params: dict[str, Any] = {
                    "model": self._model,
                    "messages": messages,
                    "tools": tools,
                }
                if tool_choice is not None:
                    api_params["tool_choice"] = tool_choice

                response = self._client.chat.completions.create(**api_params)

                message = response.choices[0].message
                content = message.content
                tool_calls = []
                usage = getattr(response, "usage", None)
                prompt_tokens = getattr(usage, "prompt_tokens", None)
                completion_tokens = getattr(
                    usage,
                    "completion_tokens",
                    None,
                )
                details = getattr(usage, "completion_tokens_details", None)
                reasoning_tokens = getattr(details, "reasoning_tokens", None)
                self._record_usage(
                    prompt_tokens,
                    completion_tokens,
                    reasoning_tokens,
                )

                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        try:
                            arguments = json.loads(tool_call.function.arguments)
                        except json.JSONDecodeError as e:
                            raise ToolCallParseError(
                                f"Failed to parse tool call arguments as JSON: "
                                f"tool={tool_call.function.name}, "
                                f"arguments={tool_call.function.arguments}"
                            ) from e

                        tool_calls.append(
                            ToolCall(
                                id=tool_call.id,
                                name=tool_call.function.name,
                                arguments=arguments,
                            )
                        )

                return AssistantTurn(
                    content=content,
                    tool_calls=tool_calls,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    reasoning_tokens=reasoning_tokens,
                )

            except OpenAIError as e:
                if not _is_retryable(e):
                    raise
                last_error = e
                if attempt >= self._max_retries - 1:
                    if isinstance(e, RateLimitError):
                        message = (
                            f"Rate limit exceeded after "
                            f"{self._max_retries} attempts"
                        )
                    else:
                        message = (
                            f"{type(e).__name__} persisted after "
                            f"{self._max_retries} attempts"
                        )
                    raise OpenAIError(message) from e
                sleep(
                    _retry_delay_seconds(
                        attempt,
                        retry_after_seconds=_retry_after_seconds(e),
                    )
                )

        # This should not be reached, but kept for type safety
        raise OpenAIError("Unexpected error in completion logic") from last_error

    def generate_text(self, prompt: str) -> str:
        """Generate text response from a simple prompt.

        This is a convenience method for simple text generation without tools.
        It adapts the simple prompt-response interface to the structured
        complete() method.

        Args:
            prompt: The input prompt text.

        Returns:
            The generated text response.

        Raises:
            ValueError: If the response content is None.
            OpenAIError: For API-related errors.
        """
        messages = [{"role": "user", "content": prompt}]
        response = self.complete(messages=messages, tools=[])

        if response.content is None:
            raise ValueError("LLM returned None content")

        return response.content


def create_provider_from_config(
    model_name: str | None = None,
    config_path: str | None = None,
) -> LLMProvider:
    """Create provider using model configuration file.

    This function attempts to resolve model configuration from a config file,
    falling back to environment variables if the model is not found.

    Args:
        model_name: Optional model name from config file.
                    If not provided, uses environment variables.
        config_path: Optional path to model configuration file.

    Returns:
        Configured LLMProvider instance.

    Raises:
        ValueError: If configuration cannot be resolved.
    """
    from patchpilot.model_config import ModelConfigManager

    manager = ModelConfigManager(config_path=Path(config_path) if config_path else None)

    if model_name:
        config = manager.get_config(model_name)
        if config:
            return LLMProvider(
                model=config.model_id,
                api_key=config.api_key,
                base_url=config.base_url,
            )

    # Fallback to environment variables
    return LLMProvider(model=model_name)
