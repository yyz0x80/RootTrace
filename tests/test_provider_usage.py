"""Tests for exact provider usage accounting."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

from patchpilot.provider import LLMProvider, ToolCallParseError


def make_provider(responses: list[SimpleNamespace]) -> LLMProvider:
    """Create a provider with a deterministic completion client."""
    provider = object.__new__(LLMProvider)
    provider._model = "test-model"
    provider._llm_call_count = 0
    provider._prompt_tokens = 0
    provider._completion_tokens = 0
    create = Mock(side_effect=responses)
    provider._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        )
    )
    return provider


def response_with_usage(
    prompt_tokens: int | None,
    completion_tokens: int | None,
) -> SimpleNamespace:
    """Build a minimal OpenAI-compatible response."""
    usage = (
        None
        if prompt_tokens is None and completion_tokens is None
        else SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
    )
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="done", tool_calls=[]),
            )
        ],
        usage=usage,
    )


def test_provider_accumulates_exact_usage() -> None:
    provider = make_provider(
        [
            response_with_usage(10, 4),
            response_with_usage(20, 6),
        ]
    )

    first = provider.complete(messages=[], tools=[])
    provider.complete(messages=[], tools=[])

    assert first.prompt_tokens == 10
    assert first.completion_tokens == 4
    assert provider.llm_call_count == 2
    assert provider.prompt_tokens == 30
    assert provider.completion_tokens == 10


def test_provider_keeps_tokens_unknown_after_missing_usage() -> None:
    provider = make_provider(
        [
            response_with_usage(10, 4),
            response_with_usage(None, None),
            response_with_usage(20, 6),
        ]
    )

    provider.complete(messages=[], tools=[])
    provider.complete(messages=[], tools=[])
    provider.complete(messages=[], tools=[])

    assert provider.llm_call_count == 3
    assert provider.prompt_tokens is None
    assert provider.completion_tokens is None


def test_provider_records_usage_before_rejecting_invalid_tool_arguments() -> None:
    response = response_with_usage(15, 5)
    response.choices[0].message.tool_calls = [
        SimpleNamespace(
            id="call-1",
            function=SimpleNamespace(name="read_file", arguments="{"),
        )
    ]
    provider = make_provider([response])

    try:
        provider.complete(messages=[], tools=[])
    except ToolCallParseError:
        pass
    else:
        raise AssertionError("Expected invalid tool arguments to be rejected")

    assert provider.llm_call_count == 1
    assert provider.prompt_tokens == 15
    assert provider.completion_tokens == 5
