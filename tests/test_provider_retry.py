"""Tests for rate-limit retry behavior in the LLM provider."""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest
from openai import OpenAIError, RateLimitError
from test_provider_usage import make_provider, response_with_usage

from patchpilot.provider import (
    BACKOFF_BASE_SECONDS,
    BACKOFF_CAP_SECONDS,
    MAX_RETRY_AFTER_SECONDS,
    LLMProvider,
    _retry_after_seconds,
    _retry_delay_seconds,
)


def _rate_limit_error(headers: dict | None = None) -> RateLimitError:
    request = httpx.Request("POST", "https://example.com/chat/completions")
    response = httpx.Response(429, request=request, headers=headers or {})
    return RateLimitError(
        "rate limited",
        response=response,
        body=None,
    )


def test_rate_limit_retries_then_succeeds() -> None:
    provider = make_provider(
        [
            _rate_limit_error(),
            _rate_limit_error(),
            response_with_usage(5, 2),
        ]
    )
    with (
        patch("patchpilot.provider.random") as rng,
        patch("patchpilot.provider.sleep") as sleeper,
    ):
        rng.uniform.side_effect = [0.1, 0.2]
        turn = provider.complete(messages=[], tools=[])

    assert turn.content == "done"
    assert provider.llm_call_count == 1
    create = provider._client.chat.completions.create
    assert create.call_count == 3
    assert sleeper.call_count == 2
    assert sleeper.call_args_list[0].args == (0.1,)
    assert sleeper.call_args_list[1].args == (0.2,)


def test_rate_limit_exhaustion_raises_openai_error() -> None:
    provider = make_provider(
        [_rate_limit_error(), _rate_limit_error(), _rate_limit_error()]
    )
    with (
        patch("patchpilot.provider.random.uniform", return_value=0.0),
        patch("patchpilot.provider.sleep") as sleeper,
        pytest.raises(OpenAIError, match="Rate limit exceeded after 3 attempts"),
    ):
        provider.complete(messages=[], tools=[])
    assert sleeper.call_count == 2


def test_retry_after_header_overrides_backoff() -> None:
    provider = make_provider(
        [
            _rate_limit_error({"Retry-After": "12"}),
            response_with_usage(1, 1),
        ]
    )
    with patch("patchpilot.provider.sleep") as sleeper:
        provider.complete(messages=[], tools=[])
    assert sleeper.call_args_list[0].args == (12.0,)


def test_retry_after_header_is_capped() -> None:
    provider = make_provider(
        [
            _rate_limit_error({"retry-after": "3600"}),
            response_with_usage(1, 1),
        ]
    )
    with patch("patchpilot.provider.sleep") as sleeper:
        provider.complete(messages=[], tools=[])
    assert sleeper.call_args_list[0].args == (MAX_RETRY_AFTER_SECONDS,)


def test_non_rate_limit_errors_are_not_retried() -> None:
    provider = make_provider([OpenAIError("boom")])
    with (
        patch("patchpilot.provider.sleep") as sleeper,
        pytest.raises(OpenAIError, match="boom"),
    ):
        provider.complete(messages=[], tools=[])
    assert sleeper.call_count == 0


def test_retry_delay_uses_capped_jittered_backoff() -> None:
    delays = [
        _retry_delay_seconds(0),
        _retry_delay_seconds(1),
        _retry_delay_seconds(2),
    ]
    assert all(0.0 <= delay <= BACKOFF_CAP_SECONDS for delay in delays)
    assert delays[0] <= BACKOFF_BASE_SECONDS
    assert delays[1] <= BACKOFF_BASE_SECONDS * 2
    assert delays[2] <= BACKOFF_BASE_SECONDS * 4


def test_retry_after_parser_handles_delta_and_date() -> None:
    assert _retry_after_seconds(_rate_limit_error({"Retry-After": "7"})) == 7.0
    assert _retry_after_seconds(_rate_limit_error({})) is None
    assert _retry_after_seconds(_rate_limit_error({"Retry-After": "not-a-date"})) is None
    assert _retry_after_seconds(_rate_limit_error({"Retry-After": "Tue, 15 Nov 1994 08:12:31 GMT"})) >= 0.0


def test_max_retries_validation() -> None:
    with (
        patch("patchpilot.provider.OpenAI") as mock_openai,
        pytest.raises(ValueError, match="max_retries"),
    ):
        LLMProvider(
            model="m",
            api_key="k",
            base_url="https://example.com",
            max_retries=0,
        )
    mock_openai.assert_not_called()


def test_jittered_retries_avoid_lockstep() -> None:
    provider = make_provider(
        [
            _rate_limit_error(),
            _rate_limit_error(),
            response_with_usage(1, 1),
        ]
    )
    with (
        patch("patchpilot.provider.random") as rng,
        patch("patchpilot.provider.sleep") as sleeper,
    ):
        rng.uniform.side_effect = [0.5, 1.0]
        provider.complete(messages=[], tools=[])
    assert sleeper.call_args_list == [((0.5,),), ((1.0,),)]
