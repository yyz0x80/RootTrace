"""Tests for in-call LLM retry behaviour and failure classification."""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    OpenAIError,
    RateLimitError,
)
from test_provider_usage import make_provider, response_with_usage

from roottrace.llm.errors import (
    BACKOFF_BASE_SECONDS,
    BACKOFF_CAP_SECONDS,
    MAX_RETRY_AFTER_SECONDS,
    MAX_TOTAL_RETRY_WAIT_SECONDS,
    is_llm_error,
    is_non_retryable_llm_error,
    is_quota_exhausted,
    is_retryable_llm_error,
    llm_retry_delay_seconds,
    retry_after_seconds,
    retry_delay_seconds,
)
from roottrace.llm.provider import LLMProvider

# ``make_provider`` configures three in-call retries, so four requests is the
# full budget for one completion.
PROVIDER_RETRIES = 3
FULL_ATTEMPTS = PROVIDER_RETRIES + 1


def _request() -> httpx.Request:
    return httpx.Request("POST", "https://example.com/chat/completions")


def _rate_limit_error(
    headers: dict | None = None,
    *,
    body: object | None = None,
) -> RateLimitError:
    response = httpx.Response(429, request=_request(), headers=headers or {})
    return RateLimitError(
        "rate limited",
        response=response,
        body=body,
    )


def _quota_exhausted_error() -> RateLimitError:
    return _rate_limit_error(
        body={
            "error": {
                "message": "You exceeded your current quota",
                "type": "insufficient_quota",
                "code": "insufficient_quota",
            }
        }
    )


def _connection_error() -> APIConnectionError:
    return APIConnectionError(request=_request())


def _status_error(
    status_code: int,
    headers: dict | None = None,
) -> APIStatusError:
    response = httpx.Response(
        status_code,
        request=_request(),
        headers=headers or {},
    )
    return APIStatusError("upstream error", response=response, body=None)


def test_rate_limit_retries_then_succeeds() -> None:
    provider = make_provider(
        [
            _rate_limit_error(),
            _rate_limit_error(),
            response_with_usage(5, 2),
        ]
    )
    with (
        patch("roottrace.llm.errors.random") as rng,
        patch("roottrace.llm.provider.sleep") as sleeper,
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


def test_rate_limit_exhaustion_reraises_rate_limit_error() -> None:
    provider = make_provider([_rate_limit_error() for _ in range(FULL_ATTEMPTS)])
    with (
        patch("roottrace.llm.errors.random.uniform", return_value=0.0),
        patch("roottrace.llm.provider.sleep") as sleeper,
        pytest.raises(RateLimitError),
    ):
        provider.complete(messages=[], tools=[])

    create = provider._client.chat.completions.create
    assert create.call_count == FULL_ATTEMPTS
    assert sleeper.call_count == PROVIDER_RETRIES


def test_quota_exhaustion_is_not_retried() -> None:
    error = _quota_exhausted_error()
    assert is_quota_exhausted(error)
    assert not is_retryable_llm_error(error)
    assert is_non_retryable_llm_error(error)

    provider = make_provider([error])
    with (
        patch("roottrace.llm.provider.sleep") as sleeper,
        pytest.raises(RateLimitError),
    ):
        provider.complete(messages=[], tools=[])

    create = provider._client.chat.completions.create
    assert create.call_count == 1
    assert sleeper.call_count == 0


def test_retry_after_header_overrides_backoff() -> None:
    provider = make_provider(
        [
            _rate_limit_error({"Retry-After": "12"}),
            response_with_usage(1, 1),
        ]
    )
    with patch("roottrace.llm.provider.sleep") as sleeper:
        provider.complete(messages=[], tools=[])
    assert sleeper.call_args_list[0].args == (12.0,)


def test_retry_after_header_is_capped() -> None:
    provider = make_provider(
        [
            _rate_limit_error({"retry-after": "3600"}),
            response_with_usage(1, 1),
        ]
    )
    with patch("roottrace.llm.provider.sleep") as sleeper:
        provider.complete(messages=[], tools=[])
    assert sleeper.call_args_list[0].args == (MAX_RETRY_AFTER_SECONDS,)


def test_total_retry_wait_is_bounded() -> None:
    provider = make_provider(
        [
            _rate_limit_error({"Retry-After": str(MAX_RETRY_AFTER_SECONDS)})
            for _ in range(FULL_ATTEMPTS)
        ]
    )
    with (
        patch("roottrace.llm.provider.sleep") as sleeper,
        pytest.raises(RateLimitError),
    ):
        provider.complete(messages=[], tools=[])

    waited = sum(call.args[0] for call in sleeper.call_args_list)
    assert waited <= MAX_TOTAL_RETRY_WAIT_SECONDS
    create = provider._client.chat.completions.create
    assert create.call_count < FULL_ATTEMPTS


def test_non_retryable_errors_are_not_retried() -> None:
    provider = make_provider([OpenAIError("boom")])
    with (
        patch("roottrace.llm.provider.sleep") as sleeper,
        pytest.raises(OpenAIError, match="boom"),
    ):
        provider.complete(messages=[], tools=[])
    assert sleeper.call_count == 0


def test_connection_error_retries_then_succeeds() -> None:
    provider = make_provider(
        [
            _connection_error(),
            _connection_error(),
            response_with_usage(5, 2),
        ]
    )
    with (
        patch("roottrace.llm.errors.random") as rng,
        patch("roottrace.llm.provider.sleep") as sleeper,
    ):
        rng.uniform.side_effect = [0.1, 0.2]
        turn = provider.complete(messages=[], tools=[])

    assert turn.content == "done"
    assert provider.llm_call_count == 1
    create = provider._client.chat.completions.create
    assert create.call_count == 3
    assert sleeper.call_count == 2
    assert sleeper.call_args_list[0].args == (0.1,)


def test_connection_error_exhaustion_reraises_connection_error() -> None:
    provider = make_provider([_connection_error() for _ in range(FULL_ATTEMPTS)])
    with (
        patch("roottrace.llm.errors.random.uniform", return_value=0.0),
        patch("roottrace.llm.provider.sleep") as sleeper,
        pytest.raises(APIConnectionError),
    ):
        provider.complete(messages=[], tools=[])
    assert sleeper.call_count == PROVIDER_RETRIES


def test_timeout_error_retries_then_succeeds() -> None:
    provider = make_provider(
        [APITimeoutError(request=_request()), response_with_usage(1, 1)]
    )
    with (
        patch("roottrace.llm.errors.random.uniform", return_value=0.0),
        patch("roottrace.llm.provider.sleep") as sleeper,
    ):
        turn = provider.complete(messages=[], tools=[])

    assert turn.content == "done"
    assert provider.llm_call_count == 1
    create = provider._client.chat.completions.create
    assert create.call_count == 2
    assert sleeper.call_count == 1


def test_server_error_retries_then_succeeds() -> None:
    provider = make_provider([_status_error(503), response_with_usage(1, 1)])
    with (
        patch("roottrace.llm.errors.random.uniform", return_value=0.0),
        patch("roottrace.llm.provider.sleep") as sleeper,
    ):
        turn = provider.complete(messages=[], tools=[])

    assert turn.content == "done"
    create = provider._client.chat.completions.create
    assert create.call_count == 2
    assert sleeper.call_count == 1


def test_server_error_retry_after_header_is_honored() -> None:
    provider = make_provider(
        [_status_error(503, {"Retry-After": "12"}), response_with_usage(1, 1)]
    )
    with patch("roottrace.llm.provider.sleep") as sleeper:
        provider.complete(messages=[], tools=[])
    assert sleeper.call_args_list[0].args == (12.0,)


def test_client_error_raises_immediately() -> None:
    response = httpx.Response(400, request=_request())
    error = BadRequestError("bad request", response=response, body=None)
    provider = make_provider([error])
    with (
        patch("roottrace.llm.provider.sleep") as sleeper,
        pytest.raises(BadRequestError, match="bad request"),
    ):
        provider.complete(messages=[], tools=[])
    create = provider._client.chat.completions.create
    assert create.call_count == 1
    assert sleeper.call_count == 0


def test_authentication_error_raises_immediately() -> None:
    response = httpx.Response(401, request=_request())
    error = AuthenticationError("invalid api key", response=response, body=None)
    provider = make_provider([error])
    with (
        patch("roottrace.llm.provider.sleep") as sleeper,
        pytest.raises(AuthenticationError, match="invalid api key"),
    ):
        provider.complete(messages=[], tools=[])
    create = provider._client.chat.completions.create
    assert create.call_count == 1
    assert sleeper.call_count == 0


def test_provider_disables_sdk_retries() -> None:
    with patch("roottrace.llm.provider.OpenAI") as mock_openai:
        LLMProvider(
            model="m",
            api_key="k",
            base_url="https://example.com",
        )
    assert mock_openai.call_args.kwargs["max_retries"] == 0


def test_retry_delay_uses_capped_jittered_backoff() -> None:
    delays = [
        retry_delay_seconds(0),
        retry_delay_seconds(1),
        retry_delay_seconds(2),
    ]
    assert all(0.0 <= delay <= BACKOFF_CAP_SECONDS for delay in delays)
    assert delays[0] <= BACKOFF_BASE_SECONDS
    assert delays[1] <= BACKOFF_BASE_SECONDS * 2
    assert delays[2] <= BACKOFF_BASE_SECONDS * 4


def test_retry_after_parser_handles_delta_and_date() -> None:
    assert retry_after_seconds(_rate_limit_error({"Retry-After": "7"})) == 7.0
    assert retry_after_seconds(_rate_limit_error({})) is None
    assert retry_after_seconds(_rate_limit_error({"Retry-After": "not-a-date"})) is None
    assert (
        retry_after_seconds(
            _rate_limit_error({"Retry-After": "Tue, 15 Nov 1994 08:12:31 GMT"})
        )
        >= 0.0
    )


def test_max_retries_validation() -> None:
    with patch("roottrace.llm.provider.OpenAI") as mock_openai:
        provider = LLMProvider(
            model="m",
            api_key="k",
            base_url="https://example.com",
            max_retries=0,
        )
        assert provider.complete is not None
        mock_openai.assert_called_once()

    for invalid in (-1, 11):
        with (
            patch("roottrace.llm.provider.OpenAI") as mock_openai,
            pytest.raises(ValueError, match="max_retries"),
        ):
            LLMProvider(
                model="m",
                api_key="k",
                base_url="https://example.com",
                max_retries=invalid,
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
        patch("roottrace.llm.errors.random") as rng,
        patch("roottrace.llm.provider.sleep") as sleeper,
    ):
        rng.uniform.side_effect = [0.5, 1.0]
        provider.complete(messages=[], tools=[])
    assert sleeper.call_args_list == [((0.5,),), ((1.0,),)]


def test_classification_matches_the_shared_taxonomy() -> None:
    assert is_llm_error(_rate_limit_error())
    assert is_retryable_llm_error(_rate_limit_error())
    assert is_retryable_llm_error(_connection_error())
    assert is_retryable_llm_error(APITimeoutError(request=_request()))
    assert is_retryable_llm_error(_status_error(503))
    assert not is_retryable_llm_error(_status_error(400))
    assert not is_retryable_llm_error(RuntimeError("not an LLM error"))
    assert not is_llm_error(RuntimeError("not an LLM error"))
    assert llm_retry_delay_seconds(_rate_limit_error({"Retry-After": "9"}), 0) == 9.0
