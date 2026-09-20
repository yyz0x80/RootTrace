"""Tests for Redis queue configuration handling."""

from __future__ import annotations

import pytest

from roottrace.queueing import (
    QUEUE_RETRY_INTERVALS_ENV_VAR,
    QUEUE_RETRY_MAX_ENV_VAR,
    REDIS_URL_ENV_VAR,
    QueueRetryConfigurationError,
    RedisConfigurationError,
    resolve_redis_url,
    resolve_retry_intervals,
    resolve_retry_max,
)


def test_resolve_redis_url_requires_environment(monkeypatch) -> None:
    monkeypatch.delenv(REDIS_URL_ENV_VAR, raising=False)

    with pytest.raises(RedisConfigurationError) as excinfo:
        resolve_redis_url()

    message = str(excinfo.value)
    assert REDIS_URL_ENV_VAR in message
    assert "redis://localhost:6379/0" in message


def test_resolve_redis_url_uses_environment(monkeypatch) -> None:
    monkeypatch.setenv(REDIS_URL_ENV_VAR, "redis://queue-host:6380/2")

    assert resolve_redis_url() == "redis://queue-host:6380/2"


def test_resolve_redis_url_prefers_explicit_value(monkeypatch) -> None:
    monkeypatch.setenv(REDIS_URL_ENV_VAR, "redis://queue-host:6380/2")

    assert resolve_redis_url("redis://explicit:6379/1") == "redis://explicit:6379/1"


def test_retry_policy_is_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv(QUEUE_RETRY_MAX_ENV_VAR, raising=False)
    monkeypatch.delenv(QUEUE_RETRY_INTERVALS_ENV_VAR, raising=False)

    assert resolve_retry_max() is None
    assert resolve_retry_intervals() is None


def test_retry_policy_prefers_explicit_values(monkeypatch) -> None:
    monkeypatch.setenv(QUEUE_RETRY_MAX_ENV_VAR, "5")
    monkeypatch.setenv(QUEUE_RETRY_INTERVALS_ENV_VAR, "90")

    assert resolve_retry_max(1) == 1
    assert resolve_retry_intervals([30, 60]) == [30, 60]


def test_retry_policy_reads_environment(monkeypatch) -> None:
    monkeypatch.setenv(QUEUE_RETRY_MAX_ENV_VAR, "2")
    monkeypatch.setenv(QUEUE_RETRY_INTERVALS_ENV_VAR, "30, 60")

    assert resolve_retry_max() == 2
    assert resolve_retry_intervals() == [30, 60]


def test_retry_policy_rejects_malformed_environment(monkeypatch) -> None:
    monkeypatch.setenv(QUEUE_RETRY_MAX_ENV_VAR, "many")
    with pytest.raises(QueueRetryConfigurationError):
        resolve_retry_max()

    monkeypatch.setenv(QUEUE_RETRY_MAX_ENV_VAR, "0")
    with pytest.raises(QueueRetryConfigurationError):
        resolve_retry_max()

    monkeypatch.delenv(QUEUE_RETRY_MAX_ENV_VAR)
    monkeypatch.setenv(QUEUE_RETRY_INTERVALS_ENV_VAR, "30,later")
    with pytest.raises(QueueRetryConfigurationError):
        resolve_retry_intervals()


def test_retry_policy_rejects_zero_explicit_budget(monkeypatch) -> None:
    monkeypatch.delenv(QUEUE_RETRY_MAX_ENV_VAR, raising=False)

    with pytest.raises(QueueRetryConfigurationError):
        resolve_retry_max(0)
