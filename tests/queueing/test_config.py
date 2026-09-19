"""Tests for Redis queue configuration handling."""

from __future__ import annotations

import pytest

from roottrace.queueing import (
    REDIS_URL_ENV_VAR,
    RedisConfigurationError,
    resolve_redis_url,
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
