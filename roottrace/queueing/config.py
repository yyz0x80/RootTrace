"""Redis configuration for the asynchronous RCA queue.

The queue connection is configured with the ``ROOTTRACE_REDIS_URL``
environment variable, and the opt-in RQ retry policy with
``ROOTTRACE_QUEUE_RETRY_MAX`` / ``ROOTTRACE_QUEUE_RETRY_INTERVALS``. No
additional configuration framework is introduced.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rq import Queue

REDIS_URL_ENV_VAR = "ROOTTRACE_REDIS_URL"
QUEUE_RETRY_MAX_ENV_VAR = "ROOTTRACE_QUEUE_RETRY_MAX"
QUEUE_RETRY_INTERVALS_ENV_VAR = "ROOTTRACE_QUEUE_RETRY_INTERVALS"
DEFAULT_RCA_QUEUE = "rca"


class RedisConfigurationError(RuntimeError):
    """Raised when no usable Redis connection URL is configured."""


class QueueRetryConfigurationError(ValueError):
    """Raised when the optional RQ retry policy configuration is malformed."""


def resolve_redis_url(redis_url: str | None = None) -> str:
    """Resolve the Redis URL from an explicit value or the environment.

    Args:
        redis_url: Optional explicit Redis URL that takes precedence.

    Returns:
        The configured Redis URL.

    Raises:
        RedisConfigurationError: If neither the argument nor the
            ``ROOTTRACE_REDIS_URL`` environment variable is set.
    """
    if redis_url:
        return redis_url
    configured = os.getenv(REDIS_URL_ENV_VAR)
    if configured:
        return configured
    raise RedisConfigurationError(
        f"{REDIS_URL_ENV_VAR} is not set, so the RCA queue has no Redis "
        "connection. Export it before enqueueing or inspecting RCA jobs, "
        f'e.g. export {REDIS_URL_ENV_VAR}="redis://localhost:6379/0".'
    )


def create_queue(
    redis_url: str | None = None,
    queue_name: str = DEFAULT_RCA_QUEUE,
) -> Queue:
    """Create the RQ queue used for complete RCA workflows.

    Args:
        redis_url: Optional explicit Redis URL, otherwise taken from the
            ``ROOTTRACE_REDIS_URL`` environment variable.
        queue_name: Queue name; defaults to the single ``rca`` queue.

    Returns:
        An RQ ``Queue`` bound to the configured Redis connection.
    """
    from redis import Redis
    from rq import Queue

    connection = Redis.from_url(resolve_redis_url(redis_url))
    return Queue(queue_name, connection=connection)


def resolve_retry_max(retry_max: int | None = None) -> int | None:
    """Resolve the RQ retry budget: explicit argument, then environment, else off.

    Job-level retries default to off because one retry re-runs the whole RCA
    workflow, so enabling them is an explicit decision by the caller or the
    deployment.

    Args:
        retry_max: Optional explicit maximum number of RQ retries.

    Returns:
        The configured retry budget, or ``None`` when retries stay disabled.

    Raises:
        QueueRetryConfigurationError: If the configured value is not a
            positive integer.
    """
    if retry_max is not None:
        if retry_max < 1:
            raise QueueRetryConfigurationError("retry_max must be at least 1")
        return retry_max

    raw = os.getenv(QUEUE_RETRY_MAX_ENV_VAR)
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise QueueRetryConfigurationError(
            f"{QUEUE_RETRY_MAX_ENV_VAR} must be an integer"
        ) from exc
    if value < 1:
        raise QueueRetryConfigurationError("retry_max must be at least 1")
    return value


def resolve_retry_intervals(
    retry_interval: int | Iterable[int] | None = None,
) -> int | list[int] | None:
    """Resolve the wait between RQ retries, supporting a comma-separated list.

    Args:
        retry_interval: Optional explicit interval or interval sequence.

    Returns:
        The configured interval(s), or ``None`` to let RQ retry immediately.

    Raises:
        QueueRetryConfigurationError: If the configured intervals are not
            non-negative integers.
    """
    if retry_interval is not None:
        return retry_interval

    raw = os.getenv(QUEUE_RETRY_INTERVALS_ENV_VAR)
    if raw is None or not raw.strip():
        return None
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    try:
        intervals = [int(part) for part in parts]
    except ValueError as exc:
        raise QueueRetryConfigurationError(
            f"{QUEUE_RETRY_INTERVALS_ENV_VAR} must be comma-separated seconds"
        ) from exc
    if not intervals or any(interval < 0 for interval in intervals):
        raise QueueRetryConfigurationError(
            f"{QUEUE_RETRY_INTERVALS_ENV_VAR} must contain non-negative seconds"
        )
    return intervals
