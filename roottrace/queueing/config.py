"""Redis configuration for the asynchronous RCA queue.

The queue connection is configured with the ``ROOTTRACE_REDIS_URL``
environment variable. No additional configuration framework is introduced.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rq import Queue

REDIS_URL_ENV_VAR = "ROOTTRACE_REDIS_URL"
DEFAULT_RCA_QUEUE = "rca"


class RedisConfigurationError(RuntimeError):
    """Raised when no usable Redis connection URL is configured."""


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
