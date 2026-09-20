"""Asynchronous execution of complete RCA workflows on one RQ queue.

The queue is deliberately coarse-grained: one complete RCA workflow (Lead,
Specialists, runtime verification, repair loop) is one RQ job on the single
``rca`` queue. Multi-agent logical boundaries are not deployment boundaries,
so RQ only adds process-level parallelism across incidents.

Enqueue a job against the Redis URL configured by ``ROOTTRACE_REDIS_URL``:

    from roottrace.queueing import enqueue_rca

    job_id = enqueue_rca(
        repo="/path/to/repo",
        issue="/path/to/issue.md",
        output_dir="/path/to/out",
    )

Start one or more native RQ workers, all consuming the same queue:

    rq worker --url "$ROOTTRACE_REDIS_URL" rca

Job-level retries are an opt-in outer safety net: the Provider already retries
transient LLM failures in place, and a failing Specialist is re-run on its own.
Enable RQ retries per job with ``retry_max`` / ``retry_interval`` or
deployment-wide with ``ROOTTRACE_QUEUE_RETRY_MAX`` and
``ROOTTRACE_QUEUE_RETRY_INTERVALS``. Deterministic failures clear the job retry
budget, so they go straight to ``FailedJobRegistry``.
"""

from roottrace.queueing.adapter import (
    JobNotFoundError,
    enqueue_rca,
    get_job,
    get_job_metadata,
    job_metadata,
)
from roottrace.queueing.config import (
    DEFAULT_RCA_QUEUE,
    QUEUE_RETRY_INTERVALS_ENV_VAR,
    QUEUE_RETRY_MAX_ENV_VAR,
    REDIS_URL_ENV_VAR,
    QueueRetryConfigurationError,
    RedisConfigurationError,
    create_queue,
    resolve_redis_url,
    resolve_retry_intervals,
    resolve_retry_max,
)
from roottrace.queueing.job import RcaJobResult, run_rca_job
from roottrace.queueing.schema import (
    RcaJobMetadata,
    RcaJobStatus,
    build_job_metadata,
    map_job_status,
)

__all__ = [
    "DEFAULT_RCA_QUEUE",
    "QUEUE_RETRY_INTERVALS_ENV_VAR",
    "QUEUE_RETRY_MAX_ENV_VAR",
    "REDIS_URL_ENV_VAR",
    "JobNotFoundError",
    "QueueRetryConfigurationError",
    "RcaJobMetadata",
    "RcaJobResult",
    "RcaJobStatus",
    "RedisConfigurationError",
    "build_job_metadata",
    "create_queue",
    "enqueue_rca",
    "get_job",
    "get_job_metadata",
    "job_metadata",
    "map_job_status",
    "resolve_redis_url",
    "resolve_retry_intervals",
    "resolve_retry_max",
    "run_rca_job",
]
