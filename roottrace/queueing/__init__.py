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
    REDIS_URL_ENV_VAR,
    RedisConfigurationError,
    create_queue,
    resolve_redis_url,
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
    "REDIS_URL_ENV_VAR",
    "JobNotFoundError",
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
    "run_rca_job",
]
