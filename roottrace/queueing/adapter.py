"""One-queue RQ adapter for asynchronous RCA execution.

Every job runs one complete RCA workflow on the single ``rca`` queue, so RQ
workers provide process-level parallelism across incidents while the Lead,
Specialists, Verifier, and repair loop stay inside one job. Job status and
timing come from RQ; this module only maps them into RootTrace terms.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from roottrace.queueing.config import create_queue
from roottrace.queueing.job import run_rca_job
from roottrace.queueing.schema import (
    RcaJobMetadata,
    bound_error,
    build_job_metadata,
    map_job_status,
)

if TYPE_CHECKING:
    from rq import Queue, Retry
    from rq.job import Job


class JobNotFoundError(LookupError):
    """Raised when an RCA job id is not present in the queue."""


def enqueue_rca(
    repo: str | Path,
    issue: str | Path,
    output_dir: str | Path,
    *,
    model: str | None = None,
    stack_trace: str | Path | None = None,
    ci_log: str | Path | None = None,
    pr_diff: str | Path | None = None,
    job_timeout: int | None = None,
    result_ttl: int | None = None,
    retry_max: int | None = None,
    retry_interval: int | Iterable[int] | None = None,
    job_id: str | None = None,
    queue: Queue | None = None,
    redis_url: str | None = None,
) -> str:
    """Enqueue one complete RCA workflow and return its unique job id.

    Args:
        repo: Path to the target repository.
        issue: Path to the local Issue Markdown/JSON file.
        output_dir: Directory that receives this job's RCA artifacts. Callers
            must give concurrent jobs distinct directories.
        model: Optional model name or model identifier from project config.
        stack_trace: Optional stack trace file.
        ci_log: Optional CI log file.
        pr_diff: Optional PR diff/context file.
        job_timeout: Optional RQ job timeout in seconds.
        result_ttl: Optional RQ result time-to-live in seconds.
        retry_max: Optional maximum number of RQ retries (at least 1) for
            transient failures such as rate limits; each retry re-runs the
            whole RCA workflow. ``None`` disables retries.
        retry_interval: Optional seconds to wait before each retry, or a
            sequence of intervals used in order; requires ``retry_max``.
        job_id: Optional explicit job id, otherwise RQ generates one.
        queue: Optional pre-built RQ queue, mainly for embedding and tests.
        redis_url: Optional explicit Redis URL, otherwise
            ``ROOTTRACE_REDIS_URL`` is used.

    Returns:
        The unique RQ job id, which also identifies the job metadata view.

    Raises:
        ValueError: If ``retry_interval`` is given without ``retry_max``.
    """
    target_queue = queue if queue is not None else create_queue(redis_url)
    job = target_queue.enqueue(
        run_rca_job,
        str(Path(repo)),
        str(Path(issue)),
        str(Path(output_dir)),
        model=model,
        stack_trace=_optional_path(stack_trace),
        ci_log=_optional_path(ci_log),
        pr_diff=_optional_path(pr_diff),
        job_timeout=job_timeout,
        result_ttl=result_ttl,
        retry=_retry_policy(retry_max, retry_interval),
        job_id=job_id,
    )
    return job.id


def get_job(
    job_id: str,
    *,
    queue: Queue | None = None,
    redis_url: str | None = None,
) -> Job:
    """Fetch one queued RCA job by id.

    Raises:
        JobNotFoundError: If the queue holds no job with this id.
    """
    target_queue = queue if queue is not None else create_queue(redis_url)
    job = target_queue.fetch_job(job_id)
    if job is None:
        raise JobNotFoundError(f"no RCA job found for id {job_id!r}")
    return job


def job_metadata(job: Job) -> RcaJobMetadata:
    """Build the benchmark metadata view of one RQ job."""
    return build_job_metadata(
        job_id=job.id,
        status=map_job_status(job.get_status(refresh=False)),
        enqueued_at=getattr(job, "enqueued_at", None),
        started_at=getattr(job, "started_at", None),
        finished_at=getattr(job, "ended_at", None),
        error=_job_error(job),
    )


def get_job_metadata(
    job_id: str,
    *,
    queue: Queue | None = None,
    redis_url: str | None = None,
) -> RcaJobMetadata:
    """Fetch one RCA job and return its status and latency metrics.

    Raises:
        JobNotFoundError: If the queue holds no job with this id.
    """
    return job_metadata(get_job(job_id, queue=queue, redis_url=redis_url))


def _optional_path(path: str | Path | None) -> str | None:
    return None if path is None else str(Path(path))


def _retry_policy(
    retry_max: int | None,
    retry_interval: int | Iterable[int] | None,
) -> Retry | None:
    """Build the RQ retry policy for one job, or ``None`` when disabled.

    The policy is handed to RQ unchanged, so retry scheduling, attempt
    counting, and the terminal failure after the last attempt stay RQ
    behaviour.
    """
    if retry_max is None:
        if retry_interval is not None:
            raise ValueError("retry_interval requires retry_max")
        return None

    from rq import Retry

    return Retry(retry_max, 0 if retry_interval is None else retry_interval)


def _job_error(job: Job) -> str | None:
    """Return bounded failure text recorded by RQ for a failed job."""
    latest_result = getattr(job, "latest_result", None)
    if callable(latest_result):
        try:
            result: Any = latest_result()
        except Exception:  # noqa: BLE001 - metadata must not mask job status
            result = None
        exc_string = getattr(result, "exc_string", None)
        if exc_string:
            return bound_error(str(exc_string))
    legacy_exc_info = getattr(job, "_exc_info", None)
    if legacy_exc_info:
        return bound_error(str(legacy_exc_info))
    return None
