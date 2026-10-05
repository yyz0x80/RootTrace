"""Job status and benchmark metadata for asynchronous RCA runs.

All timestamps are handled in UTC. The metrics reuse the timestamps recorded
by RQ (``enqueued_at``, ``started_at``, ``ended_at``) instead of persisting
duplicate timing data.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

MAX_ERROR_CHARS = 2_000


class RcaJobStatus(str, Enum):
    """Unified lifecycle status of one asynchronous RCA job."""

    QUEUED = "queued"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"


# RQ job statuses collapsed into the RootTrace queue status vocabulary.
_RQ_STATUS_MAP: dict[str, RcaJobStatus] = {
    "created": RcaJobStatus.QUEUED,
    "queued": RcaJobStatus.QUEUED,
    "deferred": RcaJobStatus.QUEUED,
    "scheduled": RcaJobStatus.QUEUED,
    "started": RcaJobStatus.RUNNING,
    "finished": RcaJobStatus.FINISHED,
    "failed": RcaJobStatus.FAILED,
    "stopped": RcaJobStatus.FAILED,
    "canceled": RcaJobStatus.FAILED,
}


class RcaJobMetadata(BaseModel):
    """Benchmark-facing status, timing, and failure view of one RCA job."""

    job_id: str
    status: RcaJobStatus
    enqueued_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    queue_wait_seconds: float | None = Field(default=None, ge=0)
    execution_seconds: float | None = Field(default=None, ge=0)
    total_latency_seconds: float | None = Field(default=None, ge=0)
    error: str | None = None


def map_job_status(status: Any) -> RcaJobStatus:
    """Map an RQ job status onto the unified RCA queue status vocabulary.

    Args:
        status: RQ ``JobStatus`` member or its string value.

    Returns:
        The unified :class:`RcaJobStatus`.

    Raises:
        ValueError: If the status is not a known RQ job status.
    """
    key = status.value if isinstance(status, Enum) else str(status)
    mapped = _RQ_STATUS_MAP.get(key)
    if mapped is None:
        raise ValueError(f"unsupported RQ job status: {key!r}")
    return mapped


def to_utc(value: datetime | None) -> datetime | None:
    """Return a timezone-aware UTC timestamp for a possibly naive datetime."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def elapsed_seconds(
    start: datetime | None,
    end: datetime | None,
) -> float | None:
    """Return the non-negative elapsed seconds between two UTC timestamps."""
    if start is None or end is None:
        return None
    return max(0.0, (end - start).total_seconds())


def bound_error(error: str | None) -> str | None:
    """Return bounded error text suitable for persisted job metadata."""
    if error is None:
        return None
    if len(error) <= MAX_ERROR_CHARS:
        return error
    keep = max(0, MAX_ERROR_CHARS - 3)
    return error[:keep] + "..."


def build_job_metadata(
    *,
    job_id: str,
    status: RcaJobStatus,
    enqueued_at: datetime | None = None,
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
    error: str | None = None,
) -> RcaJobMetadata:
    """Build the benchmark metadata view from RQ lifecycle timestamps.

    ``queue_wait_seconds`` covers the time before a worker started the job,
    ``execution_seconds`` covers the worker execution, and
    ``total_latency_seconds`` covers enqueue-to-finish end to end.
    """
    enqueued = to_utc(enqueued_at)
    started = to_utc(started_at)
    finished = to_utc(finished_at)
    return RcaJobMetadata(
        job_id=job_id,
        status=status,
        enqueued_at=enqueued,
        started_at=started,
        finished_at=finished,
        queue_wait_seconds=elapsed_seconds(enqueued, started),
        execution_seconds=elapsed_seconds(started, finished),
        total_latency_seconds=elapsed_seconds(enqueued, finished),
        error=bound_error(error),
    )
