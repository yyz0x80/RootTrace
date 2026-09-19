"""Shared in-memory fakes for queueing unit tests.

The fakes mirror the small part of the RQ API that the adapter depends on, so
the tests exercise enqueueing, status mapping, and timing without a Redis
server.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

import pytest
from rq.job import JobStatus


class FakeResult:
    """Stand-in for ``rq.results.Result``."""

    def __init__(
        self,
        *,
        exc_string: str | None = None,
        return_value: Any = None,
    ) -> None:
        self.exc_string = exc_string
        self.return_value = return_value


class FakeJob:
    """Stand-in for ``rq.job.Job`` lifecycle attributes."""

    def __init__(
        self,
        job_id: str,
        *,
        status: JobStatus = JobStatus.QUEUED,
        enqueued_at: datetime | None = None,
        started_at: datetime | None = None,
        ended_at: datetime | None = None,
        result: FakeResult | None = None,
        exc_info: str | None = None,
    ) -> None:
        self.id = job_id
        self.enqueued_at = enqueued_at
        self.started_at = started_at
        self.ended_at = ended_at
        self._status = status
        self._result = result
        self._exc_info = exc_info

    def get_status(self, refresh: bool = True) -> JobStatus:
        """Return the stored status like ``rq.job.Job.get_status``."""
        return self._status

    def latest_result(self) -> FakeResult | None:
        """Return the stored result like ``rq.job.Job.latest_result``."""
        return self._result


class FakeQueue:
    """Stand-in for ``rq.Queue`` that records enqueued calls in memory."""

    def __init__(self) -> None:
        self.enqueued: list[dict[str, Any]] = []
        self.jobs: dict[str, FakeJob] = {}

    def enqueue(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> FakeJob:
        """Create and record a fake job for one enqueued call."""
        job = FakeJob(kwargs.get("job_id") or f"fake-job-{len(self.enqueued) + 1}")
        self.jobs[job.id] = job
        self.enqueued.append(
            {"func": func, "args": args, "kwargs": kwargs, "job": job}
        )
        return job

    def fetch_job(self, job_id: str) -> FakeJob | None:
        """Return a recorded job by id."""
        return self.jobs.get(job_id)


@pytest.fixture
def fake_queue() -> FakeQueue:
    """Provide an empty in-memory RQ queue."""
    return FakeQueue()


@pytest.fixture
def make_result() -> Callable[..., FakeResult]:
    """Provide a factory for fake RQ results."""

    def _make_result(**kwargs: Any) -> FakeResult:
        return FakeResult(**kwargs)

    return _make_result


@pytest.fixture
def make_job() -> Callable[..., FakeJob]:
    """Provide a factory for fake RQ jobs."""

    def _make_job(job_id: str = "job-1", **kwargs: Any) -> FakeJob:
        return FakeJob(job_id, **kwargs)

    return _make_job
