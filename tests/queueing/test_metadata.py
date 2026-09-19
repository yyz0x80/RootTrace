"""Tests for unified job status mapping and latency metrics."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from rq.job import JobStatus

from roottrace.queueing import (
    RcaJobStatus,
    build_job_metadata,
    job_metadata,
    map_job_status,
)
from roottrace.queueing.adapter import _job_error
from roottrace.queueing.schema import MAX_ERROR_CHARS


@pytest.mark.parametrize(
    ("rq_status", "expected"),
    [
        (JobStatus.CREATED, RcaJobStatus.QUEUED),
        (JobStatus.QUEUED, RcaJobStatus.QUEUED),
        (JobStatus.DEFERRED, RcaJobStatus.QUEUED),
        (JobStatus.SCHEDULED, RcaJobStatus.QUEUED),
        (JobStatus.STARTED, RcaJobStatus.RUNNING),
        (JobStatus.FINISHED, RcaJobStatus.FINISHED),
        (JobStatus.FAILED, RcaJobStatus.FAILED),
        (JobStatus.STOPPED, RcaJobStatus.FAILED),
        (JobStatus.CANCELED, RcaJobStatus.FAILED),
    ],
)
def test_map_job_status_covers_rq_lifecycle(
    rq_status: JobStatus,
    expected: RcaJobStatus,
) -> None:
    assert map_job_status(rq_status) is expected
    assert map_job_status(rq_status.value) is expected


def test_map_job_status_rejects_unknown_status() -> None:
    with pytest.raises(ValueError, match="unsupported RQ job status"):
        map_job_status("reticulating")


def test_build_job_metadata_computes_latency_metrics() -> None:
    # Naive timestamps are what older RQ versions persisted for UTC instants.
    enqueued = datetime.fromisoformat("2026-09-20T10:00:00")
    started = enqueued + timedelta(seconds=5)
    finished = started + timedelta(seconds=15)

    metadata = build_job_metadata(
        job_id="job-1",
        status=RcaJobStatus.FINISHED,
        enqueued_at=enqueued,
        started_at=started,
        finished_at=finished,
    )

    assert metadata.queue_wait_seconds == 5.0
    assert metadata.execution_seconds == 15.0
    assert metadata.total_latency_seconds == 20.0
    for stamp in (metadata.enqueued_at, metadata.started_at, metadata.finished_at):
        assert stamp is not None
        assert stamp.tzinfo is UTC


def test_build_job_metadata_leaves_metrics_unset_before_start() -> None:
    metadata = build_job_metadata(
        job_id="job-2",
        status=RcaJobStatus.QUEUED,
        enqueued_at=datetime(2026, 9, 20, 10, 0, 0, tzinfo=UTC),
    )

    assert metadata.status is RcaJobStatus.QUEUED
    assert metadata.started_at is None
    assert metadata.finished_at is None
    assert metadata.queue_wait_seconds is None
    assert metadata.execution_seconds is None
    assert metadata.total_latency_seconds is None


def test_job_metadata_reports_failure_status_and_error(
    make_job,
    make_result,
) -> None:
    enqueued = datetime(2026, 9, 20, 10, 0, 0, tzinfo=UTC)
    job = make_job(
        "job-failed",
        status=JobStatus.FAILED,
        enqueued_at=enqueued,
        started_at=enqueued + timedelta(seconds=2),
        ended_at=enqueued + timedelta(seconds=9),
        result=make_result(exc_string="RuntimeError: boom"),
    )

    metadata = job_metadata(job)

    assert metadata.status is RcaJobStatus.FAILED
    assert metadata.error == "RuntimeError: boom"
    assert metadata.queue_wait_seconds == 2.0
    assert metadata.execution_seconds == 7.0
    assert metadata.total_latency_seconds == 9.0


def test_job_metadata_bounds_long_failure_text(make_job, make_result) -> None:
    job = make_job(
        "job-long-error",
        status=JobStatus.FAILED,
        result=make_result(exc_string="x" * (MAX_ERROR_CHARS + 500)),
    )

    metadata = job_metadata(job)

    assert metadata.error is not None
    assert len(metadata.error) == MAX_ERROR_CHARS
    assert metadata.error.endswith("...")


def test_job_error_falls_back_to_legacy_exc_info(make_job) -> None:
    job = make_job(
        "job-legacy",
        status=JobStatus.FAILED,
        exc_info="ValueError: legacy failure",
    )

    assert _job_error(job) == "ValueError: legacy failure"


def test_job_error_is_absent_for_successful_jobs(make_job, make_result) -> None:
    job = make_job(
        "job-ok",
        status=JobStatus.FINISHED,
        result=make_result(return_value={"status": "completed"}),
    )

    assert _job_error(job) is None
