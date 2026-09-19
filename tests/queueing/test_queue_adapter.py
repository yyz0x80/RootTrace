"""Tests for the one-queue RCA adapter."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from rq.job import JobStatus

from roottrace.queueing import (
    JobNotFoundError,
    RcaJobStatus,
    enqueue_rca,
    get_job,
    get_job_metadata,
    run_rca_job,
)


def test_enqueue_rca_passes_workflow_arguments_and_returns_job_id(
    fake_queue,
    tmp_path,
) -> None:
    repo = tmp_path / "repo"
    issue = tmp_path / "issue.md"
    output_dir = tmp_path / "out"
    ci_log = tmp_path / "ci.log"

    job_id = enqueue_rca(
        repo=repo,
        issue=issue,
        output_dir=output_dir,
        model="fake-model",
        ci_log=ci_log,
        job_timeout=600,
        result_ttl=900,
        queue=fake_queue,
    )

    assert len(fake_queue.enqueued) == 1
    entry = fake_queue.enqueued[0]
    assert job_id == entry["job"].id
    assert entry["func"] is run_rca_job
    assert entry["args"] == (str(repo), str(issue), str(output_dir))
    assert entry["kwargs"]["model"] == "fake-model"
    assert entry["kwargs"]["ci_log"] == str(ci_log)
    assert entry["kwargs"]["stack_trace"] is None
    assert entry["kwargs"]["pr_diff"] is None
    assert entry["kwargs"]["job_timeout"] == 600
    assert entry["kwargs"]["result_ttl"] == 900


def test_enqueue_rca_returns_unique_job_ids(fake_queue, tmp_path) -> None:
    first = enqueue_rca(
        repo=tmp_path / "repo-a",
        issue=tmp_path / "issue-a.md",
        output_dir=tmp_path / "out-a",
        queue=fake_queue,
    )
    second = enqueue_rca(
        repo=tmp_path / "repo-b",
        issue=tmp_path / "issue-b.md",
        output_dir=tmp_path / "out-b",
        queue=fake_queue,
    )

    assert first != second
    assert len(fake_queue.enqueued) == 2


def test_enqueue_rca_builds_queue_from_redis_url(monkeypatch, tmp_path) -> None:
    created: dict[str, object] = {}

    class FakeBuiltQueue:
        def enqueue(self, func, *args, **kwargs):
            job = type("FakeBuiltJob", (), {"id": "built-job"})()
            created["func"] = func
            return job

    def fake_create_queue(redis_url=None, queue_name="rca"):
        created["redis_url"] = redis_url
        created["queue_name"] = queue_name
        return FakeBuiltQueue()

    monkeypatch.setattr(
        "roottrace.queueing.adapter.create_queue",
        fake_create_queue,
    )

    job_id = enqueue_rca(
        repo=tmp_path / "repo",
        issue=tmp_path / "issue.md",
        output_dir=tmp_path / "out",
        redis_url="redis://queue-host:6380/3",
    )

    assert job_id == "built-job"
    assert created["redis_url"] == "redis://queue-host:6380/3"
    assert created["queue_name"] == "rca"
    assert created["func"] is run_rca_job


def test_get_job_returns_enqueued_job(fake_queue, tmp_path) -> None:
    job_id = enqueue_rca(
        repo=tmp_path / "repo",
        issue=tmp_path / "issue.md",
        output_dir=tmp_path / "out",
        queue=fake_queue,
    )

    assert get_job(job_id, queue=fake_queue).id == job_id


def test_get_job_raises_for_unknown_job_id(fake_queue) -> None:
    with pytest.raises(JobNotFoundError, match="no RCA job found"):
        get_job("missing-job", queue=fake_queue)


def test_get_job_metadata_reports_running_job(fake_queue, tmp_path) -> None:
    job_id = enqueue_rca(
        repo=tmp_path / "repo",
        issue=tmp_path / "issue.md",
        output_dir=tmp_path / "out",
        queue=fake_queue,
    )
    enqueued = datetime(2026, 9, 20, 10, 0, 0, tzinfo=UTC)
    job = fake_queue.jobs[job_id]
    job.enqueued_at = enqueued
    job.started_at = enqueued + timedelta(seconds=4)
    job._status = JobStatus.STARTED

    metadata = get_job_metadata(job_id, queue=fake_queue)

    assert metadata.job_id == job_id
    assert metadata.status is RcaJobStatus.RUNNING
    assert metadata.queue_wait_seconds == 4.0
    assert metadata.execution_seconds is None
    assert metadata.total_latency_seconds is None


def test_get_job_metadata_uses_explicit_redis_url(monkeypatch) -> None:
    created: dict[str, object] = {}

    class FakeBuiltQueue:
        def fetch_job(self, job_id):
            created["job_id"] = job_id

    def fake_create_queue(redis_url=None, queue_name="rca"):
        created["redis_url"] = redis_url
        return FakeBuiltQueue()

    monkeypatch.setattr(
        "roottrace.queueing.adapter.create_queue",
        fake_create_queue,
    )

    with pytest.raises(JobNotFoundError):
        get_job_metadata("job-1", redis_url="redis://queue-host:6380/3")

    assert created["redis_url"] == "redis://queue-host:6380/3"
    assert created["job_id"] == "job-1"
