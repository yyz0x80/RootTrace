"""Focused queue evaluation tests without Redis or model calls."""

from __future__ import annotations

import argparse
import json
import signal
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from evaluation import queue_runner as runner
from evaluation.manifest import ManifestCase
from evaluation.workspace import CaseWorkspace
from roottrace.queueing.adapter import JobNotFoundError
from roottrace.queueing.schema import RcaJobStatus, build_job_metadata


def _inputs(tmp_path: Path) -> tuple[argparse.Namespace, list[ManifestCase]]:
    data = tmp_path / "data"
    cases = [
        ManifestCase(
            instance_id=f"acme__demo-{index}",
            repo="acme/demo",
            base_commit=f"{index + 1:040x}",
        )
        for index in range(50)
    ]
    (data / "manifests").mkdir(parents=True)
    (data / "public").mkdir()
    (data / "gold").mkdir()
    (data / "repos").mkdir()
    (data / "manifests/dev50.json").write_text(
        json.dumps({"name": "dev50", "seed": 42, "instances": [case.model_dump() for case in cases]}),
        encoding="utf-8",
    )
    with (data / "public/verified_public.jsonl").open("w", encoding="utf-8") as stream:
        for case in cases:
            stream.write(json.dumps({**case.model_dump(), "problem_statement": "Something failed"}) + "\n")
    selected = runner.select_dev50(data / "manifests/dev50.json")
    with (data / "gold/verified_gold.jsonl").open("w", encoding="utf-8") as stream:
        for case in selected:
            stream.write(json.dumps({
                "instance_id": case.instance_id,
                "patch": "diff --git a/src/demo.py b/src/demo.py\n+++ b/src/demo.py\n",
            }) + "\n")
    return argparse.Namespace(
        data_root=data, manifest=None, output_dir=tmp_path / "output",
        redis_url="redis://localhost:6380/11", job_timeout=300, workers=2,
    ), selected


class FakeQueue:
    def __init__(self, *, failing: str | None = None, interrupt_at: int | None = None) -> None:
        self.jobs: dict[str, object] = {}
        self.submissions: list[str] = []
        self.failing = failing
        self.interrupt_at = interrupt_at
        self.prepared_roots: list[Path] = []


class FakePool:
    returncode = None

    def poll(self) -> None:
        return None


def _install(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, queue: FakeQueue) -> None:
    monkeypatch.setattr(runner, "_resolve_model", lambda: "configured-model")
    monkeypatch.setattr(runner, "POLL_SECONDS", 0)

    @contextmanager
    def pool(_output_dir: Path, redis_url: str, worker_count: int):
        assert redis_url == "redis://localhost:6380/11"
        assert worker_count == 2
        yield [FakePool(), FakePool()]

    monkeypatch.setattr(runner, "_workers", pool)

    def prepare(_cache: Path, case: ManifestCase, *, work_root: Path) -> CaseWorkspace:
        root = work_root / case.instance_id
        repo = root / "repo"
        repo.mkdir(parents=True)
        return CaseWorkspace(root=root, repo=repo, base_commit=case.base_commit)

    monkeypatch.setattr(runner, "create_case_workspace", prepare)
    monkeypatch.setattr(runner, "verify_base_boundary", lambda *_args: None)
    monkeypatch.setattr(runner, "destroy_case_workspace", lambda workspace: workspace.repo.rmdir() or workspace.root.rmdir())

    def metadata(job_id: str, *, queue: FakeQueue):
        if job_id not in queue.jobs:
            raise JobNotFoundError(job_id)
        return queue.jobs[job_id]

    monkeypatch.setattr(runner, "get_job_metadata", metadata)

    def enqueue(repo: Path, issue: Path, output_dir: Path, **kwargs) -> str:
        assert len(list((tmp_path / "output/workspaces").iterdir())) == 50
        assert repo.is_dir()
        assert kwargs["repo_identifier"] == "acme/demo"
        assert set(json.loads(issue.read_text(encoding="utf-8"))) == {"id", "title", "problem"}
        assert "patch" not in issue.read_text(encoding="utf-8")
        if queue.interrupt_at == len(queue.submissions):
            raise KeyboardInterrupt
        case_id = issue.parent.name
        queue.submissions.append(case_id)
        now = datetime.now(UTC)
        queue.jobs[kwargs["job_id"]] = build_job_metadata(
            job_id=kwargs["job_id"],
            status=RcaJobStatus.FAILED if case_id == queue.failing else RcaJobStatus.FINISHED,
            enqueued_at=now,
            started_at=now + timedelta(seconds=1),
            finished_at=now + timedelta(seconds=3),
            error="worker failure" if case_id == queue.failing else None,
        )
        if case_id != queue.failing:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "rca_report.json").write_text(
                json.dumps({"top_k_locations": [{"path": "src/demo.py"}]}),
                encoding="utf-8",
            )
            (output_dir / "run_summary.json").write_text(json.dumps({
                "usage": {"llm_calls": 2, "prompt_tokens": 10, "completion_tokens": 4, "reasoning_tokens": None},
                "synthesis_usage": {"llm_calls": 1, "prompt_tokens": 5, "completion_tokens": 2, "reasoning_tokens": None},
            }), encoding="utf-8")
        return kwargs["job_id"]

    monkeypatch.setattr(runner, "enqueue_rca", enqueue)


def test_submission_collection_and_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args, selected = _inputs(tmp_path)
    queue = FakeQueue()
    _install(monkeypatch, tmp_path, queue)
    assert runner.run_from_args(args, queue=queue) == 0
    assert queue.submissions == [case.instance_id for case in selected]
    summary = json.loads((args.output_dir / "queue_summary.json").read_text(encoding="utf-8"))
    assert summary["worker_count"] == 2
    assert summary["case_ids"] == [case.instance_id for case in selected]
    assert summary["failed_count"] == 0
    assert summary["llm_calls"] == 150
    assert summary["total_tokens"] == 1050
    assert summary["reasoning_tokens"] is None
    assert summary["jobs"][0]["queue_wait_seconds"] == 1
    assert summary["jobs"][0]["execution_seconds"] == 2
    assert summary["jobs"][0]["total_seconds"] == 3
    assert json.loads((args.output_dir / "metrics.json").read_text(encoding="utf-8"))["aggregate"]["top_1_file_accuracy"] == 1
    state = runner.QueueState.model_validate_json((args.output_dir / runner.STATE_FILE).read_text(encoding="utf-8"))
    assert all(entry.cleaned for entry in state.cases)
    assert all(not entry.workspace_root.exists() for entry in state.cases)


def test_failed_job_is_counted_and_gold_read_after_terminal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args, selected = _inputs(tmp_path)
    queue = FakeQueue(failing=selected[0].instance_id)
    _install(monkeypatch, tmp_path, queue)
    real_gold = runner.GoldStore

    class GuardedGold(real_gold):
        def gold_files(self, instance_id: str) -> list[str]:
            assert len(queue.jobs) == 50
            return super().gold_files(instance_id)

    monkeypatch.setattr(runner, "GoldStore", GuardedGold)
    assert runner.run_from_args(args, queue=queue) == 1
    summary = json.loads((args.output_dir / "queue_summary.json").read_text(encoding="utf-8"))
    assert summary["failed_count"] == 1
    assert summary["llm_calls"] is None
    assert summary["jobs"][0]["terminal_status"] == "failed"


def test_interruption_preserves_job_ids_and_repos_for_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args, _selected = _inputs(tmp_path)
    queue = FakeQueue(interrupt_at=2)
    _install(monkeypatch, tmp_path, queue)
    assert runner.run_from_args(args, queue=queue) == 130
    state = runner.QueueState.model_validate_json((args.output_dir / runner.STATE_FILE).read_text(encoding="utf-8"))
    assert len(state.cases) == 50
    assert all(entry.repo_path.exists() for entry in state.cases)
    assert sum(entry.submitted for entry in state.cases) == 2
    assert state.cases[2].job_id is not None
    queue.interrupt_at = None
    assert runner.run_from_args(args, queue=queue) == 0
    assert len(queue.submissions) == 50
    assert len(set(queue.submissions)) == 50


def test_requires_dev50_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args, _selected = _inputs(tmp_path)
    manifest_path = args.data_root / "manifests/dev50.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["instances"].pop()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert runner.run_from_args(args, queue=FakeQueue()) == 2
    assert not (args.output_dir / runner.STATE_FILE).exists()


def test_resume_rejects_unsafe_saved_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args, _selected = _inputs(tmp_path)
    queue = FakeQueue(interrupt_at=0)
    _install(monkeypatch, tmp_path, queue)
    assert runner.run_from_args(args, queue=queue) == 130
    state_path = args.output_dir / runner.STATE_FILE
    raw = json.loads(state_path.read_text(encoding="utf-8"))
    raw["cases"][0]["workspace_root"] = str(tmp_path / "outside")
    raw["cases"][0]["repo_path"] = str(tmp_path / "outside/repo")
    state_path.write_text(json.dumps(raw), encoding="utf-8")
    assert runner.run_from_args(args, queue=queue) == 2
    assert not queue.submissions


def test_worker_count_is_configurable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args, _selected = _inputs(tmp_path)
    args.workers = 4
    queue = FakeQueue()
    _install(monkeypatch, tmp_path, queue)
    started: list[int] = []

    @contextmanager
    def pool(_output_dir: Path, _redis_url: str, worker_count: int):
        started.append(worker_count)
        yield [FakePool() for _ in range(worker_count)]

    monkeypatch.setattr(runner, "_workers", pool)
    assert runner.run_from_args(args, queue=queue) == 0
    assert started == [4]
    assert runner.build_parser().parse_args([]).workers == 2


def test_resume_rejects_changed_worker_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args, _selected = _inputs(tmp_path)
    queue = FakeQueue(interrupt_at=1)
    _install(monkeypatch, tmp_path, queue)
    assert runner.run_from_args(args, queue=queue) == 130
    args.workers = 4
    assert runner.run_from_args(args, queue=queue) == 2
    assert len(queue.submissions) == 1


def test_managed_workers_are_independent_and_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    spawned: list[tuple[list[str], dict]] = []
    signals: list[tuple[int, signal.Signals]] = []

    class Process:
        def __init__(self, pid: int) -> None:
            self.pid = pid
            self.returncode = None

        def poll(self) -> None:
            return None

        def wait(self, *, timeout: int) -> int:
            assert timeout == 10
            return 0

    def popen(command: list[str], **kwargs) -> Process:
        spawned.append((command, kwargs))
        return Process(1234 + len(spawned))

    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    monkeypatch.setattr(runner.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    with runner._workers(tmp_path, "redis://localhost:6380/11", 2) as workers:
        assert len(workers) == 2
    command, kwargs = spawned[0]
    assert command[:4] == [runner.sys.executable, "-m", "rq.cli", "worker"]
    assert command[4:] == [
        "-w", "rq.worker.SpawnWorker", "-u", "redis://localhost:6380/11", "rca",
    ]
    assert len(spawned) == 2
    assert kwargs["stdin"] == runner.subprocess.DEVNULL
    assert kwargs["start_new_session"] is True
    assert signals == [(1235, signal.SIGTERM), (1236, signal.SIGTERM)]


def test_pool_exit_fails_fast(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args, _selected = _inputs(tmp_path)
    queue = FakeQueue()
    _install(monkeypatch, tmp_path, queue)

    class ExitedPool:
        returncode = 1

        def poll(self) -> int:
            return 1

    @contextmanager
    def pool(_output_dir: Path, _redis_url: str, _worker_count: int):
        yield [ExitedPool(), FakePool()]

    monkeypatch.setattr(runner, "_workers", pool)
    assert runner.run_from_args(args, queue=queue) == 2
    assert len(queue.submissions) == 50
    assert all(entry.repo_path.exists() for entry in runner.QueueState.model_validate_json(
        (args.output_dir / runner.STATE_FILE).read_text(encoding="utf-8"),
    ).cases)
