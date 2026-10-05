"""Run the fixed dev50 evaluation through managed RQ workers.

Run this command again with the same output directory to resume collection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from evaluation.adapter import (
    build_incident_input,
    load_public_cases,
    write_root_trace_input,
)
from evaluation.gold import GoldStore
from evaluation.manifest import (
    ManifestCase,
    RcaManifest,
    load_manifest,
    manifest_sha256,
)
from evaluation.metrics import (
    CASE_RESULT_SCHEMA_VERSION,
    CaseResult,
    compute_case_metrics,
    compute_eval_metrics,
)
from evaluation.report import EvalRunConfig, write_reports
from evaluation.runner import (
    DEFAULT_DATA_ROOT,
    GOLD_METADATA,
    PUBLIC_METADATA,
    _bounded_error,
    _list_artifacts,
    _read_report_json,
    _validate_manifest_against_public,
    extract_predicted_files,
)
from evaluation.variants import AblationVariant
from evaluation.workspace import (
    CaseWorkspace,
    create_case_workspace,
    destroy_case_workspace,
    verify_base_boundary,
)
from roottrace.llm.config import ModelConfigManager
from roottrace.queueing import (
    JobNotFoundError,
    create_queue,
    enqueue_rca,
    get_job_metadata,
    resolve_redis_url,
)
from roottrace.queueing.schema import RcaJobMetadata, RcaJobStatus

STATE_FILE = "queue_state.json"
VARIANT = AblationVariant.THREE_SPECIALISTS_RETRIEVAL_OFF.value
POLL_SECONDS = 2
DEFAULT_WORKERS = 2


class QueueCase(BaseModel):
    case: ManifestCase
    workspace_root: Path
    repo_path: Path
    job_id: str | None = None
    submitted: bool = False
    terminal: str | None = None
    cleaned: bool = False


class QueueState(BaseModel):
    manifest_sha256: str
    model: str
    worker_count: int = Field(ge=1)
    first_submission_at: datetime | None = None
    completed_at: datetime | None = None
    cases: list[QueueCase] = Field(default_factory=list)


def select_dev50(manifest_path: Path) -> list[ManifestCase]:
    """Use every case in the fixed development manifest, without consulting gold."""
    manifest = load_manifest(manifest_path)
    if manifest.name != "dev50" or len(manifest.instances) != 50:
        raise ValueError("queue evaluation requires the 50-case dev50 manifest")
    return manifest.instances


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _save_state(path: Path, state: QueueState) -> None:
    _write_json(path, state.model_dump(mode="json"))


def _resolve_model() -> str:
    configured = os.getenv("ROOTTRACE_MODEL")
    if configured:
        return configured
    models = ModelConfigManager().list_models()
    if len(models) == 1:
        return models[0]
    raise ValueError("set ROOTTRACE_MODEL to one configured model for queue evaluation")


@contextmanager
def _workers(
    output_dir: Path, redis_url: str, worker_count: int,
) -> Iterator[list[subprocess.Popen]]:
    """Start independent SpawnWorkers and stop them when evaluation ends."""
    env = os.environ.copy()
    project_root = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (project_root, env.get("PYTHONPATH"))))
    processes: list[subprocess.Popen] = []
    try:
        for number in range(1, worker_count + 1):
            log_path = output_dir / f"worker-{number}.log"
            with log_path.open("a", encoding="utf-8") as log:
                worker = subprocess.Popen(
                    [sys.executable, "-m", "rq.cli", "worker", "-w",
                     "rq.worker.SpawnWorker", "-u", redis_url, "rca"],
                    cwd=project_root, env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                )
            processes.append(worker)
            print(f"started worker {number}/{worker_count} (PID {worker.pid}); log: {log_path}", flush=True)
        yield processes
    finally:
        for worker in processes:
            try:
                os.killpg(worker.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for worker in processes:
            if worker.poll() is None:
                try:
                    worker.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(worker.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    worker.wait()


def _prepare_all(
    selected: list[ManifestCase], repo_cache: Path, output_dir: Path, state: QueueState,
) -> None:
    """Persist every checkout before any enqueue call or timing begins."""
    workspace_root = output_dir / "workspaces"
    workspace_root.mkdir(parents=True, exist_ok=True)
    for case in selected:
        if any(entry.case.instance_id == case.instance_id for entry in state.cases):
            continue
        workspace = create_case_workspace(repo_cache, case, work_root=workspace_root)
        state.cases.append(QueueCase(case=case, workspace_root=workspace.root, repo_path=workspace.repo))
        _save_state(output_dir / STATE_FILE, state)
    for entry in state.cases:
        if not entry.cleaned:
            verify_base_boundary(entry.repo_path, entry.case.base_commit)


def _validate_saved_state(state: QueueState, selected: list[ManifestCase], output_dir: Path) -> None:
    """Keep recovery and cleanup scoped to this run's prepared workspaces."""
    if [entry.case for entry in state.cases] != selected[:len(state.cases)]:
        raise ValueError("saved queue state differs from dev50 manifest")
    workspaces = output_dir / "workspaces"
    for entry in state.cases:
        root = entry.workspace_root
        if (
            root.parent != workspaces
            or entry.repo_path != root / "repo"
            or root.is_symlink()
            or entry.repo_path.is_symlink()
        ):
            raise ValueError("saved queue state has an unsafe workspace path")
        if entry.cleaned and root.exists():
            raise ValueError("cleaned workspace unexpectedly exists")


def _issue_for_case(case: ManifestCase, public: dict, case_dir: Path) -> Path:
    incident = build_incident_input(public[case.instance_id]).incident
    case_dir.mkdir(parents=True, exist_ok=True)
    write_root_trace_input(incident, case_dir / "root_trace_input.json")
    issue_path = case_dir / "issue.json"
    _write_json(issue_path, {"id": incident.id, "title": incident.title or incident.id, "problem": incident.problem})
    return issue_path


def _submit_all(state: QueueState, public: dict, output_dir: Path, queue: object, timeout: int) -> None:
    for entry in state.cases:
        if entry.submitted:
            continue
        case_id = entry.case.instance_id
        case_dir = output_dir / "cases" / case_id
        issue_path = _issue_for_case(entry.case, public, case_dir)
        rca_dir = case_dir / "roottrace"
        rca_dir.mkdir(parents=True, exist_ok=True)
        if entry.job_id is None:
            entry.job_id = f"roottrace-dev50-{hashlib.sha256(str(output_dir).encode()).hexdigest()[:12]}-{case_id}"
            _save_state(output_dir / STATE_FILE, state)
        try:
            existing = get_job_metadata(entry.job_id, queue=queue)
        except JobNotFoundError:
            existing = None
        if existing is None:
            if state.first_submission_at is None:
                state.first_submission_at = datetime.now(UTC)
                _save_state(output_dir / STATE_FILE, state)
            enqueue_rca(
                entry.repo_path, issue_path, rca_dir, model=state.model,
                repo_identifier=entry.case.repo,
                job_timeout=timeout, result_ttl=7 * 24 * 3600,
                job_id=entry.job_id, queue=queue,
            )
        entry.submitted = True
        _save_state(output_dir / STATE_FILE, state)
        print(f"submitted {case_id}: {entry.job_id}", flush=True)


def _wait_for_all(
    state: QueueState, output_dir: Path, queue: object, workers: list[subprocess.Popen],
) -> dict[str, RcaJobMetadata]:
    pending = {entry.case.instance_id: entry for entry in state.cases}
    metadata: dict[str, RcaJobMetadata] = {}
    while pending:
        for number, worker in enumerate(workers, start=1):
            if worker.poll() is not None:
                raise RuntimeError(
                    f"RQ worker {number} exited with code {worker.returncode}; "
                    f"see worker-{number}.log"
                )
        for case_id, entry in list(pending.items()):
            if entry.job_id is None:
                raise ValueError(f"missing job ID for {case_id}")
            detail = get_job_metadata(entry.job_id, queue=queue)
            if detail.status in {RcaJobStatus.FINISHED, RcaJobStatus.FAILED}:
                metadata[case_id] = detail
                entry.terminal = detail.status.value
                _save_state(output_dir / STATE_FILE, state)
                del pending[case_id]
                print(f"[{case_id}] {detail.status.value}", flush=True)
        if pending:
            time.sleep(POLL_SECONDS)
    if state.completed_at is None:
        state.completed_at = datetime.now(UTC)
        _save_state(output_dir / STATE_FILE, state)
    return metadata


def _usage(summary: dict | None) -> tuple[int | None, dict | None]:
    if not isinstance(summary, dict):
        return None, None
    parts = [summary.get("usage"), summary.get("synthesis_usage")]
    if any(not isinstance(part, dict) for part in parts):
        return None, None
    def total(key: str) -> int | None:
        values = [part.get(key) for part in parts]
        return sum(values) if all(type(value) is int for value in values) else None
    prompt = total("prompt_tokens")
    completion = total("completion_tokens")
    reasoning = total("reasoning_tokens")
    return total("llm_calls"), {
        "prompt_tokens": prompt, "completion_tokens": completion,
        "reasoning_tokens": reasoning,
        "total_tokens": prompt + completion if prompt is not None and completion is not None else None,
    }


def _collect(
    state: QueueState, output_dir: Path, gold: GoldStore,
    metadata: dict[str, RcaJobMetadata], config_hash: str,
) -> list[CaseResult]:
    results: list[CaseResult] = []
    for entry in state.cases:
        case = entry.case
        case_dir = output_dir / "cases" / case.instance_id
        detail = metadata[case.instance_id]
        report = _read_report_json(case_dir / "roottrace") if detail.status == RcaJobStatus.FINISHED else None
        status = "completed" if report is not None else "error"
        error = None if status == "completed" else _bounded_error(detail.error or "RCA report missing after queue completion")
        summary_path = case_dir / "roottrace" / "run_summary.json"
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            summary = None
        calls, usage = _usage(summary)
        gold_files = gold.gold_files(case.instance_id)
        predicted = extract_predicted_files(report)
        metrics = compute_case_metrics(
            instance_id=case.instance_id, predicted_files=predicted, gold_files=gold_files,
            status=status, error=error, latency_seconds=detail.execution_seconds,
            llm_calls=calls, prompt_tokens=(usage or {}).get("prompt_tokens"),
            completion_tokens=(usage or {}).get("completion_tokens"),
            reasoning_tokens=(usage or {}).get("reasoning_tokens"),
        )
        result = CaseResult(
            schema_version=CASE_RESULT_SCHEMA_VERSION, instance_id=case.instance_id,
            repo=case.repo, base_commit=case.base_commit, variant=VARIANT,
            config_hash=config_hash, status=status, error=error,
            predicted_files=predicted, gold_files=gold_files, metrics=metrics,
            latency_seconds=detail.execution_seconds, llm_calls=calls, usage=usage,
            artifacts=_list_artifacts(case_dir / "roottrace"),
        )
        _write_json(case_dir / "result.json", result.model_dump(mode="json"))
        results.append(result)
    return results


def _cleanup(state: QueueState, output_dir: Path) -> None:
    for entry in state.cases:
        if entry.terminal and not entry.cleaned:
            destroy_case_workspace(CaseWorkspace(entry.workspace_root, entry.repo_path, entry.case.base_commit))
            entry.cleaned = True
            _save_state(output_dir / STATE_FILE, state)


def run_from_args(args: argparse.Namespace, *, queue: object | None = None) -> int:
    data_root = args.data_root.expanduser().resolve()
    manifest_path = (args.manifest or data_root / "manifests/dev50.json").expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    state_path = output_dir / STATE_FILE
    try:
        if args.workers < 1:
            raise ValueError("--workers must be positive")
        selected = select_dev50(manifest_path)
        print(f"dev50: {len(selected)} cases", flush=True)
        public = load_public_cases(data_root / PUBLIC_METADATA)
        _validate_manifest_against_public(RcaManifest(name="dev50", instances=selected), public)
        repo_cache = data_root / "repos"
        if not repo_cache.is_dir():
            raise FileNotFoundError(repo_cache)
        model = _resolve_model()
        if state_path.exists():
            state = QueueState.model_validate_json(state_path.read_text(encoding="utf-8"))
            if (
                state.manifest_sha256 != manifest_sha256(manifest_path)
                or state.model != model
                or state.worker_count != args.workers
            ):
                raise ValueError("saved queue state differs from manifest, model, or worker count")
            _validate_saved_state(state, selected, output_dir)
        else:
            state = QueueState(
                manifest_sha256=manifest_sha256(manifest_path),
                model=model, worker_count=args.workers,
            )
            _save_state(state_path, state)
        _prepare_all(selected, repo_cache, output_dir, state)
        redis_url = resolve_redis_url(args.redis_url) if queue is None else args.redis_url or ""
        active_queue = queue if queue is not None else create_queue(redis_url)
        with _workers(output_dir, redis_url, args.workers) as workers:
            _submit_all(state, public, output_dir, active_queue, args.job_timeout)
            metadata = _wait_for_all(state, output_dir, active_queue, workers)
        # Gold is first touched after all jobs have reached terminal status.
        gold = GoldStore(data_root / GOLD_METADATA)
        config_hash = hashlib.sha256(
            f"{state.manifest_sha256}:{model}:{VARIANT}:{state.worker_count}".encode()
        ).hexdigest()
        results = _collect(state, output_dir, gold, metadata, config_hash)
        metrics = compute_eval_metrics(results)
        write_reports(output_dir, metrics, EvalRunConfig(
            model=model, manifest_name="dev50", manifest_sha256=state.manifest_sha256,
            variant=VARIANT, config_hash=config_hash, root_trace_mode="rq",
        ))
        enqueued_times = [detail.enqueued_at for detail in metadata.values() if detail.enqueued_at]
        finished_times = [detail.finished_at for detail in metadata.values() if detail.finished_at]
        batch_start = min(enqueued_times) if len(enqueued_times) == len(selected) else state.first_submission_at
        batch_end = max(finished_times) if finished_times else state.completed_at
        batch_seconds = (
            max(0.0, (batch_end - batch_start).total_seconds())
            if batch_end and batch_start else None
        )
        _write_json(output_dir / "queue_summary.json", {
            "case_ids": [case.instance_id for case in selected],
            "worker_count": state.worker_count,
            "variant": VARIANT, "model": model,
            "batch_elapsed_seconds": batch_seconds,
            "failed_count": sum(result.status == "error" for result in results),
            "llm_calls": sum(result.llm_calls for result in results) if all(result.llm_calls is not None for result in results) else None,
            "prompt_tokens": sum(result.usage["prompt_tokens"] for result in results) if all(result.usage and result.usage["prompt_tokens"] is not None for result in results) else None,
            "completion_tokens": sum(result.usage["completion_tokens"] for result in results) if all(result.usage and result.usage["completion_tokens"] is not None for result in results) else None,
            "total_tokens": sum(result.usage["total_tokens"] for result in results) if all(result.usage and result.usage["total_tokens"] is not None for result in results) else None,
            "reasoning_tokens": sum(result.usage["reasoning_tokens"] for result in results) if all(result.usage and result.usage["reasoning_tokens"] is not None for result in results) else None,
            "jobs": [{
                "instance_id": entry.case.instance_id, "job_id": entry.job_id,
                "terminal_status": metadata[entry.case.instance_id].status.value,
                "queue_wait_seconds": metadata[entry.case.instance_id].queue_wait_seconds,
                "execution_seconds": metadata[entry.case.instance_id].execution_seconds,
                "total_seconds": metadata[entry.case.instance_id].total_latency_seconds,
                "llm_calls": next(result.llm_calls for result in results if result.instance_id == entry.case.instance_id),
                "usage": next(result.usage for result in results if result.instance_id == entry.case.instance_id),
            } for entry in state.cases],
        })
        _cleanup(state, output_dir)
        return 0 if all(result.status == "completed" for result in results) else 1
    except KeyboardInterrupt:
        print(f"interrupted; resume with --output-dir {output_dir}", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"queue evaluation error: {exc}; resume with --output-dir {output_dir}", file=sys.stderr)
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m evaluation.queue_runner")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("output/rca-eval-queue-dev50"))
    parser.add_argument("--redis-url", default=None)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--job-timeout", type=int, default=3600)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.job_timeout < 1:
        parser.error("--job-timeout must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")
    return run_from_args(args)


if __name__ == "__main__":
    sys.exit(main())
