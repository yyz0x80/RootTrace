"""M9-A: SWE-bench-derived RCA evaluation runner.

Usage::

    python -m evaluation.rca.runner --help

The runner validates the manifest, materializes a disposable base-commit-only
workspace per case, invokes RootTrace (real in-process client for production,
fake client in tests), and only then reads the gold patch to score file
localization. A failing case is recorded per case and never aborts the
benchmark; ``--resume`` skips already-completed cases.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, Field, ValidationError

from evaluation.rca.adapter import (
    PublicCase,
    build_incident_input,
    case_from_public_record,
    write_root_trace_input,
)
from evaluation.rca.gold import GoldStore
from evaluation.rca.manifest import (
    ManifestCase,
    RcaManifest,
    load_manifest,
    manifest_sha256,
)
from evaluation.rca.metrics import (
    MAX_PREDICTED_FILES,
    CaseResult,
    compute_case_metrics,
    compute_eval_metrics,
)
from evaluation.rca.report import EvalRunConfig, write_reports
from evaluation.rca.workspace import (
    CaseWorkspace,
    create_case_workspace,
    destroy_case_workspace,
    resolve_repo_cache,
)
from patchpilot.rca.schema import IncidentInput

DEFAULT_DATA_ROOT = Path.home() / "Datasets" / "roottrace-swebench"
DEFAULT_OUTPUT_DIR = Path("output") / "rca-eval"
PUBLIC_METADATA = "public/verified_public.jsonl"
GOLD_METADATA = "gold/verified_gold.jsonl"
DEFAULT_MANIFEST = "manifests/smoke3.json"

REPORT_FILE = "rca_report.json"
RESULT_FILE = "result.json"
INPUT_FILE = "root_trace_input.json"
ROOT_TRACE_OUTPUT = "roottrace"
MAX_RESULT_ARTIFACTS = 50
MAX_ERROR_CHARS = 500


class RootTraceOutcome(BaseModel):
    """Typed outcome of one RootTrace invocation by the evaluator."""

    status: Literal["completed", "error"]
    error: str | None = None
    report: dict | None = None
    latency_seconds: float = Field(default=0.0, ge=0)
    llm_calls: int | None = Field(default=None, ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)


class RootTraceClient(Protocol):
    """Evaluator-facing RootTrace invocation boundary."""

    def run(
        self,
        *,
        case_id: str,
        repo: Path,
        incident: IncidentInput,
        output_dir: Path,
        model: str | None,
    ) -> RootTraceOutcome:
        """Run RootTrace on one case workspace and return its outcome."""


class InProcessRootTraceClient:
    """Runs the real RootTrace pipeline in-process.

    Only ``incident`` plus the workspace repository path cross this boundary;
    gold data is never passed. Token usage is taken from the RCA report and
    stays ``null`` when the provider did not return exact values.
    """

    def run(
        self,
        *,
        case_id: str,
        repo: Path,
        incident: IncidentInput,
        output_dir: Path,
        model: str | None,
    ) -> RootTraceOutcome:
        del case_id  # retained in the protocol for stable tracing
        from patchpilot.provider import create_provider_from_config
        from patchpilot.rca.cli import run_rca_pipeline
        from patchpilot.rca.incident_loader import LoadedIncident

        loaded = LoadedIncident(incident=incident)

        def provider_factory():
            return create_provider_from_config(model_name=model)

        started = time.monotonic()
        try:
            result = run_rca_pipeline(
                loaded,
                repo,
                output_dir,
                provider_factory=provider_factory,
            )
        except Exception as exc:  # noqa: BLE001 - isolate per-case RootTrace failures
            elapsed = time.monotonic() - started
            return RootTraceOutcome(
                status="error",
                error=_bounded_error(str(exc)),
                latency_seconds=elapsed,
            )
        elapsed = time.monotonic() - started
        usage = result.report.usage
        return RootTraceOutcome(
            status="completed",
            report=_read_report_json(output_dir),
            latency_seconds=elapsed,
            llm_calls=usage.llm_calls,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
        )


def _bounded_error(text: str, limit: int = MAX_ERROR_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _read_report_json(output_dir: Path) -> dict | None:
    report_path = output_dir / REPORT_FILE
    if not report_path.is_file():
        return None
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _clean_predicted_path(value: str) -> str | None:
    if not value or not value.strip():
        return None
    stripped = value.strip()
    if "\\" in stripped or stripped.startswith(("/", "~")):
        return None
    parts = stripped.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        return None
    return "/".join(parts)


def extract_predicted_files(report: dict | None) -> list[str]:
    """Extract ordered predicted source files from an RCA report."""
    if not isinstance(report, dict):
        return []
    locations = report.get("top_k_locations")
    if not isinstance(locations, list):
        return []
    files: list[str] = []
    seen: set[str] = set()
    for location in locations:
        if not isinstance(location, dict):
            continue
        raw = location.get("path")
        if not isinstance(raw, str):
            continue
        path = _clean_predicted_path(raw)
        if path is None or path in seen:
            continue
        seen.add(path)
        files.append(path)
        if len(files) >= MAX_PREDICTED_FILES:
            break
    return files


def load_public_cases(path: str | Path) -> dict[str, PublicCase]:
    """Load public SWE-bench metadata, rejecting gold fields and duplicates."""
    public_path = Path(path)
    cases: dict[str, PublicCase] = {}
    with public_path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                case = case_from_public_record(record)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    f"invalid public record at line {line_no}: {exc}"
                ) from exc
            if case.instance_id in cases:
                raise ValueError(f"duplicate public instance_id: {case.instance_id}")
            cases[case.instance_id] = case
    return cases


def _validate_manifest_against_public(
    manifest: RcaManifest,
    public: dict[str, PublicCase],
) -> None:
    missing = [
        case.instance_id for case in manifest.instances if case.instance_id not in public
    ]
    if missing:
        raise ValueError(f"manifest references unknown public cases: {sorted(missing)}")
    for case in manifest.instances:
        public_case = public[case.instance_id]
        if (public_case.repo, public_case.base_commit) != (
            case.repo,
            case.base_commit,
        ):
            raise ValueError(
                f"manifest/public metadata mismatch for {case.instance_id}: "
                f"manifest {case.repo}@{case.base_commit[:12]}, "
                f"public {public_case.repo}@{public_case.base_commit[:12]}"
            )


def _list_artifacts(output_dir: Path) -> list[str]:
    artifacts = sorted(
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file()
    )
    return artifacts[:MAX_RESULT_ARTIFACTS]


def _run_case(
    client: RootTraceClient,
    case: ManifestCase,
    public_case: PublicCase,
    workspace: CaseWorkspace,
    case_dir: Path,
    gold_store: GoldStore,
    *,
    model: str | None,
) -> CaseResult:
    adapter_result = build_incident_input(public_case)
    incident = adapter_result.incident
    write_root_trace_input(incident, case_dir / INPUT_FILE)
    root_trace_out = case_dir / ROOT_TRACE_OUTPUT
    root_trace_out.mkdir(parents=True, exist_ok=True)

    outcome = client.run(
        case_id=case.instance_id,
        repo=workspace.repo,
        incident=incident,
        output_dir=root_trace_out,
        model=model,
    )
    # Gold is read only after the RootTrace run has completed.
    gold_files = gold_store.gold_files(case.instance_id)
    predicted = extract_predicted_files(outcome.report)
    metrics = compute_case_metrics(
        instance_id=case.instance_id,
        predicted_files=predicted,
        gold_files=gold_files,
        status=outcome.status,
        error=outcome.error,
        latency_seconds=outcome.latency_seconds,
        llm_calls=outcome.llm_calls,
        prompt_tokens=outcome.prompt_tokens,
        completion_tokens=outcome.completion_tokens,
    )
    usage = None
    if outcome.prompt_tokens is not None or outcome.completion_tokens is not None:
        usage = {
            "prompt_tokens": outcome.prompt_tokens,
            "completion_tokens": outcome.completion_tokens,
            "total_tokens": metrics.total_tokens,
        }
    return CaseResult(
        schema_version="1.0",
        instance_id=case.instance_id,
        repo=case.repo,
        base_commit=case.base_commit,
        status=outcome.status,
        error=outcome.error,
        predicted_files=predicted,
        gold_files=gold_files,
        metrics=metrics,
        latency_seconds=outcome.latency_seconds,
        llm_calls=outcome.llm_calls,
        usage=usage,
        artifacts=_list_artifacts(root_trace_out),
    )


def _error_case_result(
    case: ManifestCase,
    public_case: PublicCase,
    gold_store: GoldStore,
    exc: Exception,
) -> CaseResult:
    try:
        gold_files = gold_store.gold_files(case.instance_id)
    except (OSError, ValueError):  # gold access is best-effort for error results
        gold_files = []
    metrics = compute_case_metrics(
        instance_id=case.instance_id,
        predicted_files=[],
        gold_files=gold_files,
        status="error",
        error=_bounded_error(str(exc)),
    )
    return CaseResult(
        schema_version="1.0",
        instance_id=case.instance_id,
        repo=case.repo,
        base_commit=case.base_commit,
        status="error",
        error=_bounded_error(str(exc)),
        gold_files=gold_files,
        metrics=metrics,
    )


def _write_case_result(case_dir: Path, result: CaseResult) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / RESULT_FILE).write_text(
        json.dumps(result.model_dump(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_existing_results(
    output_dir: Path,
    cases: list[ManifestCase],
) -> dict[str, CaseResult]:
    """Load persisted completed results for ``--resume``."""
    results: dict[str, CaseResult] = {}
    for case in cases:
        result_path = output_dir / "cases" / case.instance_id / RESULT_FILE
        if not result_path.is_file():
            continue
        try:
            raw = json.loads(result_path.read_text(encoding="utf-8"))
            result = CaseResult.model_validate(raw)
        except (OSError, json.JSONDecodeError, ValidationError):
            continue
        if result.status == "completed":
            results[case.instance_id] = result
    return results


def _dry_run(
    selected: list[ManifestCase],
    repo_cache: Path,
) -> int:
    print("DRY RUN: validating manifest, public metadata, and repo mirrors")
    for case in selected:
        try:
            mirror = resolve_repo_cache(repo_cache, case.repo)
        except FileNotFoundError as exc:
            print(f"  [ERROR] {case.instance_id}: {exc}", file=sys.stderr)
            return 2
        print(
            f"  would run {case.instance_id} "
            f"({case.repo}@{case.base_commit[:12]}) mirror={mirror}"
        )
    print(f"DRY RUN: {len(selected)} case(s) ready; no files were written")
    return 0


def _print_summary(metrics) -> None:
    aggregate = metrics.aggregate
    print(f"evaluation complete: {aggregate.total_cases} case(s)")
    print(f"  coverage={aggregate.coverage} invalid_rate={aggregate.invalid_output_rate}")
    print(f"  top1={aggregate.top_1_file_accuracy} any@5={aggregate.any_file_recall_at_5}")
    print(f"  failed={aggregate.failed_cases} invalid_outputs={aggregate.invalid_outputs}")


def run_from_args(
    args: argparse.Namespace,
    *,
    client: RootTraceClient | None = None,
) -> int:
    """Execute the evaluation pipeline from parsed CLI arguments.

    ``client`` is injectable for tests; production uses the in-process
    RootTrace client by default.
    """
    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.is_dir():
        print(f"evaluation error: data root is not a directory: {data_root}", file=sys.stderr)
        return 2
    manifest_path = Path(args.manifest or data_root / DEFAULT_MANIFEST).expanduser().resolve()
    repo_cache = Path(args.repo_cache or data_root / "repos").expanduser().resolve()
    gold_path = Path(args.gold_path or data_root / GOLD_METADATA).expanduser().resolve()
    public_path = data_root / PUBLIC_METADATA
    output_dir = Path(args.output_dir).expanduser().resolve()

    try:
        manifest = load_manifest(manifest_path)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        print(f"evaluation error: {exc}", file=sys.stderr)
        return 2
    for label, path in (
        ("public metadata", public_path),
        ("gold metadata", gold_path),
    ):
        if not path.is_file():
            print(f"evaluation error: {label} file not found: {path}", file=sys.stderr)
            return 2
    if not repo_cache.is_dir():
        print(f"evaluation error: repo cache is not a directory: {repo_cache}", file=sys.stderr)
        return 2

    try:
        public = load_public_cases(public_path)
        _validate_manifest_against_public(manifest, public)
    except ValueError as exc:
        print(f"evaluation error: {exc}", file=sys.stderr)
        return 2

    selected = (
        list(manifest.instances[: args.max_cases])
        if args.max_cases is not None
        else list(manifest.instances)
    )
    config = EvalRunConfig(
        model=args.model,
        manifest_name=manifest.name,
        manifest_sha256=manifest_sha256(manifest_path),
        seed=manifest.seed,
        max_cases=args.max_cases,
        resume=args.resume,
    )

    if args.dry_run:
        return _dry_run(selected, repo_cache)

    results: dict[str, CaseResult] = {}
    if args.resume:
        results = _load_existing_results(output_dir, selected)
        if results:
            print(f"resume: skipping {len(results)} completed case(s)")

    active_client = client if client is not None else InProcessRootTraceClient()
    gold_store = GoldStore(gold_path)
    for case in selected:
        case_id = case.instance_id
        if case_id in results:
            continue
        case_dir = output_dir / "cases" / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        public_case = public[case_id]
        try:
            workspace = create_case_workspace(repo_cache, case)
            try:
                result = _run_case(
                    active_client,
                    case,
                    public_case,
                    workspace,
                    case_dir,
                    gold_store,
                    model=args.model,
                )
            finally:
                destroy_case_workspace(workspace)
        except Exception as exc:  # noqa: BLE001 - a failing case must not abort the run
            result = _error_case_result(case, public_case, gold_store, exc)
        results[case_id] = result
        _write_case_result(case_dir, result)
        print(f"[{case_id}] status={result.status}")

    metrics = compute_eval_metrics(list(results.values()))
    write_reports(output_dir, metrics, config)
    _print_summary(metrics)
    return 0


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the evaluation runner CLI parser."""
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.rca.runner",
        description="SWE-bench-derived RCA evaluation pipeline (M9-A)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="directory containing public/, gold/, manifests/, and repos/",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="manifest file (default: <data-root>/manifests/smoke3.json)",
    )
    parser.add_argument(
        "--repo-cache",
        type=Path,
        default=None,
        help="local git mirror cache (default: <data-root>/repos)",
    )
    parser.add_argument(
        "--gold-path",
        type=Path,
        default=None,
        help="evaluator-only gold metadata (default: <data-root>/gold/verified_gold.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for per-case results and aggregate reports",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="model name passed to RootTrace (required for real runs)",
    )
    parser.add_argument(
        "--max-cases",
        type=_positive_int,
        default=None,
        help="run at most this many manifest cases",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate manifest/public metadata/repo mirrors without running",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip cases with a persisted completed result",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_from_args(args)


if __name__ == "__main__":
    sys.exit(main())
