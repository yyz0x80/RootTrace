"""SWE-bench-derived RCA evaluation runner.

Usage::

    python -m evaluation.runner --help

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
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field, ValidationError

from evaluation.adapter import (
    PublicCase,
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
    MAX_PREDICTED_FILES,
    CaseResult,
    compute_case_metrics,
    compute_eval_metrics,
)
from evaluation.report import EvalRunConfig, write_reports
from evaluation.variants import (
    AblationConfig,
    AblationVariant,
    VariantSettings,
    variant_settings,
)
from evaluation.workspace import (
    CaseWorkspace,
    create_case_workspace,
    destroy_case_workspace,
    resolve_repo_cache,
)
from roottrace.evidence.schema import AgentRole
from roottrace.incident.schema import IncidentInput

DEFAULT_DATA_ROOT = Path.home() / "Datasets" / "roottrace-swebench"
OUTPUT_ROOT = Path("output")
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
    errors: list[str] = Field(default_factory=list, max_length=20)
    report: dict | None = None
    latency_seconds: float = Field(default=0.0, ge=0)
    llm_calls: int | None = Field(default=None, ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)


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
    """Runs the real RootTrace pipeline in-process for one ablation variant.

    Only ``incident`` plus the workspace repository path cross this boundary;
    gold data is never passed. Token usage is taken from the RCA report and
    stays ``null`` when the provider did not return exact values.
    """

    def __init__(
        self,
        *,
        enabled_roles: frozenset[AgentRole] | None = None,
        retrieval_mode: str = "off",
        retriever: Any | None = None,
        history_excluded_ids: frozenset[str] = frozenset(),
        worker_concurrency: int = 3,
    ) -> None:
        self._enabled_roles = enabled_roles
        self._retrieval_mode = retrieval_mode
        self._retriever = retriever
        self._history_excluded_ids = history_excluded_ids
        self._worker_concurrency = worker_concurrency

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
        from roottrace.cli import run_rca_pipeline
        from roottrace.incident.loader import LoadedIncident
        from roottrace.llm.provider import create_provider_from_config

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
                worker_concurrency=self._worker_concurrency,
                enabled_roles=self._enabled_roles,
                retriever=self._retriever,
                retrieval_mode=self._retrieval_mode,
                history_excluded_ids=self._history_excluded_ids,
            )
        except Exception as exc:  # noqa: BLE001 - isolate per-case RootTrace failures
            elapsed = time.monotonic() - started
            return RootTraceOutcome(
                status="error",
                error=_bounded_error(str(exc)),
                latency_seconds=elapsed,
            )
        elapsed = time.monotonic() - started
        run_usage = result.run.usage
        report_usage = result.report.usage
        calls = (run_usage.llm_calls if run_usage else 0) + (
            report_usage.llm_calls if report_usage else 0
        )
        return RootTraceOutcome(
            status="completed",
            errors=list(result.run.errors),
            report=_read_report_json(output_dir),
            latency_seconds=elapsed,
            llm_calls=calls,
            prompt_tokens=_sum_or_none(
                run_usage.prompt_tokens if run_usage else None,
                report_usage.prompt_tokens if report_usage else None,
            ),
            completion_tokens=_sum_or_none(
                run_usage.completion_tokens if run_usage else None,
                report_usage.completion_tokens if report_usage else None,
            ),
            reasoning_tokens=_sum_or_none(
                run_usage.reasoning_tokens if run_usage else None,
                report_usage.reasoning_tokens if report_usage else None,
            ),
        )


def _bounded_error(text: str, limit: int = MAX_ERROR_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _sum_or_none(left: int | None, right: int | None) -> int | None:
    """Sum exact usage values; an unknown side keeps the total null."""
    if left is None or right is None:
        return None
    return left + right


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
    variant: AblationVariant,
    config_hash: str,
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
        reasoning_tokens=outcome.reasoning_tokens,
    )
    usage = None
    if (
        outcome.prompt_tokens is not None
        or outcome.completion_tokens is not None
        or outcome.reasoning_tokens is not None
    ):
        usage = {
            "prompt_tokens": outcome.prompt_tokens,
            "completion_tokens": outcome.completion_tokens,
            "reasoning_tokens": outcome.reasoning_tokens,
            "total_tokens": metrics.total_tokens,
        }
    return CaseResult(
        schema_version="1.0",
        instance_id=case.instance_id,
        repo=case.repo,
        base_commit=case.base_commit,
        variant=variant.value,
        config_hash=config_hash,
        status=outcome.status,
        error=outcome.error,
        rca_errors=list(outcome.errors),
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
    *,
    variant: AblationVariant,
    config_hash: str,
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
        variant=variant.value,
        config_hash=config_hash,
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
    *,
    variant: AblationVariant,
    config_hash: str,
) -> dict[str, CaseResult]:
    """Load persisted completed results matching the current variant/config."""
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
        if (
            result.status == "completed"
            and result.variant == variant.value
            and result.config_hash == config_hash
        ):
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


def _build_ablation_config(
    args: argparse.Namespace,
    manifest: RcaManifest,
    manifest_path: Path,
    selected: list[ManifestCase],
) -> AblationConfig:
    """Build the effective ablation config from CLI args and optional JSON."""
    base: dict = {}
    if args.ablation_config is not None:
        config_path = Path(args.ablation_config).expanduser().resolve()
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("ablation config must be a JSON object")
        base.update(raw)
    base["variant"] = AblationVariant(args.variant).value
    base["model"] = args.model
    base["manifest_name"] = manifest.name
    base["manifest_sha256"] = manifest_sha256(manifest_path)
    base["history_corpus"] = args.history_corpus
    base["history_index"] = args.history_index
    base["max_cases"] = args.max_cases
    base["history_excluded_ids"] = sorted(
        {case.instance_id for case in selected}
    )
    return AblationConfig(**base)


def _build_retriever(config: AblationConfig):
    """Build a historical retriever when corpus/index are configured."""
    if not config.history_corpus:
        return None
    corpus_path = Path(config.history_corpus).expanduser().resolve()
    if not corpus_path.is_file():
        print(
            "warning: history corpus configured but not found; "
            "retrieval variant will run without hints",
            file=sys.stderr,
        )
        return None
    from roottrace.history import (
        HistoricalRetriever,
        build_history_index,
        build_tfidf,
        import_corpus,
    )
    from roottrace.history.retrieval import case_text

    corpus = import_corpus(corpus_path, split="historical")
    if config.history_index:
        from roottrace.history.schema import HistoryIndex

        index = HistoryIndex.model_validate_json(
            Path(config.history_index).expanduser().resolve().read_text(
                encoding="utf-8"
            )
        )
    else:
        tfidf = build_tfidf([case_text(case) for case in corpus.cases])
        clustering = config.variant != AblationVariant.CLUSTERING_OFF
        index = build_history_index(
            corpus,
            tfidf,
            seed=42,
            clustering=clustering,
        )
    return HistoricalRetriever(corpus, index, top_k=config.retrieval_top_k)


def _make_run_client(
    settings: VariantSettings,
    config: AblationConfig,
) -> RootTraceClient:
    """Build the RootTrace client for one ablation variant."""
    if settings.deterministic:
        from evaluation.baseline import DeterministicBaselineClient

        return DeterministicBaselineClient()
    retriever = None
    retrieval_mode = settings.retrieval_mode
    if settings.retrieval_mode != "off":
        retriever = _build_retriever(config)
        if retriever is None:
            retrieval_mode = "off"
    return InProcessRootTraceClient(
        enabled_roles=frozenset(settings.enabled_roles) or None,
        retrieval_mode=retrieval_mode,
        retriever=retriever,
        history_excluded_ids=frozenset(config.history_excluded_ids),
        worker_concurrency=config.worker_concurrency,
    )


def _write_variant_config(output_dir: Path, config: AblationConfig) -> Path:
    """Persist the effective variant config for reproducibility."""
    payload = config.model_dump(mode="json")
    payload["config_hash"] = config.config_hash()
    path = output_dir / "variant.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


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
    variant = AblationVariant(args.variant)
    settings = variant_settings(variant)

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
    try:
        ablation = _build_ablation_config(args, manifest, manifest_path, selected)
    except (OSError, TypeError, ValueError) as exc:
        print(f"evaluation error: invalid ablation config: {exc}", file=sys.stderr)
        return 2
    eval_config = EvalRunConfig(
        model=ablation.model,
        manifest_name=manifest.name,
        manifest_sha256=ablation.manifest_sha256,
        seed=manifest.seed,
        max_cases=ablation.max_cases,
        resume=args.resume,
        variant=ablation.variant.value,
        config_hash=ablation.config_hash(),
        budgets=ablation.budgets.model_dump(mode="json"),
        history_corpus=ablation.history_corpus,
        root_trace_mode=(
            "deterministic_baseline" if settings.deterministic else "in_process"
        ),
    )

    if args.dry_run:
        return _dry_run(selected, repo_cache)

    output_dir = (
        OUTPUT_ROOT / f"rca-eval-{variant.value}"
        if args.output_dir is None
        else Path(args.output_dir).expanduser().resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_variant_config(output_dir, ablation)

    results: dict[str, CaseResult] = {}
    if args.resume:
        results = _load_existing_results(
            output_dir,
            selected,
            variant=variant,
            config_hash=ablation.config_hash(),
        )
        if results:
            print(f"resume: skipping {len(results)} completed case(s)")

    active_client = (
        client if client is not None else _make_run_client(settings, ablation)
    )
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
                    variant=variant,
                    config_hash=ablation.config_hash(),
                )
            finally:
                destroy_case_workspace(workspace)
        except Exception as exc:  # noqa: BLE001 - a failing case must not abort the run
            result = _error_case_result(
                case,
                public_case,
                gold_store,
                exc,
                variant=variant,
                config_hash=ablation.config_hash(),
            )
        results[case_id] = result
        _write_case_result(case_dir, result)
        print(f"[{case_id}] status={result.status}")

    metrics = compute_eval_metrics(list(results.values()))
    write_reports(output_dir, metrics, eval_config)
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
        prog="python -m evaluation.runner",
        description="SWE-bench-derived RCA benchmark runner",
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
        default=None,
        help=(
            "directory for per-case results and aggregate reports "
            "(default: output/rca-eval-<variant>)"
        ),
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
        help="skip cases with a persisted completed result for this variant/config",
    )
    parser.add_argument(
        "--variant",
        default="three_specialists_retrieval_off",
        choices=[variant.value for variant in AblationVariant],
        help="ablation variant (default: three_specialists_retrieval_off)",
    )
    parser.add_argument(
        "--ablation-config",
        type=Path,
        default=None,
        help=(
            "optional JSON file with shared ablation settings (budgets, "
            "context limits, concurrency); --variant overrides its variant"
        ),
    )
    parser.add_argument(
        "--history-corpus",
        default=None,
        help="optional historical RCA corpus JSONL for retrieval variants",
    )
    parser.add_argument(
        "--history-index",
        default=None,
        help="optional persisted history index JSON; built when omitted",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_from_args(args)


if __name__ == "__main__":
    sys.exit(main())
