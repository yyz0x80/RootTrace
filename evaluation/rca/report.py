"""Deterministic aggregate report rendering for RCA evaluation runs.

Reports are timestamp-free: the same per-case results always produce
byte-identical ``metrics.json`` and ``report.md``.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from evaluation.rca.metrics import EvalMetrics

METRICS_FILENAME = "metrics.json"
REPORT_FILENAME = "report.md"


class EvalRunConfig(BaseModel):
    """Evaluator configuration embedded in reports for reproducibility."""

    model: str | None = None
    manifest_name: str = ""
    manifest_sha256: str = ""
    seed: int | None = None
    max_cases: int | None = None
    resume: bool = False
    root_trace_mode: str = "in_process"
    variant: str = ""
    config_hash: str = ""
    budgets: dict = Field(default_factory=dict)
    history_corpus: str | None = None


def metrics_document(metrics: EvalMetrics, config: EvalRunConfig) -> dict:
    """Serialize metrics plus config into a deterministic JSON document."""
    return {
        "schema_version": metrics.schema_version,
        "config": config.model_dump(),
        "aggregate": metrics.aggregate.model_dump(),
        "per_case": [case.model_dump() for case in metrics.per_case],
    }


def _format_rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _format_seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _format_int(value: int | None) -> str:
    return "n/a" if value is None else str(value)


def render_markdown(metrics: EvalMetrics, config: EvalRunConfig) -> str:
    """Render a deterministic human-readable evaluation report."""
    aggregate = metrics.aggregate
    lines = [
        "# RootTrace SWE-bench-derived RCA Evaluation",
        "",
        "## Configuration",
        "",
        "| Setting | Value |",
        "| --- | --- |",
        f"| manifest | {config.manifest_name} |",
        f"| manifest sha256 | {config.manifest_sha256} |",
        f"| seed | {_format_int(config.seed)} |",
        f"| model | {config.model or 'n/a'} |",
        f"| max cases | {_format_int(config.max_cases)} |",
        f"| resume | {'yes' if config.resume else 'no'} |",
        f"| root trace mode | {config.root_trace_mode} |",
        f"| variant | {config.variant or 'n/a'} |",
        f"| config hash | {config.config_hash or 'n/a'} |",
        f"| budgets | {config.budgets or 'n/a'} |",
        f"| history corpus | {config.history_corpus or 'n/a'} |",
        "",
        "## Aggregate metrics",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| total cases | {aggregate.total_cases} |",
        f"| covered cases | {aggregate.covered_cases} |",
        f"| failed cases | {aggregate.failed_cases} |",
        f"| invalid outputs | {aggregate.invalid_outputs} |",
        f"| coverage | {_format_rate(aggregate.coverage)} |",
        f"| invalid output rate | {_format_rate(aggregate.invalid_output_rate)} |",
        f"| top-1 file accuracy | {_format_rate(aggregate.top_1_file_accuracy)} |",
        f"| any file recall@3 | {_format_rate(aggregate.any_file_recall_at_3)} |",
        f"| any file recall@5 | {_format_rate(aggregate.any_file_recall_at_5)} |",
        f"| all file recall@3 | {_format_rate(aggregate.all_file_recall_at_3)} |",
        f"| all file recall@5 | {_format_rate(aggregate.all_file_recall_at_5)} |",
        f"| mean gold file recall@5 | {_format_rate(aggregate.mean_gold_file_recall_at_5)} |",
        f"| latency P50 (s) | {_format_seconds(aggregate.latency_p50_seconds)} |",
        f"| latency P95 (s) | {_format_seconds(aggregate.latency_p95_seconds)} |",
        f"| mean LLM calls/case | {_format_rate(aggregate.mean_llm_calls_per_case)} |",
        f"| mean exact tokens/case | {_format_rate(aggregate.mean_total_tokens_per_case)} |",
        f"| exact token cases | {aggregate.exact_token_cases} |",
        f"| null token cases | {aggregate.null_token_cases} |",
        f"| mean reasoning tokens/case | {_format_rate(aggregate.mean_reasoning_tokens_per_case)} |",
        f"| reasoning token cases | {aggregate.reasoning_token_cases} |",
        f"| null reasoning token cases | {aggregate.null_reasoning_cases} |",
        "",
        "## Per-case results",
        "",
        "| instance | status | predicted | top-1 | any@3 | any@5 | all@5 | recall@5 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for case in metrics.per_case:
        lines.append(
            "| {id} | {status} | {predicted} | {top1} | {any3} | {any5} | {all5} "
            "| {recall5} |".format(
                id=case.instance_id,
                status=case.status,
                predicted=len(case.predicted_files),
                top1=_format_bool(case.top_1_file_accuracy),
                any3=_format_bool(case.any_file_recall_at_3),
                any5=_format_bool(case.any_file_recall_at_5),
                all5=_format_bool(case.all_file_recall_at_5),
                recall5=(
                    "n/a"
                    if case.gold_file_recall_at_5 is None
                    else f"{case.gold_file_recall_at_5:.3f}"
                ),
            )
        )
    lines.append("")
    failed = [case for case in metrics.per_case if case.status == "error"]
    if failed:
        lines.append("## Failed cases")
        lines.append("")
        for case in failed:
            lines.append(f"- `{case.instance_id}`: {case.error or 'unknown error'}")
        lines.append("")
    return "\n".join(lines)


def _format_bool(value: bool | None) -> str:
    return "n/a" if value is None else ("1" if value else "0")


def write_reports(
    output_dir: str | Path,
    metrics: EvalMetrics,
    config: EvalRunConfig,
) -> tuple[Path, Path]:
    """Write deterministic ``metrics.json`` and ``report.md``; returns paths."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / METRICS_FILENAME
    metrics_path.write_text(
        json.dumps(metrics_document(metrics, config), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    report_path = output / REPORT_FILENAME
    report_path.write_text(render_markdown(metrics, config), encoding="utf-8")
    return metrics_path, report_path
