"""Localization metrics for SWE-bench-derived RCA evaluation.

These metrics measure file localization, never natural-language "RCA
accuracy". Missing data is never treated as zero: unknown values stay
``null`` and aggregate denominators only count cases with a measurable value.
"""

from __future__ import annotations

import statistics
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, Field

MAX_PREDICTED_FILES = 10

CaseStatus = Literal["completed", "error"]


class CaseMetrics(BaseModel):
    """Per-case localization metrics."""

    instance_id: str
    status: CaseStatus
    error: str | None = None
    predicted_files: list[str] = Field(default_factory=list)
    gold_files: list[str] = Field(default_factory=list)
    top_1_file_accuracy: bool | None = None
    any_file_recall_at_3: bool | None = None
    any_file_recall_at_5: bool | None = None
    all_file_recall_at_3: bool | None = None
    all_file_recall_at_5: bool | None = None
    gold_file_recall_at_5: float | None = None
    covered: bool = False
    invalid_output: bool = False
    latency_seconds: float | None = None
    llm_calls: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class CaseResult(BaseModel):
    """Persisted per-case evaluation record (Evaluator artifact)."""

    schema_version: str = "1.0"
    instance_id: str
    repo: str | None = None
    base_commit: str | None = None
    status: CaseStatus
    error: str | None = None
    rca_errors: list[str] = Field(default_factory=list, max_length=20)
    variant: str = ""
    config_hash: str | None = None
    predicted_files: list[str] = Field(default_factory=list)
    gold_files: list[str] = Field(default_factory=list)
    metrics: CaseMetrics
    latency_seconds: float | None = None
    llm_calls: int | None = None
    usage: dict | None = None
    artifacts: list[str] = Field(default_factory=list)


class AggregateMetrics(BaseModel):
    """Aggregate localization metrics over all manifest-scope cases."""

    total_cases: int = 0
    covered_cases: int = 0
    failed_cases: int = 0
    invalid_outputs: int = 0
    coverage: float | None = None
    invalid_output_rate: float | None = None
    top_1_file_accuracy: float | None = None
    any_file_recall_at_3: float | None = None
    any_file_recall_at_5: float | None = None
    all_file_recall_at_3: float | None = None
    all_file_recall_at_5: float | None = None
    mean_gold_file_recall_at_5: float | None = None
    latency_p50_seconds: float | None = None
    latency_p95_seconds: float | None = None
    mean_llm_calls_per_case: float | None = None
    mean_total_tokens_per_case: float | None = None
    exact_token_cases: int = 0
    null_token_cases: int = 0


class EvalMetrics(BaseModel):
    """Full evaluation metrics document (deterministic)."""

    schema_version: str = "1.0"
    aggregate: AggregateMetrics = Field(default_factory=AggregateMetrics)
    per_case: list[CaseMetrics] = Field(default_factory=list)


def _normalize_predicted_path(value: str) -> str | None:
    if not value or not value.strip():
        return None
    stripped = value.strip()
    if "\\" in stripped or stripped.startswith(("/", "~")):
        return None
    parts = stripped.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        return None
    return "/".join(parts)


def normalize_predicted_files(files: list[str]) -> list[str]:
    """Deduplicate and bound predicted files, dropping invalid paths."""
    seen: set[str] = set()
    normalized: list[str] = []
    for raw in files:
        path = _normalize_predicted_path(raw)
        if path is None or path in seen:
            continue
        seen.add(path)
        normalized.append(path)
        if len(normalized) >= MAX_PREDICTED_FILES:
            break
    return normalized


def _normalize_gold_files(files: list[str]) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for raw in files:
        path = PurePosixPath(raw).as_posix()
        if path in seen:
            continue
        seen.add(path)
        normalized.append(path)
    return sorted(normalized)


def compute_case_metrics(
    *,
    instance_id: str,
    predicted_files: list[str],
    gold_files: list[str],
    status: CaseStatus,
    error: str | None = None,
    latency_seconds: float | None = None,
    llm_calls: int | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
) -> CaseMetrics:
    """Compute one case's localization metrics deterministically."""
    predicted = normalize_predicted_files(predicted_files)
    gold = _normalize_gold_files(gold_files)
    gold_set = set(gold)
    covered = status == "completed"
    invalid_output = covered and not predicted

    top_1: bool | None = None
    any_3: bool | None = None
    any_5: bool | None = None
    all_3: bool | None = None
    all_5: bool | None = None
    gold_recall_5: float | None = None
    if covered and gold_set:
        top_1 = bool(predicted and predicted[0] in gold_set)
        any_3 = bool(set(predicted[:3]) & gold_set)
        any_5 = bool(set(predicted[:5]) & gold_set)
        all_3 = bool(gold_set.issubset(set(predicted[:3])))
        all_5 = bool(gold_set.issubset(set(predicted[:5])))
        gold_recall_5 = (
            len(set(predicted[:5]) & gold_set) / len(gold_set)
        )

    total_tokens = (
        prompt_tokens + completion_tokens
        if prompt_tokens is not None and completion_tokens is not None
        else None
    )
    return CaseMetrics(
        instance_id=instance_id,
        status=status,
        error=error,
        predicted_files=predicted,
        gold_files=gold,
        top_1_file_accuracy=top_1,
        any_file_recall_at_3=any_3,
        any_file_recall_at_5=any_5,
        all_file_recall_at_3=all_3,
        all_file_recall_at_5=all_5,
        gold_file_recall_at_5=gold_recall_5,
        covered=covered,
        invalid_output=invalid_output,
        latency_seconds=latency_seconds,
        llm_calls=llm_calls,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def _rate(values: list[bool]) -> float | None:
    if not values:
        return None
    return sum(1 for value in values if value) / len(values)


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _percentile(values: list[float], percent: int) -> float | None:
    ordered = sorted(value for value in values if value is not None)
    if not ordered:
        return None
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (percent / 100.0) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def compute_eval_metrics(results: list[CaseResult]) -> EvalMetrics:
    """Aggregate per-case results into deterministic evaluation metrics."""
    ordered = sorted(results, key=lambda result: result.instance_id)
    per_case = [
        compute_case_metrics(
            instance_id=result.instance_id,
            predicted_files=result.predicted_files,
            gold_files=result.gold_files,
            status=result.status,
            error=result.error,
            latency_seconds=result.latency_seconds,
            llm_calls=result.llm_calls,
            prompt_tokens=(result.usage or {}).get("prompt_tokens"),
            completion_tokens=(result.usage or {}).get("completion_tokens"),
        )
        for result in ordered
    ]
    total = len(per_case)
    covered = [case for case in per_case if case.covered]

    token_values = [case.total_tokens for case in covered if case.total_tokens is not None]
    aggregate = AggregateMetrics(
        total_cases=total,
        covered_cases=len(covered),
        failed_cases=sum(1 for case in per_case if case.status == "error"),
        invalid_outputs=sum(1 for case in per_case if case.invalid_output),
        coverage=(len(covered) / total) if total else None,
        invalid_output_rate=(
            sum(1 for case in per_case if case.invalid_output) / total
        )
        if total
        else None,
        top_1_file_accuracy=_rate(
            [case.top_1_file_accuracy for case in per_case if case.top_1_file_accuracy is not None]
        ),
        any_file_recall_at_3=_rate(
            [case.any_file_recall_at_3 for case in per_case if case.any_file_recall_at_3 is not None]
        ),
        any_file_recall_at_5=_rate(
            [case.any_file_recall_at_5 for case in per_case if case.any_file_recall_at_5 is not None]
        ),
        all_file_recall_at_3=_rate(
            [case.all_file_recall_at_3 for case in per_case if case.all_file_recall_at_3 is not None]
        ),
        all_file_recall_at_5=_rate(
            [case.all_file_recall_at_5 for case in per_case if case.all_file_recall_at_5 is not None]
        ),
        mean_gold_file_recall_at_5=_mean(
            [
                case.gold_file_recall_at_5
                for case in per_case
                if case.gold_file_recall_at_5 is not None
            ]
        ),
        latency_p50_seconds=_percentile(
            [case.latency_seconds for case in covered], 50
        ),
        latency_p95_seconds=_percentile(
            [case.latency_seconds for case in covered], 95
        ),
        mean_llm_calls_per_case=_mean(
            [float(case.llm_calls) for case in per_case if case.llm_calls is not None]
        ),
        mean_total_tokens_per_case=_mean([float(value) for value in token_values]),
        exact_token_cases=len(token_values),
        null_token_cases=sum(1 for case in covered if case.total_tokens is None),
    )
    return EvalMetrics(aggregate=aggregate, per_case=per_case)
