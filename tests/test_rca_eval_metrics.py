"""Localization metric computation tests."""

from __future__ import annotations

import pytest

from evaluation.rca.metrics import (
    CaseMetrics,
    CaseResult,
    compute_case_metrics,
    compute_eval_metrics,
)


def _result(
    instance_id: str,
    predicted: list[str],
    gold: list[str],
    *,
    status: str = "completed",
    error: str | None = None,
    latency: float | None = None,
    calls: int | None = None,
    prompt: int | None = None,
    completion: int | None = None,
    reasoning: int | None = None,
) -> CaseResult:
    metrics = compute_case_metrics(
        instance_id=instance_id,
        predicted_files=predicted,
        gold_files=gold,
        status=status,  # type: ignore[arg-type]
        error=error,
        latency_seconds=latency,
        llm_calls=calls,
        prompt_tokens=prompt,
        completion_tokens=completion,
        reasoning_tokens=reasoning,
    )
    usage = None
    if prompt is not None or completion is not None or reasoning is not None:
        usage = {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "reasoning_tokens": reasoning,
            "total_tokens": metrics.total_tokens,
        }
    return CaseResult(
        instance_id=instance_id,
        status=status,  # type: ignore[arg-type]
        error=error,
        predicted_files=predicted,
        gold_files=gold,
        metrics=metrics,
        latency_seconds=latency,
        llm_calls=calls,
        usage=usage,
    )


def test_compute_case_metrics_hit() -> None:
    metrics = compute_case_metrics(
        instance_id="c1",
        predicted_files=["src/a.py", "src/b.py"],
        gold_files=["src/a.py"],
        status="completed",
        latency_seconds=1.5,
        llm_calls=3,
        prompt_tokens=100,
        completion_tokens=50,
    )
    assert metrics.covered is True
    assert metrics.invalid_output is False
    assert metrics.top_1_file_accuracy is True
    assert metrics.any_file_recall_at_3 is True
    assert metrics.all_file_recall_at_5 is True
    assert metrics.gold_file_recall_at_5 == 1.0
    assert metrics.total_tokens == 150


def test_compute_case_metrics_miss_and_invalid() -> None:
    miss = compute_case_metrics(
        instance_id="c2",
        predicted_files=["src/other.py"],
        gold_files=["src/a.py"],
        status="completed",
    )
    assert miss.top_1_file_accuracy is False
    assert miss.gold_file_recall_at_5 == 0.0

    invalid = compute_case_metrics(
        instance_id="c3",
        predicted_files=[],
        gold_files=["src/a.py"],
        status="completed",
    )
    assert invalid.invalid_output is True
    assert invalid.top_1_file_accuracy is False


def test_compute_case_metrics_error_is_not_covered() -> None:
    metrics = compute_case_metrics(
        instance_id="c4",
        predicted_files=[],
        gold_files=["src/a.py"],
        status="error",
        error="boom",
    )
    assert metrics.covered is False
    assert metrics.invalid_output is False
    assert metrics.top_1_file_accuracy is None
    assert metrics.any_file_recall_at_5 is None
    assert metrics.gold_file_recall_at_5 is None


def test_predicted_files_normalized_deduped_and_bounded() -> None:
    metrics = compute_case_metrics(
        instance_id="c5",
        predicted_files=[
            "src/a.py",
            "src/a.py",
            "/etc/passwd",
            "../escape.py",
            "src/b.py",
        ],
        gold_files=["src/b.py"],
        status="completed",
    )
    assert metrics.predicted_files == ["src/a.py", "src/b.py"]


def test_aggregate_metrics_values() -> None:
    results = [
        _result(
            "c1",
            ["src/a.py", "src/b.py"],
            ["src/a.py"],
            latency=1.0,
            calls=3,
            prompt=100,
            completion=50,
            reasoning=20,
        ),
        _result("c2", ["src/c.py"], ["src/a.py", "src/b.py"], latency=2.0, calls=None),
        _result("c3", [], ["src/a.py"], latency=3.0, calls=5),
        _result(
            "c4",
            [],
            ["src/a.py"],
            status="error",
            latency=4.0,
            error="boom",
        ),
    ]
    metrics = compute_eval_metrics(results)
    aggregate = metrics.aggregate
    assert aggregate.total_cases == 4
    assert aggregate.covered_cases == 3
    assert aggregate.failed_cases == 1
    assert aggregate.invalid_outputs == 1
    assert aggregate.coverage == pytest.approx(0.75)
    assert aggregate.invalid_output_rate == pytest.approx(0.25)
    assert aggregate.top_1_file_accuracy == pytest.approx(1 / 3)
    assert aggregate.any_file_recall_at_3 == pytest.approx(1 / 3)
    assert aggregate.any_file_recall_at_5 == pytest.approx(1 / 3)
    assert aggregate.all_file_recall_at_3 == pytest.approx(1 / 3)
    assert aggregate.all_file_recall_at_5 == pytest.approx(1 / 3)
    assert aggregate.mean_gold_file_recall_at_5 == pytest.approx(1 / 3)
    assert aggregate.latency_p50_seconds == pytest.approx(2.0)
    assert aggregate.latency_p95_seconds == pytest.approx(2.9)
    assert aggregate.mean_llm_calls_per_case == pytest.approx(4.0)
    assert aggregate.exact_token_cases == 1
    assert aggregate.null_token_cases == 2
    assert aggregate.mean_total_tokens_per_case == pytest.approx(150.0)
    assert aggregate.reasoning_token_cases == 1
    assert aggregate.null_reasoning_cases == 2
    assert aggregate.mean_reasoning_tokens_per_case == pytest.approx(20.0)


def test_aggregate_per_case_sorted_by_instance_id() -> None:
    results = [
        _result("z", ["a.py"], ["a.py"]),
        _result("a", ["a.py"], ["a.py"]),
    ]
    metrics = compute_eval_metrics(results)
    assert [case.instance_id for case in metrics.per_case] == ["a", "z"]


def test_no_results_yields_null_rates() -> None:
    metrics = compute_eval_metrics([])
    assert metrics.aggregate.total_cases == 0
    assert metrics.aggregate.coverage is None
    assert metrics.aggregate.top_1_file_accuracy is None
    assert metrics.per_case == []


def test_case_metrics_type_check() -> None:
    metrics = CaseMetrics(
        instance_id="c",
        status="completed",
        predicted_files=["a.py"],
        gold_files=["a.py"],
    )
    assert metrics.status == "completed"
