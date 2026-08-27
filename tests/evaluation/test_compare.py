"""Cross-variant comparison report tests."""

from __future__ import annotations

import json

from test_metrics import _result

from evaluation.compare import (
    build_comparison,
    load_variant_metrics,
    main,
    render_markdown,
    write_comparison,
)
from evaluation.metrics import compute_eval_metrics
from evaluation.report import EvalRunConfig, write_reports


def _variant_dir(tmp_path, name: str, top1: float, calls: int) -> None:
    results = [
        _result("c1", ["src/a.py"], ["src/a.py"], latency=1.0, calls=calls),
        _result("c2", ["src/b.py"], ["src/a.py"], latency=3.0, calls=calls),
    ]
    metrics = compute_eval_metrics(results)
    aggregate = metrics.aggregate
    assert aggregate.top_1_file_accuracy == top1
    config = EvalRunConfig(
        model="m",
        manifest_name="dev50",
        variant=name,
        config_hash="h" * 64,
    )
    write_reports(tmp_path / name, metrics, config)


def test_comparison_rows_cover_required_metrics(tmp_path) -> None:
    _variant_dir(tmp_path, "deterministic_baseline", top1=0.5, calls=0)
    _variant_dir(tmp_path, "lead_code", top1=0.5, calls=2)
    variants = [
        load_variant_metrics(tmp_path / "deterministic_baseline"),
        load_variant_metrics(tmp_path / "lead_code"),
    ]
    comparison = build_comparison(variants)
    assert comparison["variants"] == [
        {"variant": "deterministic_baseline", "config": variants[0]["config"]},
        {"variant": "lead_code", "config": variants[1]["config"]},
    ]
    metric_keys = {row["metric"] for row in comparison["rows"]}
    assert {
        "top_1_file_accuracy",
        "any_file_recall_at_3",
        "any_file_recall_at_5",
        "all_file_recall_at_3",
        "all_file_recall_at_5",
        "mean_gold_file_recall_at_5",
        "coverage",
        "invalid_output_rate",
        "latency_p50_seconds",
        "latency_p95_seconds",
        "mean_llm_calls_per_case",
        "mean_total_tokens_per_case",
        "mean_reasoning_tokens_per_case",
    } <= metric_keys


def test_comparison_output_is_deterministic(tmp_path) -> None:
    _variant_dir(tmp_path, "deterministic_baseline", top1=0.5, calls=0)
    _variant_dir(tmp_path, "lead_code", top1=0.5, calls=2)
    variants = [
        load_variant_metrics(tmp_path / "deterministic_baseline"),
        load_variant_metrics(tmp_path / "lead_code"),
    ]
    comparison = build_comparison(variants)
    first = write_comparison(tmp_path / "one", comparison)
    second = write_comparison(tmp_path / "two", comparison)
    assert first[0].read_bytes() == second[0].read_bytes()
    assert first[1].read_bytes() == second[1].read_bytes()


def test_markdown_never_calls_it_rca_accuracy(tmp_path) -> None:
    _variant_dir(tmp_path, "deterministic_baseline", top1=0.5, calls=0)
    variants = [load_variant_metrics(tmp_path / "deterministic_baseline")]
    text = render_markdown(build_comparison(variants))
    assert "File-localization metrics only; not RCA Accuracy." in text
    assert "| RCA Accuracy |" not in text
    assert "Top-1 File Accuracy" in text
    assert "Any File Recall@3" in text
    assert "Tokens/Case" in text


def test_compare_cli_writes_files_and_handles_missing_dir(tmp_path) -> None:
    _variant_dir(tmp_path, "deterministic_baseline", top1=0.5, calls=0)
    _variant_dir(tmp_path, "lead_code", top1=0.5, calls=2)
    output_dir = tmp_path / "out"
    exit_code = main(
        [
            "--variant-dirs",
            str(tmp_path / "deterministic_baseline"),
            str(tmp_path / "lead_code"),
            "--output-dir",
            str(output_dir),
        ]
    )
    assert exit_code == 0
    assert (output_dir / "comparison.json").is_file()
    assert (output_dir / "comparison.md").is_file()
    document = json.loads(
        (output_dir / "comparison.json").read_text(encoding="utf-8")
    )
    assert len(document["rows"]) == 13

    missing_exit = main(
        [
            "--variant-dirs",
            str(tmp_path / "does-not-exist"),
            "--output-dir",
            str(tmp_path / "out2"),
        ]
    )
    assert missing_exit == 2
