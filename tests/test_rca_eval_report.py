"""Deterministic aggregate report tests."""

from __future__ import annotations

import json

from test_rca_eval_metrics import _result

from evaluation.rca.metrics import compute_eval_metrics
from evaluation.rca.report import (
    EvalRunConfig,
    metrics_document,
    render_markdown,
    write_reports,
)


def _metrics():
    results = [
        _result("c1", ["src/a.py"], ["src/a.py"], latency=1.0, calls=2),
        _result("c2", ["src/b.py"], ["src/a.py"], latency=3.0, calls=4),
    ]
    return compute_eval_metrics(results)


def test_write_reports_is_deterministic(tmp_path) -> None:
    metrics = _metrics()
    config = EvalRunConfig(
        model="m",
        manifest_name="smoke",
        manifest_sha256="ab" * 32,
        seed=42,
    )
    first = write_reports(tmp_path / "one", metrics, config)
    second = write_reports(tmp_path / "two", metrics, config)
    assert first[0].read_bytes() == second[0].read_bytes()
    assert first[1].read_bytes() == second[1].read_bytes()


def test_metrics_document_shape_and_values() -> None:
    metrics = _metrics()
    config = EvalRunConfig(model="m", manifest_name="smoke")
    document = metrics_document(metrics, config)
    assert document["schema_version"] == "1.0"
    assert document["config"]["model"] == "m"
    assert document["aggregate"]["top_1_file_accuracy"] == 0.5
    assert document["aggregate"]["latency_p50_seconds"] == 2.0
    assert len(document["per_case"]) == 2
    assert "generated_at" not in document


def test_render_markdown_contains_summary_and_failures(tmp_path) -> None:
    results = [
        _result("c1", ["src/a.py"], ["src/a.py"]),
        _result("c2", [], ["src/a.py"], status="error", error="boom"),
    ]
    metrics = compute_eval_metrics(results)
    config = EvalRunConfig(manifest_name="smoke", manifest_sha256="x" * 64)
    text = render_markdown(metrics, config)
    assert "# RootTrace SWE-bench-derived RCA Evaluation" in text
    assert "| manifest | smoke |" in text
    assert "| c1 | completed |" in text
    assert "| c2 | error |" in text
    assert "## Failed cases" in text
    assert "boom" in text
    assert "generated_at" not in text


def test_reports_json_valid(tmp_path) -> None:
    metrics_path, _ = write_reports(
        tmp_path / "out",
        _metrics(),
        EvalRunConfig(manifest_name="smoke"),
    )
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert data["aggregate"]["covered_cases"] == 2
    assert data["per_case"][0]["instance_id"] == "c1"
