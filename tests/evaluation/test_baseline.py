"""Deterministic baseline variant tests."""

from __future__ import annotations

import json

from test_runner import _args, _setup_data, run_from_args

from evaluation.baseline import DeterministicBaselineClient
from evaluation.metrics import CaseResult
from roottrace.incident.schema import IncidentInput, Provenance


def test_baseline_client_predicts_without_model_calls(git_repo, tmp_path) -> None:
    incident = IncidentInput(
        id="inc-baseline",
        repo="target",
        base_commit=git_repo.base_sha,
        title="multiply bug",
        problem="multiply returns a+b instead of a*b",
        provenance=Provenance(source="test"),
    )
    output_dir = tmp_path / "baseline"
    outcome = DeterministicBaselineClient().run(
        case_id="inc-baseline",
        repo=git_repo.repo,
        incident=incident,
        output_dir=output_dir,
        model=None,
    )
    assert outcome.status == "completed"
    assert outcome.llm_calls == 0
    assert outcome.prompt_tokens is None
    assert outcome.completion_tokens is None
    assert (output_dir / "rca_report.json").is_file()
    predicted = [
        location["path"] for location in outcome.report["top_k_locations"]
    ]
    assert "pkg/calc.py" in predicted


def test_runner_deterministic_baseline_records_variant_without_model(
    tmp_path,
) -> None:
    data_root, _ = _setup_data(tmp_path)
    output_dir = tmp_path / "out"
    assert (
        run_from_args(
            _args(
                data_root,
                output_dir,
                variant="deterministic_baseline",
            )
        )
        == 0
    )
    for case_id in ("acme__demo-1", "acme__demo-2"):
        result = CaseResult.model_validate(
            json.loads(
                (output_dir / "cases" / case_id / "result.json").read_text(
                    encoding="utf-8"
                )
            )
        )
        assert result.status == "completed"
        assert result.variant == "deterministic_baseline"
        assert result.config_hash
        assert result.llm_calls == 0
        assert result.usage is None
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["config"]["variant"] == "deterministic_baseline"
    assert metrics["config"]["root_trace_mode"] == "deterministic_baseline"
    assert (output_dir / "variant.json").is_file()
