"""Tests for machine-readable usage and evidence artifact metadata."""

from __future__ import annotations

import json
from pathlib import Path

from patchpilot.cli import _save_prepare_summary
from patchpilot.evidence.schema import CompletionState
from patchpilot.workflow.result import WorkflowResult


class FakeProvider:
    """Provide deterministic exact usage without an external API."""

    model = "test-model"
    llm_call_count = 2
    prompt_tokens = 120
    completion_tokens = 30


def test_prepare_summary_saves_exact_provider_usage(tmp_path: Path) -> None:
    _save_prepare_summary(tmp_path, FakeProvider())

    summary = json.loads(
        (tmp_path / "prepare_summary.json").read_text(encoding="utf-8")
    )

    assert summary == {
        "phase": "prepare",
        "model": "test-model",
        "llm_call_count": 2,
        "prompt_tokens": 120,
        "completion_tokens": 30,
    }


def test_run_summary_includes_usage_and_evidence_artifact() -> None:
    result = WorkflowResult(
        run_id="run-1",
        final_status=CompletionState.VERIFIED,
        llm_call_count=3,
        prompt_tokens=200,
        completion_tokens=50,
    )

    summary = result.to_run_summary(
        task_id="task-1",
        base_commit="abc123",
        model="test-model",
        output_dir="artifacts",
    )

    assert summary.llm_call_count == 3
    assert summary.prompt_tokens == 200
    assert summary.completion_tokens == 50
    assert summary.artifacts["acceptance_evidence"] == (
        "artifacts/acceptance_evidence.json"
    )
