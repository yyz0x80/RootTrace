"""Tests for machine-readable usage and evidence artifact metadata."""

from __future__ import annotations

import json
from pathlib import Path

from patchpilot.cli import _save_prepare_summary
from patchpilot.evidence.schema import (
    AcceptanceEvidence,
    CompletionState,
    EvidenceStatus,
)
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
        "outcome_code": "READY_FOR_APPROVAL",
        "final_status": None,
        "exit_code": 0,
        "reasons": [],
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


def test_run_summary_derives_evidence_and_regression_counts() -> None:
    """Populate summary metrics when the completion decision is not retained."""
    result = WorkflowResult(
        run_id="run-1",
        final_status=CompletionState.PARTIALLY_VERIFIED,
        acceptance_evidence=[
            AcceptanceEvidence(
                criterion_id="AC-1",
                description="Verified behavior",
                status=EvidenceStatus.PASS,
                explanation="Probe passed.",
            ),
            AcceptanceEvidence(
                criterion_id="AC-2",
                description="Missing evidence",
                status=EvidenceStatus.UNVERIFIED,
                explanation="No direct check.",
            ),
        ],
        verification_report={
            "transition_summary": {
                "overall": {
                    "pre_existing_failure": 2,
                    "regression": 1,
                }
            }
        },
    )

    summary = result.to_run_summary()

    assert summary.outcome_code == "PARTIALLY_VERIFIED"
    assert summary.criterion_pass_count == 1
    assert summary.criterion_unverified_count == 1
    assert summary.pre_existing_failure_count == 2
    assert summary.new_regression_count == 1
