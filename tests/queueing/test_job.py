"""Tests for the reusable RCA job entry point."""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable
from pathlib import Path

from roottrace.incident.loader import load_incident
from roottrace.llm.schema import AssistantTurn
from roottrace.queueing import run_rca_job
from roottrace.runtime.workspace import capture_repository_fingerprint


class FakeProvider:
    """Scripted provider that returns prebuilt assistant turns."""

    model = "fake-model"

    def __init__(self, *responses: AssistantTurn) -> None:
        self._responses = list(responses)

    def complete(self, messages, tools, tool_choice=None) -> AssistantTurn:
        if not self._responses:
            raise AssertionError("unexpected provider call")
        return self._responses.pop(0)


def _turn(content: str) -> AssistantTurn:
    return AssistantTurn(
        content=content,
        tool_calls=[],
        prompt_tokens=10,
        completion_tokens=5,
    )


PLAN_JSON = json.dumps(
    {
        "questions": [
            {
                "id": "q-issue_ci-001",
                "text": "what is the failure signature?",
                "assigned_agents": ["issue_ci"],
            },
            {
                "id": "q-code-001",
                "text": "which code path implements multiply?",
                "assigned_agents": ["code"],
            },
            {
                "id": "q-git-001",
                "text": "which change is suspected?",
                "assigned_agents": ["git_history"],
            },
        ]
    }
)

ISSUE_CI_FINAL = json.dumps(
    {
        "status": "completed",
        "ranked_locations": [],
        "evidence_ids": [],
        "uncertainty": "medium",
    }
)

CODE_FINAL = json.dumps(
    {
        "status": "completed",
        "ranked_locations": [{"path": "pkg/calc.py", "symbol": "multiply"}],
        "evidence_ids": [],
        "uncertainty": "medium",
    }
)

GIT_FINAL = json.dumps(
    {
        "status": "completed",
        "ranked_locations": [{"path": "pkg/calc.py"}],
        "evidence_ids": [],
        "uncertainty": "high",
    }
)

HYPOTHESES_JSON = json.dumps(
    {
        "hypotheses": [
            {
                "statement": "multiply is implemented as addition",
                "locations": [{"path": "pkg/calc.py", "symbol": "multiply"}],
                "supporting_evidence_ids": [],
                "contradicting_evidence_ids": [],
                "verification_plan": [
                    {
                        "command": "python -m pytest -q tests/test_calc.py",
                        "description": "run the calc regression tests",
                        "timeout_seconds": 60,
                    }
                ],
                "confidence": "medium",
            }
        ]
    }
)

REPORT_JSON = json.dumps(
    {
        "conclusion": "insufficient_evidence",
        "conclusion_summary": "no hypothesis was verified in the sandbox",
        "ranked_causes": [],
        "top_k_locations": [{"path": "pkg/calc.py", "symbol": "multiply"}],
        "causal_chain": [],
        "suspected_regression": None,
        "fix_recommendation": None,
        "uncertainty": {
            "level": "high",
            "insufficient_evidence": True,
            "notes": [],
        },
    }
)


def _scripted_factory() -> Callable[[], FakeProvider]:
    providers = [
        FakeProvider(_turn(PLAN_JSON), _turn(HYPOTHESES_JSON)),
        FakeProvider(_turn(ISSUE_CI_FINAL)),
        FakeProvider(_turn(CODE_FINAL)),
        FakeProvider(_turn(GIT_FINAL)),
        FakeProvider(_turn(REPORT_JSON)),
    ]

    def factory() -> FakeProvider:
        return providers.pop(0)

    return factory


def _issue_files(tmp_path: Path) -> tuple[Path, Path]:
    issue = tmp_path / "issue.md"
    issue.write_text(
        "# multiply returns the sum\n\n"
        "When multiplying 3 and 4, the result is 7 instead of 12.\n",
        encoding="utf-8",
    )
    ci_log = tmp_path / "ci.log"
    ci_log.write_text("CI FAILURE: test_multiply failed\n", encoding="utf-8")
    return issue, ci_log


def test_run_rca_job_runs_existing_pipeline(
    git_repo,
    tmp_path: Path,
    monkeypatch,
) -> None:
    before = capture_repository_fingerprint(git_repo.repo)
    issue, ci_log = _issue_files(tmp_path)
    loaded = load_incident(issue, git_repo.repo, ci_log_path=ci_log)
    output_dir = tmp_path / "out"
    factory = _scripted_factory()
    requested_models: list[str | None] = []

    def fake_create_provider(model_name: str | None = None) -> FakeProvider:
        requested_models.append(model_name)
        return factory()

    monkeypatch.setattr(
        "roottrace.queueing.job.create_provider_from_config",
        fake_create_provider,
    )

    result = run_rca_job(
        repo=str(git_repo.repo),
        issue=str(issue),
        output_dir=str(output_dir),
        model="fake-model",
        ci_log=str(ci_log),
    )

    assert requested_models == ["fake-model"] * 5
    assert result["incident_id"] == loaded.incident.id
    assert result["status"] == "completed"
    assert result["conclusion"] == "insufficient_evidence"
    assert Path(result["output_dir"]) == output_dir.resolve()
    for name in (
        "evidence_graph.json",
        "hypotheses.json",
        "verification.json",
        "rca_report.json",
        "rca_report.md",
        "run_summary.json",
    ):
        assert name in result["artifacts"]
        assert (output_dir / name).is_file()

    verification = json.loads(
        (output_dir / "verification.json").read_text(encoding="utf-8")
    )
    assert verification["results"][0]["status"] == "passed"

    # RQ serializes return values, so the job summary must stay JSON-safe.
    assert json.loads(json.dumps(result)) == result

    after = capture_repository_fingerprint(git_repo.repo)
    assert before.model_dump(mode="json") == after.model_dump(mode="json")


def test_run_rca_job_rejects_missing_issue_file(git_repo, tmp_path: Path) -> None:
    try:
        run_rca_job(
            repo=str(git_repo.repo),
            issue=str(tmp_path / "missing.md"),
            output_dir=str(tmp_path / "out"),
        )
    except FileNotFoundError as exc:
        assert "issue file not found" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected a FileNotFoundError")


def test_run_rca_job_reference_is_worker_importable() -> None:
    """RQ imports the queued function by dotted path inside worker processes."""
    reference = f"{run_rca_job.__module__}.{run_rca_job.__qualname__}"

    assert reference == "roottrace.queueing.job.run_rca_job"
    module = importlib.import_module(run_rca_job.__module__)
    assert getattr(module, run_rca_job.__qualname__) is run_rca_job
