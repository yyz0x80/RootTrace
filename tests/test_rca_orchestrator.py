"""Tests for the RCA orchestrator: concurrency, aggregation, and failures."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from patchpilot.models import AssistantTurn, ToolCall
from patchpilot.rca.agents import PlanError
from patchpilot.rca.incident_loader import LoadedIncident
from patchpilot.rca.orchestrator import RcaOrchestrator
from patchpilot.rca.schema import (
    AgentRole,
    FindingStatus,
    HypothesisDisposition,
    IncidentInput,
    PlanBudgets,
    Provenance,
)
from patchpilot.rca.tools import RcaToolRegistry
from patchpilot.workspace import Workspace


class FakeProvider:
    """Deterministic provider; optional hook runs before each completion."""

    model = "fake-model"

    def __init__(self, *responses: AssistantTurn, on_complete=None) -> None:
        self._responses = list(responses)
        self._on_complete = on_complete
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str | None = None,
    ) -> AssistantTurn:
        self.calls.append(
            {"messages": messages, "tools": tools, "tool_choice": tool_choice}
        )
        if self._on_complete is not None:
            self._on_complete()
        if not self._responses:
            raise AssertionError("unexpected provider call")
        return self._responses.pop(0)


def turn(content: str, prompt_tokens: int = 10, completion_tokens: int = 5) -> AssistantTurn:
    return AssistantTurn(
        content=content,
        tool_calls=[],
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def tool_turn(name: str, arguments: dict[str, Any]) -> AssistantTurn:
    return AssistantTurn(
        content=None,
        tool_calls=[
            ToolCall(id=f"call-{name}", name=name, arguments=arguments)
        ],
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
        "evidence_ids": ["ev-issue_ci-001", "ev-issue_ci-002"],
        "uncertainty": "medium",
        "uncertainty_note": "failure signature is clear",
    }
)

CODE_FINAL = json.dumps(
    {
        "status": "completed",
        "ranked_locations": [{"path": "pkg/calc.py", "symbol": "multiply"}],
        "evidence_ids": ["ev-code-001"],
        "uncertainty": "medium",
        "uncertainty_note": None,
    }
)

GIT_FINAL = json.dumps(
    {
        "status": "completed",
        "ranked_locations": [{"path": "pkg/calc.py"}],
        "evidence_ids": ["ev-git_history-001"],
        "uncertainty": "high",
        "uncertainty_note": None,
    }
)

HYPOTHESES_JSON = json.dumps(
    {
        "hypotheses": [
            {
                "statement": "multiply is implemented as addition",
                "locations": [{"path": "pkg/calc.py", "symbol": "multiply"}],
                "supporting_evidence_ids": ["ev-issue_ci-001"],
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


def make_loaded(git_repo) -> LoadedIncident:
    incident = IncidentInput(
        id="inc-001",
        repo="target",
        base_commit=git_repo.base_sha,
        title="multiply returns a+b",
        problem="multiply returns a+b instead of a*b",
        logs=["Traceback (most recent call last):\nValueError: boom"],
        provenance=Provenance(source="test_fixture"),
    )
    return LoadedIncident(incident=incident)


def build_orchestrator(
    git_repo,
    tmp_path: Path,
    *,
    lead_responses: list[AssistantTurn] | None = None,
    issue_ci_responses: list[AssistantTurn] | None = None,
    code_responses: list[AssistantTurn] | None = None,
    git_responses: list[AssistantTurn] | None = None,
    budgets: PlanBudgets | None = None,
    worker_timeout_margin_seconds: float = 30.0,
):
    external_root = tmp_path / "logs"
    external_root.mkdir(exist_ok=True)
    (external_root / "ci.log").write_text("CI FAILURE\n", encoding="utf-8")
    registry = RcaToolRegistry(
        Workspace(git_repo.repo),
        external_root=external_root,
    )
    providers = {
        "lead": FakeProvider(
            *(lead_responses or [turn(PLAN_JSON), turn(HYPOTHESES_JSON)])
        ),
        "issue_ci": FakeProvider(
            *(issue_ci_responses or [turn(ISSUE_CI_FINAL)])
        ),
        "code": FakeProvider(
            *(
                code_responses
                or [
                    tool_turn("read_file", {"path": "pkg/calc.py", "raw": True}),
                    turn(CODE_FINAL),
                ]
            )
        ),
        "git": FakeProvider(
            *(
                git_responses
                or [
                    tool_turn("git_history", {"max_count": 10}),
                    turn(GIT_FINAL),
                ]
            )
        ),
    }
    orchestrator = RcaOrchestrator(
        lead_provider=providers["lead"],
        issue_ci_provider=providers["issue_ci"],
        code_provider=providers["code"],
        git_history_provider=providers["git"],
        registry=registry,
        budgets=budgets or PlanBudgets(),
        worker_timeout_margin_seconds=worker_timeout_margin_seconds,
    )
    return orchestrator, providers


def test_specialists_run_concurrently(git_repo, tmp_path: Path) -> None:
    barrier = threading.Barrier(3)

    def synchronize() -> None:
        barrier.wait(timeout=5)

    orchestrator, providers = build_orchestrator(
        git_repo,
        tmp_path,
        issue_ci_responses=[turn(ISSUE_CI_FINAL)],
        code_responses=[turn(CODE_FINAL)],
        git_responses=[turn(GIT_FINAL)],
    )
    for key in ("issue_ci", "code", "git"):
        providers[key]._on_complete = synchronize

    result = orchestrator.run(make_loaded(git_repo), git_repo.repo)
    assert result.status == FindingStatus.COMPLETED
    assert len(result.graph.findings) == 3
    assert [finding.agent for finding in result.graph.findings] == [
        AgentRole.ISSUE_CI,
        AgentRole.CODE,
        AgentRole.GIT_HISTORY,
    ]
    assert result.usage is not None
    assert result.usage.llm_calls == 5
    assert result.usage.prompt_tokens == 50
    assert result.usage.completion_tokens == 25


def test_aggregation_is_deterministic(git_repo, tmp_path: Path) -> None:
    first = build_orchestrator(git_repo, tmp_path)[0].run(
        make_loaded(git_repo), git_repo.repo
    )
    second = build_orchestrator(git_repo, tmp_path)[0].run(
        make_loaded(git_repo), git_repo.repo
    )
    assert _structural_graph(first.graph) == _structural_graph(second.graph)


def _structural_graph(graph) -> dict[str, Any]:
    """Graph payload minus wall-clock timing, which is not deterministic."""
    payload = graph.model_dump(mode="json")
    for finding in payload["findings"]:
        finding.pop("timing_seconds", None)
    return payload


def test_worker_failure_is_partial_and_preserves_others(
    git_repo, tmp_path: Path
) -> None:
    def explode() -> None:
        raise RuntimeError("code worker exploded")

    orchestrator, _ = build_orchestrator(
        git_repo,
        tmp_path,
        code_responses=None,
    )
    orchestrator._code_provider = FakeProvider(
        turn(CODE_FINAL),
        on_complete=explode,
    )
    result = orchestrator.run(make_loaded(git_repo), git_repo.repo)
    assert result.status == FindingStatus.PARTIAL
    assert any("code worker failed" in error for error in result.errors)
    findings = {finding.agent: finding for finding in result.graph.findings}
    assert findings[AgentRole.CODE].status == FindingStatus.FAILED
    assert findings[AgentRole.CODE].uncertainty.value == "high"
    assert findings[AgentRole.ISSUE_CI].status == FindingStatus.COMPLETED
    assert any(
        item.agent == AgentRole.ISSUE_CI for item in result.graph.evidence
    )


def test_worker_timeout_is_partial(git_repo, tmp_path: Path) -> None:
    orchestrator, _ = build_orchestrator(
        git_repo,
        tmp_path,
        budgets=PlanBudgets(timeout_seconds=1, max_llm_calls=5),
        worker_timeout_margin_seconds=2,
    )
    orchestrator._code_provider = FakeProvider(
        turn(CODE_FINAL),
        on_complete=lambda: time.sleep(10),
    )
    started = time.monotonic()
    result = orchestrator.run(make_loaded(git_repo), git_repo.repo)
    elapsed = time.monotonic() - started
    assert elapsed < 8
    assert result.status == FindingStatus.PARTIAL
    assert any("code worker timed out" in error for error in result.errors)
    findings = {finding.agent: finding for finding in result.graph.findings}
    assert findings[AgentRole.CODE].status == FindingStatus.PARTIAL
    assert findings[AgentRole.GIT_HISTORY].status == FindingStatus.COMPLETED


def test_planning_failure_stops_workers(git_repo, tmp_path: Path) -> None:
    orchestrator, providers = build_orchestrator(
        git_repo,
        tmp_path,
        lead_responses=[turn("not json at all")],
    )
    with pytest.raises(PlanError):
        orchestrator.run(make_loaded(git_repo), git_repo.repo)
    for key in ("issue_ci", "code", "git"):
        assert providers[key].calls == []


def test_hypotheses_are_ranked_falsifiable_and_validated(
    git_repo, tmp_path: Path
) -> None:
    rich_hypotheses = json.dumps(
        {
            "hypotheses": [
                {
                    "statement": "multiply is implemented as addition",
                    "locations": [{"path": "pkg/calc.py", "symbol": "multiply"}],
                    "supporting_evidence_ids": [
                        "ev-issue_ci-001",
                        "ev-code-001",
                    ],
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
    orchestrator, _ = build_orchestrator(
        git_repo,
        tmp_path,
        lead_responses=[turn(PLAN_JSON), turn(rich_hypotheses)],
    )
    result = orchestrator.run(
        make_loaded(git_repo), git_repo.repo
    )
    assert result.status == FindingStatus.COMPLETED
    assert len(result.graph.hypotheses) == 1
    hypothesis = result.graph.hypotheses[0]
    assert hypothesis.id == "h-001"
    assert hypothesis.disposition == HypothesisDisposition.CANDIDATE
    assert hypothesis.locations[0].path == "pkg/calc.py"
    assert hypothesis.verification_plan[0].command.startswith("python -m pytest")
    known_ids = {item.id for item in result.graph.evidence}
    assert set(hypothesis.supporting_evidence_ids) <= known_ids


def test_invalid_hypothesis_references_are_explicit(git_repo, tmp_path: Path) -> None:
    bad_hypotheses = json.dumps(
        {
            "hypotheses": [
                {
                    "statement": "unsupported claim",
                    "locations": [{"path": "pkg/calc.py"}],
                    "supporting_evidence_ids": ["ev-unknown-999"],
                    "verification_plan": [
                        {"command": "python -m pytest -q tests/test_calc.py"}
                    ],
                    "confidence": "low",
                }
            ]
        }
    )
    orchestrator, _ = build_orchestrator(
        git_repo,
        tmp_path,
        lead_responses=[turn(PLAN_JSON), turn(bad_hypotheses)],
    )
    result = orchestrator.run(make_loaded(git_repo), git_repo.repo)
    assert result.status == FindingStatus.PARTIAL
    assert any("unknown evidence ids" in error for error in result.errors)
    assert result.graph.hypotheses == []
    assert result.graph.evidence  # diagnostics are preserved


def test_malformed_hypothesis_output_is_explicit(git_repo, tmp_path: Path) -> None:
    orchestrator, _ = build_orchestrator(
        git_repo,
        tmp_path,
        lead_responses=[turn(PLAN_JSON), turn("I think the cause is X")],
    )
    result = orchestrator.run(make_loaded(git_repo), git_repo.repo)
    assert result.status == FindingStatus.PARTIAL
    assert any("malformed hypothesis" in error for error in result.errors)
    assert result.graph.evidence


def test_specialists_do_not_share_evidence(git_repo, tmp_path: Path) -> None:
    orchestrator, providers = build_orchestrator(git_repo, tmp_path)
    orchestrator.run(make_loaded(git_repo), git_repo.repo)
    for key in ("issue_ci", "code", "git"):
        assert providers[key].calls
        tool_names = {
            tool["function"]["name"]
            for tool in providers[key].calls[0]["tools"]
        }
        assert "run_command" not in tool_names
        assert "edit_file" not in tool_names
    code_prompt = providers["code"].calls[0]["messages"][1]["content"]
    git_prompt = providers["git"].calls[0]["messages"][1]["content"]
    assert "ev-issue_ci-001" not in code_prompt
    assert "ev-issue_ci-001" not in git_prompt


def test_isolation_violation_is_recorded(git_repo, tmp_path: Path) -> None:
    git_attempt = json.dumps(
        {
            "status": "completed",
            "ranked_locations": [],
            "evidence_ids": [],
            "uncertainty": "low",
            "uncertainty_note": None,
        }
    )
    orchestrator, _ = build_orchestrator(
        git_repo,
        tmp_path,
        code_responses=[
            tool_turn("git_history", {"max_count": 5}),
            turn(git_attempt),
        ],
    )
    result = orchestrator.run(make_loaded(git_repo), git_repo.repo)
    assert result.status == FindingStatus.COMPLETED
    assert result.isolation_violations == 1


def test_artifacts_are_persisted(git_repo, tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    result = build_orchestrator(git_repo, tmp_path)[0].run(
        make_loaded(git_repo),
        git_repo.repo,
        output_dir=output_dir,
    )
    assert result.status == FindingStatus.COMPLETED
    for name in (
        "investigation_plan.json",
        "evidence_graph.json",
        "hypotheses.json",
        "execution_trace.jsonl",
    ):
        assert (output_dir / name).is_file()
    graph_payload = json.loads(
        (output_dir / "evidence_graph.json").read_text(encoding="utf-8")
    )
    assert graph_payload["hypotheses"][0]["id"] == "h-001"
    hypotheses_payload = json.loads(
        (output_dir / "hypotheses.json").read_text(encoding="utf-8")
    )
    assert len(hypotheses_payload["hypotheses"]) == 1
    trace_lines = (
        output_dir / "execution_trace.jsonl"
    ).read_text(encoding="utf-8").strip().splitlines()
    assert any("planning_end" in line for line in trace_lines)
    assert any("code_start" in line for line in trace_lines)
    assert any("hypotheses_end" in line for line in trace_lines)
