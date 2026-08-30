"""Tests for the RCA orchestrator: concurrency, aggregation, and failures."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from roottrace.agents import PlanError
from roottrace.agents.schema import MAX_QUESTIONS, PlanBudgets
from roottrace.evidence.schema import (
    AgentRole,
    FindingStatus,
    HypothesisDisposition,
)
from roottrace.history.schema import RetrievalHints, RetrievedCase
from roottrace.incident.loader import LoadedIncident
from roottrace.incident.schema import IncidentInput, Provenance
from roottrace.llm.schema import AssistantTurn, ToolCall
from roottrace.orchestrator import RcaOrchestrator, _bounded_error
from roottrace.runtime.workspace import Workspace
from roottrace.tools import RcaToolRegistry


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

OVER_LIMIT_PLAN_JSON = json.dumps(
    {
        "questions": [
            {
                "id": f"q-code-{index:03d}",
                "text": f"which code evidence answers question {index}?",
                "assigned_agents": ["code"],
            }
            for index in range(1, MAX_QUESTIONS + 2)
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
    worker_concurrency: int = 3,
    enabled_roles: frozenset[AgentRole] | None = None,
    retriever=None,
    retrieval_mode: str = "off",
    history_excluded_ids: frozenset[str] = frozenset(),
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
        worker_concurrency=worker_concurrency,
        enabled_roles=enabled_roles,
        retriever=retriever,
        retrieval_mode=retrieval_mode,
        history_excluded_ids=history_excluded_ids,
    )
    return orchestrator, providers


def test_specialists_run_concurrently(git_repo, tmp_path: Path) -> None:
    barrier = threading.Barrier(3)

    def completed_without_evidence(payload: str) -> AssistantTurn:
        response = json.loads(payload)
        response["evidence_ids"] = []
        return turn(json.dumps(response))

    hypotheses = json.loads(HYPOTHESES_JSON)
    hypotheses["hypotheses"][0]["supporting_evidence_ids"] = []

    def synchronize() -> None:
        barrier.wait(timeout=5)

    orchestrator, providers = build_orchestrator(
        git_repo,
        tmp_path,
        lead_responses=[turn(PLAN_JSON), turn(json.dumps(hypotheses))],
        issue_ci_responses=[completed_without_evidence(ISSUE_CI_FINAL)],
        code_responses=[completed_without_evidence(CODE_FINAL)],
        git_responses=[completed_without_evidence(GIT_FINAL)],
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
    code_diagnostics = [
        diagnostic
        for diagnostic in result.diagnostics
        if diagnostic.agent is AgentRole.CODE
    ]
    assert len(code_diagnostics) == 1
    assert code_diagnostics[0].code == "specialist.finding_error"
    assert "finding status" not in code_diagnostics[0].message
    findings = {finding.agent: finding for finding in result.graph.findings}
    assert findings[AgentRole.CODE].status == FindingStatus.FAILED
    assert findings[AgentRole.CODE].uncertainty.value == "high"
    assert findings[AgentRole.ISSUE_CI].status == FindingStatus.COMPLETED
    assert any(
        item.agent == AgentRole.ISSUE_CI for item in result.graph.evidence
    )


def test_specialist_partial_status_degrades_run_without_worker_exception(
    git_repo, tmp_path: Path
) -> None:
    partial = json.dumps(
        {
            "status": "partial",
            "ranked_locations": [],
            "evidence_ids": [],
            "uncertainty": "high",
            "uncertainty_note": "the code evidence is incomplete",
        }
    )
    orchestrator, _ = build_orchestrator(
        git_repo,
        tmp_path,
        code_responses=[turn(partial)],
    )

    result = orchestrator.run(make_loaded(git_repo), git_repo.repo)

    assert result.status == FindingStatus.PARTIAL
    assert "code finding status: partial" in result.errors
    assert result.diagnostics[0].code == "specialist.incomplete"
    assert result.diagnostics[0].severity.value == "warning"
    findings = {finding.agent: finding for finding in result.graph.findings}
    assert findings[AgentRole.CODE].status == FindingStatus.PARTIAL
    assert findings[AgentRole.ISSUE_CI].status == FindingStatus.COMPLETED


def test_git_tool_failure_reaches_run_status_and_has_no_fake_evidence(
    git_repo,
    tmp_path: Path,
) -> None:
    empty_git_finding = json.dumps(
        {
            "status": "completed",
            "ranked_locations": [],
            "evidence_ids": [],
            "uncertainty": "high",
        }
    )
    orchestrator, _ = build_orchestrator(
        git_repo,
        tmp_path,
        git_responses=[
            tool_turn("git_show", {"revision": git_repo.head_sha}),
            turn(empty_git_finding),
        ],
    )

    result = orchestrator.run(make_loaded(git_repo), git_repo.repo)

    assert result.status == FindingStatus.PARTIAL
    git_finding = next(
        finding
        for finding in result.graph.findings
        if finding.agent is AgentRole.GIT_HISTORY
    )
    assert git_finding.status == FindingStatus.PARTIAL
    assert "tool failures" in (git_finding.error or "")
    assert not any(
        item.agent is AgentRole.GIT_HISTORY for item in result.graph.evidence
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
        lead_responses=[turn("not json at all"), turn("still not json")],
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


def test_git_policy_is_persisted_in_specialist_context(
    git_repo,
    tmp_path: Path,
) -> None:
    orchestrator, providers = build_orchestrator(git_repo, tmp_path)

    orchestrator.run(make_loaded(git_repo), git_repo.repo)

    git_prompt = providers["git"].calls[0]["messages"][1]["content"]
    assert '"enabled": false' in git_prompt
    assert '"history_depth": 1' in git_prompt
    assert '"max_tool_calls": 1' in git_prompt


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
    planning_start = json.loads(trace_lines[0])
    assert planning_start["event_type"] == "planning_start"
    assert planning_start["git_verification_policy"]["history_depth"] == 1


def test_trace_is_reset_for_each_run(git_repo, tmp_path: Path) -> None:
    output_dir = tmp_path / "reused-out"
    output_dir.mkdir()
    trace_path = output_dir / "execution_trace.jsonl"
    trace_path.write_text("stale event\n", encoding="utf-8")

    build_orchestrator(git_repo, tmp_path)[0].run(
        make_loaded(git_repo),
        git_repo.repo,
        output_dir=output_dir,
    )
    first_trace = trace_path.read_text(encoding="utf-8").splitlines()

    build_orchestrator(git_repo, tmp_path)[0].run(
        make_loaded(git_repo),
        git_repo.repo,
        output_dir=output_dir,
    )
    second_trace = trace_path.read_text(encoding="utf-8").splitlines()

    assert "stale event" not in second_trace
    assert len(second_trace) == len(first_trace)
    assert sum("planning_start" in line for line in second_trace) == 1


def test_planning_fallback_records_degradation_trace(
    git_repo,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "degraded-out"
    orchestrator, _ = build_orchestrator(
        git_repo,
        tmp_path,
        lead_responses=[
            turn(OVER_LIMIT_PLAN_JSON),
            turn(OVER_LIMIT_PLAN_JSON),
            turn(HYPOTHESES_JSON),
        ],
    )

    result = orchestrator.run(
        make_loaded(git_repo),
        git_repo.repo,
        output_dir=output_dir,
    )

    assert len(result.plan.questions) == MAX_QUESTIONS
    trace_events = [
        json.loads(line)
        for line in (output_dir / "execution_trace.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    degraded = next(
        event
        for event in trace_events
        if event["event_type"] == "planning_degraded"
    )
    assert degraded["final_status"] == "DEGRADED"
    assert degraded["retry_count"] == 1
    assert degraded["degradation"] == {
        "reason": "question_limit_exceeded",
        "original_question_count": MAX_QUESTIONS + 1,
        "retained_question_count": MAX_QUESTIONS,
        "repair_attempts": 1,
    }


def test_bounded_error_respects_limit_including_ellipsis() -> None:
    assert _bounded_error("short") == "short"
    truncated = _bounded_error("x" * 1000, limit=500)
    assert len(truncated) == 500
    assert truncated.endswith("...")
    assert truncated.startswith("x" * 497)


def test_enabled_roles_subset_runs_only_selected_specialists(
    git_repo,
    tmp_path: Path,
) -> None:
    code_hypotheses = json.dumps(
        {
            "hypotheses": [
                {
                    "statement": "multiply is implemented as addition",
                    "locations": [{"path": "pkg/calc.py", "symbol": "multiply"}],
                    "supporting_evidence_ids": ["ev-code-001"],
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
    orchestrator, providers = build_orchestrator(
        git_repo,
        tmp_path,
        enabled_roles=frozenset({AgentRole.CODE}),
        lead_responses=[turn(PLAN_JSON), turn(code_hypotheses)],
    )
    result = orchestrator.run(make_loaded(git_repo), git_repo.repo)
    assert result.status == FindingStatus.COMPLETED
    assert result.enabled_roles == ["code"]
    assert providers["code"].calls
    assert not providers["issue_ci"].calls
    assert not providers["git"].calls


def test_retrieval_hints_are_injected_and_targets_excluded(
    git_repo,
    tmp_path: Path,
) -> None:
    from unittest.mock import MagicMock

    retrieved = RetrievedCase(
        id="historical-case-1",
        similarity=0.8,
        locations=["src/mod.py"],
        summary="similar issue",
        source="fixture",
    )
    hints = RetrievalHints(
        mode="clustered",
        top_k=1,
        results=[retrieved],
        candidate_count=3,
        index_checksum="c" * 64,
    )
    retriever = MagicMock()
    retriever.retrieve.return_value = hints
    orchestrator, providers = build_orchestrator(
        git_repo,
        tmp_path,
        retriever=retriever,
        retrieval_mode="clustered",
        history_excluded_ids=frozenset({"other-target"}),
    )
    result = orchestrator.run(make_loaded(git_repo), git_repo.repo)

    retriever.retrieve.assert_called_once_with(
        make_loaded(git_repo).incident.problem,
        target_id="inc-001",
        excluded_ids=frozenset({"other-target"}),
        mode="clustered",
    )
    assert result.retrieval is not None
    assert result.retrieval.mode == "clustered"
    last_messages = providers["lead"].calls[-1]["messages"]
    assert "RETRIEVED HISTORICAL HINTS" in str(last_messages)
    assert "historical-case-1" in str(last_messages)
