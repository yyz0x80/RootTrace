"""Tests for the Lead planner and three evidence Specialists."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from roottrace.agents import (
    CodeSpecialist,
    GitHistorySpecialist,
    IssueCISpecialist,
    LeadPlanner,
    PlanError,
)
from roottrace.agents.prompts import (
    MAX_PROMPT_CHARS,
    build_code_prompt,
    build_git_history_prompt,
    build_issue_ci_prompt,
    build_lead_prompt,
)
from roottrace.agents.schema import MAX_QUESTIONS, PlanBudgets, PlanQuestion
from roottrace.evidence.schema import (
    AgentRole,
    EvidenceKind,
    FindingStatus,
    UncertaintyLevel,
)
from roottrace.incident.context import (
    ContextTruncation,
    IncidentContext,
    IncidentSignals,
    RepositoryInventory,
)
from roottrace.incident.schema import IncidentInput, Provenance
from roottrace.llm.schema import AssistantTurn, ToolCall
from roottrace.llm.usage import UsageTracker
from roottrace.runtime.workspace import RepositoryFingerprint, Workspace
from roottrace.tools import GitSearchCandidate, GitSearchSummary, RcaToolRegistry


class FakeProvider:
    """Deterministic provider used by all RCA agent tests."""

    model = "fake-model"

    def __init__(self, *responses: AssistantTurn) -> None:
        self._responses = list(responses)
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
        if not self._responses:
            raise AssertionError("unexpected provider call")
        return self._responses.pop(0)


def turn(
    content: str | None = None,
    tool_calls: list[ToolCall] | None = None,
    prompt_tokens: int | None = 10,
    completion_tokens: int | None = 5,
) -> AssistantTurn:
    return AssistantTurn(
        content=content,
        tool_calls=tool_calls or [],
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def call(call_id: str, name: str, arguments: dict[str, Any]) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=arguments)


def final_response(
    status: str = "completed",
    locations: list[dict[str, Any]] | None = None,
    evidence_ids: list[str] | None = None,
    uncertainty: str = "low",
    note: str | None = None,
) -> str:
    return json.dumps(
        {
            "status": status,
            "ranked_locations": locations or [],
            "evidence_ids": evidence_ids or [],
            "uncertainty": uncertainty,
            "uncertainty_note": note,
        }
    )


def plan_response(question_count: int) -> str:
    """Return a deterministic Lead planning response of the requested size."""
    return json.dumps(
        {
            "questions": [
                {
                    "id": f"q-{index:03d}",
                    "text": f"investigate evidence question {index}",
                    "assigned_agents": ["code"],
                }
                for index in range(1, question_count + 1)
            ]
        }
    )


def make_context(git_repo) -> IncidentContext:
    base_sha = git_repo.base_sha
    incident = IncidentInput(
        id="inc-001",
        repo="target",
        base_commit=base_sha,
        title="multiply returns a+b",
        problem="multiply returns a+b instead of a*b",
        logs=[
            (
                "Traceback (most recent call last):\n"
                "  File 'pkg/calc.py', line 8, in multiply\n"
                "ValueError: boom"
            )
        ],
        provenance=Provenance(source="test_fixture"),
    )
    inventory = RepositoryInventory(
        base_commit=base_sha,
        tracked_files=4,
        python_files=3,
        test_files=1,
        config_files=0,
        python_file_list=["pkg/__init__.py", "pkg/calc.py", "tests/test_calc.py"],
        test_file_list=["tests/test_calc.py"],
    )
    signals = IncidentSignals(
        terms=["multiply"],
        exception_names=["ValueError"],
        stack_symbols=["multiply"],
    )
    fingerprint = RepositoryFingerprint(
        head_sha=base_sha,
        status_porcelain="",
        diff_stat="",
    )
    return IncidentContext(
        incident=incident,
        repository=inventory,
        signals=signals,
        snippets=[],
        truncation=ContextTruncation(),
        fingerprint=fingerprint,
    )


@pytest.fixture
def rca_env(git_repo, tmp_path: Path):
    external_root = tmp_path / "logs"
    external_root.mkdir()
    (external_root / "ci.log").write_text(
        "CI FAILURE: test_multiply failed\n",
        encoding="utf-8",
    )
    registry = RcaToolRegistry(
        Workspace(git_repo.repo),
        external_root=external_root,
    )
    budgets = PlanBudgets(max_llm_calls=5, max_tool_calls=10, timeout_seconds=60)
    return {
        "registry": registry,
        "budgets": budgets,
        "context": make_context(git_repo),
    }


def _questions() -> list[PlanQuestion]:
    return [
        PlanQuestion(
            id="q-code-001",
            text="which code path implements multiply?",
            assigned_agents=[AgentRole.CODE],
        )
    ]


def test_role_tool_surfaces_are_least_privilege(rca_env) -> None:
    issue_ci = IssueCISpecialist(
        provider=FakeProvider(),
        registry=rca_env["registry"],
        usage=UsageTracker(),
        budgets=rca_env["budgets"],
    )
    code = CodeSpecialist(
        provider=FakeProvider(),
        registry=rca_env["registry"],
        usage=UsageTracker(),
        budgets=rca_env["budgets"],
    )
    git = GitHistorySpecialist(
        provider=FakeProvider(),
        registry=rca_env["registry"],
        usage=UsageTracker(),
        budgets=rca_env["budgets"],
    )
    assert issue_ci.allowed_tools == {"read_external_log"}
    assert code.allowed_tools == {"search_code", "read_file", "inspect_symbols"}
    assert git.allowed_tools == {"git_history", "git_blame", "git_show"}


def test_specialist_only_sees_own_tool_schemas(rca_env) -> None:
    provider = FakeProvider(turn(content=final_response()))
    CodeSpecialist(
        provider=provider,
        registry=rca_env["registry"],
        usage=UsageTracker(),
        budgets=rca_env["budgets"],
    ).run(rca_env["context"], _questions())
    schema_names = {
        tool["function"]["name"] for tool in provider.calls[0]["tools"]
    }
    assert schema_names == {"search_code", "read_file", "inspect_symbols"}


def test_specialist_tool_boundary_raises(rca_env) -> None:
    agent = IssueCISpecialist(
        provider=FakeProvider(),
        registry=rca_env["registry"],
        usage=UsageTracker(),
        budgets=rca_env["budgets"],
    )
    with pytest.raises(PermissionError):
        agent.execute_tool_call(call("c1", "search_code", {"query": "x"}))
    with pytest.raises(PermissionError):
        agent.execute_tool_call(call("c2", "git_history", {}))


def test_disallowed_tool_attempt_is_recorded_and_refused(rca_env) -> None:
    provider = FakeProvider(
        turn(
            tool_calls=[
                call("c1", "git_history", {"max_count": 5}),
            ]
        ),
        turn(content=final_response()),
    )
    output = CodeSpecialist(
        provider=provider,
        registry=rca_env["registry"],
        usage=UsageTracker(),
        budgets=rca_env["budgets"],
    ).run(rca_env["context"], _questions())
    assert output.isolation_violations == 1
    assert output.finding.status == FindingStatus.COMPLETED
    tool_message = provider.calls[1]["messages"][-1]
    assert "not allowed for role" in tool_message["content"]


def test_lead_planner_builds_validated_plan(rca_env) -> None:
    provider = FakeProvider(
        turn(
            content=json.dumps(
                {
                    "questions": [
                        {
                            "id": "q-code-001",
                            "text": "which code path implements multiply?",
                            "assigned_agents": ["code"],
                        },
                        {
                            "id": "q-issue-001",
                            "text": "what is the failure signature?",
                            "assigned_agents": ["issue_ci"],
                        },
                    ]
                }
            )
        )
    )
    planner = LeadPlanner(
        provider=provider,
        usage=UsageTracker(),
        budgets=rca_env["budgets"],
    )
    plan = planner.run(rca_env["context"])
    assert plan.id == "plan-inc-001"
    assert plan.incident_id == "inc-001"
    assert len(plan.questions) == 2
    assert set(plan.assignments) == {AgentRole.CODE, AgentRole.ISSUE_CI}
    assert plan.budgets == rca_env["budgets"]
    system_content = provider.calls[0]["messages"][0]["content"]
    assert "plan the investigation" in system_content
    assert "root cause" in system_content


def test_lead_planner_rejects_malformed_output(rca_env) -> None:
    for content in ("not json", '{"questions": []}', '{"questions": "x"}'):
        provider = FakeProvider(turn(content=content), turn(content=content))
        planner = LeadPlanner(
            provider=provider,
            usage=UsageTracker(),
            budgets=rca_env["budgets"],
        )
        with pytest.raises(PlanError):
            planner.run(rca_env["context"])


def test_lead_planner_repairs_over_limit_plan(rca_env) -> None:
    provider = FakeProvider(
        turn(content=plan_response(MAX_QUESTIONS + 3)),
        turn(content=plan_response(2)),
    )
    usage = UsageTracker()
    planner = LeadPlanner(
        provider=provider,
        usage=usage,
        budgets=rca_env["budgets"],
    )

    plan = planner.run(rca_env["context"])

    assert len(plan.questions) == 2
    assert usage.snapshot().llm_calls == 2
    assert len(provider.calls) == 2
    repair_prompt = provider.calls[1]["messages"][-1]["content"]
    assert "PREVIOUS OUTPUT REJECTED" in repair_prompt
    assert "at most 10 items" in repair_prompt


def test_lead_planner_truncates_persistently_over_limit_plan(rca_env) -> None:
    provider = FakeProvider(
        turn(content=plan_response(MAX_QUESTIONS + 3)),
        turn(content=plan_response(MAX_QUESTIONS + 1)),
    )
    planner = LeadPlanner(
        provider=provider,
        usage=UsageTracker(),
        budgets=rca_env["budgets"],
    )

    plan = planner.run(rca_env["context"])

    assert len(plan.questions) == MAX_QUESTIONS
    assert [question.id for question in plan.questions] == [
        f"q-{index:03d}" for index in range(1, MAX_QUESTIONS + 1)
    ]
    assert len(provider.calls) == 2
    assert planner.degradation is not None
    assert planner.degradation.reason == "question_limit_exceeded"
    assert planner.degradation.original_question_count == MAX_QUESTIONS + 1
    assert planner.degradation.retained_question_count == MAX_QUESTIONS
    assert planner.degradation.repair_attempts == 1


def test_lead_planner_falls_back_when_repair_is_malformed(rca_env) -> None:
    provider = FakeProvider(
        turn(content=plan_response(MAX_QUESTIONS + 3)),
        turn(content="not json"),
    )
    planner = LeadPlanner(
        provider=provider,
        usage=UsageTracker(),
        budgets=rca_env["budgets"],
    )

    plan = planner.run(rca_env["context"])

    assert len(plan.questions) == MAX_QUESTIONS
    assert planner.degradation is not None
    assert planner.degradation.original_question_count == MAX_QUESTIONS + 3
    assert planner.degradation.repair_attempts == 1


def test_issue_ci_seeds_incident_evidence(rca_env) -> None:
    provider = FakeProvider(
        turn(
            content=final_response(
                evidence_ids=["ev-issue_ci-001", "ev-issue_ci-002"],
                uncertainty="medium",
            )
        )
    )
    output = IssueCISpecialist(
        provider=provider,
        registry=rca_env["registry"],
        usage=UsageTracker(),
        budgets=rca_env["budgets"],
    ).run(rca_env["context"], _questions())
    assert output.finding.status == FindingStatus.COMPLETED
    assert output.finding.evidence_ids == ["ev-issue_ci-001", "ev-issue_ci-002"]
    kinds = {item.kind for item in output.evidence}
    assert EvidenceKind.ISSUE_TEXT in kinds
    assert EvidenceKind.STACK_TRACE in kinds
    assert all(
        item.provenance.source == "incident_input" for item in output.evidence
    )


def test_issue_ci_can_read_external_log(rca_env) -> None:
    provider = FakeProvider(
        turn(
            tool_calls=[
                call("c1", "read_external_log", {"path": "ci.log"}),
            ]
        ),
        turn(
            content=final_response(
                evidence_ids=["ev-issue_ci-003"],
                uncertainty="medium",
            )
        ),
    )
    output = IssueCISpecialist(
        provider=provider,
        registry=rca_env["registry"],
        usage=UsageTracker(),
        budgets=rca_env["budgets"],
    ).run(rca_env["context"], _questions())
    assert output.finding.status == FindingStatus.COMPLETED
    log_item = output.evidence[-1]
    assert log_item.kind == EvidenceKind.CI_LOG
    assert log_item.provenance.tool == "read_external_log"
    assert "CI FAILURE" in log_item.excerpt


def test_code_specialist_records_tool_evidence_with_provenance(rca_env) -> None:
    provider = FakeProvider(
        turn(
            tool_calls=[
                call(
                    "c1",
                    "read_file",
                    {"path": "pkg/calc.py", "raw": True},
                ),
                call("c2", "inspect_symbols", {"path": "pkg/calc.py"}),
            ]
        ),
        turn(
            content=final_response(
                locations=[{"path": "pkg/calc.py", "symbol": "multiply"}],
                evidence_ids=["ev-code-001", "ev-code-002"],
                uncertainty="medium",
            )
        ),
    )
    output = CodeSpecialist(
        provider=provider,
        registry=rca_env["registry"],
        usage=UsageTracker(),
        budgets=rca_env["budgets"],
    ).run(rca_env["context"], _questions())
    assert output.finding.status == FindingStatus.COMPLETED
    assert output.finding.ranked_locations[0].path == "pkg/calc.py"
    assert output.finding.ranked_locations[0].symbol == "multiply"
    assert output.finding.evidence_ids == ["ev-code-001", "ev-code-002"]
    assert output.evidence[0].provenance.tool == "read_file"
    assert output.evidence[1].provenance.tool == "inspect_symbols"
    assert output.evidence[1].kind == EvidenceKind.SYMBOL
    assert output.evidence[0].location is not None
    assert output.evidence[0].location.path == "pkg/calc.py"


def test_code_specialist_malformed_output_is_explicit(rca_env) -> None:
    provider = FakeProvider(turn(content="I think the bug is in calc.py"))
    output = CodeSpecialist(
        provider=provider,
        registry=rca_env["registry"],
        usage=UsageTracker(),
        budgets=rca_env["budgets"],
    ).run(rca_env["context"], _questions())
    assert output.finding.status == FindingStatus.PARTIAL
    assert output.finding.uncertainty == UncertaintyLevel.HIGH
    assert output.finding.error is not None
    assert "malformed" in output.finding.error


def test_code_specialist_rejects_invalid_location_and_unknown_ids(rca_env) -> None:
    provider = FakeProvider(
        turn(
            content=final_response(
                locations=[{"path": "../outside.py"}],
                evidence_ids=["ev-code-999"],
            )
        )
    )
    output = CodeSpecialist(
        provider=provider,
        registry=rca_env["registry"],
        usage=UsageTracker(),
        budgets=rca_env["budgets"],
    ).run(rca_env["context"], _questions())
    assert output.finding.status == FindingStatus.PARTIAL
    assert output.finding.error is not None


def test_git_history_specialist_uses_git_tools(rca_env) -> None:
    provider = FakeProvider(
        turn(
            tool_calls=[
                call("c1", "git_history", {"max_count": 10}),
                call("c2", "git_blame", {"path": "pkg/calc.py"}),
            ]
        ),
        turn(
            content=final_response(
                locations=[{"path": "pkg/calc.py"}],
                evidence_ids=["ev-git_history-001", "ev-git_history-002"],
                uncertainty="high",
            )
        ),
    )
    output = GitHistorySpecialist(
        provider=provider,
        registry=rca_env["registry"],
        usage=UsageTracker(),
        budgets=rca_env["budgets"],
    ).run(rca_env["context"], _questions())
    assert output.finding.status == FindingStatus.COMPLETED
    assert output.finding.uncertainty == UncertaintyLevel.HIGH
    assert output.evidence[0].kind == EvidenceKind.GIT_LOG
    assert output.evidence[0].provenance.command.startswith("git log")
    assert output.evidence[0].commit_ids
    assert all(len(commit_id) == 40 for commit_id in output.evidence[0].commit_ids)
    assert output.evidence[1].kind == EvidenceKind.GIT_BLAME
    assert output.evidence[1].location is not None


def test_layered_git_search_does_not_add_model_calls(git_repo) -> None:
    context = make_context(git_repo)
    incident = IncidentInput(
        id="inc-layered-search",
        repo="target",
        base_commit=git_repo.head_sha,
        title="Regression in multiply",
        problem="The behavior changed after the previous commit.",
        related_commits=[git_repo.base_sha],
        provenance=Provenance(source="test_fixture"),
    )
    context = context.model_copy(
        update={
            "incident": incident,
            "repository": context.repository.model_copy(
                update={"base_commit": git_repo.head_sha}
            ),
            "fingerprint": context.fingerprint.model_copy(
                update={"head_sha": git_repo.head_sha}
            ),
        }
    )
    registry = RcaToolRegistry(
        Workspace(git_repo.repo),
        base_commit=git_repo.head_sha,
        history_depth=50,
    )
    registry.configure_git_history(
        base_commit=git_repo.head_sha,
        history_depth=50,
        visible_depth=1,
    )
    provider = FakeProvider(turn(content=final_response()))

    output = GitHistorySpecialist(
        provider=provider,
        registry=registry,
        usage=UsageTracker(),
        budgets=PlanBudgets(
            max_llm_calls=5,
            max_tool_calls=5,
            timeout_seconds=60,
        ),
    ).run(context, _questions())

    assert len(provider.calls) == 1
    assert output.finding.git_search_summary is not None
    assert output.finding.git_search_summary.stop_reason == "explicit_commit_verified"
    prepared = [
        item
        for item in output.evidence
        if item.observation.startswith("layered Git search candidate")
    ]
    assert len(prepared) == 1
    assert prepared[0].commit_ids == [git_repo.base_sha]


def test_weak_layered_candidate_is_not_promoted_to_evidence(
    rca_env,
    git_repo,
    monkeypatch,
) -> None:
    weak_candidate = GitSearchCandidate(
        commit=git_repo.base_sha,
        depth=8,
        score=4,
        matched_paths=["pkg/calc.py"],
        matched_signals=["prepared_snippet_path:pkg/calc.py"],
        signal_kinds=["path"],
        strong_match=False,
        command=f"git log --no-walk {git_repo.base_sha}",
    )
    summary = GitSearchSummary(
        enabled=True,
        attempted_depths=[8],
        reached_depth=2,
        candidate_commits=[git_repo.base_sha],
        candidates=[weak_candidate],
        stop_reason="history_exhausted",
        commands_executed=2,
    )
    monkeypatch.setattr(
        rca_env["registry"],
        "search_git_layers",
        lambda plan: summary,
    )
    provider = FakeProvider(turn(content=final_response()))

    output = GitHistorySpecialist(
        provider=provider,
        registry=rca_env["registry"],
        usage=UsageTracker(),
        budgets=rca_env["budgets"],
    ).run(rca_env["context"], _questions())

    assert not any(
        item.observation.startswith("layered Git search candidate")
        for item in output.evidence
    )
    prompt = provider.calls[0]["messages"][1]["content"]
    assert git_repo.base_sha in prompt


def test_git_tool_failure_is_partial_and_creates_no_evidence(rca_env) -> None:
    provider = FakeProvider(
        turn(
            tool_calls=[
                call("c1", "git_show", {"revision": "not-a-valid-revision"}),
            ]
        ),
        turn(content=final_response()),
    )
    output = GitHistorySpecialist(
        provider=provider,
        registry=rca_env["registry"],
        usage=UsageTracker(),
        budgets=rca_env["budgets"],
    ).run(rca_env["context"], _questions())

    assert output.evidence == []
    assert output.finding.status == FindingStatus.PARTIAL
    assert output.finding.uncertainty == UncertaintyLevel.HIGH
    assert "tool failures" in (output.finding.error or "")


def test_specialist_budget_exhaustion_is_partial(rca_env) -> None:
    provider = FakeProvider(
        turn(
            tool_calls=[
                call("c1", "read_file", {"path": "pkg/calc.py", "raw": True})
            ]
        )
    )
    budgets = PlanBudgets(max_llm_calls=1, max_tool_calls=10, timeout_seconds=60)
    output = CodeSpecialist(
        provider=provider,
        registry=rca_env["registry"],
        usage=UsageTracker(),
        budgets=budgets,
    ).run(rca_env["context"], _questions())
    assert output.finding.status == FindingStatus.PARTIAL
    assert "llm call budget exceeded" in (output.finding.error or "")


def test_finding_reports_usage(rca_env) -> None:
    provider = FakeProvider(
        turn(content=final_response(), prompt_tokens=33, completion_tokens=7)
    )
    output = CodeSpecialist(
        provider=provider,
        registry=rca_env["registry"],
        usage=UsageTracker(),
        budgets=rca_env["budgets"],
    ).run(rca_env["context"], _questions())
    assert output.finding.usage is not None
    assert output.finding.usage.llm_calls == 1
    assert output.finding.usage.prompt_tokens == 33
    assert output.finding.usage.completion_tokens == 7


def test_prompt_builders_are_bounded_and_domain_scoped(rca_env) -> None:
    context = rca_env["context"]
    questions = _questions()
    lead_prompt = build_lead_prompt(context)
    issue_prompt = build_issue_ci_prompt(context, questions)
    code_prompt = build_code_prompt(context, questions)
    git_prompt = build_git_history_prompt(context, questions)
    for prompt in (lead_prompt, issue_prompt, code_prompt, git_prompt):
        assert len(prompt) < MAX_PROMPT_CHARS + 200
    assert "incident" in issue_prompt.lower()
    assert "snippets" in code_prompt
    assert "git" in git_prompt.lower()


def test_bounded_note_never_exceeds_agent_finding_limit() -> None:
    from roottrace.agents.specialists import _bounded_note
    from roottrace.evidence.schema import (
        MAX_NOTE_CHARS,
        AgentFinding,
        AgentRole,
        FindingStatus,
        UncertaintyLevel,
    )

    note = _bounded_note("x" * 2000)
    assert note is not None
    assert len(note) == MAX_NOTE_CHARS
    assert note.endswith("\n...[truncated]")
    # A long worker error must survive AgentFinding validation instead of
    # crashing the whole investigation with an unbounded error string.
    AgentFinding(
        agent=AgentRole.CODE,
        status=FindingStatus.PARTIAL,
        uncertainty=UncertaintyLevel.HIGH,
        uncertainty_note=note,
        error=note,
    )


def test_cap_text_keeps_excerpt_within_evidence_limit() -> None:
    from roottrace.agents.specialists import _cap_text
    from roottrace.evidence.schema import MAX_EXCERPT_CHARS

    excerpt, truncated = _cap_text(
        "y" * (MAX_EXCERPT_CHARS + 500),
        MAX_EXCERPT_CHARS,
    )
    assert truncated is True
    assert len(excerpt) == MAX_EXCERPT_CHARS
    assert excerpt.endswith("\n...[truncated]")


def test_system_prompts_contain_no_hardcoded_example_data() -> None:
    """Schema examples must not prime the model with concrete fake ids/paths."""
    from roottrace.agents.prompts import (
        CODE_SYSTEM_PROMPT,
        FINAL_REPORT_SYSTEM_PROMPT,
        GIT_HISTORY_SYSTEM_PROMPT,
        HYPOTHESES_SYSTEM_PROMPT,
        ISSUE_CI_SYSTEM_PROMPT,
        LEAD_SYSTEM_PROMPT,
    )

    prompts = {
        "lead": LEAD_SYSTEM_PROMPT,
        "issue_ci": ISSUE_CI_SYSTEM_PROMPT,
        "code": CODE_SYSTEM_PROMPT,
        "git_history": GIT_HISTORY_SYSTEM_PROMPT,
        "hypotheses": HYPOTHESES_SYSTEM_PROMPT,
        "final_report": FINAL_REPORT_SYSTEM_PROMPT,
    }
    fake_values = (
        "ev-issue_ci-001",
        "ev-code-001",
        "ev-git_history-001",
        "h-001",
        "ver-h-001-001",
        "pkg/calc.py",
        "tests/test_calc.py",
    )
    for name, prompt in prompts.items():
        for fake in fake_values:
            assert fake not in prompt, f"{name} prompt contains hardcoded {fake}"
    assert "<unique-question-id>" in LEAD_SYSTEM_PROMPT
    assert "<existing-evidence-id>" in HYPOTHESES_SYSTEM_PROMPT
    assert "<existing-hypothesis-id>" in FINAL_REPORT_SYSTEM_PROMPT
    assert "<relative-test-target>" in HYPOTHESES_SYSTEM_PROMPT
