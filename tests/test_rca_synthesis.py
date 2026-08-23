"""Tests for final RCA synthesis and the verification-to-report chain."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from patchpilot.models import AssistantTurn
from patchpilot.rca.sandbox import RuntimeVerificationSandbox
from patchpilot.rca.schema import (
    AgentRole,
    EvidenceGraph,
    EvidenceItem,
    EvidenceKind,
    Hypothesis,
    HypothesisDisposition,
    IncidentInput,
    Provenance,
    VerificationOutcome,
    VerificationStatus,
    VerificationStep,
)
from patchpilot.rca.synthesis import LeadSynthesizer, SynthesisError
from patchpilot.rca.usage import UsageTracker
from patchpilot.rca.verification import RuntimeTestVerifier, VerificationRun


def _graph(git_repo, hypotheses: list[Hypothesis]) -> EvidenceGraph:
    incident = IncidentInput(
        id="inc-001",
        repo="target",
        base_commit=git_repo.base_sha,
        title="multiply returns a+b",
        problem="multiply returns a+b instead of a*b",
        logs=[],
        provenance=Provenance(source="test_fixture"),
    )
    seed = EvidenceItem(
        id="ev-seed-001",
        agent=AgentRole.CODE,
        kind=EvidenceKind.CODE_SNIPPET,
        observation="suspicious function",
        provenance=Provenance(source="test_fixture"),
        excerpt="def multiply(a, b):\n    return a + b",
    )
    return EvidenceGraph(
        incident=incident,
        findings=[],
        evidence=[seed],
        hypotheses=hypotheses,
    )


def _hypothesis(hypothesis_id: str) -> Hypothesis:
    return Hypothesis(
        id=hypothesis_id,
        statement="multiply is implemented as addition",
        supporting_evidence_ids=["ev-seed-001"],
        verification_plan=[
            VerificationStep(
                command="python -m pytest -q tests/test_calc.py",
                description="run the calc regression tests",
                timeout_seconds=60,
            )
        ],
    )


def _report_json(git_repo, runtime_evidence_id: str | None = None) -> str:
    evidence_ids = ["ev-seed-001"]
    if runtime_evidence_id:
        evidence_ids.append(runtime_evidence_id)
    return json.dumps(
        {
            "conclusion": "root_cause_identified",
            "conclusion_summary": "multiply is implemented as addition",
            "ranked_causes": [
                {
                    "rank": 1,
                    "hypothesis_id": "h-001",
                    "confidence": "medium",
                    "rationale": "code evidence and passing regression tests",
                    "evidence_ids": evidence_ids,
                }
            ],
            "top_k_locations": [
                {"path": "pkg/calc.py", "symbol": "multiply"}
            ],
            "causal_chain": [
                {
                    "statement": "the multiply function returns a+b",
                    "hypothesis_id": "h-001",
                    "evidence_ids": ["ev-seed-001"],
                }
            ],
            "suspected_regression": {
                "commit": git_repo.base_sha,
                "summary": "initial implementation of multiply",
                "evidence_ids": ["ev-seed-001"],
                "locations": [{"path": "pkg/calc.py"}],
            },
            "fix_recommendation": {
                "scope": "review the multiply function in pkg/calc.py",
                "suggestions": [
                    "verify the arithmetic operation against expected behavior"
                ],
                "locations": [
                    {"path": "pkg/calc.py", "symbol": "multiply"}
                ],
                "evidence_ids": ["ev-seed-001"],
            },
            "uncertainty": {
                "level": "medium",
                "insufficient_evidence": False,
                "notes": [],
            },
        }
    )


class FakeProvider:
    model = "fake-model"

    def __init__(self, *responses: AssistantTurn) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def complete(self, messages, tools, tool_choice=None) -> AssistantTurn:
        self.calls.append({"messages": messages})
        if not self._responses:
            raise AssertionError("unexpected provider call")
        return self._responses.pop(0)


def turn(content: str) -> AssistantTurn:
    return AssistantTurn(
        content=content,
        tool_calls=[],
        prompt_tokens=20,
        completion_tokens=8,
    )


def _verified_run(git_repo, tmp_path: Path) -> VerificationRun:
    with RuntimeVerificationSandbox(
        git_repo.repo,
        work_dir=tmp_path,
    ) as sandbox:
        graph = _graph(git_repo, [_hypothesis("h-001")])
        return RuntimeTestVerifier(sandbox).verify(graph)


def test_synthesis_produces_validated_report(git_repo, tmp_path: Path) -> None:
    run = _verified_run(git_repo, tmp_path)
    runtime_id = run.results[0].evidence_ids[0]
    provider = FakeProvider(turn(_report_json(git_repo, runtime_id)))
    usage = UsageTracker()
    report = LeadSynthesizer(provider=provider, usage=usage).synthesize(
        run.graph,
        run,
    )

    assert report.id == "rca-inc-001"
    assert report.conclusion.value == "root_cause_identified"
    assert report.ranked_causes[0].hypothesis_id == "h-001"
    assert report.fix_recommendation is not None
    assert report.fix_recommendation.scope == (
        "review the multiply function in pkg/calc.py"
    )
    assert report.suspected_regression is not None
    assert report.suspected_regression.commit == git_repo.base_sha
    assert report.verification[0].outcome == VerificationOutcome.SUPPORTED
    assert report.verification[0].evidence_ids == [runtime_id]
    assert report.evidence_graph.hypotheses[0].disposition == (
        HypothesisDisposition.SUPPORTED
    )
    assert report.timing.verification_seconds is not None
    assert report.usage.llm_calls == 1
    assert report.usage.prompt_tokens == 20


def test_synthesis_rejects_malformed_output(git_repo, tmp_path: Path) -> None:
    run = _verified_run(git_repo, tmp_path)
    provider = FakeProvider(
        turn("the root cause is obvious"),
        turn("the root cause is obvious"),
    )
    with pytest.raises(SynthesisError):
        LeadSynthesizer(provider=provider, usage=UsageTracker()).synthesize(
            run.graph,
            run,
        )


def test_synthesis_rejects_embedded_patch_text(git_repo, tmp_path: Path) -> None:
    run = _verified_run(git_repo, tmp_path)
    payload = json.loads(_report_json(git_repo))
    payload["fix_recommendation"]["suggestions"] = [
        "diff --git a/pkg/calc.py b/pkg/calc.py\nindex 123..456"
    ]
    provider = FakeProvider(turn(json.dumps(payload)), turn(json.dumps(payload)))
    with pytest.raises(SynthesisError):
        LeadSynthesizer(provider=provider, usage=UsageTracker()).synthesize(
            run.graph,
            run,
        )


def test_synthesis_rejects_root_cause_without_ranked_causes(
    git_repo,
    tmp_path: Path,
) -> None:
    run = _verified_run(git_repo, tmp_path)
    payload = json.loads(_report_json(git_repo))
    payload["ranked_causes"] = []
    provider = FakeProvider(turn(json.dumps(payload)), turn(json.dumps(payload)))
    with pytest.raises(SynthesisError):
        LeadSynthesizer(provider=provider, usage=UsageTracker()).synthesize(
            run.graph,
            run,
        )


def test_synthesis_rejects_ranking_a_rejected_hypothesis(
    git_repo,
    tmp_path: Path,
) -> None:
    run = _verified_run(git_repo, tmp_path)
    payload = json.loads(_report_json(git_repo))
    rejected = run.graph.hypotheses[0].model_copy(
        update={"disposition": HypothesisDisposition.REJECTED}
    )
    graph = run.graph.model_copy(
        update={
            "hypotheses": [
                rejected if hypothesis.id == "h-001" else hypothesis
                for hypothesis in run.graph.hypotheses
            ]
        }
    )
    provider = FakeProvider(turn(json.dumps(payload)))
    with pytest.raises(SynthesisError):
        LeadSynthesizer(provider=provider, usage=UsageTracker()).synthesize(
            graph,
            run,
        )


def test_verification_to_report_chain(git_repo, tmp_path: Path) -> None:
    """One local case produces an evidence-backed RCA report."""
    run = _verified_run(git_repo, tmp_path)
    assert run.results[0].outcome == VerificationOutcome.SUPPORTED
    runtime_id = run.results[0].evidence_ids[0]
    provider = FakeProvider(turn(_report_json(git_repo, runtime_id)))
    report = LeadSynthesizer(provider=provider, usage=UsageTracker()).synthesize(
        run.graph,
        run,
    )
    assert report.evidence_graph.incident.id == "inc-001"
    assert any(
        item.kind == EvidenceKind.TEST_RESULT
        for item in report.evidence_graph.evidence
    )
    assert report.verification[0].status == VerificationStatus.PASSED


def test_final_report_prompt_exposes_evidence_domain_and_never_primes_ids(
    git_repo,
) -> None:
    from patchpilot.rca.prompts import (
        FINAL_REPORT_SYSTEM_PROMPT,
        build_final_report_prompt,
    )

    graph = _graph(git_repo, [_hypothesis("h-001")])
    prompt = build_final_report_prompt(graph, [])

    # The code-only graph must be told that git-history evidence is absent and
    # that invented ids are forbidden.
    assert "EVIDENCE DOMAIN RULE" in prompt
    assert "'code'" in prompt
    assert "never invent ids" in prompt
    assert "omit suspected_regression" in prompt
    # The schema must not prime the model with concrete evidence ids, which
    # caused hallucinated ids like ev-git_history-001 in partial runs.
    assert "ev-code-001" not in FINAL_REPORT_SYSTEM_PROMPT
    assert "ev-git_history-001" not in FINAL_REPORT_SYSTEM_PROMPT
    assert "<existing-evidence-id>" in FINAL_REPORT_SYSTEM_PROMPT


def test_final_report_prompt_bans_verification_ids_and_empty_commits() -> None:
    from patchpilot.rca.prompts import FINAL_REPORT_SYSTEM_PROMPT

    assert "verification result ids (ver-*)" in FINAL_REPORT_SYSTEM_PROMPT
    assert "Never emit an empty or placeholder commit" in FINAL_REPORT_SYSTEM_PROMPT


def _invalid_report_json() -> str:
    return json.dumps(
        {
            "conclusion": "insufficient_evidence",
            "conclusion_summary": "no supported cause",
            "ranked_causes": [],
            "top_k_locations": [],
            "causal_chain": [],
            "suspected_regression": {
                "commit": "",
                "summary": "empty commit",
                "evidence_ids": [],
                "locations": [],
            },
            "uncertainty": {
                "level": "medium",
                "insufficient_evidence": True,
                "notes": [],
            },
        }
    )


def test_synthesis_retries_once_with_validation_feedback(
    git_repo,
    tmp_path: Path,
) -> None:
    run = _verified_run(git_repo, tmp_path)
    runtime_id = run.results[0].evidence_ids[0]
    provider = FakeProvider(
        turn(_invalid_report_json()),
        turn(_report_json(git_repo, runtime_id)),
    )
    usage = UsageTracker()
    report = LeadSynthesizer(provider=provider, usage=usage).synthesize(
        run.graph,
        run,
    )
    assert len(provider.calls) == 2
    second_messages = provider.calls[1]["messages"][1]["content"]
    assert "PREVIOUS OUTPUT REJECTED" in second_messages
    assert "commit must be a 7-64 character hexadecimal SHA" in second_messages
    assert report.id.startswith("rca-")
    assert usage.snapshot().llm_calls == 2


def test_synthesis_raises_after_repair_retries_exhausted(
    git_repo,
    tmp_path: Path,
) -> None:
    run = _verified_run(git_repo, tmp_path)
    provider = FakeProvider(turn(_invalid_report_json()), turn(_invalid_report_json()))
    usage = UsageTracker()
    with pytest.raises(SynthesisError, match="malformed final synthesis output"):
        LeadSynthesizer(provider=provider, usage=usage).synthesize(run.graph, run)
    assert len(provider.calls) == 2
    assert usage.snapshot().llm_calls == 2
