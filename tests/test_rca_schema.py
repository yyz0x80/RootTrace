"""Focused tests for RootTrace typed RCA contracts (M1)."""

import pytest
from pydantic import ValidationError

from patchpilot.rca.artifacts import model_to_json
from patchpilot.rca.schema import (
    MAX_EXCERPT_CHARS,
    AgentFinding,
    AgentRole,
    ConfidenceLevel,
    EvidenceEdge,
    EvidenceGraph,
    EvidenceItem,
    EvidenceKind,
    EvidenceRelation,
    FindingStatus,
    FixRecommendation,
    Hypothesis,
    HypothesisDisposition,
    IncidentInput,
    InvestigationPlan,
    PlanQuestion,
    Provenance,
    RankedCause,
    RCAReport,
    ReportConclusion,
    SourceLocation,
    UncertaintyLevel,
    Usage,
    VerificationOutcome,
    VerificationResult,
    VerificationStatus,
    VerificationStep,
)


def make_provenance(**overrides) -> Provenance:
    fields = {"source": "issue.md:12", "tool": "issue_ci"}
    fields.update(overrides)
    return Provenance(**fields)


def make_location(path: str = "src/app.py", **overrides) -> SourceLocation:
    fields = {"symbol": "load", "start_line": 10, "end_line": 12}
    fields.update(overrides)
    return SourceLocation(path=path, **fields)


def make_incident(**overrides) -> IncidentInput:
    fields = {
        "id": "inc-1",
        "repo": "demo/repo",
        "base_commit": "a" * 40,
        "problem": "Crash when loading the module",
        "provenance": make_provenance(source="issue.md:1"),
    }
    fields.update(overrides)
    return IncidentInput(**fields)


def make_evidence(evidence_id: str = "ev-1", **overrides) -> EvidenceItem:
    fields = {
        "agent": AgentRole.ISSUE_CI,
        "kind": EvidenceKind.STACK_TRACE,
        "observation": "Traceback points at load()",
        "provenance": make_provenance(),
        "location": make_location(),
        "excerpt": "Traceback (most recent call last): ...",
    }
    fields.update(overrides)
    return EvidenceItem(id=evidence_id, **fields)


def make_hypothesis(hypothesis_id: str = "hyp-1", **overrides) -> Hypothesis:
    fields = {
        "statement": "load() dereferences None when config is missing",
        "locations": [make_location()],
        "supporting_evidence_ids": ["ev-1"],
        "verification_plan": [
            VerificationStep(
                command="python -m pytest tests/test_app.py::test_load",
            )
        ],
        "confidence": ConfidenceLevel.MEDIUM,
        "disposition": HypothesisDisposition.CANDIDATE,
    }
    fields.update(overrides)
    return Hypothesis(id=hypothesis_id, **fields)


def make_finding(**overrides) -> AgentFinding:
    fields = {
        "agent": AgentRole.CODE,
        "status": FindingStatus.COMPLETED,
        "ranked_locations": [make_location()],
        "evidence_ids": ["ev-1"],
        "uncertainty": UncertaintyLevel.LOW,
    }
    fields.update(overrides)
    return AgentFinding(**fields)


def make_graph(**overrides) -> EvidenceGraph:
    fields = {
        "incident": make_incident(),
        "findings": [make_finding()],
        "evidence": [make_evidence()],
        "hypotheses": [make_hypothesis()],
        "edges": [],
    }
    fields.update(overrides)
    return EvidenceGraph(**fields)


def make_report(**overrides) -> RCAReport:
    graph = make_graph()
    fields = {
        "id": "report-1",
        "incident_id": graph.incident.id,
        "evidence_graph": graph,
        "conclusion": ReportConclusion.ROOT_CAUSE_IDENTIFIED,
        "ranked_causes": [
            RankedCause(
                rank=1,
                hypothesis_id="hyp-1",
                confidence=ConfidenceLevel.MEDIUM,
                evidence_ids=["ev-1"],
            )
        ],
        "verification": [
            VerificationResult(
                id="ver-1",
                hypothesis_id="hyp-1",
                command="python -m pytest tests/test_app.py::test_load",
                status=VerificationStatus.PASSED,
                outcome=VerificationOutcome.SUPPORTED,
                evidence_ids=["ev-1"],
            )
        ],
        "fix_recommendation": FixRecommendation(
            scope="Review the load() path in src/app.py",
            suggestions=["Add a null check before accessing the config value"],
            locations=[make_location()],
            evidence_ids=["ev-1"],
        ),
    }
    fields.update(overrides)
    return RCAReport(**fields)


def test_incident_round_trip() -> None:
    incident = make_incident()
    rebuilt = IncidentInput.model_validate(incident.model_dump(mode="json"))
    assert rebuilt == incident


def test_evidence_round_trip() -> None:
    evidence = make_evidence()
    rebuilt = EvidenceItem.model_validate(evidence.model_dump(mode="json"))
    assert rebuilt == evidence
    assert rebuilt.id == "ev-1"
    assert rebuilt.provenance.source == "issue.md:12"


def test_evidence_requires_stable_id_and_provenance() -> None:
    with pytest.raises(ValidationError):
        EvidenceItem(
            id="bad id with spaces",
            agent=AgentRole.ISSUE_CI,
            kind=EvidenceKind.STACK_TRACE,
            observation="oops",
            provenance=make_provenance(),
            excerpt="x",
        )
    with pytest.raises(ValidationError):
        EvidenceItem(
            id="ev-x",
            agent=AgentRole.ISSUE_CI,
            kind=EvidenceKind.STACK_TRACE,
            observation="oops",
            excerpt="x",
        )


def test_excerpt_and_observation_are_bounded() -> None:
    with pytest.raises(ValidationError):
        make_evidence(excerpt="x" * (MAX_EXCERPT_CHARS + 1))
    with pytest.raises(ValidationError):
        make_evidence(observation="x" * 2_001)


def test_repo_relative_path_validation() -> None:
    assert make_location(path="src/app.py").path == "src/app.py"
    assert make_location(path="./src/app.py").path == "src/app.py"
    for bad in (
        "/etc/passwd",
        "../outside.py",
        "src/../../outside.py",
        "a\\b.py",
        "",
        "~/.env",
    ):
        with pytest.raises(ValidationError):
            make_location(path=bad)


def test_line_range_validation() -> None:
    with pytest.raises(ValidationError):
        make_location(start_line=5, end_line=3)
    assert make_location(start_line=3, end_line=3)


def test_duplicate_evidence_ids_rejected() -> None:
    evidence = [make_evidence("ev-1"), make_evidence("ev-1")]
    with pytest.raises(ValidationError):
        make_graph(evidence=evidence)


def test_dangling_finding_reference_rejected() -> None:
    with pytest.raises(ValidationError):
        make_graph(findings=[make_finding(evidence_ids=["ev-missing"])])


def test_dangling_hypothesis_reference_rejected() -> None:
    with pytest.raises(ValidationError):
        make_graph(
            hypotheses=[
                make_hypothesis(supporting_evidence_ids=["ev-missing"])
            ]
        )


def test_dangling_edge_rejected() -> None:
    with pytest.raises(ValidationError):
        make_graph(
            edges=[
                EvidenceEdge(
                    source="ev-1",
                    target="ev-missing",
                    relation=EvidenceRelation.SUPPORTS,
                )
            ]
        )


def test_self_loop_edge_rejected() -> None:
    with pytest.raises(ValidationError):
        EvidenceEdge(
            source="ev-1",
            target="ev-1",
            relation=EvidenceRelation.CAUSED_BY,
        )


def test_duplicate_edge_rejected() -> None:
    edge = EvidenceEdge(
        source="ev-1",
        target="ev-2",
        relation=EvidenceRelation.SUPPORTS,
    )
    graph = make_graph(evidence=[make_evidence("ev-1"), make_evidence("ev-2")])
    graph = graph.model_copy(
        update={"edges": [edge, edge.model_copy()]},
    )
    with pytest.raises(ValidationError):
        EvidenceGraph.model_validate(graph.model_dump(mode="json"))


def test_deterministic_serialization() -> None:
    graph = make_graph()
    first = model_to_json(graph)
    second = model_to_json(graph.model_copy(deep=True))
    assert first == second


def test_graph_round_trip() -> None:
    graph = make_graph()
    rebuilt = EvidenceGraph.model_validate(graph.model_dump(mode="json"))
    assert rebuilt == graph


def test_fix_recommendation_rejects_patch_content() -> None:
    with pytest.raises(ValidationError):
        FixRecommendation(
            scope="Review load()",
            suggestions=[
                (
                    "diff --git a/src/app.py b/src/app.py\n"
                    "@@ -1,2 +1,2 @@\n"
                    "-print(1)\n"
                    "+print(2)"
                )
            ],
        )
    with pytest.raises(ValidationError):
        FixRecommendation(suggestions=["run `git apply fix.patch`"])
    with pytest.raises(ValidationError):
        FixRecommendation(
            scope="Apply this patch:\n--- a/src/app.py\n+++ b/src/app.py\n"
        )


def test_fix_recommendation_accepts_advisory_text() -> None:
    recommendation = FixRecommendation(
        scope="Review the load() path in src/app.py",
        suggestions=[
            "- Consider a null check before accessing the config value",
            "Verify with python -m pytest tests/test_app.py",
        ],
        locations=[make_location()],
        evidence_ids=["ev-1"],
    )
    assert recommendation.scope is not None
    assert len(recommendation.suggestions) == 2


def test_report_round_trip_and_deterministic() -> None:
    report = make_report()
    rebuilt = RCAReport.model_validate(report.model_dump(mode="json"))
    assert rebuilt == report
    assert model_to_json(rebuilt) == model_to_json(report)


def test_report_incident_mismatch_rejected() -> None:
    graph = make_graph(incident=make_incident(id="inc-other"))
    with pytest.raises(ValidationError):
        make_report(evidence_graph=graph, incident_id="inc-1")


def test_report_dangling_verification_hypothesis_rejected() -> None:
    with pytest.raises(ValidationError):
        make_report(
            verification=[
                VerificationResult(
                    id="ver-x",
                    hypothesis_id="hyp-missing",
                    command="python -m pytest tests",
                    status=VerificationStatus.PASSED,
                    outcome=VerificationOutcome.SUPPORTED,
                )
            ]
        )


def test_report_dangling_ranked_cause_rejected() -> None:
    with pytest.raises(ValidationError):
        make_report(
            ranked_causes=[
                RankedCause(
                    rank=1,
                    hypothesis_id="hyp-missing",
                    confidence=ConfidenceLevel.HIGH,
                )
            ]
        )


def test_report_rank_duplicates_rejected() -> None:
    with pytest.raises(ValidationError):
        make_report(
            ranked_causes=[
                RankedCause(
                    rank=1,
                    hypothesis_id="hyp-1",
                    confidence=ConfidenceLevel.HIGH,
                ),
                RankedCause(
                    rank=1,
                    hypothesis_id="hyp-1",
                    confidence=ConfidenceLevel.MEDIUM,
                ),
            ]
        )


def test_report_conclusion_requires_ranked_cause() -> None:
    with pytest.raises(ValidationError):
        make_report(ranked_causes=[])


def test_plan_assignments_derived_from_questions() -> None:
    plan = InvestigationPlan(
        id="plan-1",
        incident_id="inc-1",
        questions=[
            PlanQuestion(
                id="q1",
                text="What failure signature does the stack trace show?",
                assigned_agents=[AgentRole.ISSUE_CI, AgentRole.CODE],
            ),
            PlanQuestion(
                id="q2",
                text="Which commit introduced the suspected regression?",
                assigned_agents=[AgentRole.GIT_HISTORY],
            ),
        ],
    )
    assert plan.assignments[AgentRole.ISSUE_CI] == ["q1"]
    assert plan.assignments[AgentRole.CODE] == ["q1"]
    assert plan.assignments[AgentRole.GIT_HISTORY] == ["q2"]
    data = plan.model_dump(mode="json")
    assert data["assignments"] == {
        "issue_ci": ["q1"],
        "code": ["q1"],
        "git_history": ["q2"],
    }


def test_plan_duplicate_question_ids_rejected() -> None:
    with pytest.raises(ValidationError):
        InvestigationPlan(
            id="plan-1",
            incident_id="inc-1",
            questions=[
                PlanQuestion(
                    id="q1",
                    text="question one",
                    assigned_agents=[AgentRole.CODE],
                ),
                PlanQuestion(
                    id="q1",
                    text="question two",
                    assigned_agents=[AgentRole.CODE],
                ),
            ],
        )


def test_plan_question_requires_agent() -> None:
    with pytest.raises(ValidationError):
        PlanQuestion(id="q1", text="unassigned", assigned_agents=[])


def test_base_commit_and_repo_identifier_validation() -> None:
    with pytest.raises(ValidationError):
        make_incident(base_commit="not-a-sha")
    with pytest.raises(ValidationError):
        make_incident(base_commit="a" * 65)
    with pytest.raises(ValidationError):
        make_incident(repo="/Users/me/repo")
    with pytest.raises(ValidationError):
        make_incident(repo="../repo")
    assert make_incident(repo="owner/repo").repo == "owner/repo"


def test_usage_null_semantics_round_trip() -> None:
    usage = Usage(llm_calls=3, prompt_tokens=None, completion_tokens=None)
    data = usage.model_dump(mode="json")
    assert data == {
        "llm_calls": 3,
        "prompt_tokens": None,
        "completion_tokens": None,
    }
    assert Usage.model_validate(data) == usage


def test_hypothesis_contradicting_evidence_round_trip() -> None:
    graph = make_graph(
        evidence=[make_evidence("ev-1"), make_evidence("ev-2")],
        hypotheses=[
            make_hypothesis(
                supporting_evidence_ids=["ev-1"],
                contradicting_evidence_ids=["ev-2"],
            )
        ],
    )
    rebuilt = EvidenceGraph.model_validate(graph.model_dump(mode="json"))
    assert rebuilt == graph
