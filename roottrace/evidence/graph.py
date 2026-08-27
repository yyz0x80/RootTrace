"""Validated evidence graph and deterministic aggregation."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from roottrace.evidence.schema import (
    AgentFinding,
    AgentRole,
    EvidenceEdge,
    EvidenceItem,
    Hypothesis,
)
from roottrace.incident.schema import IncidentInput

MAX_FINDINGS = 10
MAX_GRAPH_EVIDENCE = 200
MAX_GRAPH_HYPOTHESES = 20
MAX_GRAPH_EDGES = 500

_ROLE_ORDER = (AgentRole.ISSUE_CI, AgentRole.CODE, AgentRole.GIT_HISTORY)


def _unique_ids(ids, label: str) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for item in ids:
        if item in seen:
            duplicates.add(item)
        seen.add(item)
    if duplicates:
        raise ValueError(f"duplicate {label} ids: {sorted(duplicates)}")
    return seen


def missing_ids(ids, known: set[str]) -> list[str]:
    return sorted({item for item in ids if item not in known})


class EvidenceGraph(BaseModel):
    incident: IncidentInput
    findings: list[AgentFinding] = Field(default_factory=list, max_length=MAX_FINDINGS)
    evidence: list[EvidenceItem] = Field(default_factory=list, max_length=MAX_GRAPH_EVIDENCE)
    hypotheses: list[Hypothesis] = Field(default_factory=list, max_length=MAX_GRAPH_HYPOTHESES)
    edges: list[EvidenceEdge] = Field(default_factory=list, max_length=MAX_GRAPH_EDGES)

    @model_validator(mode="after")
    def _validate_references(self) -> EvidenceGraph:
        evidence_ids = _unique_ids((item.id for item in self.evidence), "evidence")
        _unique_ids((hypothesis.id for hypothesis in self.hypotheses), "hypothesis")
        for finding in self.findings:
            missing = missing_ids(finding.evidence_ids, evidence_ids)
            if missing:
                raise ValueError(
                    f"AgentFinding for {finding.agent.value} references unknown evidence ids: {missing}"
                )
        for hypothesis in self.hypotheses:
            referenced = (*hypothesis.supporting_evidence_ids, *hypothesis.contradicting_evidence_ids)
            missing = missing_ids(referenced, evidence_ids)
            if missing:
                raise ValueError(
                    f"Hypothesis {hypothesis.id} references unknown evidence ids: {missing}"
                )
        edge_keys: set[tuple[str, str, str]] = set()
        for edge in self.edges:
            missing = missing_ids((edge.source, edge.target), evidence_ids)
            if missing:
                raise ValueError(f"EvidenceEdge references unknown evidence ids: {missing}")
            key = (edge.source, edge.target, edge.relation.value)
            if key in edge_keys:
                raise ValueError(f"duplicate evidence edge: {key}")
            edge_keys.add(key)
        return self


def aggregate_evidence(incident: IncidentInput, outputs: dict) -> EvidenceGraph:
    """Merge specialist outputs in stable role order into an EvidenceGraph."""
    ordered = [outputs[role] for role in _ROLE_ORDER if role in outputs]
    return EvidenceGraph(
        incident=incident,
        findings=[output.finding for output in ordered],
        evidence=[item for output in ordered for item in output.evidence],
    )
