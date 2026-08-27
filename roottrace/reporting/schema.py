"""Final RootTrace report contracts."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator

from roottrace.evidence.graph import EvidenceGraph, missing_ids
from roottrace.evidence.schema import (
    MAX_EVIDENCE_IDS,
    MAX_LOCATIONS,
    MAX_NOTE_CHARS,
    MAX_STATEMENT_CHARS,
    BoundedNote,
    BoundedSuggestion,
    ConfidenceLevel,
    SourceLocation,
    UncertaintyLevel,
)
from roottrace.incident.schema import StableId, validate_commit_sha
from roottrace.llm.schema import Usage
from roottrace.verification.schema import VerificationResult

MAX_CAUSES = 5
MAX_LINKS = 5
MAX_NOTES = 10
MAX_SUGGESTIONS = 5
MAX_VERIFICATION_RESULTS = 10

_FORBIDDEN_PATCH_PREFIXES = ("diff --git", "--- a/", "+++ b/", "@@ ")
_FORBIDDEN_MODIFICATION_COMMANDS = (
    "git apply", "git am", "git reset --hard", "git checkout --",
    "patch -p", "apply_patch",
)


class ReportConclusion(str, Enum):
    ROOT_CAUSE_IDENTIFIED = "root_cause_identified"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class RankedCause(BaseModel):
    rank: int = Field(ge=1)
    hypothesis_id: StableId
    confidence: ConfidenceLevel
    rationale: str | None = Field(default=None, max_length=MAX_NOTE_CHARS)
    evidence_ids: list[StableId] = Field(default_factory=list, max_length=MAX_EVIDENCE_IDS)


class CauseLink(BaseModel):
    statement: str = Field(max_length=MAX_STATEMENT_CHARS)
    hypothesis_id: StableId | None = None
    evidence_ids: list[StableId] = Field(default_factory=list, max_length=MAX_EVIDENCE_IDS)


class RegressionChange(BaseModel):
    commit: str
    summary: str | None = Field(default=None, max_length=MAX_STATEMENT_CHARS)
    evidence_ids: list[StableId] = Field(default_factory=list, max_length=MAX_EVIDENCE_IDS)
    locations: list[SourceLocation] = Field(default_factory=list, max_length=MAX_LOCATIONS)

    @field_validator("commit")
    @classmethod
    def _validate_commit(cls, value: str) -> str:
        return validate_commit_sha(value)


def assert_advisory_text(value: str) -> str:
    """Reject text that embeds executable patch content or edit commands."""
    for index, line in enumerate(value.splitlines(), start=1):
        stripped = line.lstrip()
        if stripped.startswith(_FORBIDDEN_PATCH_PREFIXES):
            raise ValueError(
                f"fix recommendation must be advisory text; line {index} looks like an embedded patch"
            )
        if stripped.startswith("\\ No newline"):
            raise ValueError(
                f"fix recommendation must be advisory text; line {index} is a patch artifact"
            )
        if any(command in line.lower() for command in _FORBIDDEN_MODIFICATION_COMMANDS):
            raise ValueError(
                f"fix recommendation must not contain code modification commands (line {index})"
            )
    return value


class FixRecommendation(BaseModel):
    scope: str | None = Field(default=None, max_length=MAX_STATEMENT_CHARS)
    suggestions: list[BoundedSuggestion] = Field(default_factory=list, max_length=MAX_SUGGESTIONS)
    locations: list[SourceLocation] = Field(default_factory=list, max_length=MAX_LOCATIONS)
    evidence_ids: list[StableId] = Field(default_factory=list, max_length=MAX_EVIDENCE_IDS)

    @field_validator("scope")
    @classmethod
    def _validate_scope(cls, value: str | None) -> str | None:
        if value is not None:
            assert_advisory_text(value)
        return value

    @field_validator("suggestions")
    @classmethod
    def _validate_suggestions(cls, values: list[str]) -> list[str]:
        for value in values:
            assert_advisory_text(value)
        return values


class UncertaintySummary(BaseModel):
    level: UncertaintyLevel = UncertaintyLevel.LOW
    insufficient_evidence: bool = False
    notes: list[BoundedNote] = Field(default_factory=list, max_length=MAX_NOTES)


class Timing(BaseModel):
    total_seconds: float | None = Field(default=None, ge=0)
    model_seconds: float | None = Field(default=None, ge=0)
    verification_seconds: float | None = Field(default=None, ge=0)


class RCAReport(BaseModel):
    id: StableId
    incident_id: StableId
    evidence_graph: EvidenceGraph
    conclusion: ReportConclusion
    conclusion_summary: str | None = Field(default=None, max_length=MAX_STATEMENT_CHARS)
    ranked_causes: list[RankedCause] = Field(default_factory=list, max_length=MAX_CAUSES)
    top_k_locations: list[SourceLocation] = Field(default_factory=list, max_length=MAX_LOCATIONS)
    causal_chain: list[CauseLink] = Field(default_factory=list, max_length=MAX_LINKS)
    verification: list[VerificationResult] = Field(default_factory=list, max_length=MAX_VERIFICATION_RESULTS)
    suspected_regression: RegressionChange | None = None
    fix_recommendation: FixRecommendation | None = None
    uncertainty: UncertaintySummary = Field(default_factory=UncertaintySummary)
    timing: Timing = Field(default_factory=Timing)
    usage: Usage = Field(default_factory=Usage)

    @model_validator(mode="after")
    def _validate_references(self) -> RCAReport:
        graph = self.evidence_graph
        if self.incident_id != graph.incident.id:
            raise ValueError("incident_id must match evidence_graph.incident.id")
        hypothesis_ids = {hypothesis.id for hypothesis in graph.hypotheses}
        evidence_ids = {item.id for item in graph.evidence}
        if self.conclusion is ReportConclusion.ROOT_CAUSE_IDENTIFIED and not self.ranked_causes:
            raise ValueError("conclusion root_cause_identified requires at least one ranked cause")
        ranks = [cause.rank for cause in self.ranked_causes]
        if len(set(ranks)) != len(ranks):
            raise ValueError("ranked cause ranks must be unique")
        for cause in self.ranked_causes:
            if cause.hypothesis_id not in hypothesis_ids:
                raise ValueError(f"RankedCause references unknown hypothesis: {cause.hypothesis_id}")
            missing = missing_ids(cause.evidence_ids, evidence_ids)
            if missing:
                raise ValueError(f"RankedCause {cause.rank} references unknown evidence ids: {missing}")
        for link in self.causal_chain:
            if link.hypothesis_id is not None and link.hypothesis_id not in hypothesis_ids:
                raise ValueError(f"CauseLink references unknown hypothesis: {link.hypothesis_id}")
            missing = missing_ids(link.evidence_ids, evidence_ids)
            if missing:
                raise ValueError(f"CauseLink references unknown evidence ids: {missing}")
        verification_ids: set[str] = set()
        for result in self.verification:
            if result.id in verification_ids:
                raise ValueError(f"duplicate verification result id: {result.id}")
            verification_ids.add(result.id)
            if result.hypothesis_id not in hypothesis_ids:
                raise ValueError(f"VerificationResult references unknown hypothesis: {result.hypothesis_id}")
            missing = missing_ids(result.evidence_ids, evidence_ids)
            if missing:
                raise ValueError(f"VerificationResult {result.id} references unknown evidence ids: {missing}")
        if self.suspected_regression is not None:
            missing = missing_ids(self.suspected_regression.evidence_ids, evidence_ids)
            if missing:
                raise ValueError(f"suspected_regression references unknown evidence ids: {missing}")
        if self.fix_recommendation is not None:
            missing = missing_ids(self.fix_recommendation.evidence_ids, evidence_ids)
            if missing:
                raise ValueError(f"fix_recommendation references unknown evidence ids: {missing}")
        return self
