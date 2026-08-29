"""Evidence, hypothesis, and specialist-finding domain contracts."""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import (
    BaseModel,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from roottrace.incident.schema import Provenance, StableId, validate_commit_sha
from roottrace.llm.schema import Usage
from roottrace.runtime.paths import validate_relative_path

MAX_COMMAND_CHARS = 500
MAX_SYMBOL_CHARS = 512
MAX_EXCERPT_CHARS = 8_000
MAX_OBSERVATION_CHARS = 2_000
MAX_STATEMENT_CHARS = 2_000
MAX_NOTE_CHARS = 500
MAX_PROPOSAL_CHARS = 4_000
MAX_LOCATIONS = 10
MAX_EVIDENCE_IDS = 50
MAX_STEPS = 5
MAX_SUGGESTIONS = 5
MAX_GRAPH_EVIDENCE = 200
MAX_COMMIT_IDS = 20

BoundedNote = Annotated[str, StringConstraints(max_length=MAX_NOTE_CHARS)]
BoundedSuggestion = Annotated[str, StringConstraints(max_length=MAX_PROPOSAL_CHARS)]


class AgentRole(str, Enum):
    """Agents that may produce findings or evidence in a RootTrace run."""

    LEAD = "lead"
    ISSUE_CI = "issue_ci"
    CODE = "code"
    GIT_HISTORY = "git_history"
    RUNTIME_TEST = "runtime_test"


class EvidenceKind(str, Enum):
    ISSUE_TEXT = "issue_text"
    STACK_TRACE = "stack_trace"
    CI_LOG = "ci_log"
    FAILURE_SIGNATURE = "failure_signature"
    CODE_SNIPPET = "code_snippet"
    SYMBOL = "symbol"
    GIT_LOG = "git_log"
    GIT_BLAME = "git_blame"
    GIT_DIFF = "git_diff"
    TEST_RESULT = "test_result"
    OTHER = "other"


class EvidenceRelation(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    CAUSED_BY = "caused_by"


class FindingStatus(str, Enum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


class UncertaintyLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ConfidenceLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class HypothesisDisposition(str, Enum):
    CANDIDATE = "candidate"
    SUPPORTED = "supported"
    REJECTED = "rejected"
    UNVERIFIED = "unverified"


class SourceLocation(BaseModel):
    path: str
    symbol: str | None = Field(default=None, max_length=MAX_SYMBOL_CHARS)
    start_line: int | None = Field(default=None, ge=1)
    end_line: int | None = Field(default=None, ge=1)

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        return validate_relative_path(value)

    @model_validator(mode="after")
    def _validate_line_range(self) -> SourceLocation:
        if (
            self.start_line is not None
            and self.end_line is not None
            and self.end_line < self.start_line
        ):
            raise ValueError("end_line must be >= start_line")
        return self


class EvidenceItem(BaseModel):
    id: StableId
    agent: AgentRole
    kind: EvidenceKind
    observation: str = Field(max_length=MAX_OBSERVATION_CHARS)
    provenance: Provenance
    location: SourceLocation | None = None
    excerpt: str = Field(max_length=MAX_EXCERPT_CHARS)
    commit_ids: list[str] = Field(default_factory=list, max_length=MAX_COMMIT_IDS)

    @field_validator("commit_ids")
    @classmethod
    def _validate_commit_ids(cls, values: list[str]) -> list[str]:
        normalized = [validate_commit_sha(value).lower() for value in values]
        if len(set(normalized)) != len(normalized):
            raise ValueError("evidence commit ids must be unique")
        return normalized

    @model_validator(mode="after")
    def _validate_commit_provenance(self) -> EvidenceItem:
        git_kinds = {
            EvidenceKind.GIT_LOG,
            EvidenceKind.GIT_DIFF,
            EvidenceKind.GIT_BLAME,
        }
        if self.commit_ids and (
            self.agent is not AgentRole.GIT_HISTORY or self.kind not in git_kinds
        ):
            raise ValueError(
                "commit ids are allowed only on Git History evidence"
            )
        return self


class EvidenceEdge(BaseModel):
    source: StableId
    target: StableId
    relation: EvidenceRelation

    @model_validator(mode="after")
    def _validate_endpoints(self) -> EvidenceEdge:
        if self.source == self.target:
            raise ValueError("evidence edge must connect distinct evidence items")
        return self


class AgentFinding(BaseModel):
    agent: AgentRole
    status: FindingStatus
    ranked_locations: list[SourceLocation] = Field(default_factory=list, max_length=MAX_LOCATIONS)
    evidence_ids: list[StableId] = Field(default_factory=list, max_length=MAX_EVIDENCE_IDS)
    uncertainty: UncertaintyLevel = UncertaintyLevel.LOW
    uncertainty_note: str | None = Field(default=None, max_length=MAX_NOTE_CHARS)
    timing_seconds: float | None = Field(default=None, ge=0)
    usage: Usage | None = None
    error: str | None = Field(default=None, max_length=MAX_NOTE_CHARS)


class VerificationStep(BaseModel):
    command: str = Field(max_length=MAX_COMMAND_CHARS)
    description: str | None = Field(default=None, max_length=MAX_NOTE_CHARS)
    timeout_seconds: int | None = Field(default=None, gt=0, le=3_600)
    expect_failure: bool = Field(
        default=False,
        description=(
            "True when the command should exit non-zero to confirm the "
            "hypothesis (e.g., reproducing the reported failure)."
        ),
    )


class Hypothesis(BaseModel):
    id: StableId
    statement: str = Field(max_length=MAX_STATEMENT_CHARS)
    locations: list[SourceLocation] = Field(default_factory=list, max_length=MAX_LOCATIONS)
    supporting_evidence_ids: list[StableId] = Field(default_factory=list, max_length=MAX_EVIDENCE_IDS)
    contradicting_evidence_ids: list[StableId] = Field(default_factory=list, max_length=MAX_EVIDENCE_IDS)
    verification_plan: list[VerificationStep] = Field(default_factory=list, max_length=MAX_STEPS)
    confidence: ConfidenceLevel = ConfidenceLevel.LOW
    disposition: HypothesisDisposition = HypothesisDisposition.CANDIDATE
