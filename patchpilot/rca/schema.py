"""Typed RCA data contracts for RootTrace.

Pydantic models persisted across RCA artifacts. Guarantees:

- every evidence item has a stable ID and reproducible provenance;
- all source paths are repository-relative and validated;
- supporting/contradicting evidence references are explicit;
- ``EvidenceGraph`` rejects dangling references and duplicate IDs;
- ``RCAReport`` rejects references to unknown hypotheses/evidence;
- ``FixRecommendation`` is advisory text only and cannot carry a patch.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from enum import Enum
from pathlib import PurePosixPath
from typing import Annotated

from pydantic import (
    BaseModel,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

# ---------------------------------------------------------------------------
# Size budgets: every persisted string and collection is bounded.
# ---------------------------------------------------------------------------

MAX_TITLE_CHARS = 200
MAX_PROBLEM_CHARS = 20_000
MAX_LOG_CHARS = 20_000
MAX_DIFF_CHARS = 100_000
MAX_SOURCE_CHARS = 1_000
MAX_TOOL_CHARS = 200
MAX_COMMAND_CHARS = 500
MAX_SYMBOL_CHARS = 512
MAX_EXCERPT_CHARS = 8_000
MAX_OBSERVATION_CHARS = 2_000
MAX_QUESTION_CHARS = 500
MAX_STATEMENT_CHARS = 2_000
MAX_NOTE_CHARS = 500
MAX_PROPOSAL_CHARS = 4_000

MAX_QUESTIONS = 10
MAX_LOCATIONS = 10
MAX_EVIDENCE_IDS = 50
MAX_STEPS = 5
MAX_LOGS = 10
MAX_CAUSES = 5
MAX_LINKS = 5
MAX_NOTES = 10
MAX_SUGGESTIONS = 5
MAX_FINDINGS = 10
MAX_GRAPH_EVIDENCE = 200
MAX_GRAPH_HYPOTHESES = 20
MAX_GRAPH_EDGES = 500
MAX_VERIFICATION_RESULTS = 10


StableId = Annotated[
    str,
    StringConstraints(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$",
        max_length=128,
    ),
]
BoundedLog = Annotated[str, StringConstraints(max_length=MAX_LOG_CHARS)]
BoundedNote = Annotated[str, StringConstraints(max_length=MAX_NOTE_CHARS)]
BoundedSuggestion = Annotated[str, StringConstraints(max_length=MAX_PROPOSAL_CHARS)]

_SHA_PATTERN = re.compile(r"^[0-9a-fA-F]{7,64}$")

# Markers that indicate embedded executable patch content or commands that
# would modify the analyzed repository. FixRecommendation must stay advisory.
_FORBIDDEN_PATCH_PREFIXES = ("diff --git", "--- a/", "+++ b/", "@@ ")
_FORBIDDEN_MODIFICATION_COMMANDS = (
    "git apply",
    "git am",
    "git reset --hard",
    "git checkout --",
    "patch -p",
    "apply_patch",
)


def validate_relative_path(value: str) -> str:
    """Validate and normalize a repository-relative path.

    Rejects absolute paths, ``.``/``..`` segments, backslashes, and home
    shortcuts so persisted locations never leak host paths.
    """
    if not value or not value.strip():
        raise ValueError("repository-relative path must not be empty")
    if "\\" in value:
        raise ValueError("repository-relative path must use forward slashes")
    if value.startswith("~"):
        raise ValueError("repository-relative path must not start with '~'")
    try:
        path = PurePosixPath(value)
    except ValueError as exc:
        raise ValueError("invalid repository-relative path") from exc
    if path.is_absolute():
        raise ValueError("repository-relative path must not be absolute")
    if not path.parts or any(part in (".", "..") for part in path.parts):
        raise ValueError(
            "repository-relative path must not contain '.' or '..' segments"
        )
    return path.as_posix()


def _validate_commit_sha(value: str) -> str:
    if not _SHA_PATTERN.fullmatch(value):
        raise ValueError("commit must be a 7-64 character hexadecimal SHA")
    return value


def _unique_ids(ids: Iterable[str], label: str) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for item in ids:
        if item in seen:
            duplicates.add(item)
        seen.add(item)
    if duplicates:
        raise ValueError(f"duplicate {label} ids: {sorted(duplicates)}")
    return seen


def _missing_ids(ids: Iterable[str], known: set[str]) -> list[str]:
    return sorted({item for item in ids if item not in known})


# ---------------------------------------------------------------------------
# Roles, kinds, and status enums
# ---------------------------------------------------------------------------


class AgentRole(str, Enum):
    """Agents that may produce findings or evidence in an RCA run."""

    LEAD = "lead"
    ISSUE_CI = "issue_ci"
    CODE = "code"
    GIT_HISTORY = "git_history"
    RUNTIME_TEST = "runtime_test"


class EvidenceKind(str, Enum):
    """Typed evidence categories used by the RCA specialists."""

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
    """Relation between two evidence items in the evidence graph."""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    CAUSED_BY = "caused_by"


class FindingStatus(str, Enum):
    """Status of a specialist's evidence-gathering run."""

    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


class UncertaintyLevel(str, Enum):
    """Explicit uncertainty carried by findings and the report."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ConfidenceLevel(str, Enum):
    """Ranking of hypothesis/cause confidence.

    Confidence ranks evidence; it never replaces evidence.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class HypothesisDisposition(str, Enum):
    """Lifecycle of a hypothesis during an RCA run."""

    CANDIDATE = "candidate"
    SUPPORTED = "supported"
    REJECTED = "rejected"
    UNVERIFIED = "unverified"


class VerificationOutcome(str, Enum):
    """Semantic conclusion of a runtime verification run."""

    SUPPORTED = "supported"
    REJECTED = "rejected"
    UNVERIFIED = "unverified"


class VerificationStatus(str, Enum):
    """Raw status of the verification command itself."""

    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"


class ReportConclusion(str, Enum):
    """Final report conclusion."""

    ROOT_CAUSE_IDENTIFIED = "root_cause_identified"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


# ---------------------------------------------------------------------------
# Core contracts
# ---------------------------------------------------------------------------


class Provenance(BaseModel):
    """Reproducible origin of an evidence item or incident input."""

    source: str = Field(max_length=MAX_SOURCE_CHARS)
    tool: str | None = Field(default=None, max_length=MAX_TOOL_CHARS)
    command: str | None = Field(default=None, max_length=MAX_COMMAND_CHARS)
    commit: str | None = None

    @field_validator("commit")
    @classmethod
    def _validate_commit(cls, value: str | None) -> str | None:
        if value is not None:
            _validate_commit_sha(value)
        return value


class SourceLocation(BaseModel):
    """A repository-relative source location with optional symbol/line range."""

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


class IncidentInput(BaseModel):
    """Normalized incident input for one RCA run."""

    id: StableId
    repo: str = Field(max_length=MAX_SOURCE_CHARS)
    base_commit: str
    title: str | None = Field(default=None, max_length=MAX_TITLE_CHARS)
    problem: str = Field(max_length=MAX_PROBLEM_CHARS)
    logs: list[BoundedLog] = Field(default_factory=list, max_length=MAX_LOGS)
    diff: str | None = Field(default=None, max_length=MAX_DIFF_CHARS)
    provenance: Provenance

    @field_validator("repo")
    @classmethod
    def _validate_repo(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("repo must be a non-empty repository identifier")
        if "\\" in value or re.match(r"^[A-Za-z]:", value):
            raise ValueError("repo must be a repository identifier, not a host path")
        try:
            path = PurePosixPath(value)
        except ValueError as exc:
            raise ValueError("invalid repository identifier") from exc
        if (
            path.is_absolute()
            or not path.parts
            or any(part in (".", "..") for part in path.parts)
        ):
            raise ValueError("repo must not contain '.' or '..' segments")
        return value

    @field_validator("base_commit")
    @classmethod
    def _validate_base_commit(cls, value: str) -> str:
        return _validate_commit_sha(value)


class EvidenceItem(BaseModel):
    """One typed, provenance-backed piece of RCA evidence."""

    id: StableId
    agent: AgentRole
    kind: EvidenceKind
    observation: str = Field(max_length=MAX_OBSERVATION_CHARS)
    provenance: Provenance
    location: SourceLocation | None = None
    excerpt: str = Field(max_length=MAX_EXCERPT_CHARS)


class EvidenceEdge(BaseModel):
    """Directed relation between two evidence items."""

    source: StableId
    target: StableId
    relation: EvidenceRelation

    @model_validator(mode="after")
    def _validate_endpoints(self) -> EvidenceEdge:
        if self.source == self.target:
            raise ValueError("evidence edge must connect distinct evidence items")
        return self


class Usage(BaseModel):
    """Exact or null token usage; null means the provider did not return it."""

    llm_calls: int = Field(default=0, ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)


class AgentFinding(BaseModel):
    """Typed output of one evidence-gathering specialist."""

    agent: AgentRole
    status: FindingStatus
    ranked_locations: list[SourceLocation] = Field(
        default_factory=list,
        max_length=MAX_LOCATIONS,
    )
    evidence_ids: list[StableId] = Field(
        default_factory=list,
        max_length=MAX_EVIDENCE_IDS,
    )
    uncertainty: UncertaintyLevel = UncertaintyLevel.LOW
    uncertainty_note: str | None = Field(default=None, max_length=MAX_NOTE_CHARS)
    timing_seconds: float | None = Field(default=None, ge=0)
    usage: Usage | None = None
    error: str | None = Field(default=None, max_length=MAX_NOTE_CHARS)


class PlanQuestion(BaseModel):
    """One investigation question assigned to one or more specialists."""

    id: StableId
    text: str = Field(max_length=MAX_QUESTION_CHARS)
    assigned_agents: list[AgentRole] = Field(
        default_factory=list,
        min_length=1,
        max_length=3,
    )


class PlanBudgets(BaseModel):
    """Bounded reasoning budgets for the investigation."""

    max_llm_calls: int = Field(default=7, gt=0, le=50)
    max_evidence_items: int = Field(default=50, gt=0, le=500)
    max_tool_calls: int = Field(default=50, gt=0, le=500)
    timeout_seconds: int = Field(default=120, gt=0, le=3_600)


class InvestigationPlan(BaseModel):
    """Lead-generated investigation plan; never a premature final cause."""

    id: StableId
    incident_id: StableId
    questions: list[PlanQuestion] = Field(default_factory=list, max_length=MAX_QUESTIONS)
    budgets: PlanBudgets = Field(default_factory=PlanBudgets)
    assignments: dict[AgentRole, list[str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _finalize_assignments(self) -> InvestigationPlan:
        seen: set[str] = set()
        for question in self.questions:
            if question.id in seen:
                raise ValueError(f"duplicate plan question id: {question.id}")
            seen.add(question.id)
        assignments: dict[AgentRole, list[str]] = {}
        for question in self.questions:
            for agent in question.assigned_agents:
                assignments.setdefault(agent, []).append(question.id)
        self.assignments = assignments
        return self


class VerificationStep(BaseModel):
    """One bounded, sandbox-runnable verification action for a hypothesis."""

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
    """A falsifiable root-cause hypothesis with explicit evidence links."""

    id: StableId
    statement: str = Field(max_length=MAX_STATEMENT_CHARS)
    locations: list[SourceLocation] = Field(
        default_factory=list,
        max_length=MAX_LOCATIONS,
    )
    supporting_evidence_ids: list[StableId] = Field(
        default_factory=list,
        max_length=MAX_EVIDENCE_IDS,
    )
    contradicting_evidence_ids: list[StableId] = Field(
        default_factory=list,
        max_length=MAX_EVIDENCE_IDS,
    )
    verification_plan: list[VerificationStep] = Field(
        default_factory=list,
        max_length=MAX_STEPS,
    )
    confidence: ConfidenceLevel = ConfidenceLevel.LOW
    disposition: HypothesisDisposition = HypothesisDisposition.CANDIDATE


class EvidenceGraph(BaseModel):
    """Validated, auditable shared state for one RCA run."""

    incident: IncidentInput
    findings: list[AgentFinding] = Field(default_factory=list, max_length=MAX_FINDINGS)
    evidence: list[EvidenceItem] = Field(
        default_factory=list,
        max_length=MAX_GRAPH_EVIDENCE,
    )
    hypotheses: list[Hypothesis] = Field(
        default_factory=list,
        max_length=MAX_GRAPH_HYPOTHESES,
    )
    edges: list[EvidenceEdge] = Field(
        default_factory=list,
        max_length=MAX_GRAPH_EDGES,
    )

    @model_validator(mode="after")
    def _validate_references(self) -> EvidenceGraph:
        evidence_ids = _unique_ids((item.id for item in self.evidence), "evidence")
        _unique_ids((hypothesis.id for hypothesis in self.hypotheses), "hypothesis")

        for finding in self.findings:
            missing = _missing_ids(finding.evidence_ids, evidence_ids)
            if missing:
                raise ValueError(
                    f"AgentFinding for {finding.agent.value} references unknown "
                    f"evidence ids: {missing}"
                )

        for hypothesis in self.hypotheses:
            referenced = (
                *hypothesis.supporting_evidence_ids,
                *hypothesis.contradicting_evidence_ids,
            )
            missing = _missing_ids(referenced, evidence_ids)
            if missing:
                raise ValueError(
                    f"Hypothesis {hypothesis.id} references unknown evidence ids: "
                    f"{missing}"
                )

        edge_keys: set[tuple[str, str, str]] = set()
        for edge in self.edges:
            missing = _missing_ids((edge.source, edge.target), evidence_ids)
            if missing:
                raise ValueError(
                    f"EvidenceEdge references unknown evidence ids: {missing}"
                )
            key = (edge.source, edge.target, edge.relation.value)
            if key in edge_keys:
                raise ValueError(f"duplicate evidence edge: {key}")
            edge_keys.add(key)
        return self


class RankedCause(BaseModel):
    """A ranked cause selection from the final Lead synthesis."""

    rank: int = Field(ge=1)
    hypothesis_id: StableId
    confidence: ConfidenceLevel
    rationale: str | None = Field(default=None, max_length=MAX_NOTE_CHARS)
    evidence_ids: list[StableId] = Field(
        default_factory=list,
        max_length=MAX_EVIDENCE_IDS,
    )


class CauseLink(BaseModel):
    """One step of the causal chain, tied to evidence."""

    statement: str = Field(max_length=MAX_STATEMENT_CHARS)
    hypothesis_id: StableId | None = None
    evidence_ids: list[StableId] = Field(
        default_factory=list,
        max_length=MAX_EVIDENCE_IDS,
    )


class VerificationResult(BaseModel):
    """Outcome of verifying one hypothesis in the disposable sandbox."""

    id: StableId
    hypothesis_id: StableId
    command: str = Field(max_length=MAX_COMMAND_CHARS)
    status: VerificationStatus
    outcome: VerificationOutcome
    evidence_ids: list[StableId] = Field(
        default_factory=list,
        max_length=MAX_EVIDENCE_IDS,
    )
    output_excerpt: str | None = Field(default=None, max_length=MAX_EXCERPT_CHARS)
    exit_code: int | None = None
    duration_seconds: float | None = Field(default=None, ge=0)


class RegressionChange(BaseModel):
    """A suspected regression commit, reported only when supported."""

    commit: str
    summary: str | None = Field(default=None, max_length=MAX_STATEMENT_CHARS)
    evidence_ids: list[StableId] = Field(
        default_factory=list,
        max_length=MAX_EVIDENCE_IDS,
    )
    locations: list[SourceLocation] = Field(
        default_factory=list,
        max_length=MAX_LOCATIONS,
    )

    @field_validator("commit")
    @classmethod
    def _validate_commit(cls, value: str) -> str:
        return _validate_commit_sha(value)


def assert_advisory_text(value: str) -> str:
    """Reject text that embeds executable patch content or edit commands."""
    lines = value.splitlines()
    for index, line in enumerate(lines, start=1):
        stripped = line.lstrip()
        if stripped.startswith(_FORBIDDEN_PATCH_PREFIXES):
            raise ValueError(
                f"fix recommendation must be advisory text; line {index} looks "
                "like an embedded patch"
            )
        if stripped.startswith("\\ No newline"):
            raise ValueError(
                f"fix recommendation must be advisory text; line {index} is a "
                "patch artifact"
            )
        lowered = line.lower()
        if any(command in lowered for command in _FORBIDDEN_MODIFICATION_COMMANDS):
            raise ValueError(
                f"fix recommendation must not contain code modification commands "
                f"(line {index})"
            )
    return value


class FixRecommendation(BaseModel):
    """Advisory fix scope and suggestions; never an executable patch."""

    scope: str | None = Field(default=None, max_length=MAX_STATEMENT_CHARS)
    suggestions: list[BoundedSuggestion] = Field(
        default_factory=list,
        max_length=MAX_SUGGESTIONS,
    )
    locations: list[SourceLocation] = Field(
        default_factory=list,
        max_length=MAX_LOCATIONS,
    )
    evidence_ids: list[StableId] = Field(
        default_factory=list,
        max_length=MAX_EVIDENCE_IDS,
    )

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
    """Final report uncertainty and insufficiency summary."""

    level: UncertaintyLevel = UncertaintyLevel.LOW
    insufficient_evidence: bool = False
    notes: list[BoundedNote] = Field(default_factory=list, max_length=MAX_NOTES)


class Timing(BaseModel):
    """Separate wall-clock, model-call, and verification durations."""

    total_seconds: float | None = Field(default=None, ge=0)
    model_seconds: float | None = Field(default=None, ge=0)
    verification_seconds: float | None = Field(default=None, ge=0)


class RCAReport(BaseModel):
    """Final evidence-backed RCA report with an embedded, validated graph."""

    id: StableId
    incident_id: StableId
    evidence_graph: EvidenceGraph
    conclusion: ReportConclusion
    conclusion_summary: str | None = Field(default=None, max_length=MAX_STATEMENT_CHARS)
    ranked_causes: list[RankedCause] = Field(
        default_factory=list,
        max_length=MAX_CAUSES,
    )
    top_k_locations: list[SourceLocation] = Field(
        default_factory=list,
        max_length=MAX_LOCATIONS,
    )
    causal_chain: list[CauseLink] = Field(
        default_factory=list,
        max_length=MAX_LINKS,
    )
    verification: list[VerificationResult] = Field(
        default_factory=list,
        max_length=MAX_VERIFICATION_RESULTS,
    )
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
            raise ValueError(
                "conclusion root_cause_identified requires at least one ranked cause"
            )

        ranks = [cause.rank for cause in self.ranked_causes]
        if len(set(ranks)) != len(ranks):
            raise ValueError("ranked cause ranks must be unique")

        for cause in self.ranked_causes:
            if cause.hypothesis_id not in hypothesis_ids:
                raise ValueError(
                    f"RankedCause references unknown hypothesis: {cause.hypothesis_id}"
                )
            missing = _missing_ids(cause.evidence_ids, evidence_ids)
            if missing:
                raise ValueError(
                    f"RankedCause {cause.rank} references unknown evidence ids: "
                    f"{missing}"
                )

        for link in self.causal_chain:
            if link.hypothesis_id is not None and link.hypothesis_id not in hypothesis_ids:
                raise ValueError(
                    f"CauseLink references unknown hypothesis: {link.hypothesis_id}"
                )
            missing = _missing_ids(link.evidence_ids, evidence_ids)
            if missing:
                raise ValueError(
                    f"CauseLink references unknown evidence ids: {missing}"
                )

        verification_ids: set[str] = set()
        for result in self.verification:
            if result.id in verification_ids:
                raise ValueError(f"duplicate verification result id: {result.id}")
            verification_ids.add(result.id)
            if result.hypothesis_id not in hypothesis_ids:
                raise ValueError(
                    f"VerificationResult references unknown hypothesis: "
                    f"{result.hypothesis_id}"
                )
            missing = _missing_ids(result.evidence_ids, evidence_ids)
            if missing:
                raise ValueError(
                    f"VerificationResult {result.id} references unknown evidence "
                    f"ids: {missing}"
                )

        if self.suspected_regression is not None:
            missing = _missing_ids(
                self.suspected_regression.evidence_ids,
                evidence_ids,
            )
            if missing:
                raise ValueError(
                    "suspected_regression references unknown evidence ids: "
                    f"{missing}"
                )

        if self.fix_recommendation is not None:
            missing = _missing_ids(
                self.fix_recommendation.evidence_ids,
                evidence_ids,
            )
            if missing:
                raise ValueError(
                    "fix_recommendation references unknown evidence ids: "
                    f"{missing}"
                )
        return self
