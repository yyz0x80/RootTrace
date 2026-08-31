"""Final RCA synthesis from evidence and verification results.

``LeadSynthesizer`` asks the Lead to produce the final evidence-backed
``RCAReport``: ranked causes, causal chain, suspected regression (only when
supported), and an advisory, non-executable fix recommendation. The wrapper
injects the verified evidence graph, verification results, timing, and usage;
the model cannot invent graph references because ``RCAReport`` validation
rejects unknown evidence/hypothesis ids and embedded patch content.
"""

from __future__ import annotations

from pydantic import ValidationError

from roottrace.agents.prompts import (
    FINAL_REPORT_SYSTEM_PROMPT,
    build_final_report_prompt,
)
from roottrace.agents.specialists import ProviderProtocol, extract_json_object
from roottrace.diagnostics import DiagnosticSeverity, PipelineDiagnostic
from roottrace.evidence.graph import EvidenceGraph
from roottrace.evidence.schema import (
    MAX_LOCATIONS,
    AgentRole,
    HypothesisDisposition,
    SourceLocation,
)
from roottrace.llm.usage import UsageTracker
from roottrace.reporting.schema import (
    RCAReport,
    ReportConclusion,
    has_qualified_git_regression_evidence,
)
from roottrace.verification.schema import VerificationOutcome
from roottrace.verification.verifier import VerificationRun


class SynthesisError(Exception):
    """Raised when the final synthesis output is unusable."""


MAX_SYNTHESIS_ATTEMPTS = 2
MAX_FEEDBACK_CHARS = 2_000


def _repair_feedback(error: Exception) -> str:
    """Build bounded validation feedback for one synthesis repair retry."""
    text = str(error)
    if len(text) > MAX_FEEDBACK_CHARS:
        text = text[:MAX_FEEDBACK_CHARS] + "..."
    return (
        "\n\nPREVIOUS OUTPUT REJECTED:\n"
        "Fix exactly these validation errors and output the corrected JSON "
        f"object only:\n{text}"
    )


def _normalize_suspected_regression(
    data: dict[str, object],
    graph: EvidenceGraph,
) -> dict[str, object]:
    """Normalize unsupported object-shaped regression claims to ``None``."""
    normalized = dict(data)
    regression = normalized.get("suspected_regression")
    if regression == {} or (
        isinstance(regression, dict)
        and not has_qualified_git_regression_evidence(graph)
    ):
        normalized["suspected_regression"] = None
    return normalized


def _has_missing_regression_evidence_ids(
    data: dict[str, object] | None,
) -> bool:
    """Return whether a regression object omitted its required evidence ids."""
    if data is None:
        return False
    regression = data.get("suspected_regression")
    if not isinstance(regression, dict) or not regression:
        return False
    evidence_ids = regression.get("evidence_ids")
    return evidence_ids is None or evidence_ids == []


def _validate_supported_causes(report: RCAReport) -> None:
    """Require runtime support and linked evidence for confirmed causes."""
    if report.conclusion is not ReportConclusion.ROOT_CAUSE_IDENTIFIED:
        return

    supported_evidence_by_hypothesis: dict[str, set[str]] = {}
    for result in report.verification:
        if result.outcome is VerificationOutcome.SUPPORTED:
            supported_evidence_by_hypothesis.setdefault(
                result.hypothesis_id,
                set(),
            ).update(result.evidence_ids)

    for cause in report.ranked_causes:
        supported_evidence = supported_evidence_by_hypothesis.get(
            cause.hypothesis_id,
            set(),
        )
        if not supported_evidence:
            raise ValueError(
                f"RankedCause {cause.rank} requires a supported verification "
                f"result for hypothesis {cause.hypothesis_id}"
            )
        if not supported_evidence.intersection(cause.evidence_ids):
            raise ValueError(
                f"RankedCause {cause.rank} must cite evidence from a supported "
                "verification result"
            )


def _validate_ranked_causes(report: RCAReport) -> None:
    """Reject final causes that point at hypotheses already disproved."""
    dispositions = {
        hypothesis.id: hypothesis.disposition
        for hypothesis in report.evidence_graph.hypotheses
    }
    for cause in report.ranked_causes:
        if dispositions.get(cause.hypothesis_id) is HypothesisDisposition.REJECTED:
            raise SynthesisError(
                f"ranked cause {cause.rank} references rejected "
                f"hypothesis {cause.hypothesis_id}"
            )


def _fallback_top_k_locations(graph: EvidenceGraph) -> list[SourceLocation]:
    """Collect stable evidence-grounded locations when synthesis omits them."""
    candidates: list[SourceLocation] = []
    for hypothesis in graph.hypotheses:
        if hypothesis.disposition is not HypothesisDisposition.REJECTED:
            candidates.extend(hypothesis.locations)

    candidates.extend(
        location
        for finding in graph.findings
        if finding.agent is AgentRole.CODE
        for location in finding.ranked_locations
    )
    candidates.extend(
        location
        for finding in graph.findings
        if finding.agent is not AgentRole.CODE
        for location in finding.ranked_locations
    )

    locations: list[SourceLocation] = []
    seen_paths: set[str] = set()
    for location in candidates:
        if location.path in seen_paths:
            continue
        seen_paths.add(location.path)
        locations.append(location)
        if len(locations) >= MAX_LOCATIONS:
            break
    return locations


def _build_report(
    data: dict[str, object],
    graph: EvidenceGraph,
    verification: VerificationRun,
    usage: UsageTracker,
) -> RCAReport:
    """Build and fully validate one candidate final report."""
    report = RCAReport.model_validate(
        {
            **data,
            "id": f"rca-{graph.incident.id}",
            "incident_id": graph.incident.id,
            "evidence_graph": graph,
            "verification": [
                result.model_dump(mode="json")
                for result in verification.results
            ],
            "timing": {
                "total_seconds": None,
                "model_seconds": None,
                "verification_seconds": verification.timing_seconds,
            },
            "usage": usage.snapshot(),
        }
    )
    _validate_ranked_causes(report)
    _validate_supported_causes(report)
    return report


class LeadSynthesizer:
    """Generate the final RCA report from graph plus verification."""

    def __init__(
        self,
        *,
        provider: ProviderProtocol,
        usage: UsageTracker,
        max_attempts: int = MAX_SYNTHESIS_ATTEMPTS,
    ) -> None:
        if not 1 <= max_attempts <= 3:
            raise ValueError("max_attempts must be between 1 and 3")
        self._provider = provider
        self._usage = usage
        self._max_attempts = max_attempts
        self._diagnostics: list[PipelineDiagnostic] = []

    @property
    def diagnostics(self) -> list[PipelineDiagnostic]:
        """Return diagnostics emitted while producing the final report."""
        return list(self._diagnostics)

    def _try_regression_salvage(
        self,
        data: dict[str, object] | None,
        graph: EvidenceGraph,
        verification: VerificationRun,
    ) -> RCAReport | None:
        """Drop only an invalid regression claim when all other fields validate."""
        if not _has_missing_regression_evidence_ids(data):
            return None
        assert data is not None
        candidate = dict(data)
        candidate["suspected_regression"] = None
        try:
            report = _build_report(candidate, graph, verification, self._usage)
        except (SynthesisError, TypeError, ValueError, ValidationError):
            return None
        self._diagnostics.append(
            PipelineDiagnostic(
                code="synthesis.suspected_regression_removed",
                stage="synthesis",
                severity=DiagnosticSeverity.RECOVERABLE,
                message=(
                    "Invalid suspected_regression was removed after "
                    "synthesis repair failed"
                ),
                agent=AgentRole.LEAD,
            )
        )
        return report

    def synthesize(
        self,
        graph: EvidenceGraph,
        verification: VerificationRun,
    ) -> RCAReport:
        """Produce a validated final RCA report.

        One bounded repair retry is allowed: when the model output fails
        validation, the exact error is appended as feedback and the call is
        retried once. Persistent failures still raise ``SynthesisError`` so
        malformed outputs stay explicit and usage stays honest.
        """
        self._diagnostics = []
        prompt = build_final_report_prompt(graph, verification.results)
        report: RCAReport | None = None
        last_error: Exception | None = None
        for attempt in range(self._max_attempts):
            if attempt > 0:
                prompt = prompt + _repair_feedback(last_error)
            turn = self._provider.complete(
                messages=[
                    {"role": "system", "content": FINAL_REPORT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                tools=[],
            )
            self._usage.record(
                turn.prompt_tokens,
                turn.completion_tokens,
                turn.reasoning_tokens,
            )
            if turn.content is None or not turn.content.strip():
                raise SynthesisError("final synthesis returned no content")
            data: dict[str, object] | None = None
            try:
                data = extract_json_object(turn.content)
                if not isinstance(data, dict):
                    raise TypeError("final synthesis output must be a JSON object")
                data = _normalize_suspected_regression(data, graph)
                report = _build_report(data, graph, verification, self._usage)
            except (TypeError, ValueError, ValidationError) as exc:
                last_error = exc
                if attempt < self._max_attempts - 1:
                    continue
                if attempt > 0:
                    report = self._try_regression_salvage(
                        data,
                        graph,
                        verification,
                    )
                    if report is not None:
                        break
                raise SynthesisError(
                    f"malformed final synthesis output: {exc}"
                ) from exc
            break

        if report is None:
            raise SynthesisError("final synthesis produced no valid report")
        if not report.top_k_locations:
            report = report.model_copy(
                update={
                    "top_k_locations": _fallback_top_k_locations(
                        report.evidence_graph
                    )
                }
            )
        return report
