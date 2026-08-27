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
from roottrace.evidence.graph import EvidenceGraph
from roottrace.evidence.schema import HypothesisDisposition
from roottrace.llm.usage import UsageTracker
from roottrace.reporting.schema import RCAReport
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
            try:
                data = extract_json_object(turn.content)
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
                        "usage": self._usage.snapshot(),
                    }
                )
            except (ValueError, ValidationError) as exc:
                last_error = exc
                if attempt < self._max_attempts - 1:
                    continue
                raise SynthesisError(
                    f"malformed final synthesis output: {exc}"
                ) from exc
            break

        if report is None:
            raise SynthesisError("final synthesis produced no valid report")
        dispositions = {
            hypothesis.id: hypothesis.disposition
            for hypothesis in graph.hypotheses
        }
        for cause in report.ranked_causes:
            if (
                dispositions.get(cause.hypothesis_id)
                is HypothesisDisposition.REJECTED
            ):
                raise SynthesisError(
                    f"ranked cause {cause.rank} references rejected "
                    f"hypothesis {cause.hypothesis_id}"
                )
        return report
