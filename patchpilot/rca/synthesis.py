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

from patchpilot.rca.agents import ProviderProtocol, extract_json_object
from patchpilot.rca.prompts import (
    FINAL_REPORT_SYSTEM_PROMPT,
    build_final_report_prompt,
)
from patchpilot.rca.schema import EvidenceGraph, RCAReport
from patchpilot.rca.usage import UsageTracker
from patchpilot.rca.verification import VerificationRun


class SynthesisError(Exception):
    """Raised when the final synthesis output is unusable."""


class LeadSynthesizer:
    """Generate the final RCA report from graph plus verification."""

    def __init__(
        self,
        *,
        provider: ProviderProtocol,
        usage: UsageTracker,
    ) -> None:
        self._provider = provider
        self._usage = usage

    def synthesize(
        self,
        graph: EvidenceGraph,
        verification: VerificationRun,
    ) -> RCAReport:
        """Produce a validated final RCA report."""
        prompt = build_final_report_prompt(graph, verification.results)
        turn = self._provider.complete(
            messages=[
                {"role": "system", "content": FINAL_REPORT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            tools=[],
        )
        self._usage.record(turn.prompt_tokens, turn.completion_tokens)
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
            raise SynthesisError(
                f"malformed final synthesis output: {exc}"
            ) from exc
        return report
