"""Runtime test verification for RootTrace RCA hypotheses.

``RuntimeTestVerifier`` is the only RCA component allowed to execute test
commands, and it executes them only inside the ephemeral verification sandbox.
Commands come from each hypothesis's verification plan; the sandbox enforces
the parsed ``python -m pytest`` allowlist with ``shell=False``, so no general
shell is ever exposed.

Outcome rules are deterministic:
- every executed step passes (exit 0, no timeout) -> hypothesis ``supported``;
- any executed step fails (non-zero exit) -> hypothesis ``rejected``;
- any executed step errors or times out -> hypothesis ``unverified``;
- a hypothesis without executed steps stays ``unverified``.

Every executed command is recorded as a ``TEST_RESULT`` evidence item with
reproducible provenance (command, commit, bounded output). The original target
repository is never modified; the sandbox copy is disposable.
"""

from __future__ import annotations

import shlex
import time

from pydantic import BaseModel, Field

from roottrace.evidence.graph import EvidenceGraph
from roottrace.evidence.schema import (
    MAX_EXCERPT_CHARS,
    AgentRole,
    EvidenceItem,
    EvidenceKind,
    Hypothesis,
    HypothesisDisposition,
    VerificationStep,
)
from roottrace.incident.schema import Provenance
from roottrace.runtime.sandbox import (
    RuntimeVerificationSandbox,
    SandboxCommandResult,
)
from roottrace.verification.schema import (
    VerificationOutcome,
    VerificationResult,
    VerificationStatus,
)

DEFAULT_STEP_TIMEOUT_SECONDS = 120


def _cap(text: str, limit: int = MAX_EXCERPT_CHARS) -> str:
    """Cap text to ``limit`` characters including the truncation marker."""
    if len(text) <= limit:
        return text
    keep = max(0, limit - len("\n...[truncated]"))
    return text[:keep] + "\n...[truncated]"


class VerificationRun(BaseModel):
    """Typed output of one verification pass over the evidence graph."""

    graph: EvidenceGraph
    results: list[VerificationResult] = Field(
        default_factory=list,
        max_length=10,
    )
    evidence: list[EvidenceItem] = Field(
        default_factory=list,
        max_length=25,
    )
    timing_seconds: float | None = Field(default=None, ge=0)


class RuntimeTestVerifier:
    """Execute hypothesis verification plans inside the disposable sandbox."""

    def __init__(self, sandbox: RuntimeVerificationSandbox) -> None:
        self.sandbox = sandbox

    def verify(
        self,
        graph: EvidenceGraph,
        *,
        max_steps: int = 10,
    ) -> VerificationRun:
        """Verify every hypothesis plan and return an updated graph."""
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        started = time.monotonic()
        self._seen_evidence_ids: set[str] = set()
        self._seen_result_ids: set[str] = set()
        results: list[VerificationResult] = []
        evidence: list[EvidenceItem] = []
        dispositions: dict[str, HypothesisDisposition] = {}
        remaining = max_steps

        for hypothesis in graph.hypotheses:
            step_outcomes: list[VerificationOutcome] = []
            for step in hypothesis.verification_plan:
                if remaining <= 0:
                    break
                remaining -= 1
                result, item = self._run_step(hypothesis, step)
                results.append(result)
                if item is not None:
                    evidence.append(item)
                step_outcomes.append(result.outcome)
            dispositions[hypothesis.id] = _disposition_for(step_outcomes)

        updated_hypotheses = [
            hypothesis.model_copy(
                update={
                    "disposition": dispositions.get(
                        hypothesis.id,
                        HypothesisDisposition.UNVERIFIED,
                    )
                }
            )
            for hypothesis in graph.hypotheses
        ]
        updated_graph = graph.model_copy(
            update={
                "evidence": [*graph.evidence, *evidence],
                "hypotheses": updated_hypotheses,
            }
        )
        return VerificationRun(
            graph=updated_graph,
            results=results,
            evidence=evidence,
            timing_seconds=round(time.monotonic() - started, 3),
        )

    def _run_step(
        self,
        hypothesis: Hypothesis,
        step: VerificationStep,
    ) -> tuple[VerificationResult, EvidenceItem | None]:
        command = step.command
        try:
            argv = shlex.split(command)
            sandbox_result = self.sandbox.run(
                argv,
                timeout_seconds=step.timeout_seconds or DEFAULT_STEP_TIMEOUT_SECONDS,
            )
        except (ValueError, RuntimeError) as exc:
            return self._rejected_command(hypothesis, command, str(exc)), None
        return self._record_command(
            hypothesis,
            command,
            sandbox_result,
            expect_failure=step.expect_failure,
        )

    def _record_command(
        self,
        hypothesis: Hypothesis,
        command: str,
        sandbox_result: SandboxCommandResult,
        *,
        expect_failure: bool,
    ) -> tuple[VerificationResult, EvidenceItem]:
        exit_code = sandbox_result.exit_code
        timed_out = sandbox_result.timed_out
        if timed_out:
            status = VerificationStatus.ERROR
            outcome = VerificationOutcome.UNVERIFIED
        elif exit_code == 0:
            status = VerificationStatus.PASSED
            outcome = (
                VerificationOutcome.REJECTED
                if expect_failure
                else VerificationOutcome.SUPPORTED
            )
        else:
            status = VerificationStatus.FAILED
            outcome = (
                VerificationOutcome.SUPPORTED
                if expect_failure
                else VerificationOutcome.REJECTED
            )
        output = "\n".join(
            part for part in (sandbox_result.stdout, sandbox_result.stderr) if part
        )
        item = EvidenceItem(
            id=f"ev-runtime_test-{len(self._seen_evidence_ids) + 1:03d}",
            agent=AgentRole.RUNTIME_TEST,
            kind=EvidenceKind.TEST_RESULT,
            observation=(
                f"{command} exited {exit_code}"
                if not timed_out
                else f"{command} timed out"
            ),
            provenance=Provenance(
                source="verification_sandbox",
                tool="runtime_test",
                command=command,
                commit=self.sandbox.head_sha,
            ),
            excerpt=_cap(output),
        )
        self._seen_evidence_ids.add(item.id)
        result = VerificationResult(
            id=f"ver-{hypothesis.id}-{len(self._seen_result_ids) + 1:03d}",
            hypothesis_id=hypothesis.id,
            command=command,
            status=status,
            outcome=outcome,
            evidence_ids=[item.id],
            output_excerpt=_cap(output),
            exit_code=exit_code,
            duration_seconds=round(sandbox_result.duration_seconds, 3),
        )
        self._seen_result_ids.add(result.id)
        return result, item

    def _rejected_command(
        self,
        hypothesis: Hypothesis,
        command: str,
        error: str,
    ) -> VerificationResult:
        result = VerificationResult(
            id=f"ver-{hypothesis.id}-{len(self._seen_result_ids) + 1:03d}",
            hypothesis_id=hypothesis.id,
            command=command,
            status=VerificationStatus.ERROR,
            outcome=VerificationOutcome.UNVERIFIED,
            output_excerpt=_cap(error),
            exit_code=None,
            duration_seconds=None,
        )
        # Register the id so later steps never reuse the same result id; a
        # rejected step still consumes a slot in the global id sequence.
        self._seen_result_ids.add(result.id)
        return result


def _disposition_for(
    outcomes: list[VerificationOutcome],
) -> HypothesisDisposition:
    if not outcomes:
        return HypothesisDisposition.UNVERIFIED
    if any(outcome is VerificationOutcome.REJECTED for outcome in outcomes):
        return HypothesisDisposition.REJECTED
    if any(outcome is VerificationOutcome.UNVERIFIED for outcome in outcomes):
        return HypothesisDisposition.UNVERIFIED
    return HypothesisDisposition.SUPPORTED
