"""RCA evidence orchestration and hypothesis generation.

The orchestrator owns the investigation pipeline: it plans with the Lead,
runs the three evidence Specialists concurrently, aggregates their outputs
into a validated ``EvidenceGraph``, and asks the Lead to generate ranked,
falsifiable root-cause hypotheses from that shared state.

Guarantees:
- Specialists never communicate directly; the typed ``EvidenceGraph`` is the
  only shared state between them.
- No Specialist receives runtime-test execution or write-capable tools.
- Worker timeout or failure yields partial evidence and higher uncertainty;
  exceptions never erase the diagnostics collected by other workers.
- Aggregation is deterministic: findings and evidence are ordered by stable
  role order, never by thread completion order.
"""

from __future__ import annotations

import concurrent.futures
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from patchpilot.rca.agents import (
    CodeSpecialist,
    GitHistorySpecialist,
    IssueCISpecialist,
    LeadPlanner,
    PlanError,
    ProviderProtocol,
    SpecialistOutput,
    extract_json_object,
)
from patchpilot.rca.artifacts import (
    ARTIFACT_EVIDENCE_GRAPH,
    ARTIFACT_HYPOTHESES,
    ARTIFACT_INVESTIGATION_PLAN,
    ArtifactWriter,
)
from patchpilot.rca.context import IncidentContext
from patchpilot.rca.context_builder import build_incident_context
from patchpilot.rca.incident_loader import LoadedIncident
from patchpilot.rca.prompts import (
    HYPOTHESES_SYSTEM_PROMPT,
    build_hypotheses_prompt,
)
from patchpilot.rca.schema import (
    AgentFinding,
    AgentRole,
    BoundedNote,
    EvidenceGraph,
    FindingStatus,
    Hypothesis,
    HypothesisDisposition,
    IncidentInput,
    InvestigationPlan,
    PlanBudgets,
    PlanQuestion,
    UncertaintyLevel,
    Usage,
)
from patchpilot.rca.tools import RcaToolRegistry
from patchpilot.rca.usage import UsageTracker
from patchpilot.workflow.trace import TraceEvent, TraceWriter

_ROLE_ORDER = (
    AgentRole.ISSUE_CI,
    AgentRole.CODE,
    AgentRole.GIT_HISTORY,
)
_MAX_HYPOTHESES = 5
_TRACE_WORKFLOW_STAGE = "RCA"


class RcaRunResult(BaseModel):
    """Typed outcome of one RCA orchestration run."""

    incident_id: str
    plan: InvestigationPlan
    graph: EvidenceGraph
    status: FindingStatus
    errors: list[BoundedNote] = Field(default_factory=list, max_length=20)
    timing_seconds: float | None = Field(default=None, ge=0)
    usage: Usage | None = None
    isolation_violations: int = Field(default=0, ge=0)


def _bounded_error(text: str, limit: int = 500) -> str:
    if len(text) <= limit:
        return text
    keep = max(0, limit - 3)
    return text[:keep] + "..."


def _merge_usage(*snapshots: Usage) -> Usage:
    prompt_tokens: list[int | None] = [
        snapshot.prompt_tokens for snapshot in snapshots
    ]
    completion_tokens: list[int | None] = [
        snapshot.completion_tokens for snapshot in snapshots
    ]
    return Usage(
        llm_calls=sum(snapshot.llm_calls for snapshot in snapshots),
        prompt_tokens=(
            None
            if any(value is None for value in prompt_tokens)
            else sum(value for value in prompt_tokens if value is not None)
        ),
        completion_tokens=(
            None
            if any(value is None for value in completion_tokens)
            else sum(value for value in completion_tokens if value is not None)
        ),
    )


def aggregate_evidence(
    incident: IncidentInput,
    outputs: dict[AgentRole, SpecialistOutput],
) -> EvidenceGraph:
    """Merge specialist outputs in stable role order into an EvidenceGraph."""
    ordered = [outputs[role] for role in _ROLE_ORDER if role in outputs]
    findings = [output.finding for output in ordered]
    evidence = [
        item for output in ordered for item in output.evidence
    ]
    return EvidenceGraph(
        incident=incident,
        findings=findings,
        evidence=evidence,
    )


class RcaOrchestrator:
    """Run one bounded RCA investigation from incident to hypotheses."""

    def __init__(
        self,
        *,
        lead_provider: ProviderProtocol,
        issue_ci_provider: ProviderProtocol,
        code_provider: ProviderProtocol,
        git_history_provider: ProviderProtocol,
        registry: RcaToolRegistry,
        budgets: PlanBudgets,
        worker_timeout_margin_seconds: float = 30.0,
    ) -> None:
        self._lead_provider = lead_provider
        self._issue_ci_provider = issue_ci_provider
        self._code_provider = code_provider
        self._git_history_provider = git_history_provider
        self._registry = registry
        self._budgets = budgets
        self._worker_timeout_margin_seconds = worker_timeout_margin_seconds

    def run(
        self,
        loaded: LoadedIncident,
        repo: str | Path,
        output_dir: str | Path | None = None,
    ) -> RcaRunResult:
        """Run the full evidence pipeline for one incident."""
        started = time.monotonic()
        incident = loaded.incident
        context = build_incident_context(loaded, repo)
        writer = (
            TraceWriter(Path(output_dir) / "execution_trace.jsonl")
            if output_dir is not None
            else None
        )
        lead_usage = UsageTracker()
        issue_ci_usage = UsageTracker()
        code_usage = UsageTracker()
        git_history_usage = UsageTracker()
        self._trace(writer, incident.id, "planning_start", self._lead_provider.model)

        lead = LeadPlanner(
            provider=self._lead_provider,
            usage=lead_usage,
            budgets=self._budgets,
        )
        try:
            plan = lead.run(context)
        except PlanError:
            self._trace(
                writer,
                incident.id,
                "planning_failed",
                self._lead_provider.model,
                final_status="FAILED",
            )
            raise
        self._trace(
            writer,
            incident.id,
            "planning_end",
            self._lead_provider.model,
            final_status="COMPLETED",
            prompt_tokens=lead_usage.snapshot().prompt_tokens,
            completion_tokens=lead_usage.snapshot().completion_tokens,
        )

        questions = self._questions_by_role(plan)
        agents: dict[AgentRole, Any] = {
            AgentRole.ISSUE_CI: IssueCISpecialist(
                provider=self._issue_ci_provider,
                registry=self._registry,
                usage=issue_ci_usage,
                budgets=self._budgets,
            ),
            AgentRole.CODE: CodeSpecialist(
                provider=self._code_provider,
                registry=self._registry,
                usage=code_usage,
                budgets=self._budgets,
            ),
            AgentRole.GIT_HISTORY: GitHistorySpecialist(
                provider=self._git_history_provider,
                registry=self._registry,
                usage=git_history_usage,
                budgets=self._budgets,
            ),
        }
        outputs, errors, isolation_violations = self._run_workers(
            context,
            incident,
            agents,
            questions,
            writer,
        )
        graph = aggregate_evidence(incident, outputs)

        self._trace(
            writer,
            incident.id,
            "hypotheses_start",
            self._lead_provider.model,
        )
        hypotheses, hypothesis_errors = self._generate_hypotheses(
            context,
            graph,
            lead_usage,
        )
        self._trace(
            writer,
            incident.id,
            "hypotheses_end",
            self._lead_provider.model,
            final_status="COMPLETED" if not hypothesis_errors else "PARTIAL",
            prompt_tokens=lead_usage.snapshot().prompt_tokens,
            completion_tokens=lead_usage.snapshot().completion_tokens,
        )
        errors.extend(hypothesis_errors)
        graph = graph.model_copy(update={"hypotheses": hypotheses})

        result = RcaRunResult(
            incident_id=incident.id,
            plan=plan,
            graph=graph,
            status=(
                FindingStatus.COMPLETED
                if not errors
                else FindingStatus.PARTIAL
            ),
            errors=errors,
            timing_seconds=round(time.monotonic() - started, 3),
            usage=_merge_usage(
                lead_usage.snapshot(),
                issue_ci_usage.snapshot(),
                code_usage.snapshot(),
                git_history_usage.snapshot(),
            ),
            isolation_violations=isolation_violations,
        )
        if output_dir is not None:
            self._persist(Path(output_dir), plan, graph)
        return result

    @staticmethod
    def _questions_by_role(
        plan: InvestigationPlan,
    ) -> dict[AgentRole, list[PlanQuestion]]:
        by_id = {question.id: question for question in plan.questions}
        return {
            role: [
                by_id[question_id]
                for question_id in plan.assignments.get(role, [])
                if question_id in by_id
            ]
            for role in _ROLE_ORDER
        }

    def _run_workers(
        self,
        context: IncidentContext,
        incident: IncidentInput,
        agents: dict[AgentRole, Any],
        questions: dict[AgentRole, list[PlanQuestion]],
        writer: TraceWriter | None,
    ) -> tuple[dict[AgentRole, SpecialistOutput], list[str], int]:
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=3,
            thread_name_prefix="rca-specialist",
        )
        futures: dict[AgentRole, concurrent.futures.Future[SpecialistOutput]] = {}
        try:
            for role in _ROLE_ORDER:
                futures[role] = executor.submit(
                    self._run_specialist,
                    context,
                    incident,
                    role,
                    agents[role],
                    questions[role],
                    writer,
                    self._provider_model(role),
                )
            timeout = (
                self._budgets.timeout_seconds
                + self._worker_timeout_margin_seconds
            )
            outputs: dict[AgentRole, SpecialistOutput] = {}
            errors: list[str] = []
            isolation_violations = 0
            for role in _ROLE_ORDER:
                try:
                    output = futures[role].result(timeout=timeout)
                except TimeoutError:
                    error = _bounded_error(
                        f"{role.value} worker timed out after {timeout}s"
                    )
                    errors.append(error)
                    outputs[role] = self._failed_output(
                        role,
                        status=FindingStatus.PARTIAL,
                        error=error,
                    )
                # Worker failure isolation: any worker exception becomes a
                # partial finding instead of aborting the whole investigation.
                except Exception as exc:  # noqa: BLE001
                    error = _bounded_error(
                        f"{role.value} worker failed: {exc}"
                    )
                    errors.append(error)
                    outputs[role] = self._failed_output(
                        role,
                        status=FindingStatus.FAILED,
                        error=error,
                    )
                else:
                    isolation_violations += output.isolation_violations
                    outputs[role] = output
            return outputs, errors, isolation_violations
        finally:
            for future in futures.values():
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)

    def _run_specialist(
        self,
        context: IncidentContext,
        incident: IncidentInput,
        role: AgentRole,
        agent: Any,
        questions: list[PlanQuestion],
        writer: TraceWriter | None,
        provider_model: str,
    ) -> SpecialistOutput:
        self._trace(
            writer,
            incident.id,
            f"{role.value}_start",
            provider_model,
        )
        try:
            output = agent.run(context, questions)
        except Exception:
            self._trace(
                writer,
                incident.id,
                f"{role.value}_end",
                provider_model,
                final_status="FAILED",
            )
            raise
        self._trace(
            writer,
            incident.id,
            f"{role.value}_end",
            provider_model,
            final_status=output.finding.status.value.upper(),
            prompt_tokens=(
                output.finding.usage.prompt_tokens
                if output.finding.usage is not None
                else None
            ),
            completion_tokens=(
                output.finding.usage.completion_tokens
                if output.finding.usage is not None
                else None
            ),
        )
        return output

    def _provider_model(self, role: AgentRole) -> str:
        providers = {
            AgentRole.ISSUE_CI: self._issue_ci_provider,
            AgentRole.CODE: self._code_provider,
            AgentRole.GIT_HISTORY: self._git_history_provider,
        }
        return providers[role].model

    def _failed_output(
        self,
        role: AgentRole,
        *,
        status: FindingStatus,
        error: str,
    ) -> SpecialistOutput:
        return SpecialistOutput(
            finding=AgentFinding(
                agent=role,
                status=status,
                uncertainty=UncertaintyLevel.HIGH,
                error=error,
                timing_seconds=0.0,
            )
        )

    def _generate_hypotheses(
        self,
        context: IncidentContext,
        graph: EvidenceGraph,
        lead_usage: UsageTracker,
    ) -> tuple[list[Hypothesis], list[str]]:
        prompt = build_hypotheses_prompt(graph)
        turn = self._lead_provider.complete(
            messages=[
                {"role": "system", "content": HYPOTHESES_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            tools=[],
        )
        lead_usage.record(turn.prompt_tokens, turn.completion_tokens)
        if turn.content is None or not turn.content.strip():
            return [], ["hypothesis generation returned no content"]
        try:
            data = extract_json_object(turn.content)
            raw_hypotheses = data.get("hypotheses", [])
        except (ValueError, ValidationError) as exc:
            return [], [_bounded_error(f"malformed hypothesis output: {exc}")]
        known_ids = {item.id for item in graph.evidence}
        hypotheses: list[Hypothesis] = []
        errors: list[str] = []
        if not raw_hypotheses:
            return [], ["hypothesis generation returned no hypotheses"]
        for index, raw in enumerate(raw_hypotheses[: _MAX_HYPOTHESES], start=1):
            try:
                hypothesis = Hypothesis.model_validate(
                    {**raw, "id": f"h-{index:03d}"}
                )
            except ValidationError as exc:
                errors.append(_bounded_error(f"hypothesis {index} invalid: {exc}"))
                continue
            referenced = (
                *hypothesis.supporting_evidence_ids,
                *hypothesis.contradicting_evidence_ids,
            )
            unknown = sorted(
                {item for item in referenced if item not in known_ids}
            )
            if unknown:
                errors.append(
                    _bounded_error(
                        f"hypothesis {index} references unknown evidence ids: "
                        f"{unknown}"
                    )
                )
                continue
            commands = [step.command for step in hypothesis.verification_plan]
            if not commands or not all(
                command.startswith("python -m pytest")
                for command in commands
            ):
                errors.append(
                    _bounded_error(
                        f"hypothesis {index} has no sandbox-runnable pytest "
                        "verification plan"
                    )
                )
                continue
            hypotheses.append(
                hypothesis.model_copy(
                    update={
                        "disposition": HypothesisDisposition.CANDIDATE,
                    }
                )
            )
        return hypotheses, errors

    @staticmethod
    def _persist(
        output_dir: Path,
        plan: InvestigationPlan,
        graph: EvidenceGraph,
    ) -> None:
        writer = ArtifactWriter(output_dir)
        writer.write_model(ARTIFACT_INVESTIGATION_PLAN, plan)
        writer.write_model(ARTIFACT_EVIDENCE_GRAPH, graph)
        writer.write_dict(
            ARTIFACT_HYPOTHESES,
            {
                "hypotheses": [
                    hypothesis.model_dump(mode="json")
                    for hypothesis in graph.hypotheses
                ]
            },
        )

    @staticmethod
    def _trace(
        writer: TraceWriter | None,
        run_id: str,
        event_type: str,
        model: str | None,
        *,
        final_status: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> None:
        if writer is None:
            return
        writer.write(
            TraceEvent(
                run_id=run_id,
                event_type=event_type,
                workflow_stage=_TRACE_WORKFLOW_STAGE,
                model=model,
                final_status=final_status,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        )
