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

from pydantic import BaseModel, Field, ValidationError, model_validator

from roottrace.agents import (
    CodeSpecialist,
    GitHistorySpecialist,
    IssueCISpecialist,
    LeadPlanner,
    PlanError,
    ProviderProtocol,
    SpecialistOutput,
    extract_json_object,
)
from roottrace.agents.prompts import (
    HYPOTHESES_SYSTEM_PROMPT,
    build_hypotheses_prompt,
)
from roottrace.agents.schema import InvestigationPlan, PlanBudgets, PlanQuestion
from roottrace.artifacts import (
    ARTIFACT_EVIDENCE_GRAPH,
    ARTIFACT_HYPOTHESES,
    ARTIFACT_INVESTIGATION_PLAN,
    ArtifactWriter,
)
from roottrace.diagnostics import (
    DiagnosticSeverity,
    PipelineDiagnostic,
    deduplicate_diagnostics,
    diagnostics_from_legacy,
    project_diagnostics,
)
from roottrace.evidence.graph import EvidenceGraph, aggregate_evidence
from roottrace.evidence.schema import (
    AgentFinding,
    AgentRole,
    BoundedNote,
    FindingStatus,
    Hypothesis,
    HypothesisDisposition,
    UncertaintyLevel,
)
from roottrace.history.retrieval import HistoricalRetriever
from roottrace.history.schema import RetrievalHints
from roottrace.incident.builder import build_incident_context
from roottrace.incident.context import IncidentContext
from roottrace.incident.loader import LoadedIncident
from roottrace.incident.schema import IncidentInput
from roottrace.llm.schema import Usage
from roottrace.llm.usage import UsageTracker
from roottrace.tools.repository import RcaToolRegistry
from roottrace.tracing import TraceEvent, TraceWriter

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
    diagnostics: list[PipelineDiagnostic] = Field(default_factory=list, max_length=20)
    errors: list[BoundedNote] = Field(default_factory=list, max_length=20)
    timing_seconds: float | None = Field(default=None, ge=0)
    usage: Usage | None = None
    isolation_violations: int = Field(default=0, ge=0)
    enabled_roles: list[str] | None = None
    retrieval: RetrievalHints | None = None

    @model_validator(mode="after")
    def _project_legacy_errors(self) -> RcaRunResult:
        """Keep the legacy error list as a deterministic diagnostic projection."""
        if "diagnostics" in self.model_fields_set:
            self.diagnostics = deduplicate_diagnostics(self.diagnostics)
        elif self.errors:
            self.diagnostics = diagnostics_from_legacy(self.errors)
        self.errors = project_diagnostics(self.diagnostics)
        return self


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
    reasoning_tokens: list[int | None] = [
        snapshot.reasoning_tokens for snapshot in snapshots
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
        reasoning_tokens=(
            None
            if any(value is None for value in reasoning_tokens)
            else sum(value for value in reasoning_tokens if value is not None)
        ),
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
        worker_concurrency: int = 3,
        enabled_roles: frozenset[AgentRole] | None = None,
        retriever: HistoricalRetriever | None = None,
        retrieval_mode: str = "off",
        history_excluded_ids: frozenset[str] = frozenset(),
    ) -> None:
        if not 1 <= worker_concurrency <= 3:
            raise ValueError("worker_concurrency must be between 1 and 3")
        if retrieval_mode not in {"off", "clustered", "flat"}:
            raise ValueError("retrieval_mode must be off, clustered, or flat")
        self._lead_provider = lead_provider
        self._issue_ci_provider = issue_ci_provider
        self._code_provider = code_provider
        self._git_history_provider = git_history_provider
        self._registry = registry
        self._budgets = budgets
        self._worker_timeout_margin_seconds = worker_timeout_margin_seconds
        self._worker_concurrency = worker_concurrency
        self._enabled_roles = (
            frozenset(_ROLE_ORDER) if enabled_roles is None else frozenset(enabled_roles)
        )
        self._retriever = retriever
        self._retrieval_mode = retrieval_mode
        self._history_excluded_ids = frozenset(history_excluded_ids)

    def _specialist_roles(self) -> list[AgentRole]:
        """Return the enabled evidence-specialist roles in stable order."""
        return [role for role in _ROLE_ORDER if role in self._enabled_roles]

    def run(
        self,
        loaded: LoadedIncident,
        repo: str | Path,
        output_dir: str | Path | None = None,
    ) -> RcaRunResult:
        """Run the full evidence pipeline for one incident."""
        started = time.monotonic()
        incident = loaded.incident
        self._registry.configure_git_history(
            base_commit=incident.base_commit,
            history_depth=incident.git_verification_policy.history_depth,
            visible_depth=1,
        )
        writer = (
            TraceWriter(Path(output_dir) / "execution_trace.jsonl")
            if output_dir is not None
            else None
        )
        if writer is not None:
            writer.start_run()
        context = build_incident_context(loaded, repo)
        lead_usage = UsageTracker()
        issue_ci_usage = UsageTracker()
        code_usage = UsageTracker()
        git_history_usage = UsageTracker()
        self._trace(
            writer,
            incident.id,
            "planning_start",
            self._lead_provider.model,
            git_verification_policy=incident.git_verification_policy.model_dump(
                mode="json"
            ),
        )

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
        if lead.degradation is not None:
            self._trace(
                writer,
                incident.id,
                "planning_degraded",
                self._lead_provider.model,
                final_status="DEGRADED",
                retry_count=lead.degradation.repair_attempts,
                degradation=lead.degradation.model_dump(mode="json"),
            )
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
        specialist_builders: dict[AgentRole, tuple[Any, UsageTracker]] = {
            AgentRole.ISSUE_CI: (
                IssueCISpecialist,
                issue_ci_usage,
            ),
            AgentRole.CODE: (
                CodeSpecialist,
                code_usage,
            ),
            AgentRole.GIT_HISTORY: (
                GitHistorySpecialist,
                git_history_usage,
            ),
        }
        role_providers = {
            AgentRole.ISSUE_CI: self._issue_ci_provider,
            AgentRole.CODE: self._code_provider,
            AgentRole.GIT_HISTORY: self._git_history_provider,
        }
        agents: dict[AgentRole, Any] = {}
        for role in self._specialist_roles():
            specialist_cls, usage = specialist_builders[role]
            specialist_budgets = self._budgets
            if role is AgentRole.GIT_HISTORY:
                specialist_budgets = self._budgets.model_copy(
                    update={
                        "max_tool_calls": min(
                            self._budgets.max_tool_calls,
                            incident.git_verification_policy.max_tool_calls,
                        )
                    }
                )
            agents[role] = specialist_cls(
                provider=role_providers[role],
                registry=self._registry,
                usage=usage,
                budgets=specialist_budgets,
            )
        outputs, isolation_violations = self._run_workers(
            context,
            incident,
            agents,
            questions,
            writer,
        )
        diagnostics: list[PipelineDiagnostic] = []
        for role in _ROLE_ORDER:
            output = outputs.get(role)
            if output is None:
                continue
            finding = output.finding
            if finding.error:
                diagnostics.append(
                    PipelineDiagnostic(
                        code="specialist.finding_error",
                        stage="specialist",
                        severity=DiagnosticSeverity.RECOVERABLE,
                        message=finding.error,
                        agent=role,
                    )
                )
            elif finding.status != FindingStatus.COMPLETED:
                diagnostics.append(
                    PipelineDiagnostic(
                        code="specialist.incomplete",
                        stage="specialist",
                        severity=DiagnosticSeverity.WARNING,
                        message=(
                            f"{role.value} finding status: "
                            f"{finding.status.value}"
                        ),
                        agent=role,
                    )
                )
        graph = aggregate_evidence(incident, outputs)

        retrieval_hints: RetrievalHints | None = None
        if self._retriever is not None:
            retrieval_hints = self._retriever.retrieve(
                incident.problem,
                target_id=incident.id,
                excluded_ids=self._history_excluded_ids,
                mode=self._retrieval_mode,
            )
        self._trace(
            writer,
            incident.id,
            "hypotheses_start",
            self._lead_provider.model,
        )
        hypotheses, hypothesis_diagnostics = self._generate_hypotheses(
            context,
            graph,
            lead_usage,
            retrieval_hints=retrieval_hints,
        )
        self._trace(
            writer,
            incident.id,
            "hypotheses_end",
            self._lead_provider.model,
            final_status=(
                "COMPLETED" if not hypothesis_diagnostics else "PARTIAL"
            ),
            prompt_tokens=lead_usage.snapshot().prompt_tokens,
            completion_tokens=lead_usage.snapshot().completion_tokens,
        )
        diagnostics.extend(hypothesis_diagnostics)
        graph = graph.model_copy(update={"hypotheses": hypotheses})
        diagnostics = deduplicate_diagnostics(diagnostics)
        legacy_errors = project_diagnostics(diagnostics)

        result = RcaRunResult(
            incident_id=incident.id,
            plan=plan,
            graph=graph,
            status=(
                FindingStatus.COMPLETED
                if not legacy_errors
                else FindingStatus.PARTIAL
            ),
            diagnostics=diagnostics,
            timing_seconds=round(time.monotonic() - started, 3),
            usage=_merge_usage(
                lead_usage.snapshot(),
                issue_ci_usage.snapshot(),
                code_usage.snapshot(),
                git_history_usage.snapshot(),
            ),
            isolation_violations=isolation_violations,
            enabled_roles=[role.value for role in self._specialist_roles()],
            retrieval=retrieval_hints,
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
    ) -> tuple[dict[AgentRole, SpecialistOutput], int]:
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=min(self._worker_concurrency, max(len(agents), 1)),
            thread_name_prefix="rca-specialist",
        )
        futures: dict[AgentRole, concurrent.futures.Future[SpecialistOutput]] = {}
        role_order = [role for role in _ROLE_ORDER if role in agents]
        try:
            for role in role_order:
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
            isolation_violations = 0
            for role in role_order:
                try:
                    output = futures[role].result(timeout=timeout)
                except TimeoutError:
                    error = _bounded_error(
                        f"{role.value} worker timed out after {timeout}s"
                    )
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
                    outputs[role] = self._failed_output(
                        role,
                        status=FindingStatus.FAILED,
                        error=error,
                    )
                else:
                    isolation_violations += output.isolation_violations
                    outputs[role] = output
            return outputs, isolation_violations
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
            git_search_summary=(
                output.finding.git_search_summary.model_dump(mode="json")
                if output.finding.git_search_summary is not None
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
        *,
        retrieval_hints: RetrievalHints | None = None,
    ) -> tuple[list[Hypothesis], list[PipelineDiagnostic]]:
        prompt = build_hypotheses_prompt(
            graph,
            retrieval_hints=retrieval_hints,
        )
        turn = self._lead_provider.complete(
            messages=[
                {"role": "system", "content": HYPOTHESES_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            tools=[],
        )
        lead_usage.record(
            turn.prompt_tokens,
            turn.completion_tokens,
            turn.reasoning_tokens,
        )
        if turn.content is None or not turn.content.strip():
            return [], [
                PipelineDiagnostic(
                    code="hypotheses.empty",
                    stage="hypotheses",
                    severity=DiagnosticSeverity.RECOVERABLE,
                    message="hypothesis generation returned no content",
                    agent=AgentRole.LEAD,
                )
            ]
        try:
            data = extract_json_object(turn.content)
            raw_hypotheses = data.get("hypotheses", [])
        except (ValueError, ValidationError) as exc:
            return [], [
                PipelineDiagnostic(
                    code="hypotheses.malformed",
                    stage="hypotheses",
                    severity=DiagnosticSeverity.RECOVERABLE,
                    message=_bounded_error(f"malformed hypothesis output: {exc}"),
                    agent=AgentRole.LEAD,
                )
            ]
        known_ids = {item.id for item in graph.evidence}
        hypotheses: list[Hypothesis] = []
        diagnostics: list[PipelineDiagnostic] = []
        if not raw_hypotheses:
            return [], [
                PipelineDiagnostic(
                    code="hypotheses.empty",
                    stage="hypotheses",
                    severity=DiagnosticSeverity.WARNING,
                    message="hypothesis generation returned no hypotheses",
                    agent=AgentRole.LEAD,
                )
            ]
        for index, raw in enumerate(raw_hypotheses[: _MAX_HYPOTHESES], start=1):
            try:
                hypothesis = Hypothesis.model_validate(
                    {**raw, "id": f"h-{index:03d}"}
                )
            except ValidationError as exc:
                diagnostics.append(
                    PipelineDiagnostic(
                        code="hypotheses.invalid",
                        stage="hypotheses",
                        severity=DiagnosticSeverity.RECOVERABLE,
                        message=_bounded_error(
                            f"hypothesis {index} invalid: {exc}"
                        ),
                        agent=AgentRole.LEAD,
                    )
                )
                continue
            referenced = (
                *hypothesis.supporting_evidence_ids,
                *hypothesis.contradicting_evidence_ids,
            )
            unknown = sorted(
                {item for item in referenced if item not in known_ids}
            )
            if unknown:
                diagnostics.append(
                    PipelineDiagnostic(
                        code="hypotheses.unknown_evidence",
                        stage="hypotheses",
                        severity=DiagnosticSeverity.RECOVERABLE,
                        message=_bounded_error(
                            f"hypothesis {index} references unknown evidence ids: "
                            f"{unknown}"
                        ),
                        agent=AgentRole.LEAD,
                    )
                )
                continue
            commands = [step.command for step in hypothesis.verification_plan]
            if not commands or not all(
                command.startswith("python -m pytest")
                for command in commands
            ):
                diagnostics.append(
                    PipelineDiagnostic(
                        code="hypotheses.invalid_verification",
                        stage="hypotheses",
                        severity=DiagnosticSeverity.RECOVERABLE,
                        message=_bounded_error(
                            f"hypothesis {index} has no sandbox-runnable pytest "
                            "verification plan"
                        ),
                        agent=AgentRole.LEAD,
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
        return hypotheses, diagnostics

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
        retry_count: int = 0,
        degradation: dict[str, Any] | None = None,
        git_verification_policy: dict[str, Any] | None = None,
        git_search_summary: dict[str, Any] | None = None,
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
                retry_count=retry_count,
                degradation=degradation,
                git_verification_policy=git_verification_policy,
                git_search_summary=git_search_summary,
            )
        )
