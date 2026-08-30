"""Lead planner and three evidence Specialists for RootTrace RCA.

Typed RCA Agents with strict least-privilege tool binding:

- ``LeadPlanner`` plans investigation questions/assignments only; it never
  commits to a root cause before evidence.
- ``IssueCISpecialist`` may call only ``read_external_log``.
- ``CodeSpecialist`` may call only ``search_code``/``read_file``/
  ``inspect_symbols``.
- ``GitHistorySpecialist`` may call only ``git_history``/``git_blame``/
  ``git_show``.

No Specialist receives runtime-test execution or write-capable tools.
Tool results are recorded as provenance-backed ``EvidenceItem`` objects with
stable ids; every finding cites evidence ids and explicit uncertainty.
Malformed outputs are explicit failures, never silent estimates.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from typing import Any, Protocol

from pydantic import BaseModel, Field, ValidationError

from roottrace.agents.prompts import (
    CODE_SYSTEM_PROMPT,
    GIT_HISTORY_SYSTEM_PROMPT,
    ISSUE_CI_SYSTEM_PROMPT,
    build_code_prompt,
    build_git_history_prompt,
    build_issue_ci_prompt,
)
from roottrace.agents.schema import PlanBudgets, PlanQuestion
from roottrace.evidence.schema import (
    MAX_EVIDENCE_IDS,
    MAX_EXCERPT_CHARS,
    MAX_GRAPH_EVIDENCE,
    MAX_NOTE_CHARS,
    AgentFinding,
    AgentRole,
    EvidenceItem,
    EvidenceKind,
    FindingStatus,
    SourceLocation,
    UncertaintyLevel,
)
from roottrace.incident.context import IncidentContext
from roottrace.incident.schema import IncidentInput, Provenance, validate_commit_sha
from roottrace.llm.schema import AssistantTurn, ToolCall
from roottrace.llm.usage import UsageTracker
from roottrace.tools.git_search import (
    GitSearchSummary,
    build_git_search_plan,
)
from roottrace.tools.repository import RcaToolRegistry

_TOOL_EVIDENCE_KIND: dict[str, EvidenceKind] = {
    "search_code": EvidenceKind.CODE_SNIPPET,
    "read_file": EvidenceKind.CODE_SNIPPET,
    "inspect_symbols": EvidenceKind.SYMBOL,
    "git_history": EvidenceKind.GIT_LOG,
    "git_blame": EvidenceKind.GIT_BLAME,
    "git_show": EvidenceKind.GIT_DIFF,
    "read_external_log": EvidenceKind.CI_LOG,
}
_GIT_EVIDENCE_TOOLS = frozenset({"git_history", "git_blame", "git_show"})


class ProviderProtocol(Protocol):
    """Minimal provider surface used by RCA Agents (satisfied by LLMProvider)."""

    model: str

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str | None = None,
    ) -> AssistantTurn: ...


class SpecialistResponse(BaseModel):
    """Structured final answer expected from an evidence Specialist."""

    status: FindingStatus = FindingStatus.COMPLETED
    ranked_locations: list[SourceLocation] = Field(
        default_factory=list,
        max_length=10,
    )
    evidence_ids: list[str] = Field(default_factory=list, max_length=MAX_EVIDENCE_IDS)
    uncertainty: UncertaintyLevel = UncertaintyLevel.LOW
    uncertainty_note: str | None = Field(default=None, max_length=MAX_NOTE_CHARS)


class SpecialistOutput(BaseModel):
    """Typed output of one evidence-gathering Specialist run."""

    finding: AgentFinding
    evidence: list[EvidenceItem] = Field(
        default_factory=list,
        max_length=MAX_GRAPH_EVIDENCE,
    )
    isolation_violations: int = Field(default=0, ge=0)


def _cap_text(text: str, limit: int, marker: str = "\n...[truncated]") -> tuple[str, bool]:
    """Cap text to ``limit`` characters including the truncation marker."""
    if len(text) <= limit:
        return text, False
    keep = max(0, limit - len(marker))
    return text[:keep] + marker, True


def _bounded_note(text: str | None, limit: int = MAX_NOTE_CHARS) -> str | None:
    if text is None or not text.strip():
        return None
    return _cap_text(text, limit)[0]


def _extract_commit_ids(tool: str, content: str) -> list[str]:
    """Extract commit ids from machine-formatted Git tool output."""
    if tool not in _GIT_EVIDENCE_TOOLS:
        return []
    patterns = {
        "git_history": re.compile(r"^\s*([0-9a-fA-F]{7,64})(?:\s|$)"),
        "git_show": re.compile(r"^\s*commit\s+([0-9a-fA-F]{7,64})(?:\s|$)"),
        "git_blame": re.compile(r"^\s*\^?([0-9a-fA-F]{7,64})(?:\s|$)"),
    }
    pattern = patterns[tool]
    commit_ids: set[str] = set()
    for line in content.splitlines():
        match = pattern.match(line)
        if match:
            try:
                commit_ids.add(validate_commit_sha(match.group(1)).lower())
            except ValueError:
                continue
    return sorted(commit_ids)


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object from a model response."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found in model output")
    return json.loads(cleaned[start : end + 1])


def _assistant_message(turn: AssistantTurn) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": turn.content,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments),
                },
            }
            for call in turn.tool_calls
        ],
    }


class _Specialist:
    """Shared typed loop for the three evidence Specialists."""

    def __init__(
        self,
        *,
        agent_id: str,
        role: AgentRole,
        allowed_tools: frozenset[str],
        system_prompt: str,
        prompt_builder: Callable[[IncidentContext, list[PlanQuestion]], str],
        provider: ProviderProtocol,
        registry: RcaToolRegistry,
        usage: UsageTracker,
        budgets: PlanBudgets,
    ) -> None:
        self.agent_id = agent_id
        self.role = role
        self.allowed_tools = allowed_tools
        self._system_prompt = system_prompt
        self._prompt_builder = prompt_builder
        self._provider = provider
        self._registry = registry
        self._usage = usage
        self._budgets = budgets
        self._evidence: list[EvidenceItem] = []
        self._seed: list[EvidenceItem] = []
        self._counter = 0
        self._isolation_violations = 0
        self._base_commit = ""
        self._tool_failures: list[str] = []
        self._prepared_evidence: list[EvidenceItem] = []
        self._git_search_summary: GitSearchSummary | None = None
        self._prepared_context = ""

    def _next_id(self) -> str:
        self._counter += 1
        return f"ev-{self.agent_id}-{self._counter:03d}"

    def _seed_evidence(self, context: IncidentContext) -> None:
        """Issue/CI seed evidence comes from the incident itself."""
        if self.role is not AgentRole.ISSUE_CI:
            return
        incident: IncidentInput = context.incident
        problem_excerpt, _ = _cap_text(incident.problem, MAX_EXCERPT_CHARS)
        problem = EvidenceItem(
            id=self._next_id(),
            agent=self.role,
            kind=EvidenceKind.ISSUE_TEXT,
            observation="incident problem text",
            provenance=Provenance(source="incident_input"),
            excerpt=problem_excerpt,
        )
        self._seed.append(problem)
        self._evidence.append(problem)
        for log in incident.logs:
            kind = (
                EvidenceKind.STACK_TRACE
                if "Traceback (most recent call last)" in log
                else EvidenceKind.CI_LOG
            )
            log_excerpt, _ = _cap_text(log, MAX_EXCERPT_CHARS)
            item = EvidenceItem(
                id=self._next_id(),
                agent=self.role,
                kind=kind,
                observation="incident log entry",
                provenance=Provenance(source="incident_input"),
                excerpt=log_excerpt,
            )
            self._seed.append(item)
            self._evidence.append(item)

    def _prepare_evidence(self, context: IncidentContext) -> None:
        """Prepare deterministic evidence before the first model call."""

    def _prepared_prompt(self) -> str:
        """Return bounded prepared evidence context for the first model call."""
        sections: list[str] = []
        if self._seed:
            sections.append(
                json.dumps(
                    [
                        {
                            "id": item.id,
                            "kind": item.kind.value,
                            "observation": item.observation,
                        }
                        for item in self._seed
                    ],
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        if self._prepared_evidence:
            sections.append(
                json.dumps(
                    [
                        {
                            "id": item.id,
                            "kind": item.kind.value,
                            "observation": item.observation,
                            "excerpt": item.excerpt,
                            "commit_ids": item.commit_ids,
                        }
                        for item in self._prepared_evidence
                    ],
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        if self._prepared_context:
            sections.append(self._prepared_context)
        return "\n\n".join(sections)

    def _evidence_from_tool(
        self,
        tool: str,
        result: Any,
        arguments: dict[str, Any],
    ) -> EvidenceItem:
        location = None
        path = arguments.get("path")
        if path and tool in ("read_file", "inspect_symbols", "git_blame", "git_show"):
            try:
                location = SourceLocation(path=path)
            except (ValueError, ValidationError):
                location = None
        excerpt, _ = _cap_text(result.content, MAX_EXCERPT_CHARS)
        return EvidenceItem(
            id=self._next_id(),
            agent=self.role,
            kind=_TOOL_EVIDENCE_KIND[tool],
            observation=f"{tool} returned {len(result.content)} chars",
            provenance=Provenance(
                source="rca_tool",
                tool=tool,
                command=getattr(result, "command", None) or None,
                commit=self._base_commit,
            ),
            location=location,
            excerpt=excerpt,
            commit_ids=_extract_commit_ids(tool, result.content),
        )

    def _run_tool_call(self, call: ToolCall) -> str:
        """Execute one tool call; disallowed tools become refusal feedback."""
        if call.name not in self.allowed_tools:
            self._isolation_violations += 1
            return (
                f"tool '{call.name}' is not allowed for role "
                f"'{self.role.value}'"
            )
        result = self._registry.execute(call.name, call.arguments)
        if result.ok:
            if result.empty:
                return result.content
            evidence = self._evidence_from_tool(
                call.name,
                result,
                call.arguments,
            )
            self._evidence.append(evidence)
            return f"Evidence id: {evidence.id}\n{result.content}"
        failure = _bounded_note(f"{call.name} failed: {result.content}")
        if failure is not None:
            self._tool_failures.append(failure)
        return result.content

    def execute_tool_call(self, call: ToolCall) -> str:
        """Strict tool boundary: raises for any tool outside the role surface."""
        if call.name not in self.allowed_tools:
            raise PermissionError(
                f"tool '{call.name}' is not allowed for role '{self.role.value}'"
            )
        return self._run_tool_call(call)

    def _parse_final(self, content: str | None) -> SpecialistResponse:
        if content is None or not content.strip():
            raise ValueError("specialist returned no content")
        return SpecialistResponse.model_validate(extract_json_object(content))

    def run(
        self,
        context: IncidentContext,
        questions: list[PlanQuestion],
    ) -> SpecialistOutput:
        """Gather evidence for the assigned questions and return a finding."""
        started = time.monotonic()
        self._evidence = []
        self._seed = []
        self._counter = 0
        self._isolation_violations = 0
        self._base_commit = context.incident.base_commit
        self._tool_failures = []
        self._prepared_evidence = []
        self._git_search_summary = None
        self._prepared_context = ""
        self._seed_evidence(context)
        try:
            self._prepare_evidence(context)
        except Exception as exc:  # noqa: BLE001
            self._tool_failures.append(
                _bounded_note(f"prepared evidence failed: {exc}")
                or "prepared evidence failed"
            )

        prompt = self._prompt_builder(context, questions)
        prepared_prompt = self._prepared_prompt()
        if prepared_prompt:
            prompt += "\n\nPREPARED EVIDENCE:\n" + prepared_prompt
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": prompt},
        ]
        schemas = [
            schema
            for schema in self._registry.get_tool_schemas()
            if schema["function"]["name"] in self.allowed_tools
        ]

        llm_calls = 0
        tool_calls_made = 0
        status = FindingStatus.COMPLETED
        error: str | None = None
        response: SpecialistResponse | None = None

        while True:
            if time.monotonic() - started > self._budgets.timeout_seconds:
                status = FindingStatus.PARTIAL
                error = "specialist timed out"
                break
            if llm_calls >= self._budgets.max_llm_calls:
                status = FindingStatus.PARTIAL
                error = "llm call budget exceeded"
                break
            turn = self._provider.complete(
                messages=messages,
                tools=schemas,
                tool_choice="auto",
            )
            self._usage.record(
                turn.prompt_tokens,
                turn.completion_tokens,
                turn.reasoning_tokens,
            )
            llm_calls += 1
            if turn.tool_calls:
                tool_calls_made += len(turn.tool_calls)
                if tool_calls_made > self._budgets.max_tool_calls:
                    status = FindingStatus.PARTIAL
                    error = "tool call budget exceeded"
                    break
                messages.append(_assistant_message(turn))
                for call in turn.tool_calls:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": self._run_tool_call(call),
                        }
                    )
                continue
            try:
                response = self._parse_final(turn.content)
            except (ValueError, ValidationError) as exc:
                status = FindingStatus.PARTIAL
                error = f"malformed specialist output: {exc}"
            break

        known_ids = {item.id for item in self._evidence}
        if response is None:
            ranked_locations: list[SourceLocation] = []
            evidence_ids = [item.id for item in self._evidence][:MAX_EVIDENCE_IDS]
            uncertainty = UncertaintyLevel.HIGH
            uncertainty_note = _bounded_note(error)
        else:
            ranked_locations = response.ranked_locations
            unknown = [
                item for item in response.evidence_ids if item not in known_ids
            ]
            evidence_ids = [
                item for item in response.evidence_ids if item in known_ids
            ]
            if unknown:
                status = FindingStatus.PARTIAL
                error = (
                    f"specialist referenced unknown evidence ids: "
                    f"{sorted(unknown)}"
                )
            elif response.status == FindingStatus.FAILED:
                status = FindingStatus.FAILED
                error = response.uncertainty_note or "specialist reported failure"
            else:
                status = response.status
            uncertainty = response.uncertainty
            uncertainty_note = _bounded_note(response.uncertainty_note or error)

        if self._tool_failures:
            tool_error = _bounded_note(
                "tool failures: " + "; ".join(self._tool_failures)
            )
            if status == FindingStatus.COMPLETED:
                status = FindingStatus.PARTIAL
            if error:
                error = _bounded_note(f"{error}; {tool_error}")
            else:
                error = tool_error
            uncertainty = UncertaintyLevel.HIGH
            uncertainty_note = _bounded_note(error)

        finding = AgentFinding(
            agent=self.role,
            status=status,
            ranked_locations=ranked_locations,
            evidence_ids=evidence_ids,
            uncertainty=uncertainty,
            uncertainty_note=uncertainty_note,
            timing_seconds=round(time.monotonic() - started, 3),
            usage=self._usage.snapshot(),
            error=_bounded_note(error) if error else None,
            git_search_summary=self._git_search_summary,
        )
        return SpecialistOutput(
            finding=finding,
            evidence=self._evidence,
            isolation_violations=self._isolation_violations,
        )


class IssueCISpecialist(_Specialist):
    """Gathers issue text, stack trace, CI, and failure-signature evidence."""

    def __init__(
        self,
        *,
        provider: ProviderProtocol,
        registry: RcaToolRegistry,
        usage: UsageTracker,
        budgets: PlanBudgets,
        agent_id: str = "issue_ci",
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.ISSUE_CI,
            allowed_tools=frozenset({"read_external_log"}),
            system_prompt=ISSUE_CI_SYSTEM_PROMPT,
            prompt_builder=build_issue_ci_prompt,
            provider=provider,
            registry=registry,
            usage=usage,
            budgets=budgets,
        )


class CodeSpecialist(_Specialist):
    """Localizes likely files, functions, symbols, and code paths."""

    def __init__(
        self,
        *,
        provider: ProviderProtocol,
        registry: RcaToolRegistry,
        usage: UsageTracker,
        budgets: PlanBudgets,
        agent_id: str = "code",
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.CODE,
            allowed_tools=frozenset({"search_code", "read_file", "inspect_symbols"}),
            system_prompt=CODE_SYSTEM_PROMPT,
            prompt_builder=build_code_prompt,
            provider=provider,
            registry=registry,
            usage=usage,
            budgets=budgets,
        )


class GitHistorySpecialist(_Specialist):
    """Reports relevant history, blame, and suspected regression changes."""

    def __init__(
        self,
        *,
        provider: ProviderProtocol,
        registry: RcaToolRegistry,
        usage: UsageTracker,
        budgets: PlanBudgets,
        agent_id: str = "git_history",
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.GIT_HISTORY,
            allowed_tools=frozenset({"git_history", "git_blame", "git_show"}),
            system_prompt=GIT_HISTORY_SYSTEM_PROMPT,
            prompt_builder=build_git_history_prompt,
            provider=provider,
            registry=registry,
            usage=usage,
            budgets=budgets,
        )

    def _prepare_evidence(self, context: IncidentContext) -> None:
        """Run layered Git search and prepare only the final Top-K candidates."""
        plan = build_git_search_plan(context)
        summary = self._registry.search_git_layers(plan)
        self._git_search_summary = summary
        for error in summary.errors:
            self._tool_failures.append(f"prepared Git search: {error}")

        for candidate in summary.candidates:
            if not candidate.strong_match:
                continue
            location = None
            if candidate.matched_paths:
                location = SourceLocation(path=candidate.matched_paths[0])
            excerpt = (
                f"prepared Git candidate at depth {candidate.depth}: "
                f"{candidate.commit} {candidate.subject}; "
                f"score={candidate.score}; "
                f"signals={', '.join(candidate.matched_signals)}"
            )
            evidence = EvidenceItem(
                id=self._next_id(),
                agent=self.role,
                kind=EvidenceKind.GIT_LOG,
                observation=(
                    f"layered Git search candidate at depth {candidate.depth}"
                ),
                provenance=Provenance(
                    source="rca_tool",
                    tool="git_history",
                    command=candidate.command or None,
                    commit=self._base_commit,
                ),
                location=location,
                excerpt=_cap_text(excerpt, MAX_EXCERPT_CHARS)[0],
                commit_ids=[candidate.commit],
            )
            self._prepared_evidence.append(evidence)
            self._evidence.append(evidence)

        self._prepared_context = json.dumps(
            {
                "git_search_summary": summary.model_dump(mode="json"),
                "prepared_evidence_ids": [
                    item.id for item in self._prepared_evidence
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
