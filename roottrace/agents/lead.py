"""Lead investigation planning agent."""

from pydantic import ValidationError

from roottrace.agents.prompts import LEAD_SYSTEM_PROMPT, build_lead_prompt
from roottrace.agents.schema import InvestigationPlan, PlanBudgets, PlanQuestion
from roottrace.agents.specialists import ProviderProtocol, extract_json_object
from roottrace.evidence.schema import AgentRole
from roottrace.incident.context import IncidentContext
from roottrace.llm.usage import UsageTracker


class PlanError(Exception):
    """Raised when the Lead produces an unusable investigation plan."""


class LeadPlanner:
    """Plans the investigation; never commits to a root cause."""

    def __init__(
        self,
        *,
        provider: ProviderProtocol,
        usage: UsageTracker,
        budgets: PlanBudgets,
        agent_id: str = "lead",
    ) -> None:
        self.agent_id = agent_id
        self.role = AgentRole.LEAD
        self._provider = provider
        self._usage = usage
        self._budgets = budgets

    def run(self, context: IncidentContext) -> InvestigationPlan:
        prompt = build_lead_prompt(context)
        turn = self._provider.complete(
            messages=[
                {"role": "system", "content": LEAD_SYSTEM_PROMPT},
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
            raise PlanError("Lead returned no content")
        try:
            data = extract_json_object(turn.content)
            questions = [
                PlanQuestion.model_validate(question)
                for question in data.get("questions", [])
            ]
        except (ValueError, ValidationError) as exc:
            raise PlanError(f"malformed Lead output: {exc}") from exc
        if not questions:
            raise PlanError("Lead plan contains no questions")
        return InvestigationPlan(
            id=f"plan-{context.incident.id}",
            incident_id=context.incident.id,
            questions=questions,
            budgets=self._budgets,
        )
