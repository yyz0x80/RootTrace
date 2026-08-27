"""Lead investigation planning agent."""

from pydantic import BaseModel, Field, ValidationError

from roottrace.agents.prompts import LEAD_SYSTEM_PROMPT, build_lead_prompt
from roottrace.agents.schema import (
    MAX_QUESTIONS,
    InvestigationPlan,
    PlanBudgets,
    PlanQuestion,
)
from roottrace.agents.specialists import ProviderProtocol, extract_json_object
from roottrace.evidence.schema import AgentRole
from roottrace.incident.context import IncidentContext
from roottrace.llm.usage import UsageTracker


class PlanError(Exception):
    """Raised when the Lead produces an unusable investigation plan."""


MAX_PLANNING_ATTEMPTS = 2
MAX_FEEDBACK_CHARS = 2_000


class PlanningDegradation(BaseModel):
    """Structured audit metadata for a deterministic planning fallback."""

    reason: str
    original_question_count: int = Field(gt=0)
    retained_question_count: int = Field(gt=0)
    repair_attempts: int = Field(ge=0)


def _repair_feedback(error: Exception) -> str:
    """Build bounded validation feedback for one planning repair retry."""
    text = str(error)
    if len(text) > MAX_FEEDBACK_CHARS:
        text = text[:MAX_FEEDBACK_CHARS] + "..."
    return (
        "\n\nPREVIOUS OUTPUT REJECTED:\n"
        "Fix exactly these validation errors and output the corrected JSON "
        f"object only:\n{text}"
    )


class LeadPlanner:
    """Plans the investigation; never commits to a root cause."""

    def __init__(
        self,
        *,
        provider: ProviderProtocol,
        usage: UsageTracker,
        budgets: PlanBudgets,
        agent_id: str = "lead",
        max_attempts: int = MAX_PLANNING_ATTEMPTS,
    ) -> None:
        if not 1 <= max_attempts <= 3:
            raise ValueError("max_attempts must be between 1 and 3")
        self.agent_id = agent_id
        self.role = AgentRole.LEAD
        self._provider = provider
        self._usage = usage
        self._budgets = budgets
        self._max_attempts = min(max_attempts, budgets.max_llm_calls)
        self._degradation: PlanningDegradation | None = None

    @property
    def degradation(self) -> PlanningDegradation | None:
        """Return fallback metadata when the accepted plan was degraded."""
        return self._degradation

    def run(self, context: IncidentContext) -> InvestigationPlan:
        """Produce a validated plan, with one bounded repair retry by default."""
        self._degradation = None
        prompt = build_lead_prompt(context)
        last_error: Exception | None = None
        fallback_questions: list[PlanQuestion] | None = None
        for attempt in range(self._max_attempts):
            attempt_prompt = prompt
            if attempt > 0 and last_error is not None:
                attempt_prompt += _repair_feedback(last_error)
            turn = self._provider.complete(
                messages=[
                    {"role": "system", "content": LEAD_SYSTEM_PROMPT},
                    {"role": "user", "content": attempt_prompt},
                ],
                tools=[],
            )
            self._usage.record(
                turn.prompt_tokens,
                turn.completion_tokens,
                turn.reasoning_tokens,
            )
            try:
                if turn.content is None or not turn.content.strip():
                    raise ValueError("Lead returned no content")
                data = extract_json_object(turn.content)
                raw_questions = data.get("questions", [])
                if not isinstance(raw_questions, list):
                    raise TypeError("Lead plan questions must be a list")
                questions = [
                    PlanQuestion.model_validate(question)
                    for question in raw_questions
                ]
                if not questions:
                    raise ValueError("Lead plan contains no questions")
                if len(questions) > MAX_QUESTIONS:
                    fallback_questions = questions
                return InvestigationPlan(
                    id=f"plan-{context.incident.id}",
                    incident_id=context.incident.id,
                    questions=questions,
                    budgets=self._budgets,
                )
            except (TypeError, ValueError, ValidationError) as exc:
                last_error = exc
                if attempt < self._max_attempts - 1:
                    continue
                if fallback_questions is not None:
                    try:
                        plan = InvestigationPlan(
                            id=f"plan-{context.incident.id}",
                            incident_id=context.incident.id,
                            questions=fallback_questions[:MAX_QUESTIONS],
                            budgets=self._budgets,
                        )
                    except ValidationError as fallback_error:
                        raise PlanError(
                            f"malformed Lead output: {fallback_error}"
                        ) from fallback_error
                    self._degradation = PlanningDegradation(
                        reason="question_limit_exceeded",
                        original_question_count=len(fallback_questions),
                        retained_question_count=len(plan.questions),
                        repair_attempts=attempt,
                    )
                    return plan
                raise PlanError(f"malformed Lead output: {exc}") from exc

        raise PlanError("Lead produced no valid investigation plan")
