"""Lead planning contracts owned by the agent capability."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from roottrace.evidence.schema import AgentRole
from roottrace.incident.schema import StableId

MAX_QUESTION_CHARS = 500
MAX_QUESTIONS = 10


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
