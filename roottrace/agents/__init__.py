"""Lead and evidence-specialist reasoning capability."""

from roottrace.agents.lead import LeadPlanner, PlanError
from roottrace.agents.schema import InvestigationPlan, PlanBudgets, PlanQuestion
from roottrace.agents.specialists import (
    CodeSpecialist,
    GitHistorySpecialist,
    IssueCISpecialist,
    ProviderProtocol,
    SpecialistOutput,
    extract_json_object,
)
from roottrace.agents.synthesis import LeadSynthesizer, SynthesisError

__all__ = [
    "CodeSpecialist", "GitHistorySpecialist", "InvestigationPlan",
    "IssueCISpecialist", "LeadPlanner", "LeadSynthesizer", "PlanBudgets",
    "PlanError", "PlanQuestion", "ProviderProtocol", "SpecialistOutput",
    "SynthesisError", "extract_json_object",
]
