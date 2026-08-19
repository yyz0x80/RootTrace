from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class ChangeAction(str, Enum):
    """Action type for a planned change."""

    CREATE = "create"
    MODIFY = "modify"
    DELETE = "delete"


class PlanDisposition(str, Enum):
    """Disposition of the overall change plan."""

    CHANGE_REQUIRED = "change_required"
    ALREADY_SATISFIED = "already_satisfied"
    REPOSITORY_MISMATCH = "repository_mismatch"
    BLOCKED = "blocked"


class CriterionPlanDetail(str, Enum):
    """Detailed disposition for a single acceptance criterion."""

    TO_IMPLEMENT = "to_implement"
    TO_PRESERVE = "to_preserve"
    ALREADY_SATISFIED = "already_satisfied"
    CANNOT_VERIFY = "cannot_verify"


class VerificationSpec(BaseModel):
    """Verification specification for acceptance criteria.

    Each verification spec defines how to verify a criterion,
    including the command to run and the expected result.
    """

    criterion_id: str
    command: str
    expected_result: str = Field(default="")
    baseline_evidence: str = Field(default="")


class CriterionPlan(BaseModel):
    """Plan for a single acceptance criterion.

    Each criterion plan specifies the disposition of the criterion
    and relevant source files for structural criteria.
    """

    criterion_id: str
    disposition: CriterionPlanDetail
    relevant_source_files: list[str] = Field(default_factory=list)
    baseline_evidence: str = Field(default="")


class PlannedChange(BaseModel):
    """Represents a single planned code change.

    Each planned change specifies a target file path, an action type,
    a description of what will change, and acceptance criteria to verify the change is correct.
    """

    path: str
    action: ChangeAction
    description: str
    acceptance_criteria: list[str] = Field(default_factory=list)
    criterion_ids: list[str] = Field(default_factory=list)


class PlannedTest(BaseModel):
    """Represents a planned test execution for verification.

    Each planned test specifies a command to run, the purpose of the test,
    and acceptance criteria to verify the test results.
    """

    command: str
    purpose: str
    acceptance_criteria: list[str] = Field(default_factory=list)
    criterion_ids: list[str] = Field(default_factory=list)


class AcceptanceProbeSpec(BaseModel):
    """Specification for an acceptance probe.

    Acceptance probes are model-generated verification scripts that test
    specific aspects of code changes without becoming part of the patch itself.
    """

    probe_id: str
    target_function: str
    probe_type: Literal["function_io", "exception", "state_change", "invariant", "return_structure"]
    criterion_ids: list[str] = Field(default_factory=list)
    steps: list[dict[str, Any]] = Field(default_factory=list)
    setup_code: str = ""
    teardown_code: str = ""


class StructuralCheckSpec(BaseModel):
    """Specification for a structural check.

    Structural checks use AST analysis to verify code structure without execution.
    """

    check_id: str
    check_type: Literal[
        "function_exists",
        "signature_preserved",
        "call_relationship",
        "no_new_imports",
        "method_exists",
        "decorator_exists",
    ]
    target: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    criterion_ids: list[str] = Field(default_factory=list)
    file_path: str


class ChangePlan(BaseModel):
    """Comprehensive plan for addressing a normalized issue.

    The change plan includes relevant files to examine, specific changes to make,
    tests to run for verification, items explicitly out of scope, and a risk assessment.
    """

    base_commit: str = ""
    repository_match: bool = True
    repository_mismatch_reason: str | None = None
    relevant_files: list[str] = Field(default_factory=list)

    planned_changes: list[PlannedChange] = Field(default_factory=list)

    planned_tests: list[PlannedTest] = Field(default_factory=list)

    out_of_scope: list[str] = Field(default_factory=list)

    risk_level: Literal["low", "medium", "high"]

    # New fields for enhanced AC planning
    criterion_plans: list[CriterionPlan] = Field(default_factory=list)
    verification_specs: list[VerificationSpec] = Field(default_factory=list)
    plan_disposition: PlanDisposition = PlanDisposition.CHANGE_REQUIRED

    # Specialized verification specifications
    acceptance_probes: list[AcceptanceProbeSpec] = Field(default_factory=list)
    structural_checks: list[StructuralCheckSpec] = Field(default_factory=list)
