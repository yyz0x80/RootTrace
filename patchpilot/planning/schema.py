from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    """Declarative behavior probe executed outside the generated patch."""

    model_config = ConfigDict(extra="forbid")

    probe_id: str
    module: str = Field(pattern=r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)*$")
    target: str = Field(pattern=r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)*$")
    probe_type: Literal["function_io", "exception", "state_change", "invariant", "return_structure"]
    criterion_ids: list[str] = Field(default_factory=list)
    constructor_args: list[Any] = Field(default_factory=list)
    constructor_kwargs: dict[str, Any] = Field(default_factory=dict)
    arguments: list[Any] = Field(default_factory=list)
    keyword_arguments: dict[str, Any] = Field(default_factory=dict)
    assertion: Literal[
        "equals",
        "attribute_equals",
        "raises",
        "truthy",
        "falsy",
    ]
    expected: Any = None
    attribute: str = Field(default="", pattern=r"^$|^[A-Za-z_]\w*(\.[A-Za-z_]\w*)*$")
    exception: str = Field(default="", pattern=r"^$|^[A-Za-z_]\w*$")

    @model_validator(mode="after")
    def validate_assertion_details(self) -> "AcceptanceProbeSpec":
        """Require the operands used by the selected assertion."""
        if self.assertion == "attribute_equals" and not self.attribute:
            raise ValueError("attribute_equals probes require attribute")
        if self.assertion == "raises" and not self.exception:
            raise ValueError("raises probes require exception")
        return self


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
        "dataclass_field",
        "method_parameter",
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
