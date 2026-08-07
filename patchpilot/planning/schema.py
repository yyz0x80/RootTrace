from typing import Literal

from pydantic import BaseModel, Field


class PlannedChange(BaseModel):
    """Represents a single planned code change.

    Each planned change specifies a target file, a description of what will change,
    and acceptance criteria to verify the change is correct.
    """

    file: str
    description: str
    acceptance_criteria: list[str] = Field(default_factory=list)


class PlannedTest(BaseModel):
    """Represents a planned test execution for verification.

    Each planned test specifies a command to run, the purpose of the test,
    and acceptance criteria to verify the test results.
    """

    command: str
    purpose: str
    acceptance_criteria: list[str] = Field(default_factory=list)


class ChangePlan(BaseModel):
    """Comprehensive plan for addressing a normalized issue.

    The change plan includes relevant files to examine, specific changes to make,
    tests to run for verification, items explicitly out of scope, and a risk assessment.
    """

    relevant_files: list[str] = Field(default_factory=list)

    planned_changes: list[PlannedChange] = Field(default_factory=list)

    planned_tests: list[PlannedTest] = Field(default_factory=list)

    out_of_scope: list[str] = Field(default_factory=list)

    risk_level: Literal["low", "medium", "high"]
