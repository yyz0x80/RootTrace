from typing import Literal

from pydantic import BaseModel, Field


class AcceptanceCriterion(BaseModel):
    """Represents a single acceptance criterion for validating task completion.

    Each criterion should be testable and verifiable through code or tests.
    """
    id: str
    description: str


class NormalizedIssue(BaseModel):
    """Structured representation of a normalized issue for PatchPilot processing.

    This schema converts raw natural language issues into structured format
    that guides the Agent's execution and verification process.
    """
    title: str

    task_type: Literal[
        "bug",
        "feature",
        "test",
        "refactor",
        "dependency",
        "other",
    ]

    problem_statement: str

    acceptance_criteria: list[AcceptanceCriterion] = Field(
        default_factory=list
    )

    constraints: list[str] = Field(default_factory=list)

    ambiguous_points: list[str] = Field(default_factory=list)

    expected_test_areas: list[str] = Field(default_factory=list)

    implementation_notes: list[str] = Field(default_factory=list)