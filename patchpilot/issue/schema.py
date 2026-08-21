from typing import Literal

from pydantic import BaseModel, Field

CriterionKind = Literal["behavior", "preservation", "structural"]
"""Classification of acceptance criterion type.

behavior: Final program behavior (e.g., "reject invalid input")
preservation: Existing behavior must not regress (e.g., "keep function signature")
structural: Required source code structure (e.g., "must call normalize_email")
"""

ConstraintKind = Literal["READ_SCOPE", "WRITE_SCOPE", "COMMAND", "NETWORK", "OTHER"]
"""Classification of constraint type.

READ_SCOPE: Restrictions on what files can be read
WRITE_SCOPE: Restrictions on what files can be modified
COMMAND: Restrictions on commands that can be run
NETWORK: Restrictions on network access
OTHER: Other execution boundary constraints
"""


class AcceptanceCriterion(BaseModel):
    """Represents a single acceptance criterion for validating task completion.

    Each criterion should be testable and verifiable through code or tests.
    """
    id: str
    description: str
    kind: CriterionKind = Field(default="behavior")
    required: bool = Field(default=True)


class TaskConstraint(BaseModel):
    """Represents an execution boundary constraint for the Agent.

    Constraints define what the Agent must NOT do during execution.
    """
    id: str
    description: str
    kind: ConstraintKind


class ArtifactRequirement(BaseModel):
    """Describe a file-level deliverable requested by the issue author."""

    kind: Literal["target_test_change", "documentation", "configuration", "other"]
    description: str
    required: bool = Field(default=True)


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

    constraints: list[TaskConstraint] = Field(default_factory=list)

    verification_requirements: list[str] = Field(default_factory=list)

    artifact_requirements: list[ArtifactRequirement] = Field(default_factory=list)

    ambiguous_points: list[str] = Field(default_factory=list)

    expected_test_areas: list[str] = Field(default_factory=list)

    implementation_notes: list[str] = Field(default_factory=list)
