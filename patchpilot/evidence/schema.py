"""Data schemas for acceptance evidence mapping and completion states."""

from enum import Enum

from pydantic import BaseModel, Field


class BehaviorChangeStatus(str, Enum):
    """Status for behavior change verification (baseline to post-patch)."""

    PASS = "PASS"
    ALREADY_SATISFIED = "ALREADY_SATISFIED"
    FAIL = "FAIL"
    UNVERIFIED = "UNVERIFIED"


class BehaviorPreservationStatus(str, Enum):
    """Status for behavior preservation verification."""

    PASS = "PASS"
    FAIL = "FAIL"
    UNVERIFIED = "UNVERIFIED"


class StructuralContractStatus(str, Enum):
    """Status for structural contract verification (AST/mock checks)."""

    PASS = "PASS"
    FAIL = "FAIL"
    UNVERIFIED = "UNVERIFIED"


class ConstraintStatus(str, Enum):
    """Status for constraint verification (policy compliance)."""

    COMPLIANT = "COMPLIANT"
    VIOLATED = "VIOLATED"
    UNSUPPORTED = "UNSUPPORTED"
    ADVISORY = "ADVISORY"


class ConstraintSeverity(str, Enum):
    """Severity level for constraint violations or unsupported status."""

    CRITICAL = "CRITICAL"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class CriterionRequirement(str, Enum):
    """Requirement level for acceptance criteria."""

    REQUIRED = "REQUIRED"
    OPTIONAL = "OPTIONAL"


class EvidenceStatus(str, Enum):
    """Verification status for a single acceptance criterion."""

    PASS = "PASS"
    FAIL = "FAIL"
    UNVERIFIED = "UNVERIFIED"


class CompletionState(str, Enum):
    """Overall completion state for a task based on acceptance evidence.

    States are evaluated by Workflow in fixed priority order:
    1. NEEDS_CLARIFICATION - Issue has unresolved ambiguities
    2. BLOCKED - Environment, permission, or scope issues prevent execution
    3. FAILED - Any AC is FAIL or unrecoverable code failure occurred
    4. VERIFIED - Deterministic verification passed
    5. PARTIALLY_VERIFIED - Deterministic verification did not fully pass
    """

    VERIFIED = "VERIFIED"
    PARTIALLY_VERIFIED = "PARTIALLY_VERIFIED"
    BLOCKED = "BLOCKED"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    FAILED = "FAILED"
    REGRESSION = "REGRESSION"


class FailureType(str, Enum):
    """Type of failure when completion state is FAILED."""

    CONSTRAINT_VIOLATION = "CONSTRAINT_VIOLATION"
    ACCEPTANCE_CRITERION_FAILURE = "ACCEPTANCE_CRITERION_FAILURE"
    EXECUTION_FAILURE = "EXECUTION_FAILURE"


class BehaviorChangeEvidence(BaseModel):
    """Evidence for behavior change verification.

    Tracks verification results from baseline to post-patch execution.

    Attributes:
        status: Behavior change status based on baseline and post-patch results.
        baseline_passed: Whether baseline checks passed.
        post_patch_passed: Whether post-patch checks passed.
        explanation: Human-readable explanation of the status determination.
    """

    status: BehaviorChangeStatus
    baseline_passed: bool
    post_patch_passed: bool
    explanation: str


class BehaviorPreservationEvidence(BaseModel):
    """Evidence for behavior preservation verification.

    Tracks that existing behavior is not broken by changes.

    Attributes:
        status: Behavior preservation status.
        baseline_passed: Whether baseline checks passed.
        post_patch_passed: Whether post-patch checks passed.
        explanation: Human-readable explanation of the status determination.
    """

    status: BehaviorPreservationStatus
    baseline_passed: bool
    post_patch_passed: bool
    explanation: str


class StructuralContractEvidence(BaseModel):
    """Evidence for structural contract verification.

    Tracks AST/mock verification results.

    Attributes:
        status: Structural contract status.
        has_specialized_check: Whether specialized AST/mock checks were run.
        check_passed: Whether specialized checks passed.
        has_pytest_only: Whether only pytest was available.
        explanation: Human-readable explanation of the status determination.
    """

    status: StructuralContractStatus
    has_specialized_check: bool
    check_passed: bool
    has_pytest_only: bool
    explanation: str


class ConstraintEvidence(BaseModel):
    """Evidence for constraint verification.

    Tracks policy compliance during code changes.

    Attributes:
        status: Constraint compliance status.
        severity: Severity level for the constraint status.
        has_hard_policy_violation: Whether hard policy was violated.
        has_attempted_violation: Whether violation was attempted but rejected.
        has_compilation_error: Whether compilation failed.
        has_advisory: Whether advisory issues exist.
        explanation: Human-readable explanation of the status determination.
    """

    status: ConstraintStatus
    severity: ConstraintSeverity = ConstraintSeverity.MEDIUM
    has_hard_policy_violation: bool
    has_attempted_violation: bool
    has_compilation_error: bool
    has_advisory: bool
    explanation: str


class AcceptanceEvidence(BaseModel):
    """Evidence mapping for a single acceptance criterion.

    Represents the verification status and supporting evidence for one
    acceptance criterion, linking it to actual code changes, test results,
    and command outputs.

    Attributes:
        criterion_id: ID reference to the original AcceptanceCriterion.
        description: Human-readable description of the acceptance criterion.
        status: Verification status based on concrete evidence.
        required: Whether this criterion is required for completion.
        changed_files: List of files that were modified to satisfy this criterion.
        tests: List of test names/paths that verify this criterion.
        command_results: List of verification command outputs supporting this criterion.
        explanation: Human-readable explanation of why this status was assigned.
        behavior_change: Evidence for behavior change verification.
        behavior_preservation: Evidence for behavior preservation verification.
        structural_contract: Evidence for structural contract verification.
        constraint: Evidence for constraint verification.
    """

    criterion_id: str
    description: str
    status: EvidenceStatus
    required: bool = True
    changed_files: list[str] = Field(default_factory=list)
    tests: list[str] = Field(default_factory=list)
    command_results: list[str] = Field(default_factory=list)
    explanation: str
    behavior_change: BehaviorChangeEvidence | None = None
    behavior_preservation: BehaviorPreservationEvidence | None = None
    structural_contract: StructuralContractEvidence | None = None
    constraint: ConstraintEvidence | None = None


class AcceptanceCoverageReport(BaseModel):
    """Complete acceptance criteria coverage report.

    Aggregates evidence for all acceptance criteria and provides the overall
    completion state for the task.

    Attributes:
        acceptance_evidence: List of evidence for each acceptance criterion.
        completion_state: Overall task completion state.
        summary: Human-readable summary of the verification results.
    """

    acceptance_evidence: list[AcceptanceEvidence] = Field(default_factory=list)
    completion_state: CompletionState = CompletionState.NEEDS_CLARIFICATION
    summary: str = ""
