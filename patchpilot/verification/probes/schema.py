"""Schema definitions for Acceptance Probes.

This module defines the structured data models for acceptance probes,
which are model-generated verification scripts that test specific aspects
of code changes without becoming part of the patch itself.

Probes support various verification types:
- Function input/output validation
- Exception handling verification
- State change detection
- Invariant checking
- Return structure validation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ProbeType(StrEnum):
    """Classification of probe verification types."""

    FUNCTION_IO = "function_io"
    EXCEPTION = "exception"
    STATE_CHANGE = "state_change"
    INVARIANT = "invariant"
    RETURN_STRUCTURE = "return_structure"


@dataclass
class ProbeStep:
    """Single step in a probe execution sequence.

    Attributes:
        description: Human-readable description of what this step tests
        code: Python code to execute for this step
        expected_outcome: Expected result (e.g., "no_exception", "specific_value")
        tolerance: Optional tolerance for numeric comparisons
    """

    description: str
    code: str
    expected_outcome: str
    tolerance: float | None = None


@dataclass
class AcceptanceProbe:
    """Structured acceptance probe for verification.

    An acceptance probe is a model-generated verification script that tests
    specific aspects of a code change. Probes are executed in temporary
    directories and do not become part of the final patch.

    Attributes:
        id: Unique identifier for this probe
        name: Human-readable name for the probe
        description: Detailed description of what the probe verifies
        probe_type: Type of verification this probe performs
        target_function: Target function or method being tested
        steps: Sequence of verification steps
        setup_code: Optional setup code to run before steps
        teardown_code: Optional teardown code to run after steps
        subject_ids: Acceptance criteria IDs this probe validates
        artifacts: List of artifact files generated during execution
    """

    id: str
    name: str
    description: str
    probe_type: ProbeType
    target_function: str
    steps: list[ProbeStep]
    setup_code: str = ""
    teardown_code: str = ""
    subject_ids: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert probe to dictionary for serialization.

        Returns:
            Dictionary representation of the probe
        """
        from dataclasses import asdict

        return asdict(self)


@dataclass
class ProbeExecutionResult:
    """Result of executing an acceptance probe.

    Attributes:
        probe_id: ID of the probe that was executed
        passed: Whether the probe passed all steps
        step_results: Results for each individual step
        execution_time_seconds: Total execution time
        output: Captured stdout/stderr from execution
        error: Error message if execution failed
        artifacts: Paths to artifact files generated
    """

    probe_id: str
    passed: bool
    step_results: list[StepResult]
    execution_time_seconds: float
    output: str
    error: str | None = None
    artifacts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for serialization.

        Returns:
            Dictionary representation of the execution result
        """
        from dataclasses import asdict

        return asdict(self)


@dataclass
class StepResult:
    """Result of executing a single probe step.

    Attributes:
        step_index: Index of the step in the probe
        description: Description of the step
        passed: Whether the step passed
        expected_outcome: What was expected
        actual_outcome: What was actually observed
        error: Error message if step failed
    """

    step_index: int
    description: str
    passed: bool
    expected_outcome: str
    actual_outcome: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert step result to dictionary for serialization.

        Returns:
            Dictionary representation of the step result
        """
        from dataclasses import asdict

        return asdict(self)
