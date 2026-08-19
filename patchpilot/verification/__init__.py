"""Verification module for PatchPilot.

This module provides tools for verifying code changes through:
- Error parsing and analysis
- Test execution and reporting
- Verification result aggregation
- Report generation and persistence
- Failure fingerprinting for repair loop optimization
- Deterministic verification with Ruff and pytest
- Acceptance Probes for model-generated verification
- Structural Checkers for AST-based validation
"""

from patchpilot.verification.error_parser import (
    FailureSummary,
    parse_failure,
)
from patchpilot.verification.probes import (
    AcceptanceProbe,
    ProbeExecutionResult,
    ProbeRunner,
    ProbeStep,
    ProbeType,
    ProbeValidationError,
    ProbeValidator,
    StepResult,
)
from patchpilot.verification.report import (
    CheckReport,
    VerificationReport,
    failure_fingerprint,
)
from patchpilot.verification.structural import (
    ASTChecker,
    CheckResult as StructuralCheckResult,
    CheckType,
    StructuralCheck,
    StructuralReport,
    StructuralRunner,
)
from patchpilot.verification.verifier import Verifier

__all__ = [
    "AcceptanceProbe",
    "CheckReport",
    "FailureSummary",
    "ProbeExecutionResult",
    "ProbeRunner",
    "ProbeStep",
    "ProbeType",
    "ProbeValidationError",
    "ProbeValidator",
    "StepResult",
    "ASTChecker",
    "StructuralCheck",
    "StructuralCheckResult",
    "StructuralReport",
    "StructuralRunner",
    "CheckType",
    "VerificationReport",
    "Verifier",
    "failure_fingerprint",
    "parse_failure",
]
