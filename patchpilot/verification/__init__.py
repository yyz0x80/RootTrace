"""Verification module for PatchPilot.

This module provides tools for verifying code changes through:
- Error parsing and analysis
- Test execution and reporting
- Verification result aggregation
- Report generation and persistence
- Failure fingerprinting for repair loop optimization
- Deterministic verification with Ruff and pytest
"""

from patchpilot.verification.error_parser import (
    FailureSummary,
    parse_failure,
)
from patchpilot.verification.report import (
    CheckReport,
    VerificationReport,
    failure_fingerprint,
)
from patchpilot.verification.verifier import Verifier

__all__ = [
    "CheckReport",
    "FailureSummary",
    "VerificationReport",
    "Verifier",
    "failure_fingerprint",
    "parse_failure",
]
