"""Verification module for PatchPilot.

This module provides tools for verifying code changes through:
- Error parsing and analysis
- Test execution and reporting
- Verification result aggregation
"""

from patchpilot.verification.error_parser import (
    FailureSummary,
    parse_failure,
)

__all__ = [
    "FailureSummary",
    "parse_failure",
]
