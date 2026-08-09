"""Failure classification for verification results.

This module provides functionality to classify different types of failures
that can occur during code verification. It analyzes failure summaries to
determine the root cause category, enabling targeted remediation strategies.

Classification categories include:
- Code failures: Syntax errors, type errors, import errors
- Test failures: Assertion failures, test-specific errors
- Environment failures: Missing dependencies, network issues
- Permission failures: Access denied errors
- Timeouts: Commands that exceeded time limits
- Model failures: AI agent generation issues
- Requirement ambiguity: Unclear or conflicting specifications
"""

from __future__ import annotations

from enum import Enum

from patchpilot.verification.error_parser import FailureSummary


class FailureType(str, Enum):
    """Enumeration of possible failure types."""

    CODE_FAILURE = "CODE_FAILURE"
    TEST_FAILURE = "TEST_FAILURE"
    ENVIRONMENT_FAILURE = "ENVIRONMENT_FAILURE"
    PERMISSION_FAILURE = "PERMISSION_FAILURE"
    TIMEOUT = "TIMEOUT"
    MODEL_FAILURE = "MODEL_FAILURE"
    REQUIREMENT_AMBIGUITY = "REQUIREMENT_AMBIGUITY"


def classify_failure(summary: FailureSummary) -> FailureType:
    """Classify a failure summary into a specific failure type.

    Analyzes the failure summary to determine the most likely root cause
    category based on error patterns, command type, and output content.

    Args:
        summary: FailureSummary containing command execution details

    Returns:
        FailureType enum value indicating the classified failure category
    """
    text = summary.relevant_output.lower()

    # Check for timeout first as it's a distinct category
    if summary.timed_out:
        return FailureType.TIMEOUT

    # Check for permission-related errors
    permission_markers = ("permission denied", "operation not permitted")
    if any(marker in text for marker in permission_markers):
        return FailureType.PERMISSION_FAILURE

    # Check for environment-related failures
    environment_markers = (
        "command not found",
        "modulenotfounderror",
        "could not resolve host",
        "temporary failure in name resolution",
        "connection refused",
    )
    if any(marker in text for marker in environment_markers):
        return FailureType.ENVIRONMENT_FAILURE

    # Ruff lint errors are code quality issues
    if summary.command.strip().startswith("ruff"):
        return FailureType.CODE_FAILURE

    # Failed tests indicate test failures
    if summary.failed_tests:
        return FailureType.TEST_FAILURE

    # Specific Python error types indicate code failures
    code_error_types = {
        "SyntaxError",
        "IndentationError",
        "NameError",
        "TypeError",
        "ImportError",
    }
    if summary.error_type in code_error_types:
        return FailureType.CODE_FAILURE

    # Pytest commands without explicit test failures are test-related
    if summary.command.strip().startswith("pytest"):
        return FailureType.TEST_FAILURE

    # Default to code failure for unclassified errors
    return FailureType.CODE_FAILURE
