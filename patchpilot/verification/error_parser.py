"""Error parsing utilities for verification command results.

This module provides functions to parse and extract meaningful information
from command execution results, particularly for pytest and ruff failures.
It identifies failed tests, error types, and extracts relevant output for
debugging and reporting.

The parser handles:
- Pytest failure patterns
- Ruff lint errors
- Timeout detection
- Common Python error types
- Relevant output extraction
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from patchpilot.sandbox.docker_runner import CommandResult


@dataclass
class FailureSummary:
    """Structured summary of a command execution failure.
    
    Attributes:
        command: The command string that was executed
        exit_code: The process exit code
        failed_tests: List of test names that failed (for pytest)
        error_type: Categorized error type (e.g., "AssertionError", "Timeout")
        relevant_output: Extracted relevant output lines for debugging
        timed_out: Whether the command timed out
    """
    command: str
    exit_code: int
    failed_tests: list[str]
    error_type: str
    relevant_output: str
    timed_out: bool = False


def _find_failed_tests(text: str) -> list[str]:
    """Extract failed test names from pytest output.
    
    Args:
        text: Combined stdout and stderr from command execution
        
    Returns:
        List of test names that failed, as extracted from FAILED lines
    """
    return re.findall(
        r"FAILED\s+([^\s]+)",
        text,
        flags=re.MULTILINE,
    )


def _find_error_type(text: str) -> str:
    """Identify the primary error type from command output.
    
    Searches for common Python exception types in the output text
    and returns the first match found.
    
    Args:
        text: Combined stdout and stderr from command execution
        
    Returns:
        The identified error type, or "UnknownError" if no match found
    """
    candidates = [
        "AssertionError",
        "SyntaxError",
        "IndentationError",
        "TypeError",
        "ValueError",
        "NameError",
        "ImportError",
        "ModuleNotFoundError",
        "PermissionError",
    ]

    for error_type in candidates:
        if error_type in text:
            return error_type

    return "UnknownError"


def _extract_relevant_output(
    text: str,
    max_lines: int = 20,
) -> str:
    """Extract relevant output lines for error diagnosis.
    
    Filters output to include lines containing error markers, falling back
    to the last N lines if no markers are found. Limits output to prevent
    excessive size.
    
    Args:
        text: Combined stdout and stderr from command execution
        max_lines: Maximum number of lines to include in output
        
    Returns:
        Extracted relevant output, truncated to 3000 characters
    """
    lines = [
        line
        for line in text.splitlines()
        if line.strip()
    ]

    markers = (
        "FAILED",
        "ERROR",
        "Error",
        "AssertionError",
        "SyntaxError",
        "TypeError",
        "ValueError",
        "E ",
    )

    relevant = [
        line
        for line in lines
        if any(marker in line for marker in markers)
    ]

    if not relevant:
        relevant = lines[-max_lines:]

    return "\n".join(relevant[-max_lines:])[:3000]


def parse_failure(
    result: CommandResult,
) -> FailureSummary:
    """Parse a command failure result into a structured summary.
    
    Analyzes the command result to extract key information about the failure,
    including error type, failed tests, and relevant output.
    
    Args:
        result: CommandResult from a failed command execution
        
    Returns:
        FailureSummary with structured information about the failure
    """
    text = f"{result.stdout}\n{result.stderr}"

    if result.timed_out:
        error_type = "Timeout"

    elif result.command.strip().startswith("ruff"):
        error_type = "LintError"

    else:
        error_type = _find_error_type(text)

    return FailureSummary(
        command=result.command,
        exit_code=result.exit_code,
        failed_tests=_find_failed_tests(text),
        error_type=error_type,
        relevant_output=_extract_relevant_output(text),
        timed_out=result.timed_out,
    )
