"""Verification report generation and management.

This module provides data structures for generating and managing verification
reports from code change validation. It captures the results of individual
verification checks (like pytest and ruff) and aggregates them into a comprehensive
verification report that can be saved to disk for later analysis.

The report structures support:
- Individual check results with pass/fail status
- Detailed error information and classification
- Execution timing and retry tracking
- JSON serialization for persistence
- Aggregated verification status
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CheckReport:
    """Report for a single verification check execution.

    Attributes:
        level: Verification level (e.g., "quick", "standard", "comprehensive")
        command: The command string that was executed
        passed: Whether the check passed (exit code 0)
        exit_code: The process exit code from command execution
        duration_seconds: Execution time in seconds
        failure_type: Categorized failure type if check failed (e.g., "AssertionError")
        summary: Additional structured summary data for the check result
    """

    level: str
    command: str
    passed: bool
    exit_code: int
    duration_seconds: float
    failure_type: str | None = None
    summary: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the check report to a dictionary for serialization.

        Returns:
            Dictionary representation of the check report
        """
        return asdict(self)


@dataclass
class VerificationReport:
    """Comprehensive verification report for code change validation.

    Aggregates multiple check reports into a single verification result,
    tracking overall status, retry attempts, and failure classification.

    Attributes:
        run_id: Unique identifier for this verification run
        passed: Overall verification status (True if all checks passed)
        checks: List of individual check reports
        retry_count: Number of retry attempts for failed checks
        failed_level: The verification level at which failure occurred
        failure_type: Primary failure type classification from workflow
        patch: Git diff patch containing all changes made
    """

    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    passed: bool = True
    checks: list[CheckReport] = field(default_factory=list)
    retry_count: int = 0
    failed_level: str | None = None
    failure_type: str | None = None
    patch: str = ""

    def add_check(self, check: CheckReport) -> None:
        """Add a check report to the verification report.

        Updates the overall passed status based on the new check result.

        Args:
            check: CheckReport to add to the verification report
        """
        self.checks.append(check)
        if not check.passed:
            self.passed = False
            self.failed_level = check.level
            self.failure_type = check.failure_type

    def get_failed_checks(self) -> list[CheckReport]:
        """Retrieve all checks that failed.

        Returns:
            List of CheckReport objects where passed is False
        """
        return [check for check in self.checks if not check.passed]

    def get_passed_checks(self) -> list[CheckReport]:
        """Retrieve all checks that passed.

        Returns:
            List of CheckReport objects where passed is True
        """
        return [check for check in self.checks if check.passed]

    def get_checks_by_level(self, level: str) -> list[CheckReport]:
        """Retrieve all checks for a specific verification level.

        Args:
            level: The verification level to filter by

        Returns:
            List of CheckReport objects matching the specified level
        """
        return [check for check in self.checks if check.level == level]

    def total_duration(self) -> float:
        """Calculate total execution time across all checks.

        Returns:
            Sum of duration_seconds for all checks
        """
        return sum(check.duration_seconds for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        """Convert the verification report to a dictionary for serialization.

        Returns:
            Dictionary representation with checks converted to dicts
        """
        data = asdict(self)
        data["checks"] = [check.to_dict() for check in self.checks]
        return data

    def save(self, path: Path) -> None:
        """Save the verification report to a JSON file.

        Creates parent directories if they don't exist. The file is written
        with UTF-8 encoding and pretty-printed with 2-space indentation.

        Args:
            path: Path where the JSON report should be saved
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text(
            json.dumps(
                self.to_dict(),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> VerificationReport:
        """Load a verification report from a JSON file.

        Args:
            path: Path to the JSON report file to load

        Returns:
            VerificationReport instance loaded from the file

        Raises:
            FileNotFoundError: If the specified file does not exist
            json.JSONDecodeError: If the file contains invalid JSON
        """
        data = json.loads(path.read_text(encoding="utf-8"))

        # Convert check dicts back to CheckReport objects
        checks = [
            CheckReport(**check_data) for check_data in data.get("checks", [])
        ]

        return cls(
            run_id=data.get("run_id", str(uuid.uuid4())),
            passed=data.get("passed", True),
            checks=checks,
            retry_count=data.get("retry_count", 0),
            failed_level=data.get("failed_level"),
            failure_type=data.get("failure_type"),
            patch=data.get("patch", ""),
        )


def failure_fingerprint(report: VerificationReport) -> tuple:
    """Generate a fingerprint for failure detection to detect repeated failures.

    Creates a unique identifier based on the failure characteristics to determine
    if a repair attempt has failed with the same error, indicating that further
    repair attempts would likely be futile.

    Args:
        report: VerificationReport containing the failure information

    Returns:
        A tuple containing the failure type, failed tests, error type, and
        relevant output (truncated to 500 characters) for fingerprint comparison
    """
    if report.passed or not report.checks:
        return ()

    # Get the most recent failed check
    failed_checks = report.get_failed_checks()
    if not failed_checks:
        return ()

    failed = failed_checks[-1]
    summary = failed.summary or {}

    return (
        report.failure_type,
        tuple(summary.get("failed_tests", [])),
        summary.get("error_type"),
        summary.get("relevant_output", "")[:500],
    )
