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
        verification_id: Unique identifier for this verification check
        method: Verification method (e.g., "ruff", "pytest", "acceptance_probe", "structural_check")
        phase: Verification phase (e.g., "baseline", "post_patch", "constraint_audit")
        level: Verification level (e.g., "quick", "standard", "comprehensive")
        tier: Verification tier (e.g., "required", "affected", "optional")
        command: The command string that was executed
        passed: Whether the check passed (exit code 0)
        exit_code: The process exit code from command execution
        duration_seconds: Execution time in seconds
        timeout_seconds: Timeout budget that was configured for this check
        failure_type: Categorized failure type if check failed (e.g., "AssertionError")
        summary: Additional structured summary data for the check result
        subject_ids: List of acceptance criteria or constraint IDs associated with this check
        direct: Whether this check provides direct evidence for the subject_ids
        selection_reason: Reason why this test was selected (for tiered verification)
        test_node: Test node identifier for pytest checks (e.g., "tests/test_example.py::test_func")
        transition: Transition type from baseline to post-patch (for post-patch checks)
        baseline_check_id: Verification ID of the matching baseline check (for post-patch checks)
        failure_fingerprint: Stable identifier for failure pattern (for failed checks)
    """

    method: str
    phase: str
    level: str
    command: str
    passed: bool
    exit_code: int
    duration_seconds: float
    timeout_seconds: int = 60
    verification_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    failure_type: str | None = None
    summary: dict[str, Any] | None = None
    subject_ids: list[str] = field(default_factory=list)
    direct: bool = False
    tier: str = ""
    selection_reason: str = ""
    test_node: str = ""
    transition: str = ""
    baseline_check_id: str = ""
    failure_fingerprint: str = ""

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
        baseline_checks: List of baseline verification checks (before changes)
        post_patch_checks: List of post-patch verification checks (after changes)
        retry_count: Number of retry attempts for failed checks
        failed_level: The verification level at which failure occurred
        failure_type: Primary failure type classification from workflow
        patch: Git diff patch containing all changes made
        strategy: Verification strategy used (strict, balanced, focused)
        verification_status: Detailed verification status (VERIFIED, PARTIALLY_VERIFIED, FAILED)
        regression_coverage: Regression evidence coverage (FULL or INCOMPLETE)
        tier_summary: Summary of check results by tier
        transition_summary: Summary of check transitions from baseline to post-patch
    """

    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    passed: bool = True
    checks: list[CheckReport] = field(default_factory=list)
    baseline_checks: list[CheckReport] = field(default_factory=list)
    post_patch_checks: list[CheckReport] = field(default_factory=list)
    retry_count: int = 0
    failed_level: str | None = None
    failure_type: str | None = None
    patch: str = ""
    strategy: str = ""
    verification_status: str = ""
    regression_coverage: str = "FULL"
    tier_summary: dict[str, dict[str, Any]] = field(default_factory=dict)
    transition_summary: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Populate phase-specific lists from checks after initialization."""
        # Ensure phase-specific lists are populated from checks
        for check in self.checks:
            if check.phase == "baseline" and check not in self.baseline_checks:
                self.baseline_checks.append(check)
            elif check.phase == "post_patch" and check not in self.post_patch_checks:
                self.post_patch_checks.append(check)

    def add_check(self, check: CheckReport) -> None:
        """Add a check report to the verification report.

        Updates the overall passed status based on the new check result.

        Args:
            check: CheckReport to add to the verification report
        """
        self.checks.append(check)
        
        # Also add to phase-specific lists
        if check.phase == "baseline":
            self.baseline_checks.append(check)
        elif check.phase == "post_patch":
            self.post_patch_checks.append(check)
        
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

    def get_checks_by_phase(self, phase: str) -> list[CheckReport]:
        """Retrieve all checks for a specific verification phase.

        Args:
            phase: The verification phase to filter by (e.g., "baseline", "post_patch")

        Returns:
            List of CheckReport objects matching the specified phase
        """
        return [check for check in self.checks if check.phase == phase]

    def get_baseline_checks(self) -> list[CheckReport]:
        """Retrieve all baseline verification checks.

        Returns:
            List of CheckReport objects from the baseline phase
        """
        return self.baseline_checks

    def get_post_patch_checks(self) -> list[CheckReport]:
        """Retrieve all post-patch verification checks.

        Returns:
            List of CheckReport objects from the post-patch phase
        """
        return self.post_patch_checks

    def merge_baseline(self, baseline_report: VerificationReport) -> None:
        """Merge baseline checks into this report for comparison.

        This method integrates baseline verification checks into the current
        (typically post-patch) report to enable behavior change analysis.
        The merged report contains both baseline and post-patch checks while
        maintaining internal consistency across all check lists.

        Args:
            baseline_report: VerificationReport containing baseline checks to merge

        Notes:
            - Baseline checks are added to the master `checks` list
            - `baseline_checks` is populated from the baseline report
            - `post_patch_checks` remains unchanged (contains only post-patch checks)
            - The overall `passed` status is NOT affected by baseline failures
              (baseline failures are for comparison, not for determining final success)
            - Duplicate checks are avoided by checking verification_id
        """
        # Get baseline checks from the baseline report
        baseline_checks_to_merge = baseline_report.get_baseline_checks()

        # Add each baseline check to this report if not already present
        for baseline_check in baseline_checks_to_merge:
            # Check if this check is already in our baseline_checks (avoid duplicates)
            if not any(
                existing.verification_id == baseline_check.verification_id
                for existing in self.baseline_checks
            ):
                # Add to master checks list
                self.checks.insert(0, baseline_check)  # Insert at beginning for chronological order
                # Add to baseline_checks list
                self.baseline_checks.append(baseline_check)

        # Ensure post_patch_checks is correctly populated from existing checks
        # (in case it wasn't already)
        self.post_patch_checks = [
            check for check in self.checks if check.phase == "post_patch"
        ]

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
        checks = []
        for check_data in data.get("checks", []):
            # Handle backward compatibility for missing tier and selection_reason fields
            if "tier" not in check_data:
                check_data["tier"] = ""
            if "selection_reason" not in check_data:
                check_data["selection_reason"] = ""
            # Handle backward compatibility for new baseline-delta fields
            if "test_node" not in check_data:
                check_data["test_node"] = ""
            if "transition" not in check_data:
                check_data["transition"] = ""
            if "baseline_check_id" not in check_data:
                check_data["baseline_check_id"] = ""
            if "failure_fingerprint" not in check_data:
                check_data["failure_fingerprint"] = ""
            checks.append(CheckReport(**check_data))
        
        # Handle backward compatibility for reports without phase-specific lists
        baseline_checks = [
            CheckReport(**check_data) for check_data in data.get("baseline_checks", [])
        ]
        post_patch_checks = [
            CheckReport(**check_data) for check_data in data.get("post_patch_checks", [])
        ]
        
        # If phase-specific lists are empty but checks exist, populate them
        if not baseline_checks and not post_patch_checks and checks:
            for check in checks:
                if check.phase == "baseline":
                    baseline_checks.append(check)
                elif check.phase == "post_patch":
                    post_patch_checks.append(check)

        return cls(
            run_id=data.get("run_id", str(uuid.uuid4())),
            passed=data.get("passed", True),
            checks=checks,
            baseline_checks=baseline_checks,
            post_patch_checks=post_patch_checks,
            retry_count=data.get("retry_count", 0),
            failed_level=data.get("failed_level"),
            failure_type=data.get("failure_type"),
            patch=data.get("patch", ""),
            strategy=data.get("strategy", ""),
            verification_status=data.get("verification_status", ""),
            regression_coverage=data.get("regression_coverage", "FULL"),
            tier_summary=data.get("tier_summary", {}),
            transition_summary=data.get("transition_summary", {}),
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
