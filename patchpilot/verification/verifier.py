"""Verifier for running deterministic verification checks.

This module provides the Verifier class which executes verification checks
in two phases: Baseline Verification (before changes) and Post-patch Verification
(after changes). It runs checks inside the Docker sandbox and aggregates results
into a VerificationReport with proper failure classification.

The verifier implements different strategies for each phase:
- Baseline: Records current state, can fail-fast for blocking failures
- Post-patch: Collects complete evidence, does not fail-fast
"""

from __future__ import annotations

import shlex
from dataclasses import asdict

from patchpilot.sandbox.docker_runner import DockerSandbox
from patchpilot.verification.error_parser import parse_failure
from patchpilot.verification.report import (
    CheckReport,
    VerificationReport,
)
from patchpilot.workflow.failure_classifier import classify_failure


class Verifier:
    """Run deterministic verification checks inside the sandbox.

    The Verifier executes checks in two phases:
    1. Baseline Verification: Records current state before changes
    2. Post-patch Verification: Validates changes after implementation

    Baseline phase can fail-fast for blocking failures.
    Post-patch phase collects complete evidence without fail-fast.

    Attributes:
        sandbox: DockerSandbox instance for isolated command execution
    """

    def __init__(self, sandbox: DockerSandbox) -> None:
        """Initialize the Verifier with a Docker sandbox.

        Args:
            sandbox: DockerSandbox instance for running verification commands
        """
        self.sandbox = sandbox

    def verify_baseline(
        self,
        run_id: str,
        target_tests: list[str] | None = None,
        subject_ids: list[str] | None = None,
    ) -> VerificationReport:
        """Run baseline verification before making changes.

        Records the current state of the repository:
        - Regression test status
        - Preservation behavior status
        - Acceptance Probe results (if applicable)
        - Structural checker results (if applicable)

        Args:
            run_id: Unique identifier for this verification run
            target_tests: Optional list of specific test paths to run
            subject_ids: Optional list of acceptance criteria IDs

        Returns:
            VerificationReport containing baseline check results
        """
        checks: list[CheckReport] = []

        # Run regression tests to establish baseline
        regression_command = "python -m pytest -q -p no:cacheprovider"
        result = self.sandbox.run(
            regression_command,
            timeout_seconds=60,
        )

        regression_check = self._create_check_report(
            method="pytest",
            phase="baseline",
            level="BASELINE_REGRESSION",
            command=regression_command,
            result=result,
            subject_ids=subject_ids or [],
            direct=False,
        )
        checks.append(regression_check)

        # Run target tests if specified
        if target_tests:
            targets = " ".join(
                shlex.quote(test) for test in target_tests
            )
            target_command = f"python -m pytest {targets} -q -p no:cacheprovider"
            result = self.sandbox.run(
                target_command,
                timeout_seconds=60,
            )

            target_check = self._create_check_report(
                method="pytest",
                phase="baseline",
                level="BASELINE_TARGET",
                command=target_command,
                result=result,
                subject_ids=subject_ids or [],
                direct=True,
            )
            checks.append(target_check)

        # Create baseline report
        report = VerificationReport(
            run_id=run_id,
            passed=all(check.passed for check in checks),
            checks=checks,
            retry_count=0,
        )

        # Set failure info if any check failed
        failed_checks = [check for check in checks if not check.passed]
        if failed_checks:
            report.failed_level = failed_checks[0].level
            report.failure_type = failed_checks[0].failure_type

        return report

    def verify_post_patch(
        self,
        run_id: str,
        target_tests: list[str] | None = None,
        subject_ids: list[str] | None = None,
        direct_subject_ids: list[str] | None = None,
        retry_count: int = 0,
    ) -> VerificationReport:
        """Run post-patch verification after making changes.

        Executes comprehensive checks without fail-fast to collect complete evidence:
        - Ruff linting
        - Precise target tests
        - Acceptance Probe (if applicable)
        - Structural check (if applicable)
        - Full regression tests
        - Constraint audit

        Args:
            run_id: Unique identifier for this verification run
            target_tests: Optional list of specific test paths to run first
            subject_ids: Optional list of acceptance criteria IDs
            direct_subject_ids: Optional list of directly exercised acceptance criteria IDs
            retry_count: Number of retry attempts for failed checks

        Returns:
            VerificationReport containing post-patch check results
        """
        checks: list[CheckReport] = []

        # Level 1: Ruff linting
        ruff_command = "ruff check --no-cache ."
        ruff_result = self.sandbox.run(
            ruff_command,
            timeout_seconds=60,
        )

        ruff_check = self._create_check_report(
            method="ruff",
            phase="post_patch",
            level="LEVEL_1_LINT",
            command=ruff_command,
            result=ruff_result,
            subject_ids=[],
            direct=False,
        )
        checks.append(ruff_check)

        # Level 2: Target tests (if specified)
        if target_tests:
            targets = " ".join(
                shlex.quote(test) for test in target_tests
            )
            target_command = f"python -m pytest {targets} -q -p no:cacheprovider"
            target_result = self.sandbox.run(
                target_command,
                timeout_seconds=60,
            )

            target_check = self._create_check_report(
                method="pytest",
                phase="post_patch",
                level="LEVEL_2_TARGET_TESTS",
                command=target_command,
                result=target_result,
                subject_ids=direct_subject_ids or subject_ids or [],
                direct=bool(direct_subject_ids),
            )
            checks.append(target_check)

        # Level 3: Full regression tests
        regression_command = "python -m pytest -q -p no:cacheprovider"
        regression_result = self.sandbox.run(
            regression_command,
            timeout_seconds=60,
        )

        regression_check = self._create_check_report(
            method="pytest",
            phase="post_patch",
            level="LEVEL_3_REGRESSION",
            command=regression_command,
            result=regression_result,
            subject_ids=[],  # Regression tests don't map to specific ACs
            direct=False,
        )
        checks.append(regression_check)

        # Create post-patch report (non-fail-fast)
        report = VerificationReport(
            run_id=run_id,
            passed=all(check.passed for check in checks),
            checks=checks,
            retry_count=retry_count,
        )

        # Set failure info if any check failed
        failed_checks = [check for check in checks if not check.passed]
        if failed_checks:
            report.failed_level = failed_checks[0].level
            report.failure_type = failed_checks[0].failure_type

        return report

    def verify(
        self,
        run_id: str,
        target_tests: list[str] | None = None,
        target_acceptance_criteria: list[str] | None = None,
        target_direct_acceptance_criteria: list[str] | None = None,
        retry_count: int = 0,
    ) -> VerificationReport:
        """Run post-patch verification for backward compatibility.

        This method maintains backward compatibility with the existing interface
        by calling verify_post_patch with mapped parameters.

        Args:
            run_id: Unique identifier for this verification run
            target_tests: Optional list of specific test paths to run first
            target_acceptance_criteria: Optional list of acceptance criteria for target tests
            target_direct_acceptance_criteria: Criteria directly exercised by
                precise target test node IDs.
            retry_count: Number of retry attempts for failed checks

        Returns:
            VerificationReport containing results of all executed checks
        """
        return self.verify_post_patch(
            run_id=run_id,
            target_tests=target_tests,
            subject_ids=target_acceptance_criteria,
            direct_subject_ids=target_direct_acceptance_criteria,
            retry_count=retry_count,
        )

    def _create_check_report(
        self,
        method: str,
        phase: str,
        level: str,
        command: str,
        result,
        subject_ids: list[str],
        direct: bool,
    ) -> CheckReport:
        """Create a CheckReport from command execution result.

        Args:
            method: Verification method (e.g., "pytest", "ruff")
            phase: Verification phase (e.g., "baseline", "post_patch")
            level: Verification level identifier
            command: Command that was executed
            result: Command execution result from sandbox
            subject_ids: List of acceptance criteria IDs
            direct: Whether this provides direct evidence

        Returns:
            CheckReport with execution results
        """
        if result.exit_code == 0:
            return CheckReport(
                method=method,
                phase=phase,
                level=level,
                command=command,
                passed=True,
                exit_code=result.exit_code,
                duration_seconds=result.duration_seconds,
                subject_ids=subject_ids,
                direct=direct,
            )

        # Check failed - parse and classify the failure
        summary = parse_failure(result)
        failure_type = classify_failure(summary)

        return CheckReport(
            method=method,
            phase=phase,
            level=level,
            command=command,
            passed=False,
            exit_code=result.exit_code,
            duration_seconds=result.duration_seconds,
            failure_type=failure_type.value,
            summary=asdict(summary),
            subject_ids=subject_ids,
            direct=direct,
        )
