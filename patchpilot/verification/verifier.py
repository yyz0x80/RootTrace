"""Verifier for running deterministic verification checks.

This module provides the Verifier class which executes verification checks
in a fixed order: Ruff linting, target tests, and full regression tests.
It runs checks inside the Docker sandbox and aggregates results into a
VerificationReport with proper failure classification.

The verifier implements fail-fast behavior: if a check fails, subsequent
checks are not executed and the failure is immediately reported.
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

    The Verifier executes checks in a fixed order with fail-fast behavior:
    1. Level 1: Ruff linting
    2. Level 2: Target tests (if specified)
    3. Level 3: Full regression tests

    If any check fails, subsequent checks are skipped and the failure is
    immediately reported with proper classification.

    Attributes:
        sandbox: DockerSandbox instance for isolated command execution
    """

    def __init__(self, sandbox: DockerSandbox) -> None:
        """Initialize the Verifier with a Docker sandbox.

        Args:
            sandbox: DockerSandbox instance for running verification commands
        """
        self.sandbox = sandbox

    def verify(
        self,
        run_id: str,
        target_tests: list[str] | None = None,
        target_acceptance_criteria: list[str] | None = None,
        retry_count: int = 0,
    ) -> VerificationReport:
        """Run lint, target tests, and full regression tests.

        Executes verification checks in order with fail-fast behavior.
        Each check is run inside the Docker sandbox with a timeout.

        Args:
            run_id: Unique identifier for this verification run
            target_tests: Optional list of specific test paths to run first
            target_acceptance_criteria: Optional list of acceptance criteria for target tests
            retry_count: Number of retry attempts for failed checks

        Returns:
            VerificationReport containing results of all executed checks
        """
        checks: list[CheckReport] = []

        # Level 1: Ruff linting
        commands: list[tuple[str, str]] = [
            (
                "LEVEL_1_LINT",
                "ruff check --no-cache .",
            )
        ]

        # Level 2: Target tests (if specified)
        if target_tests:
            targets = " ".join(
                shlex.quote(test) for test in target_tests
            )
            commands.append(
                (
                    "LEVEL_2_TARGET_TESTS",
                    f"pytest {targets} -q -p no:cacheprovider",
                )
            )

        # Level 3: Full regression tests
        commands.append(
            (
                "LEVEL_3_REGRESSION",
                "pytest -q -p no:cacheprovider",
            )
        )

        # Execute checks in order with fail-fast behavior
        for level, command in commands:
            result = self.sandbox.run(
                command,
                timeout_seconds=60,
            )

            # Check passed
            if result.exit_code == 0:
                checks.append(
                    CheckReport(
                        level=level,
                        command=command,
                        passed=True,
                        exit_code=result.exit_code,
                        duration_seconds=result.duration_seconds,
                        acceptance_criteria=target_acceptance_criteria or []
                        if level == "LEVEL_2_TARGET_TESTS"
                        else [],
                    )
                )
                continue

            # Check failed - parse and classify the failure
            summary = parse_failure(result)
            failure_type = classify_failure(summary)

            checks.append(
                CheckReport(
                    level=level,
                    command=command,
                    passed=False,
                    exit_code=result.exit_code,
                    duration_seconds=result.duration_seconds,
                    failure_type=failure_type.value,
                    summary=asdict(summary),
                    acceptance_criteria=target_acceptance_criteria or []
                    if level == "LEVEL_2_TARGET_TESTS"
                    else [],
                )
            )

            # Fail-fast: return immediately on first failure
            return VerificationReport(
                run_id=run_id,
                passed=False,
                checks=checks,
                retry_count=retry_count,
                failed_level=level,
                failure_type=failure_type.value,
            )

        # All checks passed
        return VerificationReport(
            run_id=run_id,
            passed=True,
            checks=checks,
            retry_count=retry_count,
        )
