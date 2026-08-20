"""Verifier for running deterministic verification checks.

This module provides the Verifier class which executes verification checks
in two phases: Baseline Verification (before changes) and Post-patch Verification
(after changes). It runs checks inside the Docker sandbox and aggregates results
into a VerificationReport with proper failure classification.

The verifier implements different strategies for each phase:
- Baseline: Records current state, can fail-fast for blocking failures
- Post-patch: Collects complete evidence, does not fail-fast

The verifier now supports specialized verification checks when a ChangePlan
includes acceptance probes or structural checks:
- Acceptance Probes: Model-generated verification scripts executed in isolated
  temporary directories to test specific aspects of code changes
- Structural Checks: AST-based verification to ensure code structure meets
  requirements without execution
- Constraint Audit: Deterministic policy validation against actual git diff

Specialized checks are only executed when the approved ChangePlan explicitly
defines sufficient validated information for those checks. Missing optional
specialized checks result in UNVERIFIED evidence rather than invented PASS.
"""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from patchpilot.evidence.schema import CheckTransition
from patchpilot.sandbox.docker_runner import DockerSandbox
from patchpilot.verification.baseline_delta import (
    apply_baseline_delta_evaluation,
    classify_transition,
    compute_failure_fingerprint,
    compute_transition_summary,
    match_baseline_checks,
)
from patchpilot.verification.config import (
    VerificationStrategy,
    VerificationTimeouts,
)
from patchpilot.verification.error_parser import parse_failure
from patchpilot.verification.report import (
    CheckReport,
    VerificationReport,
)
from patchpilot.verification.targets import (
    SelectionReasonType,
    TargetTestSelection,
)
from patchpilot.workflow.failure_classifier import classify_failure

if TYPE_CHECKING:
    from patchpilot.planning.schema import ChangePlan


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

    def __init__(
        self,
        sandbox: DockerSandbox,
        workspace_root: Path | None = None,
        timeouts: VerificationTimeouts | None = None,
        strategy: VerificationStrategy = VerificationStrategy.BALANCED,
    ) -> None:
        """Initialize the Verifier with a Docker sandbox.

        Args:
            sandbox: DockerSandbox instance for running verification commands
            workspace_root: Optional workspace root path for specialized checks
            timeouts: Optional VerificationTimeouts configuration (uses defaults if None)
            strategy: Verification strategy (strict, balanced, focused)
        """
        self.sandbox = sandbox
        self.workspace_root = workspace_root
        self.timeouts = timeouts or VerificationTimeouts()
        self.strategy = strategy
        self._specialized_verifier = None

    def _get_specialized_verifier(self):
        """Get or create the specialized verifier instance.

        Returns:
            SpecializedVerifier instance if workspace_root is set, None otherwise
        """
        if self.workspace_root is None:
            return None

        if self._specialized_verifier is None:
            from patchpilot.verification.specialized import SpecializedVerifier

            self._specialized_verifier = SpecializedVerifier(
                self.workspace_root,
                timeouts=self.timeouts,
            )

        return self._specialized_verifier

    def verify_baseline(
        self,
        run_id: str,
        target_tests: list[str] | None = None,
        subject_ids: list[str] | None = None,
        change_plan: ChangePlan | None = None,
    ) -> VerificationReport:
        """Run baseline verification before making changes.

        Records the current state of the repository:
        - Regression test status
        - Preservation behavior status
        - Acceptance Probe results (if ChangePlan includes acceptance_probes)
        - Structural checker results (if ChangePlan includes structural_checks)

        Specialized checks are only executed when the ChangePlan contains
        validated acceptance probe or structural check specifications.

        Args:
            run_id: Unique identifier for this verification run
            target_tests: Optional list of specific test paths to run
            subject_ids: Optional list of acceptance criteria IDs
            change_plan: Optional ChangePlan with specialized verification specs

        Returns:
            VerificationReport containing baseline check results
        """
        checks: list[CheckReport] = []

        # Run regression tests to establish baseline
        regression_command = "python -m pytest -q -p no:cacheprovider"
        result = self.sandbox.run(
            regression_command,
            timeout_seconds=self.timeouts.regression_tests,
        )

        regression_check = self._create_check_report(
            method="pytest",
            phase="baseline",
            level="BASELINE_REGRESSION",
            command=regression_command,
            result=result,
            timeout_seconds=self.timeouts.regression_tests,
            subject_ids=subject_ids or [],
            direct=False,
            test_node="",  # Baseline regression doesn't target specific nodes
        )
        # Add failure fingerprint for baseline checks
        if not regression_check.passed:
            regression_check.failure_fingerprint = compute_failure_fingerprint(regression_check)
        checks.append(regression_check)

        # Run target tests if specified
        if target_tests:
            targets = " ".join(
                shlex.quote(test) for test in target_tests
            )
            target_command = f"python -m pytest {targets} -q -p no:cacheprovider"
            result = self.sandbox.run(
                target_command,
                timeout_seconds=self.timeouts.target_tests,
            )

            target_check = self._create_check_report(
                method="pytest",
                phase="baseline",
                level="BASELINE_TARGET",
                command=target_command,
                result=result,
                timeout_seconds=self.timeouts.target_tests,
                subject_ids=subject_ids or [],
                direct=True,
                test_node=target_tests[0] if target_tests else "",  # Use first test as node identifier
            )
            # Add failure fingerprint for baseline checks
            if not target_check.passed:
                target_check.failure_fingerprint = compute_failure_fingerprint(target_check)
            checks.append(target_check)

        # Run specialized checks if change plan provides them
        specialized_verifier = self._get_specialized_verifier()
        if specialized_verifier and change_plan and specialized_verifier.has_specialized_checks(change_plan):
            specialized_checks = specialized_verifier.execute_specialized_checks(
                change_plan,
                phase="baseline",
            )
            # Add failure fingerprints for baseline specialized checks
            for check in specialized_checks:
                if not check.passed:
                    check.failure_fingerprint = compute_failure_fingerprint(check)
            checks.extend(specialized_checks)

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
        change_plan: ChangePlan | None = None,
        baseline_report: VerificationReport | None = None,
    ) -> VerificationReport:
        """Run post-patch verification after making changes.

        Executes comprehensive checks without fail-fast to collect complete evidence:
        - Ruff linting
        - Precise target tests
        - Acceptance Probe (if ChangePlan includes acceptance_probes)
        - Structural check (if ChangePlan includes structural_checks)
        - Full regression tests
        - Constraint audit (deterministic policy validation against git diff)

        Specialized checks are only executed when the ChangePlan contains
        validated acceptance probe or structural check specifications.
        Constraint audit is performed deterministically against the actual
        git diff using policy validation, not model judgment.

        Args:
            run_id: Unique identifier for this verification run
            target_tests: Optional list of specific test paths to run first
            subject_ids: Optional list of acceptance criteria IDs
            direct_subject_ids: Optional list of directly exercised acceptance criteria IDs
            retry_count: Number of retry attempts for failed checks
            change_plan: Optional ChangePlan with specialized verification specs
            baseline_report: Optional baseline VerificationReport for delta comparison

        Returns:
            VerificationReport containing post-patch check results
        """
        checks: list[CheckReport] = []

        # Level 1: Ruff linting
        ruff_command = "ruff check --no-cache ."
        ruff_result = self.sandbox.run(
            ruff_command,
            timeout_seconds=self.timeouts.ruff,
        )

        ruff_check = self._create_check_report(
            method="ruff",
            phase="post_patch",
            level="LEVEL_1_LINT",
            command=ruff_command,
            result=ruff_result,
            timeout_seconds=self.timeouts.ruff,
            subject_ids=[],
            direct=False,
            test_node="",
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
                timeout_seconds=self.timeouts.target_tests,
            )

            target_check = self._create_check_report(
                method="pytest",
                phase="post_patch",
                level="LEVEL_2_TARGET_TESTS",
                command=target_command,
                result=target_result,
                timeout_seconds=self.timeouts.target_tests,
                subject_ids=direct_subject_ids or subject_ids or [],
                direct=bool(direct_subject_ids),
                test_node=target_tests[0] if target_tests else "",
            )
            checks.append(target_check)

        # Level 2.5: Specialized checks (if change plan provides them)
        specialized_verifier = self._get_specialized_verifier()
        if specialized_verifier and change_plan and specialized_verifier.has_specialized_checks(change_plan):
            specialized_checks = specialized_verifier.execute_specialized_checks(
                change_plan,
                phase="post_patch",
            )
            checks.extend(specialized_checks)

        # Level 3: Full regression tests
        regression_command = "python -m pytest -q -p no:cacheprovider"
        regression_result = self.sandbox.run(
            regression_command,
            timeout_seconds=self.timeouts.regression_tests,
        )

        regression_check = self._create_check_report(
            method="pytest",
            phase="post_patch",
            level="LEVEL_3_REGRESSION",
            command=regression_command,
            result=regression_result,
            timeout_seconds=self.timeouts.regression_tests,
            subject_ids=[],  # Regression tests don't map to specific ACs
            direct=False,
            test_node="",
        )
        checks.append(regression_check)

        # Level 4: Constraint audit (if policy information available)
        if change_plan:
            constraint_checks = self._create_constraint_audit_checks(
                run_id,
                change_plan,
            )
            checks.extend(constraint_checks)

        # Apply baseline-delta comparison if baseline checks available
        baseline_checks = baseline_report.get_baseline_checks() if baseline_report else []
        if baseline_checks:
            # Match post-patch checks to baseline
            baseline_matches = match_baseline_checks(
                post_patch_checks=checks,
                baseline_checks=baseline_checks,
            )

            # Classify transitions for each post-patch check
            for check in checks:
                if check.phase == "post_patch":
                    baseline_check = baseline_matches.get(check.verification_id)
                    transition, baseline_check_id = classify_transition(
                        baseline_check=baseline_check,
                        post_patch_check=check,
                    )
                    check.transition = transition
                    check.baseline_check_id = baseline_check_id
                    # Add failure fingerprint for failed checks
                    if not check.passed:
                        check.failure_fingerprint = compute_failure_fingerprint(check)

            # Compute transition summary
            transition_summary = compute_transition_summary(checks)
        else:
            # No baseline available, mark all as NEW_OR_UNCOMPARED
            for check in checks:
                if check.phase == "post_patch":
                    check.transition = CheckTransition.NEW_OR_UNCOMPARED.value
                    check.baseline_check_id = ""
                    if not check.passed:
                        check.failure_fingerprint = compute_failure_fingerprint(check)
            transition_summary = compute_transition_summary(checks)

        # Create post-patch report with transition information
        report = VerificationReport(
            run_id=run_id,
            passed=True,  # Will be updated by baseline-delta evaluation
            checks=checks,
            retry_count=retry_count,
            transition_summary=transition_summary,
        )

        # Apply baseline-delta evaluation to determine final status
        verification_status, passed = apply_baseline_delta_evaluation(
            report=report,
            strategy=self.strategy.value,
        )
        report.verification_status = verification_status
        report.passed = passed

        # Set failure info if any check failed
        failed_checks = [check for check in checks if not check.passed]
        if failed_checks:
            report.failed_level = failed_checks[0].level
            report.failure_type = failed_checks[0].failure_type

        return report

    def verify_post_patch_tiered(
        self,
        run_id: str,
        target_selection: TargetTestSelection,
        changed_files: list[str] | None = None,
        subject_ids: list[str] | None = None,
        direct_subject_ids: list[str] | None = None,
        retry_count: int = 0,
        change_plan: ChangePlan | None = None,
        repo_root: Path | None = None,
        python_files: list[str] | None = None,
        baseline_report: VerificationReport | None = None,
    ) -> VerificationReport:
        """Run post-patch verification with tiered test classification.

        Executes verification checks in tier order (REQUIRED, AFFECTED, OPTIONAL)
        and applies baseline-delta evaluation to determine final status.

        Args:
            run_id: Unique identifier for this verification run
            target_selection: TargetTestSelection with classified test selection
            changed_files: List of actually changed source file paths
            subject_ids: Optional list of acceptance criteria IDs
            direct_subject_ids: Optional list of directly exercised acceptance criteria IDs
            retry_count: Number of retry attempts for failed checks
            change_plan: Optional ChangePlan with specialized verification specs
            repo_root: Path to repository root for dependency analysis
            python_files: List of all Python files in repository
            baseline_report: Optional baseline VerificationReport for delta comparison

        Returns:
            VerificationReport containing tiered check results and baseline-delta evaluation
        """
        checks: list[CheckReport] = []
        tier_results: dict[str, dict[str, Any]] = {
            "required": {"total": 0, "passed": 0, "failed": 0, "skipped": 0},
            "affected": {"total": 0, "passed": 0, "failed": 0, "skipped": 0},
            "optional": {"total": 0, "passed": 0, "failed": 0, "skipped": 0},
        }

        # Store baseline checks for delta comparison
        baseline_checks = baseline_report.get_baseline_checks() if baseline_report else []

        # Level 1: Ruff linting (always runs, marked as required since it's blocking)
        ruff_command = "ruff check --no-cache ."
        ruff_result = self.sandbox.run(
            ruff_command,
            timeout_seconds=self.timeouts.ruff,
        )

        ruff_check = self._create_check_report(
            method="ruff",
            phase="post_patch",
            level="LEVEL_1_LINT",
            command=ruff_command,
            result=ruff_result,
            timeout_seconds=self.timeouts.ruff,
            subject_ids=[],
            direct=False,
            test_node="",
        )
        ruff_check.tier = "required"
        ruff_check.selection_reason = "Ruff linting - blocking style check"
        checks.append(ruff_check)
        tier_results["required"]["total"] += 1
        if ruff_check.passed:
            tier_results["required"]["passed"] += 1
        else:
            tier_results["required"]["failed"] += 1

        # Classify tests into tiers based on target selection
        required_tests: list[str] = []
        affected_tests: list[str] = []
        optional_tests: list[str] = []

        for selected_test in target_selection.selected_tests:
            if selected_test.reason.classification == SelectionReasonType.DIRECT:
                required_tests.append(selected_test.test_id)
            elif selected_test.reason.classification == SelectionReasonType.AFFECTED:
                affected_tests.append(selected_test.test_id)
            else:  # UNRELATED
                optional_tests.append(selected_test.test_id)

        # Tier 1: REQUIRED tests (direct target tests, explicit acceptance tests)
        if required_tests:
            required_checks = self._execute_tier(
                "required",
                required_tests,
                target_selection,
                tier_results,
                subject_ids=direct_subject_ids or subject_ids or [],
                direct=True,
            )
            checks.extend(required_checks)

        # Tier 2: AFFECTED tests (dependency analysis)
        if affected_tests:
            affected_checks = self._execute_tier(
                "affected",
                affected_tests,
                target_selection,
                tier_results,
                subject_ids=subject_ids or [],
                direct=False,
            )
            checks.extend(affected_checks)

        # Tier 2.5: Specialized checks (if change plan provides them)
        specialized_verifier = self._get_specialized_verifier()
        if specialized_verifier and change_plan and specialized_verifier.has_specialized_checks(change_plan):
            specialized_checks = specialized_verifier.execute_specialized_checks(
                change_plan,
                phase="post_patch",
            )
            # Mark specialized checks as required (they provide direct evidence)
            for check in specialized_checks:
                check.tier = "required"
                check.selection_reason = "Specialized verification check from ChangePlan"
            checks.extend(specialized_checks)
            tier_results["required"]["total"] += len(specialized_checks)
            tier_results["required"]["passed"] += sum(1 for c in specialized_checks if c.passed)
            tier_results["required"]["failed"] += sum(1 for c in specialized_checks if not c.passed)

        # Tier 3: OPTIONAL tests (remaining regression tests)
        if optional_tests:
            optional_checks = self._execute_tier(
                "optional",
                optional_tests,
                target_selection,
                tier_results,
                subject_ids=[],
                direct=False,
            )
            checks.extend(optional_checks)

        # If no classified tests, fall back to full regression suite
        if not required_tests and not affected_tests and not optional_tests:
            regression_command = "python -m pytest -q -p no:cacheprovider"
            regression_result = self.sandbox.run(
                regression_command,
                timeout_seconds=self.timeouts.regression_tests,
            )

            regression_check = self._create_check_report(
                method="pytest",
                phase="post_patch",
                level="LEVEL_3_REGRESSION",
                command=regression_command,
                result=regression_result,
                timeout_seconds=self.timeouts.regression_tests,
                subject_ids=[],
                direct=False,
                test_node="",
            )
            regression_check.tier = "optional"
            regression_check.selection_reason = "Fallback full regression suite (no classified tests)"
            checks.append(regression_check)
            tier_results["optional"]["total"] += 1
            if regression_check.passed:
                tier_results["optional"]["passed"] += 1
            else:
                tier_results["optional"]["failed"] += 1

        # Level 4: Constraint audit (if policy information available)
        if change_plan:
            constraint_checks = self._create_constraint_audit_checks(
                run_id,
                change_plan,
            )
            # Mark constraint audit checks as required (they are blocking)
            for check in constraint_checks:
                check.tier = "required"
                check.selection_reason = "Constraint audit - deterministic policy validation"
            checks.extend(constraint_checks)
            tier_results["required"]["total"] += len(constraint_checks)
            tier_results["required"]["passed"] += sum(1 for c in constraint_checks if c.passed)
            tier_results["required"]["failed"] += sum(1 for c in constraint_checks if not c.passed)

        # Apply baseline-delta comparison if baseline checks available
        if baseline_checks:
            # Match post-patch checks to baseline
            baseline_matches = match_baseline_checks(
                post_patch_checks=checks,
                baseline_checks=baseline_checks,
            )

            # Classify transitions for each post-patch check
            for check in checks:
                if check.phase == "post_patch":
                    baseline_check = baseline_matches.get(check.verification_id)
                    transition, baseline_check_id = classify_transition(
                        baseline_check=baseline_check,
                        post_patch_check=check,
                    )
                    check.transition = transition
                    check.baseline_check_id = baseline_check_id
                    # Add failure fingerprint for failed checks
                    if not check.passed:
                        check.failure_fingerprint = compute_failure_fingerprint(check)

            # Compute transition summary
            transition_summary = compute_transition_summary(checks)
        else:
            # No baseline available, mark all as NEW_OR_UNCOMPARED
            for check in checks:
                if check.phase == "post_patch":
                    check.transition = CheckTransition.NEW_OR_UNCOMPARED.value
                    check.baseline_check_id = ""
                    if not check.passed:
                        check.failure_fingerprint = compute_failure_fingerprint(check)
            transition_summary = compute_transition_summary(checks)

        # Create post-patch report with transition information
        report = VerificationReport(
            run_id=run_id,
            passed=True,  # Will be updated by baseline-delta evaluation
            checks=checks,
            retry_count=retry_count,
            strategy=self.strategy.value,
            verification_status="",  # Will be set by baseline-delta evaluation
            tier_summary=tier_results,
            transition_summary=transition_summary,
        )

        # Apply baseline-delta evaluation to determine final status
        verification_status, passed = apply_baseline_delta_evaluation(
            report=report,
            strategy=self.strategy.value,
        )
        report.verification_status = verification_status
        report.passed = passed

        # Set failure info if any check failed
        failed_checks = [check for check in checks if not check.passed]
        if failed_checks:
            report.failed_level = failed_checks[0].level
            report.failure_type = failed_checks[0].failure_type

        return report

    def _execute_tier(
        self,
        tier: str,
        test_paths: list[str],
        target_selection: TargetTestSelection,
        tier_results: dict[str, dict[str, Any]],
        subject_ids: list[str],
        direct: bool,
    ) -> list[CheckReport]:
        """Execute tests for a specific tier.

        Args:
            tier: Tier name (required, affected, optional)
            test_paths: List of test paths to execute
            target_selection: TargetTestSelection with classification metadata
            tier_results: Dictionary to track tier statistics
            subject_ids: Acceptance criteria IDs for these tests
            direct: Whether these tests provide direct evidence

        Returns:
            List of CheckReport objects for this tier
        """
        checks: list[CheckReport] = []

        for test_path in test_paths:
            # Find the selection reason for this test
            selection_reason = ""
            for selected_test in target_selection.selected_tests:
                if selected_test.test_id == test_path:
                    selection_reason = selected_test.reason.description
                    break

            # Execute the test
            targets = " ".join(shlex.quote(test) for test in [test_path])
            command = f"python -m pytest {targets} -q -p no:cacheprovider"
            result = self.sandbox.run(
                command,
                timeout_seconds=self.timeouts.target_tests,
            )

            check = self._create_check_report(
                method="pytest",
                phase="post_patch",
                level=f"TIER_{tier.upper()}",
                command=command,
                result=result,
                timeout_seconds=self.timeouts.target_tests,
                subject_ids=subject_ids,
                direct=direct,
                test_node=test_path,  # Use test path as node identifier
            )
            check.tier = tier
            check.selection_reason = selection_reason
            checks.append(check)

            # Update tier statistics
            tier_results[tier]["total"] += 1
            if check.passed:
                tier_results[tier]["passed"] += 1
            else:
                tier_results[tier]["failed"] += 1

        return checks

    def verify(
        self,
        run_id: str,
        target_tests: list[str] | None = None,
        target_acceptance_criteria: list[str] | None = None,
        target_direct_acceptance_criteria: list[str] | None = None,
        retry_count: int = 0,
        change_plan: ChangePlan | None = None,
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
            change_plan: Optional ChangePlan with specialized verification specs

        Returns:
            VerificationReport containing results of all executed checks
        """
        return self.verify_post_patch(
            run_id=run_id,
            target_tests=target_tests,
            subject_ids=target_acceptance_criteria,
            direct_subject_ids=target_direct_acceptance_criteria,
            retry_count=retry_count,
            change_plan=change_plan,
        )

    def _create_check_report(
        self,
        method: str,
        phase: str,
        level: str,
        command: str,
        result,
        timeout_seconds: int,
        subject_ids: list[str],
        direct: bool,
        test_node: str = "",
    ) -> CheckReport:
        """Create a CheckReport from command execution result.

        Args:
            method: Verification method (e.g., "pytest", "ruff")
            phase: Verification phase (e.g., "baseline", "post_patch")
            level: Verification level identifier
            command: Command that was executed
            result: Command execution result from sandbox
            timeout_seconds: Timeout budget that was configured for this check
            subject_ids: List of acceptance criteria IDs
            direct: Whether this provides direct evidence
            test_node: Test node identifier for pytest checks

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
                timeout_seconds=timeout_seconds,
                subject_ids=subject_ids,
                direct=direct,
                test_node=test_node,
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
            timeout_seconds=timeout_seconds,
            failure_type=failure_type.value,
            summary=asdict(summary),
            subject_ids=subject_ids,
            direct=direct,
            test_node=test_node,
        )

    def _create_constraint_audit_checks(
        self,
        run_id: str,
        change_plan: ChangePlan,
    ) -> list[CheckReport]:
        """Create constraint audit checks based on policy and final diff.

        This method performs deterministic policy validation against the actual
        changes made, ensuring compliance with security boundaries without
        allowing the model to decide compliance.

        Args:
            run_id: Unique identifier for this verification run
            change_plan: The approved ChangePlan for policy validation

        Returns:
            List of CheckReport objects for constraint audit
        """
        checks: list[CheckReport] = []

        # Import runtime audit here to avoid circular dependencies
        try:
            from patchpilot.policy.builtins import get_builtin_policies
            from patchpilot.policy.runtime_audit import audit_git_diff
        except ImportError:
            # If policy modules are not available, skip constraint audit
            return checks

        if not self.workspace_root:
            return checks

        # Get policy set for constraint validation
        try:
            policy_set = get_builtin_policies()
        except (ImportError, RuntimeError):
            # If policy compilation fails, skip constraint audit
            return checks

        # Get planned files from change plan
        planned_files = {change.path for change in change_plan.planned_changes}

        # Run runtime audit against actual git diff
        try:
            audit_result = audit_git_diff(
                workspace_root=self.workspace_root,
                policy_set=policy_set,
                planned_files=planned_files,
            )
        except (subprocess.CalledProcessError, RuntimeError, OSError):
            # If audit fails, create a failure check
            checks.append(
                CheckReport(
                    method="constraint_audit",
                    phase="constraint_audit",
                    level="CONSTRAINT_AUDIT",
                    command="git_diff_audit",
                    passed=False,
                    exit_code=1,
                    duration_seconds=0.0,
                    timeout_seconds=0,  # Constraint audit doesn't use timeout
                    failure_type="audit_error",
                    summary={
                        "error": "Constraint audit execution failed",
                    },
                    subject_ids=[],
                    direct=False,
                )
            )
            return checks

        # Create constraint audit check report
        if audit_result.passed:
            checks.append(
                CheckReport(
                    method="constraint_audit",
                    phase="constraint_audit",
                    level="CONSTRAINT_AUDIT",
                    command="git_diff_audit",
                    passed=True,
                    exit_code=0,
                    duration_seconds=0.0,
                    timeout_seconds=0,  # Constraint audit doesn't use timeout
                    subject_ids=[],
                    direct=False,
                )
            )
        else:
            # Create failure check with bounded violation output
            violations_output = "\n".join(audit_result.violations)
            if len(violations_output) > 2000:
                violations_output = violations_output[:2000] + "\n... (truncated)"

            checks.append(
                CheckReport(
                    method="constraint_audit",
                    phase="constraint_audit",
                    level="CONSTRAINT_AUDIT",
                    command="git_diff_audit",
                    passed=False,
                    exit_code=1,
                    duration_seconds=0.0,
                    timeout_seconds=0,  # Constraint audit doesn't use timeout
                    failure_type="policy_violation",
                    summary={
                        "violation_type": "hard_policy",
                        "violations": audit_result.violations,
                        "output": violations_output,
                    },
                    subject_ids=[],
                    direct=False,
                )
            )

        return checks
