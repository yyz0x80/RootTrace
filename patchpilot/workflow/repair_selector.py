"""Repair candidate selector for relevance-aware repair loop.

This module implements deterministic selection of repairable failures from
verification results. The selector ensures that the Repair Agent only receives
failures that are plausibly caused by the current patch or represent unmet
required behavior, excluding pre-existing unrelated failures.

The selector implements these rules:
- REQUIRED test still failing after the patch → repairable
- REQUIRED or AFFECTED PASS → FAIL regression → repairable  
- REQUIRED or AFFECTED worsened failure → repairable
- Patch-introduced Ruff violation → repairable
- Deterministic structural/acceptance check failure related to approved AC → repairable
- Unchanged pre-existing unrelated failures → excluded
- Environment/permission/sandbox failures → excluded (non-repairable)
- Missing dependencies requiring unapproved installation → excluded
- Optional failures excluded by configured strategy → excluded
- Failures requiring out-of-scope files → excluded
"""

from __future__ import annotations

from dataclasses import dataclass, field

from patchpilot.evidence.schema import CheckTransition
from patchpilot.planning.schema import ChangePlan
from patchpilot.verification.report import CheckReport, VerificationReport
from patchpilot.workflow.failure_classifier import FailureType


@dataclass
class RepairCandidate:
    """A failure that is a candidate for repair.

    Attributes:
        check: The failed CheckReport
        reason: Why this failure was selected for repair
        tier: Verification tier (required, affected, optional)
        transition: Baseline-to-post-patch transition
        fingerprint: Failure fingerprint for detecting repeated failures
        bounded_output: Bounded failure output for repair prompt
    """

    check: CheckReport
    reason: str
    tier: str
    transition: str
    fingerprint: str
    bounded_output: str


@dataclass
class ExcludedFailure:
    """A failure that was excluded from repair consideration.

    Attributes:
        check: The failed CheckReport
        reason: Why this failure was excluded from repair
        tier: Verification tier (required, affected, optional)
        transition: Baseline-to-post-patch transition
        is_blocking: Whether this excluded failure should block completion
    """

    check: CheckReport
    reason: str
    tier: str
    transition: str
    is_blocking: bool


@dataclass
class RepairSelection:
    """Result of repair candidate selection.

    Attributes:
        repair_candidates: List of repairable failures
        excluded_failures: List of excluded failures with reasons
        should_repair: Whether repair should be attempted
        should_stop: Whether the repair loop should stop
        stop_reason: Reason for stopping if should_stop is True
        completion_hint: Hint for completion state determination
    """

    repair_candidates: list[RepairCandidate] = field(default_factory=list)
    excluded_failures: list[ExcludedFailure] = field(default_factory=list)
    should_repair: bool = False
    should_stop: bool = False
    stop_reason: str = ""
    completion_hint: str = ""


class RepairSelector:
    """Deterministic selector for repairable verification failures.

    The selector implements fixed rules to determine which failures should
    be sent to the Repair Agent, ensuring that only plausibly patch-related
    failures are repaired while pre-existing unrelated failures are reported
    without causing out-of-scope edits.

    Attributes:
        strategy: Verification strategy (strict, balanced, focused)
        changed_files: List of files that were actually changed
        approved_files: Set of files approved for modification in the change plan
    """

    def __init__(
        self,
        strategy: str = "balanced",
        changed_files: list[str] | None = None,
        approved_files: set[str] | None = None,
    ) -> None:
        """Initialize the repair selector.

        Args:
            strategy: Verification strategy (strict, balanced, focused)
            changed_files: List of files that were actually changed
            approved_files: Set of files approved for modification in the change plan
        """
        self.strategy = strategy
        self.changed_files = changed_files or []
        self.approved_files = approved_files or set()

    def select_repair_candidates(
        self,
        report: VerificationReport,
        change_plan: ChangePlan | None = None,
    ) -> RepairSelection:
        """Select repairable failures from verification report.

        Applies deterministic rules to determine which failures should be
        sent to the Repair Agent and which should be excluded.

        Args:
            report: VerificationReport containing failed checks
            change_plan: Optional ChangePlan for scope validation

        Returns:
            RepairSelection with repair candidates and excluded failures
        """
        selection = RepairSelection()

        # Get failed checks
        failed_checks = report.get_failed_checks()

        if not failed_checks:
            # No failures - no repair needed
            selection.should_stop = True
            selection.stop_reason = "No failures to repair"
            selection.completion_hint = "VERIFIED"
            return selection

        # Categorize failures
        for check in failed_checks:
            candidate_or_excluded = self._evaluate_check(
                check,
                change_plan,
            )
            if isinstance(candidate_or_excluded, RepairCandidate):
                selection.repair_candidates.append(candidate_or_excluded)
            else:
                selection.excluded_failures.append(candidate_or_excluded)

        # Determine if repair should be attempted
        selection.should_repair = len(selection.repair_candidates) > 0

        # Check if we should stop without repair
        if not selection.should_repair:
            selection.should_stop = True
            selection.stop_reason = self._determine_stop_reason(selection.excluded_failures)
            selection.completion_hint = self._determine_completion_hint(selection.excluded_failures)

        return selection

    def _evaluate_check(
        self,
        check: CheckReport,
        change_plan: ChangePlan | None,
    ) -> RepairCandidate | ExcludedFailure:
        """Evaluate a single failed check for repair candidacy.

        Args:
            check: Failed CheckReport to evaluate
            change_plan: Optional ChangePlan for scope validation

        Returns:
            Either RepairCandidate (if repairable) or ExcludedFailure (if not)
        """
        tier = check.tier or "unknown"
        transition = check.transition or CheckTransition.NEW_OR_UNCOMPARED.value
        fingerprint = check.failure_fingerprint or ""

        # Check for non-repairable failure types first
        if self._is_non_repairable_failure(check):
            return ExcludedFailure(
                check=check,
                reason="Non-repairable failure type (environment, permission, or timeout)",
                tier=tier,
                transition=transition,
                is_blocking=True,
            )

        # Check for out-of-scope file requirements
        if self._requires_out_of_scope_files(check, change_plan):
            return ExcludedFailure(
                check=check,
                reason="Failure requires files outside approved change plan",
                tier=tier,
                transition=transition,
                is_blocking=True,
            )

        # Check for patch-introduced Ruff violations
        if check.method == "ruff" and transition in (
            CheckTransition.NEW_OR_UNCOMPARED.value,
            CheckTransition.REGRESSION.value,
        ):
            return RepairCandidate(
                check=check,
                reason="Patch-introduced Ruff violation",
                tier=tier,
                transition=transition,
                fingerprint=fingerprint,
                bounded_output=self._get_bounded_output(check),
            )

        # Check for REQUIRED tier failures
        if tier == "required":
            return self._evaluate_required_failure(check, tier, transition, fingerprint)

        # Check for AFFECTED tier failures
        if tier == "affected":
            return self._evaluate_affected_failure(check, tier, transition, fingerprint)

        # Check for OPTIONAL tier failures
        if tier == "optional":
            return self._evaluate_optional_failure(check, tier, transition, fingerprint)

        # Default: exclude if no clear tier classification
        return ExcludedFailure(
            check=check,
            reason="Failure without clear tier classification",
            tier=tier,
            transition=transition,
            is_blocking=False,
        )

    def _is_non_repairable_failure(self, check: CheckReport) -> bool:
        """Check if failure type is non-repairable.

        Non-repairable failures include environment failures, permission failures,
        timeouts, and other infrastructure issues that cannot be fixed by code changes.

        Args:
            check: CheckReport to evaluate

        Returns:
            True if failure is non-repairable
        """
        failure_type = check.failure_type or ""

        non_repairable_types = {
            FailureType.ENVIRONMENT_FAILURE.value,
            FailureType.PERMISSION_FAILURE.value,
            FailureType.TIMEOUT.value,
        }

        return failure_type in non_repairable_types

    def _requires_out_of_scope_files(
        self,
        check: CheckReport,
        change_plan: ChangePlan | None,
    ) -> bool:
        """Check if failure requires files outside approved scope.

        Args:
            check: CheckReport to evaluate
            change_plan: Optional ChangePlan for approved files

        Returns:
            True if failure requires out-of-scope files
        """
        if not change_plan or not self.approved_files:
            return False

        # Extract file paths from failure output
        summary = check.summary or {}
        relevant_output = summary.get("relevant_output", "")

        # Check if any mentioned files are outside approved scope
        for approved_file in self.approved_files:
            if approved_file in relevant_output:
                return False

        # If no approved files are mentioned, it might be out of scope
        # This is a conservative check - could be refined
        return bool(relevant_output and self.approved_files)

    def _evaluate_required_failure(
        self,
        check: CheckReport,
        tier: str,
        transition: str,
        fingerprint: str,
    ) -> RepairCandidate | ExcludedFailure:
        """Evaluate a REQUIRED tier failure.

        REQUIRED failures are generally repairable unless they are pre-existing
        and unchanged.

        Args:
            check: CheckReport to evaluate
            tier: Verification tier
            transition: Baseline transition
            fingerprint: Failure fingerprint

        Returns:
            RepairCandidate or ExcludedFailure
        """
        # REQUIRED test still failing after patch → repairable
        if transition in (
            CheckTransition.NEW_OR_UNCOMPARED.value,
            CheckTransition.WORSENED.value,
        ):
            return RepairCandidate(
                check=check,
                reason="REQUIRED test failure (new or worsened)",
                tier=tier,
                transition=transition,
                fingerprint=fingerprint,
                bounded_output=self._get_bounded_output(check),
            )

        # REQUIRED regression (PASS → FAIL) → repairable
        if transition == CheckTransition.REGRESSION.value:
            return RepairCandidate(
                check=check,
                reason="REQUIRED regression (PASS → FAIL)",
                tier=tier,
                transition=transition,
                fingerprint=fingerprint,
                bounded_output=self._get_bounded_output(check),
            )

        # Pre-existing REQUIRED failure → potentially repairable if worsened
        if transition == CheckTransition.PRE_EXISTING_FAILURE.value:
            # Still repairable since it's REQUIRED and needs to be fixed
            return RepairCandidate(
                check=check,
                reason="REQUIRED pre-existing failure (needs fix)",
                tier=tier,
                transition=transition,
                fingerprint=fingerprint,
                bounded_output=self._get_bounded_output(check),
            )

        # Default to repairable for REQUIRED
        return RepairCandidate(
            check=check,
            reason="REQUIRED tier failure",
            tier=tier,
            transition=transition,
            fingerprint=fingerprint,
            bounded_output=self._get_bounded_output(check),
        )

    def _evaluate_affected_failure(
        self,
        check: CheckReport,
        tier: str,
        transition: str,
        fingerprint: str,
    ) -> RepairCandidate | ExcludedFailure:
        """Evaluate an AFFECTED tier failure.

        AFFECTED regressions and worsened failures are repairable, but
        pre-existing unchanged failures are excluded.

        Args:
            check: CheckReport to evaluate
            tier: Verification tier
            transition: Baseline transition
            fingerprint: Failure fingerprint

        Returns:
            RepairCandidate or ExcludedFailure
        """
        # AFFECTED regression (PASS → FAIL) → repairable
        if transition == CheckTransition.REGRESSION.value:
            return RepairCandidate(
                check=check,
                reason="AFFECTED regression (PASS → FAIL)",
                tier=tier,
                transition=transition,
                fingerprint=fingerprint,
                bounded_output=self._get_bounded_output(check),
            )

        # AFFECTED worsened failure → repairable
        if transition == CheckTransition.WORSENED.value:
            return RepairCandidate(
                check=check,
                reason="AFFECTED worsened failure",
                tier=tier,
                transition=transition,
                fingerprint=fingerprint,
                bounded_output=self._get_bounded_output(check),
            )

        # Pre-existing unchanged AFFECTED failure → exclude
        if transition == CheckTransition.PRE_EXISTING_FAILURE.value:
            return ExcludedFailure(
                check=check,
                reason="Pre-existing unchanged AFFECTED failure (unrelated)",
                tier=tier,
                transition=transition,
                is_blocking=False,
            )

        # New AFFECTED failure → repairable
        if transition == CheckTransition.NEW_OR_UNCOMPARED.value:
            return RepairCandidate(
                check=check,
                reason="New AFFECTED failure",
                tier=tier,
                transition=transition,
                fingerprint=fingerprint,
                bounded_output=self._get_bounded_output(check),
            )

        # Default: exclude
        return ExcludedFailure(
            check=check,
            reason="AFFECTED failure not meeting repair criteria",
            tier=tier,
            transition=transition,
            is_blocking=False,
        )

    def _evaluate_optional_failure(
        self,
        check: CheckReport,
        tier: str,
        transition: str,
        fingerprint: str,
    ) -> RepairCandidate | ExcludedFailure:
        """Evaluate an OPTIONAL tier failure.

        OPTIONAL failures are handled according to the configured strategy.
        Pre-existing failures are always excluded.

        Args:
            check: CheckReport to evaluate
            tier: Verification tier
            transition: Baseline transition
            fingerprint: Failure fingerprint

        Returns:
            RepairCandidate or ExcludedFailure
        """
        # Pre-existing unchanged OPTIONAL failure → always exclude
        if transition == CheckTransition.PRE_EXISTING_FAILURE.value:
            return ExcludedFailure(
                check=check,
                reason="Pre-existing unchanged OPTIONAL failure (unrelated)",
                tier=tier,
                transition=transition,
                is_blocking=False,
            )

        # Strategy-based handling of OPTIONAL failures
        if self.strategy == "strict":
            # STRICT: Repair OPTIONAL regressions
            if transition in (
                CheckTransition.REGRESSION.value,
                CheckTransition.WORSENED.value,
                CheckTransition.NEW_OR_UNCOMPARED.value,
            ):
                return RepairCandidate(
                    check=check,
                    reason="OPTIONAL failure (strict strategy)",
                    tier=tier,
                    transition=transition,
                    fingerprint=fingerprint,
                    bounded_output=self._get_bounded_output(check),
                )
        elif self.strategy == "balanced":
            # BALANCED: Don't repair OPTIONAL failures (exclude as non-blocking)
            return ExcludedFailure(
                check=check,
                reason="OPTIONAL failure (balanced strategy - non-blocking)",
                tier=tier,
                transition=transition,
                is_blocking=False,
            )
        elif self.strategy == "focused":
            # FOCUSED: Don't repair OPTIONAL failures (exclude as non-blocking)
            return ExcludedFailure(
                check=check,
                reason="OPTIONAL failure (focused strategy - non-blocking)",
                tier=tier,
                transition=transition,
                is_blocking=False,
            )

        # Default: exclude OPTIONAL failures
        return ExcludedFailure(
            check=check,
            reason="OPTIONAL failure (strategy-based exclusion)",
            tier=tier,
            transition=transition,
            is_blocking=False,
        )

    def _get_bounded_output(self, check: CheckReport) -> str:
        """Get bounded failure output for repair prompt.

        Args:
            check: CheckReport to extract output from

        Returns:
            Bounded output string (max 2000 characters)
        """
        summary = check.summary or {}
        relevant_output = summary.get("relevant_output") or summary.get("error") or ""

        if len(relevant_output) > 2000:
            return relevant_output[:2000] + "\n... (truncated)"

        return relevant_output

    def _determine_stop_reason(self, excluded_failures: list[ExcludedFailure]) -> str:
        """Determine the reason for stopping repair loop.

        Args:
            excluded_failures: List of excluded failures

        Returns:
            Human-readable stop reason
        """
        # Check for blocking failures
        blocking_failures = [f for f in excluded_failures if f.is_blocking]
        if blocking_failures:
            return f"Blocking non-repairable failures: {', '.join(f.reason for f in blocking_failures)}"

        # Check for non-blocking failures only
        if excluded_failures:
            return f"Only non-repairable or excluded failures remain: {', '.join(f.reason for f in excluded_failures)}"

        return "No repairable failures found"

    def _determine_completion_hint(self, excluded_failures: list[ExcludedFailure]) -> str:
        """Determine completion state hint based on excluded failures.

        Args:
            excluded_failures: List of excluded failures

        Returns:
            Completion state hint (VERIFIED, PARTIALLY_VERIFIED, FAILED, BLOCKED)
        """
        # Check for blocking failures
        blocking_failures = [f for f in excluded_failures if f.is_blocking]
        if blocking_failures:
            return "BLOCKED"

        # Check for non-blocking failures
        if excluded_failures:
            return "PARTIALLY_VERIFIED"

        return "VERIFIED"
