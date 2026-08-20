"""Repair loop with early stopping logic for repeated failures.

This module implements a repair loop that attempts to fix verification failures
while detecting when the same failure persists across repair attempts. When a
failure fingerprint remains unchanged, the loop stops early to avoid wasting
LLM calls on futile repair attempts.

The repair loop is now relevance-aware:
1. Runs the agent to implement a fix
2. Verifies the fix with deterministic checks
3. Selects repairable failures using RepairSelector
4. If verification fails, generates a failure fingerprint
5. Compares with previous failure fingerprint
6. Stops early if the same failure recurs or no repairable failures remain
7. Builds the next prompt from selected repair candidates only
8. Otherwise continues with another repair attempt
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from patchpilot.agent_loop import AgentLoop
from patchpilot.planning.schema import ChangePlan
from patchpilot.prompts import REPAIR_PROMPT, REPAIR_SYSTEM_PROMPT
from patchpilot.verification.report import VerificationReport
from patchpilot.workflow.repair_selector import (
    RepairSelection,
    RepairSelector,
)

logger = logging.getLogger(__name__)

MAX_REPAIR_TASK_CHARS = 6_000
MAX_REPAIR_FAILURE_CHARS = 2_000

Verifier = Callable[[], VerificationReport]
RepairPromptBuilder = Callable[[str, VerificationReport, RepairSelection], str]


def _truncate_text(text: str, limit: int) -> str:
    """Return bounded prompt text while preserving its beginning and end."""
    normalized = text.strip()
    if len(normalized) <= limit:
        return normalized

    marker = "\n... repair context truncated ...\n"
    remaining = limit - len(marker)
    head_size = (remaining * 2) // 3
    return normalized[:head_size] + marker + normalized[-(remaining - head_size):]


def _read_plan_context(issue: str) -> tuple[str, str, str, str, str]:
    """Extract a compact repair boundary from a serialized change plan."""
    try:
        plan = ChangePlan.model_validate_json(issue)
    except ValueError:
        return (
            issue,
            "Preserve the original task intent.",
            "Use only source files allowed by the original task.",
            "Do not broaden the requested behavior or modify tests.",
            "Use the acceptance criteria from the original task.",
        )

    change_intent = "\n".join(
        f"- {change.action.value} {change.path}: {change.description}"
        for change in plan.planned_changes
    )
    allowed_files = "\n".join(
        f"- {change.path}"
        for change in plan.planned_changes
    )
    constraints = "\n".join(
        f"- Out of scope: {item}"
        for item in plan.out_of_scope
    )
    acceptance_criteria = "\n".join(
        f"- {criterion}"
        for change in plan.planned_changes
        for criterion in change.acceptance_criteria
    )

    return (
        "Complete the approved change plan.",
        change_intent or "Preserve the approved change intent.",
        allowed_files or "Use only source files allowed by the approved plan.",
        constraints or "Do not broaden the approved scope or modify tests.",
        acceptance_criteria or "Use the approved plan acceptance criteria.",
    )


def _format_repair_candidates(selection: RepairSelection) -> str:
    """Format selected repair candidates as actionable repair evidence.

    Args:
        selection: RepairSelection with repair candidates

    Returns:
        Formatted repair candidates string
    """
    if not selection.repair_candidates:
        return "No repairable failures selected."

    candidate_details = []
    for i, candidate in enumerate(selection.repair_candidates, 1):
        details = [
            f"Failure {i}:",
            f"  Command: {candidate.check.command}",
            f"  Tier: {candidate.tier}",
            f"  Transition: {candidate.transition}",
            f"  Reason: {candidate.reason}",
            f"  Failure Type: {candidate.check.failure_type or 'unknown'}",
            f"  Exit Code: {candidate.check.exit_code}",
        ]

        summary: dict[str, Any] = candidate.check.summary or {}
        failed_tests = summary.get("failed_tests")
        if isinstance(failed_tests, list) and failed_tests:
            details.append(
                "  Failed Tests: " + ", ".join(str(test) for test in failed_tests)
            )

        error_type = summary.get("error_type")
        if error_type:
            details.append(f"  Error Type: {error_type}")

        if candidate.bounded_output:
            details.append(
                "  Relevant Output:\n"
                + _truncate_text(candidate.bounded_output, MAX_REPAIR_FAILURE_CHARS)
            )

        candidate_details.append("\n".join(details))

    return "\n\n".join(candidate_details)


def build_failure_repair_prompt(
    issue: str,
    failure_report: VerificationReport,
    selection: RepairSelection,
) -> str:
    """Build a focused retry prompt from selected repair candidates.

    Args:
        issue: The original issue or plan description
        failure_report: VerificationReport with failure information
        selection: RepairSelection with repair candidates

    Returns:
        Formatted repair prompt with only relevant failures
    """
    (
        task_goal,
        change_intent,
        allowed_files,
        constraints,
        acceptance_criteria,
    ) = _read_plan_context(issue)
    return REPAIR_PROMPT.format(
        task_goal=_truncate_text(task_goal, MAX_REPAIR_TASK_CHARS),
        change_intent=_truncate_text(change_intent, MAX_REPAIR_TASK_CHARS),
        allowed_files=allowed_files,
        task_constraints=constraints,
        acceptance_criteria=acceptance_criteria,
        current_patch=(
            "Inspect the current workspace diff and preserve correct changes."
        ),
        failure=_format_repair_candidates(selection),
    )


class RepairLoopError(RuntimeError):
    """Base exception for Repair Loop failures."""


class RepairLoopLimitError(RepairLoopError):
    """Raised when the repair loop exceeds the configured attempt limit."""


class RepairLoopStalledError(RepairLoopError):
    """Raised when the same failure repeats across repair attempts."""


class RepairLoop:
    """Coordinate repair attempts with early stopping for repeated failures.

    The repair loop runs the agent to fix verification failures and detects
    when the same error persists, indicating that further attempts would be futile.
    The loop is now relevance-aware, using RepairSelector to filter failures.

    Attributes:
        agent_loop: The AgentLoop instance for running repair attempts
        max_attempts: Maximum number of repair attempts
        verifier: Function to run verification and return a VerificationReport
        repair_selector: RepairSelector for filtering repairable failures
        strategy: Verification strategy (strict, balanced, focused)
        changed_files: List of files that were actually changed
        approved_files: Set of files approved for modification
    """

    def __init__(
        self,
        agent_loop: AgentLoop,
        max_attempts: int = 3,
        verifier: Verifier | None = None,
        strategy: str = "balanced",
        changed_files: list[str] | None = None,
        approved_files: set[str] | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

        self.agent_loop = agent_loop
        self.max_attempts = max_attempts
        self.verifier = verifier
        self.strategy = strategy
        self.changed_files = changed_files or []
        self.approved_files = approved_files or set()
        self.repair_selector = RepairSelector(
            strategy=strategy,
            changed_files=self.changed_files,
            approved_files=self.approved_files,
        )

    def run(
        self,
        issue: str,
        repair_prompt_builder: RepairPromptBuilder | None = None,
        change_plan: ChangePlan | None = None,
    ) -> tuple[str, VerificationReport | None]:
        """Run the repair loop until success, stall detection, or limit.

        Args:
            issue: The original issue or plan description
            repair_prompt_builder: Optional override taking
                (issue, failure_report, selection) and returning a prompt string. The
                default builder includes structured failure evidence from selected candidates.
            change_plan: Optional ChangePlan for scope validation

        Returns:
            Tuple of (final_agent_response, final_verification_report)

        Raises:
            RepairLoopLimitError: If max_attempts is exceeded
            RepairLoopStalledError: If the same failure repeats
            RepairLoopError: For other repair loop failures
        """
        if not issue.strip():
            raise ValueError("issue must not be empty")

        previous_fingerprints: set[str] = set()
        current_issue = issue
        prompt_builder = repair_prompt_builder or build_failure_repair_prompt

        for attempt in range(1, self.max_attempts + 1):
            logger.info(
                "Starting repair attempt %d/%d",
                attempt,
                self.max_attempts,
            )

            # Each attempt is an independent Agent run. Retry attempts use a
            # focused system prompt and the latest verification evidence.
            run_options: dict[str, Any] = {"reset_state": True}
            if attempt > 1:
                run_options["system_prompt"] = REPAIR_SYSTEM_PROMPT
            agent_response = self.agent_loop.run(
                issue=current_issue,
                **run_options,
            )

            # Run verification if a verifier is provided
            verification_report = None
            if self.verifier:
                verification_report = self.verifier()
                verification_report.retry_count = attempt - 1

                # If verification passed, return success
                if verification_report.passed:
                    logger.info(
                        "Repair succeeded on attempt %d",
                        attempt,
                    )
                    return agent_response, verification_report

                # Select repair candidates using RepairSelector
                selection = self.repair_selector.select_repair_candidates(
                    verification_report,
                    change_plan,
                )

                # Check if we should stop without repair
                if selection.should_stop:
                    # If there are repair candidates but agent made no changes, give a second chance
                    if selection.repair_candidates and attempt == 1:
                        logger.warning(
                            "Agent had repair candidates but made no changes on attempt %d, retrying with stronger prompt",
                            attempt,
                        )
                        # Continue to next attempt instead of stopping
                        continue

                    logger.warning(
                        "Repair stopped: %s (attempt %d)",
                        selection.stop_reason,
                        attempt,
                    )
                    # Return with the last verification report
                    return agent_response, verification_report

                # Generate failure fingerprints for stall detection
                current_fingerprints = {
                    candidate.fingerprint for candidate in selection.repair_candidates
                }

                # Check if the same failures are repeating
                if previous_fingerprints and current_fingerprints == previous_fingerprints:
                    logger.warning(
                        "Repair stalled: same failures repeated on attempt %d",
                        attempt,
                    )
                    raise RepairLoopStalledError(
                        f"Repair stalled after {attempt} attempt(s): "
                        f"same failure fingerprints detected"
                    )

                previous_fingerprints = current_fingerprints

                # Rebuild the next prompt from selected repair candidates
                if attempt < self.max_attempts:
                    current_issue = prompt_builder(issue, verification_report, selection)
                    if (
                        not isinstance(current_issue, str)
                        or not current_issue.strip()
                    ):
                        raise RepairLoopError(
                            "repair prompt builder returned an empty prompt"
                        )
            else:
                # No verifier means single-pass mode - return immediately
                return agent_response, None

        # If we've exhausted attempts without success
        raise RepairLoopLimitError(
            f"Repair loop exceeded maximum of {self.max_attempts} attempts"
        )


def run_repair_loop(
    agent_loop: AgentLoop,
    issue: str,
    max_attempts: int = 3,
    verifier: Verifier | None = None,
    repair_prompt_builder: RepairPromptBuilder | None = None,
    strategy: str = "balanced",
    changed_files: list[str] | None = None,
    approved_files: set[str] | None = None,
    change_plan: ChangePlan | None = None,
) -> tuple[str, VerificationReport | None]:
    """Convenience function to run a repair loop with default configuration.

    Args:
        agent_loop: The AgentLoop instance for running repair attempts
        issue: The original issue or plan description
        max_attempts: Maximum number of repair attempts (default: 3)
        verifier: Optional function to run verification
        repair_prompt_builder: Optional function to build repair prompts
        strategy: Verification strategy (strict, balanced, focused)
        changed_files: List of files that were actually changed
        approved_files: Set of files approved for modification
        change_plan: Optional ChangePlan for scope validation

    Returns:
        Tuple of (final_agent_response, final_verification_report)

    Raises:
        RepairLoopLimitError: If max_attempts is exceeded
        RepairLoopStalledError: If the same failure repeats
        RepairLoopError: For other repair loop failures
    """
    repair_loop = RepairLoop(
        agent_loop=agent_loop,
        max_attempts=max_attempts,
        verifier=verifier,
        strategy=strategy,
        changed_files=changed_files,
        approved_files=approved_files,
    )

    return repair_loop.run(
        issue=issue,
        repair_prompt_builder=repair_prompt_builder,
        change_plan=change_plan,
    )
