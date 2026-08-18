"""Repair loop with early stopping logic for repeated failures.

This module implements a repair loop that attempts to fix verification failures
while detecting when the same failure persists across repair attempts. When a
failure fingerprint remains unchanged, the loop stops early to avoid wasting
LLM calls on futile repair attempts.

The repair loop:
1. Runs the agent to implement a fix
2. Verifies the fix with deterministic checks
3. If verification fails, generates a failure fingerprint
4. Compares with previous failure fingerprint
5. Stops early if the same failure recurs
6. Builds the next prompt from the latest deterministic failure
7. Otherwise continues with another repair attempt
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from patchpilot.agent_loop import AgentLoop
from patchpilot.planning.schema import ChangePlan
from patchpilot.prompts import REPAIR_PROMPT, REPAIR_SYSTEM_PROMPT
from patchpilot.verification.report import VerificationReport, failure_fingerprint

logger = logging.getLogger(__name__)

MAX_REPAIR_TASK_CHARS = 6_000
MAX_REPAIR_FAILURE_CHARS = 2_000

Verifier = Callable[[], VerificationReport]
RepairPromptBuilder = Callable[[str, VerificationReport], str]


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


def _format_failure(report: VerificationReport) -> str:
    """Format the latest deterministic failure as actionable repair evidence."""
    failed_checks = report.get_failed_checks()
    if not failed_checks:
        return "No specific failure details available."

    failed = failed_checks[-1]
    details = [
        f"Command: {failed.command}",
        f"Failure Type: {failed.failure_type or report.failure_type or 'unknown'}",
        f"Exit Code: {failed.exit_code}",
    ]
    summary: dict[str, Any] = failed.summary or {}
    failed_tests = summary.get("failed_tests")
    if isinstance(failed_tests, list) and failed_tests:
        details.append(
            "Failed Tests: " + ", ".join(str(test) for test in failed_tests)
        )
    error_type = summary.get("error_type")
    if error_type:
        details.append(f"Error Type: {error_type}")
    relevant_output = summary.get("relevant_output") or summary.get("error")
    if relevant_output:
        details.append(
            "Relevant Output:\n"
            + _truncate_text(str(relevant_output), MAX_REPAIR_FAILURE_CHARS)
        )
    return "\n".join(details)


def build_failure_repair_prompt(
    issue: str,
    failure_report: VerificationReport,
) -> str:
    """Build a focused retry prompt from the latest verification failure."""
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
        failure=_format_failure(failure_report),
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

    Attributes:
        agent_loop: The AgentLoop instance for running repair attempts
        max_attempts: Maximum number of repair attempts
        verifier: Function to run verification and return a VerificationReport
    """

    def __init__(
        self,
        agent_loop: AgentLoop,
        max_attempts: int = 3,
        verifier: Verifier | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

        self.agent_loop = agent_loop
        self.max_attempts = max_attempts
        self.verifier = verifier

    def run(
        self,
        issue: str,
        repair_prompt_builder: RepairPromptBuilder | None = None,
    ) -> tuple[str, VerificationReport | None]:
        """Run the repair loop until success, stall detection, or limit.

        Args:
            issue: The original issue or plan description
            repair_prompt_builder: Optional override taking
                (issue, failure_report) and returning a prompt string. The
                default builder includes structured failure evidence.

        Returns:
            Tuple of (final_agent_response, final_verification_report)

        Raises:
            RepairLoopLimitError: If max_attempts is exceeded
            RepairLoopStalledError: If the same failure repeats
            RepairLoopError: For other repair loop failures
        """
        if not issue.strip():
            raise ValueError("issue must not be empty")

        previous_fingerprint = None
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

                # Generate failure fingerprint for stall detection
                current_fingerprint = failure_fingerprint(verification_report)

                # Check if the same failure is repeating
                if (
                    previous_fingerprint is not None
                    and current_fingerprint == previous_fingerprint
                ):
                    logger.warning(
                        "Repair stalled: same failure repeated on attempt %d",
                        attempt,
                    )
                    raise RepairLoopStalledError(
                        f"Repair stalled after {attempt} attempt(s): "
                        f"same failure fingerprint detected"
                    )

                previous_fingerprint = current_fingerprint

                # Rebuild the next prompt from the newest failure rather than
                # repeating the original implementation request.
                if attempt < self.max_attempts:
                    current_issue = prompt_builder(issue, verification_report)
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
) -> tuple[str, VerificationReport | None]:
    """Convenience function to run a repair loop with default configuration.

    Args:
        agent_loop: The AgentLoop instance for running repair attempts
        issue: The original issue or plan description
        max_attempts: Maximum number of repair attempts (default: 3)
        verifier: Optional function to run verification
        repair_prompt_builder: Optional function to build repair prompts

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
    )

    return repair_loop.run(
        issue=issue,
        repair_prompt_builder=repair_prompt_builder,
    )
