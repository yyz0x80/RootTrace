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
6. Otherwise continues with another repair attempt
"""

from __future__ import annotations

import logging

from patchpilot.agent_loop import AgentLoop
from patchpilot.verification.report import VerificationReport, failure_fingerprint

logger = logging.getLogger(__name__)


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
        verifier: callable | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

        self.agent_loop = agent_loop
        self.max_attempts = max_attempts
        self.verifier = verifier

    def run(
        self,
        issue: str,
        repair_prompt_builder: callable | None = None,
    ) -> tuple[str, VerificationReport | None]:
        """Run the repair loop until success, stall detection, or limit.

        Args:
            issue: The original issue or plan description
            repair_prompt_builder: Optional function to build repair prompts
                taking (issue, failure_report) and returning a prompt string

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

        for attempt in range(1, self.max_attempts + 1):
            logger.info(
                "Starting repair attempt %d/%d",
                attempt,
                self.max_attempts,
            )

            # Run the agent to attempt a repair
            agent_response = self.agent_loop.run(issue=current_issue)

            # Run verification if a verifier is provided
            verification_report = None
            if self.verifier:
                verification_report = self.verifier()

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

                # Build repair prompt for next attempt if builder provided
                if repair_prompt_builder and attempt < self.max_attempts:
                    current_issue = repair_prompt_builder(
                        issue, verification_report
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
    verifier: callable | None = None,
    repair_prompt_builder: callable | None = None,
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