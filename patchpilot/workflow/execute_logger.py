"""Structured logging for PatchPilot execute workflow.

This module provides a structured logging helper that produces clean,
section-based logging output during the execute workflow, following
the format specified in the day3 development plan.

Sections:
- PRECHECK: Repository and baseline validation
- WORKSPACE: Temporary workspace setup
- SANDBOX: Docker container initialization
- CODING: Agent execution rounds
- CHANGES: File modification summary
- SCOPE: Runtime scope validation
- VERIFY: Verification results
- RESULT: Final outcome and artifacts
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ExecuteLogger:
    """Structured logger for PatchPilot execute workflow.

    Provides section-based logging with consistent formatting
    for better visibility and debugging of the execution process.
    """

    _SECTION_WIDTH = 80
    _STATUS_WIDTH = 5

    @staticmethod
    def _section_header(title: str) -> str:
        """Generate a formatted section header.

        Args:
            title: Section title

        Returns:
            Formatted section header string
        """
        title_centered = title.center(ExecuteLogger._SECTION_WIDTH - 2)
        return f"[{title_centered}]"

    @staticmethod
    def _status_line(status: str, message: str) -> str:
        """Generate a formatted status line.

        Args:
            status: Status indicator (PASS, FAIL, etc.)
            message: Status message

        Returns:
            Formatted status line string
        """
        status_formatted = status.ljust(ExecuteLogger._STATUS_WIDTH)
        return f"{status_formatted}: {message}"

    @staticmethod
    def log_section(title: str) -> None:
        """Log a section header.

        Args:
            title: Section title
        """
        logger.info(ExecuteLogger._section_header(title))

    @staticmethod
    def log_issue(issue_title: str) -> None:
        """Log the loaded issue.

        Args:
            issue_title: Issue title or description
        """
        logger.info("Loading normalized issue...")
        logger.info("Loaded issue: %s", issue_title)

    @staticmethod
    def log_plan(base_commit: str, planned_changes_count: int) -> None:
        """Log the approved plan details.

        Args:
            base_commit: Base commit hash
            planned_changes_count: Number of planned changes
        """
        logger.info("Loading approved plan...")
        logger.info("Base commit: %s", base_commit)
        logger.info("Planned changes: %d", planned_changes_count)

    @staticmethod
    def log_precheck(
        git_repo: bool,
        working_tree_clean: bool,
        base_commit_match: bool,
    ) -> None:
        """Log precheck validation results.

        Args:
            git_repo: Whether target is a valid Git repository
            working_tree_clean: Whether working tree is clean
            base_commit_match: Whether HEAD matches base commit
        """
        ExecuteLogger.log_section("PRECHECK")
        logger.info(
            ExecuteLogger._status_line(
                "PASS" if git_repo else "FAIL",
                "Git repository"
            )
        )
        logger.info(
            ExecuteLogger._status_line(
                "PASS" if working_tree_clean else "FAIL",
                "Working tree clean"
            )
        )
        logger.info(
            ExecuteLogger._status_line(
                "PASS" if base_commit_match else "FAIL",
                "Base commit match"
            )
        )

    @staticmethod
    def log_workspace_setup(workspace_path: str) -> None:
        """Log workspace setup.

        Args:
            workspace_path: Path to the temporary workspace
        """
        ExecuteLogger.log_section("WORKSPACE")
        logger.info("Creating repository snapshot from HEAD...")
        logger.info("Temporary Git baseline created")
        logger.info("Workspace path: %s", workspace_path)

    @staticmethod
    def log_sandbox_start() -> None:
        """Log Docker sandbox startup."""
        ExecuteLogger.log_section("SANDBOX")
        logger.info("Starting isolated Docker container")

    @staticmethod
    def log_coding_round(round_number: int, tool_name: str, args: dict[str, Any]) -> None:
        """Log a coding agent round.

        Args:
            round_number: Round number
            tool_name: Name of the tool being called
            args: Tool arguments
        """
        args_str = ", ".join(f"{k}={v}" for k, v in args.items())
        logger.info("Round %d: %s(%s)", round_number, tool_name, args_str)

    @staticmethod
    def log_coding_complete(round_number: int) -> None:
        """Log coding agent completion.

        Args:
            round_number: Final round number
        """
        logger.info("Round %d: final answer", round_number)

    @staticmethod
    def log_changes(modified: list[str], created: list[str], deleted: list[str]) -> None:
        """Log file changes summary.

        Args:
            modified: List of modified files
            created: List of created files
            deleted: List of deleted files
        """
        ExecuteLogger.log_section("CHANGES")
        for file in modified:
            logger.info("M %s", file)
        for file in created:
            logger.info("A %s", file)
        for file in deleted:
            logger.info("D %s", file)

    @staticmethod
    def log_scope_validation(allowed: bool, violations: list[str] | None = None) -> None:
        """Log scope validation results.

        Args:
            allowed: Whether changes are within scope
            violations: Optional list of scope violations
        """
        ExecuteLogger.log_section("SCOPE")
        if allowed:
            logger.info("All changes match approved plan: PASS")
        else:
            logger.info("All changes match approved plan: FAIL")
            if violations:
                for violation in violations:
                    logger.info("Violation: %s", violation)

    @staticmethod
    def log_verification(results: dict[str, bool]) -> None:
        """Log verification results.

        Args:
            results: Dictionary mapping check names to pass/fail status
        """
        ExecuteLogger.log_section("VERIFY")
        for check_name, passed in results.items():
            status = "PASS" if passed else "FAIL"
            logger.info("%s: %s", check_name, status)

    @staticmethod
    def log_result(
        passed: bool,
        artifacts: dict[str, str],
    ) -> None:
        """Log final result and artifacts.

        Args:
            passed: Whether the overall verification passed
            artifacts: Dictionary mapping artifact names to file paths
        """
        ExecuteLogger.log_section("RESULT")
        logger.info("VERIFIED" if passed else "FAILED")
        logger.info("Generated:")
        for artifact_name, artifact_path in artifacts.items():
            logger.info("%s: %s", artifact_name, artifact_path)

    @staticmethod
    def log_repair_attempt(attempt: int, max_attempts: int) -> None:
        """Log a repair attempt.

        Args:
            attempt: Current attempt number
            max_attempts: Maximum number of attempts
        """
        logger.info("Starting repair attempt %d/%d", attempt, max_attempts)

    @staticmethod
    def log_repair_stopped(reason: str) -> None:
        """Log repair loop stop.

        Args:
            reason: Reason for stopping repair loop
        """
        logger.info("Repair loop stopped: %s", reason)

    @staticmethod
    def log_issue_loading(issue_path: str) -> None:
        """Log issue loading for prepare workflow.

        Args:
            issue_path: Path to the issue file
        """
        ExecuteLogger.log_section("ISSUE")
        logger.info("Loading issue from: %s", issue_path)

    @staticmethod
    def log_issue_normalization(success: bool, ambiguous_points: list[str] | None = None) -> None:
        """Log issue normalization result.

        Args:
            success: Whether normalization succeeded
            ambiguous_points: Optional list of ambiguous points
        """
        if success:
            logger.info("Issue normalization: PASS")
        else:
            logger.info("Issue normalization: FAIL")
            if ambiguous_points:
                logger.info("Ambiguous requirements:")
                for i, point in enumerate(ambiguous_points, start=1):
                    logger.info("%d. %s", i, point)

    @staticmethod
    def log_repository_validation(
        is_valid: bool,
        head_sha: str | None = None,
        error: str | None = None,
    ) -> None:
        """Log repository validation result.

        Args:
            is_valid: Whether repository validation passed
            head_sha: Optional HEAD commit SHA
            error: Optional error message
        """
        ExecuteLogger.log_section("REPOSITORY")
        if is_valid:
            logger.info("Repository validation: PASS")
            if head_sha:
                logger.info("Base commit: %s", head_sha[:8])
        else:
            logger.info("Repository validation: FAIL")
            if error:
                logger.info("Error: %s", error)

    @staticmethod
    def log_repository_analysis(
        python_files_count: int,
        test_files_count: int,
        keyword_matches_count: int,
    ) -> None:
        """Log repository analysis results.

        Args:
            python_files_count: Number of Python files found
            test_files_count: Number of test files found
            keyword_matches_count: Number of keyword matches
        """
        ExecuteLogger.log_section("ANALYSIS")
        logger.info("Python files: %d", python_files_count)
        logger.info("Test files: %d", test_files_count)
        logger.info("Relevant matches: %d", keyword_matches_count)

    @staticmethod
    def log_plan_creation(planned_changes: list[str], planned_tests: list[str]) -> None:
        """Log plan creation details.

        Args:
            planned_changes: List of planned file changes
            planned_tests: List of planned test commands
        """
        ExecuteLogger.log_section("PLAN")
        logger.info("Planned changes:")
        for change in planned_changes:
            logger.info("  %s", change)
        if planned_tests:
            logger.info("Planned tests:")
            for test in planned_tests:
                logger.info("  %s", test)

    @staticmethod
    def log_plan_validation(
        allowed: bool,
        violations: list[str] | None = None,
        warnings: list[str] | None = None,
    ) -> None:
        """Log plan validation and scope gate results.

        Args:
            allowed: Whether plan passed scope gate
            violations: Optional list of scope violations
            warnings: Optional list of scope warnings
        """
        ExecuteLogger.log_section("VALIDATION")
        if allowed:
            logger.info("Scope gate: PASS")
        else:
            logger.info("Scope gate: FAIL")
            if violations:
                logger.info("Violations:")
                for i, violation in enumerate(violations, start=1):
                    logger.info("%d. %s", i, violation)
            if warnings:
                logger.info("Warnings:")
                for i, warning in enumerate(warnings, start=1):
                    logger.info("%d. %s", i, warning)

    @staticmethod
    def log_artifacts(artifact_paths: list[str]) -> None:
        """Log generated artifacts.

        Args:
            artifact_paths: List of generated artifact file paths
        """
        ExecuteLogger.log_section("ARTIFACTS")
        logger.info("Generated artifacts:")
        for path in artifact_paths:
            logger.info("  %s", path)
