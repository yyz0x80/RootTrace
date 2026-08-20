"""Tests for the repair loop with early stopping logic."""

import json
from unittest.mock import Mock

import pytest

from patchpilot.prompts import REPAIR_SYSTEM_PROMPT
from patchpilot.verification.report import CheckReport, VerificationReport
from patchpilot.workflow.repair_loop import (
    RepairLoop,
    RepairLoopError,
    RepairLoopLimitError,
    RepairLoopStalledError,
    build_failure_repair_prompt,
    run_repair_loop,
)
from patchpilot.workflow.repair_selector import RepairSelection


class TestRepairLoopInit:
    """Tests for RepairLoop initialization."""

    def test_init_with_valid_parameters(self):
        """Test initialization with valid parameters."""
        mock_agent_loop = Mock()
        mock_verifier = Mock()

        repair_loop = RepairLoop(
            agent_loop=mock_agent_loop,
            max_attempts=5,
            verifier=mock_verifier,
        )

        assert repair_loop.agent_loop == mock_agent_loop
        assert repair_loop.max_attempts == 5
        assert repair_loop.verifier == mock_verifier

    def test_init_with_default_max_attempts(self):
        """Test initialization with default max_attempts."""
        mock_agent_loop = Mock()

        repair_loop = RepairLoop(agent_loop=mock_agent_loop)

        assert repair_loop.max_attempts == 3

    def test_init_with_invalid_max_attempts(self):
        """Test that max_attempts must be at least 1."""
        mock_agent_loop = Mock()

        with pytest.raises(ValueError, match="max_attempts must be at least 1"):
            RepairLoop(agent_loop=mock_agent_loop, max_attempts=0)

        with pytest.raises(ValueError, match="max_attempts must be at least 1"):
            RepairLoop(agent_loop=mock_agent_loop, max_attempts=-1)


class TestRepairLoopRun:
    """Tests for RepairLoop.run method."""

    def test_run_with_empty_issue(self):
        """Test that empty issue raises ValueError."""
        mock_agent_loop = Mock()
        mock_verifier = Mock()

        repair_loop = RepairLoop(
            agent_loop=mock_agent_loop,
            verifier=mock_verifier,
        )

        with pytest.raises(ValueError, match="issue must not be empty"):
            repair_loop.run("")

        with pytest.raises(ValueError, match="issue must not be empty"):
            repair_loop.run("   ")

    def test_run_immediate_success(self):
        """Test successful repair on first attempt."""
        mock_agent_loop = Mock()
        mock_agent_loop.run.return_value = "Fixed successfully"
        
        mock_verifier = Mock()
        mock_report = VerificationReport(passed=True)
        mock_verifier.return_value = mock_report

        repair_loop = RepairLoop(
            agent_loop=mock_agent_loop,
            max_attempts=3,
            verifier=mock_verifier,
        )

        result, report = repair_loop.run("Fix the bug")

        assert result == "Fixed successfully"
        assert report.passed is True
        assert mock_agent_loop.run.call_count == 1
        assert mock_verifier.call_count == 1

    def test_run_max_attempts_exceeded(self):
        """Test that exceeding max attempts raises RepairLoopLimitError."""
        mock_agent_loop = Mock()
        mock_agent_loop.run.return_value = "Attempted fix"
        
        mock_verifier = Mock()
        
        # Create different failures to avoid stall detection
        report1 = VerificationReport(passed=False)
        report1.add_check(
            CheckReport(
                method="pytest",
                phase="post_patch",
                level="standard",
                command="pytest tests/",
                passed=False,
                exit_code=1,
                duration_seconds=1.0,
                failure_type="AssertionError",
                tier="required",
                transition="NEW_OR_UNCOMPARED",
                failure_fingerprint="req1",
                summary={"error_type": "AssertionError", "failed_tests": ["test_1"]},
            )
        )
        
        report2 = VerificationReport(passed=False)
        report2.add_check(
            CheckReport(
                method="pytest",
                phase="post_patch",
                level="standard",
                command="pytest tests/",
                passed=False,
                exit_code=1,
                duration_seconds=1.0,
                failure_type="TypeError",
                tier="required",
                transition="NEW_OR_UNCOMPARED",
                failure_fingerprint="req2",
                summary={"error_type": "TypeError", "failed_tests": ["test_2"]},
            )
        )
        
        mock_verifier.side_effect = [report1, report2]

        repair_loop = RepairLoop(
            agent_loop=mock_agent_loop,
            max_attempts=2,
            verifier=mock_verifier,
        )

        with pytest.raises(RepairLoopLimitError, match="exceeded maximum of 2 attempts"):
            repair_loop.run("Fix the bug")

        assert mock_agent_loop.run.call_count == 2
        assert mock_verifier.call_count == 2

    def test_run_stalled_detection(self):
        """Test that repeated failures raise RepairLoopStalledError."""
        mock_agent_loop = Mock()
        mock_agent_loop.run.return_value = "Attempted fix"
        
        mock_verifier = Mock()
        
        # Create consistent failure fingerprint
        mock_report = VerificationReport(passed=False)
        mock_report.add_check(
            CheckReport(
                method="pytest",
                phase="post_patch",
                level="standard",
                command="pytest tests/",
                passed=False,
                exit_code=1,
                duration_seconds=1.0,
                failure_type="AssertionError",
                tier="required",
                transition="NEW_OR_UNCOMPARED",
                failure_fingerprint="same123",
                summary={
                    "failed_tests": ["test_example"],
                    "error_type": "AssertionError",
                    "relevant_output": "expected 'high', got None",
                },
            )
        )
        mock_verifier.return_value = mock_report

        repair_loop = RepairLoop(
            agent_loop=mock_agent_loop,
            max_attempts=3,
            verifier=mock_verifier,
        )

        with pytest.raises(RepairLoopStalledError, match="same failure fingerprint"):
            repair_loop.run("Fix the bug")

        # Should stop after 2 attempts (first fails, second repeats, stops)
        assert mock_agent_loop.run.call_count == 2
        assert mock_verifier.call_count == 2

    def test_run_with_repair_prompt_builder(self):
        """Retry attempts should use custom failure feedback in repair mode."""
        mock_agent_loop = Mock()
        mock_agent_loop.run.side_effect = ["Initial attempt", "Fixed successfully"]

        mock_verifier = Mock()
        failed_report = VerificationReport(passed=False)
        failed_report.add_check(
            CheckReport(
                method="pytest",
                phase="post_patch",
                level="standard",
                command="python -m pytest tests/ -q",
                passed=False,
                exit_code=1,
                duration_seconds=1.0,
                failure_type="TEST_FAILURE",
                tier="required",
                transition="NEW_OR_UNCOMPARED",
                failure_fingerprint="abc123",
            )
        )
        passed_report = VerificationReport(passed=True)
        mock_verifier.side_effect = [failed_report, passed_report]

        repair_loop = RepairLoop(
            agent_loop=mock_agent_loop,
            max_attempts=3,
            verifier=mock_verifier,
        )

        # Custom prompt builder with new signature
        def build_prompt(issue, failure_report, selection):
            return f"REPAIR: {issue}"

        result, report = repair_loop.run(
            "Fix the bug",
            repair_prompt_builder=build_prompt,
        )

        assert result == "Fixed successfully"
        assert report is passed_report
        assert report.retry_count == 1
        assert mock_agent_loop.run.call_args_list[0].kwargs == {
            "issue": "Fix the bug",
            "reset_state": True,
        }
        assert mock_agent_loop.run.call_args_list[1].kwargs == {
            "issue": "REPAIR: Fix the bug",
            "reset_state": True,
            "system_prompt": REPAIR_SYSTEM_PROMPT,
        }

    def test_run_builds_default_failure_prompt_for_retry(self):
        """A retry should receive the latest failure without a custom builder."""
        mock_agent_loop = Mock()
        mock_agent_loop.run.side_effect = ["Initial attempt", "Fixed successfully"]
        failed_report = VerificationReport(passed=False)
        failed_report.add_check(
            CheckReport(
                method="pytest",
                phase="post_patch",
                level="standard",
                command="python -m pytest tests/test_math.py -q",
                passed=False,
                exit_code=1,
                duration_seconds=1.0,
                failure_type="TEST_FAILURE",
                tier="required",
                transition="NEW_OR_UNCOMPARED",
                failure_fingerprint="abc123",
                summary={
                    "failed_tests": ["tests/test_math.py::test_total"],
                    "error_type": "AssertionError",
                    "relevant_output": "assert 3 == 4",
                },
            )
        )
        mock_verifier = Mock(
            side_effect=[failed_report, VerificationReport(passed=True)]
        )
        plan = json.dumps(
            {
                "planned_changes": [
                    {
                        "path": "src/math.py",
                        "action": "modify",
                        "description": "Correct total calculation.",
                        "acceptance_criteria": ["AC-1"],
                    }
                ],
                "out_of_scope": ["Do not change public APIs."],
                "risk_level": "low",
            }
        )
        repair_loop = RepairLoop(
            agent_loop=mock_agent_loop,
            max_attempts=2,
            verifier=mock_verifier,
        )

        repair_loop.run(plan)

        retry_call = mock_agent_loop.run.call_args_list[1]
        retry_prompt = retry_call.kwargs["issue"]
        assert "src/math.py" in retry_prompt
        assert "python -m pytest tests/test_math.py -q" in retry_prompt
        assert "tests/test_math.py::test_total" in retry_prompt
        assert "assert 3 == 4" in retry_prompt
        assert "Do not change public APIs." in retry_prompt
        assert "AC-1" in retry_prompt
        assert retry_call.kwargs["system_prompt"] == REPAIR_SYSTEM_PROMPT

    def test_run_rejects_empty_repair_prompt(self):
        """An invalid prompt builder should fail before another model call."""
        mock_agent_loop = Mock()
        mock_agent_loop.run.return_value = "Initial attempt"
        failed_report = VerificationReport(passed=False)
        failed_report.add_check(
            CheckReport(
                method="ruff",
                phase="post_patch",
                level="standard",
                command="ruff check",
                passed=False,
                exit_code=1,
                duration_seconds=1.0,
                tier="required",
                transition="NEW_OR_UNCOMPARED",
                failure_fingerprint="ruff123",
            )
        )
        repair_loop = RepairLoop(
            agent_loop=mock_agent_loop,
            verifier=Mock(return_value=failed_report),
        )

        with pytest.raises(
            RepairLoopError,
            match="repair prompt builder returned an empty prompt",
        ):
            repair_loop.run(
                "Fix the bug",
                repair_prompt_builder=lambda _issue, _report, _selection: "   ",
            )

        assert mock_agent_loop.run.call_count == 1

    def test_run_without_verifier(self):
        """Test repair loop without verifier (single pass)."""
        mock_agent_loop = Mock()
        mock_agent_loop.run.return_value = "Fixed successfully"

        repair_loop = RepairLoop(
            agent_loop=mock_agent_loop,
            max_attempts=3,
            verifier=None,
        )

        result, report = repair_loop.run("Fix the bug")

        assert result == "Fixed successfully"
        assert report is None
        assert mock_agent_loop.run.call_count == 1

    def test_run_stops_when_no_repairable_failures(self):
        """Test that repair loop stops when selector finds no repairable failures."""
        mock_agent_loop = Mock()
        mock_agent_loop.run.return_value = "Initial attempt"

        # Create a report with only pre-existing failures (non-repairable)
        failed_report = VerificationReport(passed=False)
        failed_report.add_check(
            CheckReport(
                method="pytest",
                phase="post_patch",
                level="standard",
                command="pytest tests/",
                passed=False,
                exit_code=1,
                duration_seconds=1.0,
                failure_type="AssertionError",
                tier="affected",
                transition="PRE_EXISTING_FAILURE",
                failure_fingerprint="preexisting123",
                summary={"error_type": "AssertionError", "failed_tests": ["test_old"]},
            )
        )

        mock_verifier = Mock(return_value=failed_report)
        repair_loop = RepairLoop(
            agent_loop=mock_agent_loop,
            max_attempts=3,
            verifier=mock_verifier,
        )

        result, report = repair_loop.run("Fix the bug")

        # Should stop after initial attempt without retry
        assert result == "Initial attempt"
        assert report == failed_report
        assert mock_agent_loop.run.call_count == 1
        assert mock_verifier.call_count == 1

    def test_run_stops_on_repeated_relevant_fingerprints(self):
        """Test that repair loop stops when same relevant failures repeat."""
        mock_agent_loop = Mock()
        mock_agent_loop.run.side_effect = ["Initial attempt", "Repair attempt"]

        # Create reports with same repairable failure fingerprint
        failed_report = VerificationReport(passed=False)
        failed_report.add_check(
            CheckReport(
                method="pytest",
                phase="post_patch",
                level="standard",
                command="pytest tests/test.py",
                passed=False,
                exit_code=1,
                duration_seconds=1.0,
                failure_type="AssertionError",
                tier="required",
                transition="NEW_OR_UNCOMPARED",
                failure_fingerprint="same123",
                summary={"error_type": "AssertionError", "failed_tests": ["test_example"]},
            )
        )

        mock_verifier = Mock(return_value=failed_report)
        repair_loop = RepairLoop(
            agent_loop=mock_agent_loop,
            max_attempts=3,
            verifier=mock_verifier,
        )

        with pytest.raises(RepairLoopStalledError, match="same failure fingerprints"):
            repair_loop.run("Fix the bug")

        # Should stop after 2 attempts (initial + 1 repair that repeats)
        assert mock_agent_loop.run.call_count == 2
        assert mock_verifier.call_count == 2


class TestRunRepairLoop:
    """Tests for the convenience run_repair_loop function."""

    def test_run_repair_loop_convenience(self):
        """Test the convenience function with default parameters."""
        mock_agent_loop = Mock()
        mock_agent_loop.run.return_value = "Fixed successfully"
        
        mock_verifier = Mock()
        mock_report = VerificationReport(passed=True)
        mock_verifier.return_value = mock_report

        result, report = run_repair_loop(
            agent_loop=mock_agent_loop,
            issue="Fix the bug",
            max_attempts=2,
            verifier=mock_verifier,
        )

        assert result == "Fixed successfully"
        assert report.passed is True
        assert mock_agent_loop.run.call_count == 1

    def test_run_repair_loop_with_prompt_builder(self):
        """Test the convenience function with prompt builder."""
        mock_agent_loop = Mock()
        mock_agent_loop.run.return_value = "Fixed successfully"

        mock_verifier = Mock()
        mock_report = VerificationReport(passed=True)
        mock_verifier.return_value = mock_report

        def build_prompt(issue, failure_report, selection):
            return f"REPAIR: {issue}"

        result, report = run_repair_loop(
            agent_loop=mock_agent_loop,
            issue="Fix the bug",
            max_attempts=2,
            verifier=mock_verifier,
            repair_prompt_builder=build_prompt,
        )

        assert result == "Fixed successfully"
        assert report.passed is True


class TestBuildFailureRepairPrompt:
    """Tests for default verification-driven repair context."""

    def test_plain_issue_uses_bounded_failure_evidence(self):
        """Plain issue prompts should include actionable failure details."""
        report = VerificationReport(passed=False)
        report.add_check(
            CheckReport(
                method="ruff",
                phase="post_patch",
                level="quick",
                command="ruff check",
                passed=False,
                exit_code=1,
                duration_seconds=0.1,
                failure_type="CODE_FAILURE",
                tier="required",
                transition="NEW_OR_UNCOMPARED",
                failure_fingerprint="abc123",
                summary={"error": "F821 undefined name 'value'"},
            )
        )

        # Create a mock selection with repair candidates
        from patchpilot.workflow.repair_selector import RepairCandidate, RepairSelection
        selection = RepairSelection(
            repair_candidates=[
                RepairCandidate(
                    check=report.checks[0],
                    reason="REQUIRED test failure (new or worsened)",
                    tier="required",
                    transition="NEW_OR_UNCOMPARED",
                    fingerprint="abc123",
                    bounded_output="F821 undefined name 'value'",
                )
            ],
            should_repair=True,
        )

        prompt = build_failure_repair_prompt("Fix the parser", report, selection)

        assert "Fix the parser" in prompt
        assert "ruff check" in prompt
        assert "CODE_FAILURE" in prompt
        assert "F821 undefined name 'value'" in prompt


class TestRepairLoopExceptions:
    """Tests for repair loop exception handling."""

    def test_repair_loop_error_base_class(self):
        """Test that RepairLoopError is a RuntimeError."""
        assert issubclass(RepairLoopError, RuntimeError)

    def test_repair_loop_limit_error_inheritance(self):
        """Test that RepairLoopLimitError inherits from RepairLoopError."""
        assert issubclass(RepairLoopLimitError, RepairLoopError)

    def test_repair_loop_stalled_error_inheritance(self):
        """Test that RepairLoopStalledError inherits from RepairLoopError."""
        assert issubclass(RepairLoopStalledError, RepairLoopError)
