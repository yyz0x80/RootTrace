"""Tests for the repair loop with early stopping logic."""

from unittest.mock import Mock

import pytest

from patchpilot.verification.report import CheckReport, VerificationReport
from patchpilot.workflow.repair_loop import (
    RepairLoop,
    RepairLoopError,
    RepairLoopLimitError,
    RepairLoopStalledError,
    run_repair_loop,
)


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
                level="standard",
                command="pytest tests/",
                passed=False,
                exit_code=1,
                duration_seconds=1.0,
                failure_type="AssertionError",
                summary={"error_type": "AssertionError", "failed_tests": ["test_1"]},
            )
        )
        
        report2 = VerificationReport(passed=False)
        report2.add_check(
            CheckReport(
                level="standard",
                command="pytest tests/",
                passed=False,
                exit_code=1,
                duration_seconds=1.0,
                failure_type="TypeError",
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
                level="standard",
                command="pytest tests/",
                passed=False,
                exit_code=1,
                duration_seconds=1.0,
                failure_type="AssertionError",
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
        """Test repair loop with custom prompt builder."""
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

        # Custom prompt builder
        def build_prompt(issue, failure_report):
            return f"REPAIR: {issue}"

        result, _report = repair_loop.run(
            "Fix the bug",
            repair_prompt_builder=build_prompt,
        )

        assert result == "Fixed successfully"
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

        def build_prompt(issue, failure_report):
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