"""Tests for the Verifier class."""

from __future__ import annotations

from unittest.mock import MagicMock

from patchpilot.sandbox.docker_runner import CommandResult
from patchpilot.verification.verifier import Verifier


def test_verifier_initialization() -> None:
    """Test that Verifier initializes correctly with a sandbox."""
    sandbox_mock = MagicMock()
    verifier = Verifier(sandbox=sandbox_mock)
    assert verifier.sandbox == sandbox_mock


def test_verify_all_checks_pass() -> None:
    """Test verify when all checks pass."""
    sandbox_mock = MagicMock()
    # Setup mock to return success for all commands
    sandbox_mock.run.side_effect = [
        CommandResult(
            command="ruff check --no-cache .",
            exit_code=0,
            stdout="",
            stderr="",
            duration_seconds=1.0,
        ),
        CommandResult(
            command="pytest -q -p no:cacheprovider",
            exit_code=0,
            stdout="",
            stderr="",
            duration_seconds=2.0,
        ),
    ]

    verifier = Verifier(sandbox=sandbox_mock)
    report = verifier.verify(run_id="test-run-1")

    assert report.passed is True
    assert report.run_id == "test-run-1"
    assert len(report.checks) == 2
    assert report.checks[0].level == "LEVEL_1_LINT"
    assert report.checks[0].passed is True
    assert report.checks[1].level == "LEVEL_3_REGRESSION"
    assert report.checks[1].passed is True


def test_verify_with_target_tests() -> None:
    """Test verify with specific target tests."""
    sandbox_mock = MagicMock()
    # Setup mock to return success for all commands
    sandbox_mock.run.side_effect = [
        CommandResult(
            command="ruff check --no-cache .",
            exit_code=0,
            stdout="",
            stderr="",
            duration_seconds=1.0,
        ),
        CommandResult(
            command="pytest tests/test_specific.py -q -p no:cacheprovider",
            exit_code=0,
            stdout="",
            stderr="",
            duration_seconds=1.5,
        ),
        CommandResult(
            command="pytest -q -p no:cacheprovider",
            exit_code=0,
            stdout="",
            stderr="",
            duration_seconds=2.0,
        ),
    ]

    verifier = Verifier(sandbox=sandbox_mock)
    report = verifier.verify(
        run_id="test-run-2",
        target_tests=["tests/test_specific.py"],
        target_acceptance_criteria=["AC-1", "AC-2"],
    )

    assert report.passed is True
    assert len(report.checks) == 3
    assert report.checks[0].level == "LEVEL_1_LINT"
    assert report.checks[1].level == "LEVEL_2_TARGET_TESTS"
    assert report.checks[2].level == "LEVEL_3_REGRESSION"
    assert report.checks[0].acceptance_criteria == []
    assert report.checks[1].acceptance_criteria == [
        "AC-1",
        "AC-2",
    ]
    assert report.checks[2].acceptance_criteria == []


def test_verify_ruff_fails_fail_fast() -> None:
    """Test verify fails fast when ruff check fails."""
    sandbox_mock = MagicMock()
    # Setup mock to return failure for ruff
    sandbox_mock.run.return_value = CommandResult(
        command="ruff check --no-cache .",
        exit_code=1,
        stdout="",
        stderr="E501 line too long",
        duration_seconds=0.5,
    )

    verifier = Verifier(sandbox=sandbox_mock)
    report = verifier.verify(run_id="test-run-3")

    assert report.passed is False
    assert len(report.checks) == 1
    assert report.checks[0].level == "LEVEL_1_LINT"
    assert report.checks[0].passed is False
    assert report.failed_level == "LEVEL_1_LINT"
    assert report.failure_type is not None

    # Verify only ruff was called (fail-fast)
    assert sandbox_mock.run.call_count == 1


def test_verify_target_test_fails_fail_fast() -> None:
    """Test verify fails fast when target tests fail."""
    sandbox_mock = MagicMock()
    # Setup mock: ruff passes, target tests fail
    sandbox_mock.run.side_effect = [
        CommandResult(
            command="ruff check --no-cache .",
            exit_code=0,
            stdout="",
            stderr="",
            duration_seconds=1.0,
        ),
        CommandResult(
            command="pytest tests/test_failing.py -q -p no:cacheprovider",
            exit_code=1,
            stdout="FAILED tests/test_failing.py::test_example",
            stderr="AssertionError",
            duration_seconds=1.5,
        ),
    ]

    verifier = Verifier(sandbox=sandbox_mock)
    report = verifier.verify(
        run_id="test-run-4",
        target_tests=["tests/test_failing.py"],
        target_acceptance_criteria=["AC-1"],
    )

    assert report.passed is False
    assert len(report.checks) == 2
    assert report.checks[0].level == "LEVEL_1_LINT"
    assert report.checks[0].passed is True
    assert report.checks[1].level == "LEVEL_2_TARGET_TESTS"
    assert report.checks[1].passed is False
    assert report.failed_level == "LEVEL_2_TARGET_TESTS"
    assert report.checks[1].acceptance_criteria == ["AC-1"]

    # Verify ruff and target tests were called, but not full regression
    assert sandbox_mock.run.call_count == 2


def test_verify_with_retry_count() -> None:
    """Test verify includes retry count in report."""
    sandbox_mock = MagicMock()
    sandbox_mock.run.return_value = CommandResult(
        command="ruff check --no-cache .",
        exit_code=0,
        stdout="",
        stderr="",
        duration_seconds=1.0,
    )

    verifier = Verifier(sandbox=sandbox_mock)
    report = verifier.verify(run_id="test-run-5", retry_count=2)

    assert report.retry_count == 2


def test_verify_command_arguments() -> None:
    """Test verify passes correct command arguments to sandbox."""
    sandbox_mock = MagicMock()
    sandbox_mock.run.side_effect = [
        CommandResult(
            command="ruff check --no-cache .",
            exit_code=0,
            stdout="",
            stderr="",
            duration_seconds=1.0,
        ),
        CommandResult(
            command="pytest -q -p no:cacheprovider",
            exit_code=0,
            stdout="",
            stderr="",
            duration_seconds=2.0,
        ),
    ]

    verifier = Verifier(sandbox=sandbox_mock)
    verifier.verify(run_id="test-run-6")

    # Verify sandbox.run was called with timeout for each command
    assert sandbox_mock.run.call_count == 2
    for call in sandbox_mock.run.call_args_list:
        assert call[1]["timeout_seconds"] == 60
