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
            command="python -m pytest -q -p no:cacheprovider",
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
    assert report.checks[0].method == "ruff"
    assert report.checks[0].phase == "post_patch"
    assert report.checks[0].passed is True
    assert report.checks[1].level == "LEVEL_3_REGRESSION"
    assert report.checks[1].method == "pytest"
    assert report.checks[1].phase == "post_patch"
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
            command="python -m pytest tests/test_specific.py -q -p no:cacheprovider",
            exit_code=0,
            stdout="",
            stderr="",
            duration_seconds=1.5,
        ),
        CommandResult(
            command="python -m pytest -q -p no:cacheprovider",
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
    assert report.checks[0].subject_ids == []
    assert report.checks[1].subject_ids == [
        "AC-1",
        "AC-2",
    ]
    assert report.checks[2].subject_ids == []


def test_verify_ruff_fails_no_fail_fast() -> None:
    """Test verify does not fail fast when ruff check fails in post-patch phase."""
    sandbox_mock = MagicMock()
    # Setup mock to return failure for ruff, but continue with other checks
    sandbox_mock.run.side_effect = [
        CommandResult(
            command="ruff check --no-cache .",
            exit_code=1,
            stdout="",
            stderr="E501 line too long",
            duration_seconds=0.5,
        ),
        CommandResult(
            command="python -m pytest -q -p no:cacheprovider",
            exit_code=0,
            stdout="",
            stderr="",
            duration_seconds=2.0,
        ),
    ]

    verifier = Verifier(sandbox=sandbox_mock)
    report = verifier.verify(run_id="test-run-3")

    assert report.passed is False
    assert len(report.checks) == 2  # Both checks run despite failure
    assert report.checks[0].level == "LEVEL_1_LINT"
    assert report.checks[0].passed is False
    assert report.checks[1].level == "LEVEL_3_REGRESSION"
    assert report.checks[1].passed is True
    assert report.failed_level == "LEVEL_1_LINT"
    assert report.failure_type is not None

    # Verify both checks were called (no fail-fast in post-patch)
    assert sandbox_mock.run.call_count == 2


def test_verify_target_test_fails_no_fail_fast() -> None:
    """Test verify does not fail fast when target tests fail in post-patch phase."""
    sandbox_mock = MagicMock()
    # Setup mock: ruff passes, target tests fail, regression still runs
    sandbox_mock.run.side_effect = [
        CommandResult(
            command="ruff check --no-cache .",
            exit_code=0,
            stdout="",
            stderr="",
            duration_seconds=1.0,
        ),
        CommandResult(
            command="python -m pytest tests/test_failing.py -q -p no:cacheprovider",
            exit_code=1,
            stdout="FAILED tests/test_failing.py::test_example",
            stderr="AssertionError",
            duration_seconds=1.5,
        ),
        CommandResult(
            command="python -m pytest -q -p no:cacheprovider",
            exit_code=0,
            stdout="",
            stderr="",
            duration_seconds=2.0,
        ),
    ]

    verifier = Verifier(sandbox=sandbox_mock)
    report = verifier.verify(
        run_id="test-run-4",
        target_tests=["tests/test_failing.py"],
        target_acceptance_criteria=["AC-1"],
    )

    assert report.passed is False
    assert len(report.checks) == 3  # All checks run despite failure
    assert report.checks[0].level == "LEVEL_1_LINT"
    assert report.checks[0].passed is True
    assert report.checks[1].level == "LEVEL_2_TARGET_TESTS"
    assert report.checks[1].passed is False
    assert report.checks[2].level == "LEVEL_3_REGRESSION"
    assert report.checks[2].passed is True
    assert report.failed_level == "LEVEL_2_TARGET_TESTS"
    assert report.checks[1].subject_ids == ["AC-1"]

    # Verify all checks were called (no fail-fast in post-patch)
    assert sandbox_mock.run.call_count == 3


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
            command="python -m pytest -q -p no:cacheprovider",
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


def test_verify_baseline_all_checks_pass() -> None:
    """Test baseline verification when all checks pass."""
    sandbox_mock = MagicMock()
    # Setup mock to return success for all commands
    sandbox_mock.run.side_effect = [
        CommandResult(
            command="python -m pytest -q -p no:cacheprovider",
            exit_code=0,
            stdout="",
            stderr="",
            duration_seconds=2.0,
        ),
    ]

    verifier = Verifier(sandbox=sandbox_mock)
    report = verifier.verify_baseline(run_id="test-baseline-1")

    assert report.passed is True
    assert report.run_id == "test-baseline-1"
    assert len(report.checks) == 1
    assert report.checks[0].level == "BASELINE_REGRESSION"
    assert report.checks[0].method == "pytest"
    assert report.checks[0].phase == "baseline"
    assert report.checks[0].passed is True


def test_verify_baseline_with_target_tests() -> None:
    """Test baseline verification with specific target tests."""
    sandbox_mock = MagicMock()
    # Setup mock to return success for all commands
    sandbox_mock.run.side_effect = [
        CommandResult(
            command="python -m pytest -q -p no:cacheprovider",
            exit_code=0,
            stdout="",
            stderr="",
            duration_seconds=2.0,
        ),
        CommandResult(
            command="python -m pytest tests/test_specific.py -q -p no:cacheprovider",
            exit_code=0,
            stdout="",
            stderr="",
            duration_seconds=1.5,
        ),
    ]

    verifier = Verifier(sandbox=sandbox_mock)
    report = verifier.verify_baseline(
        run_id="test-baseline-2",
        target_tests=["tests/test_specific.py"],
        subject_ids=["AC-1", "AC-2"],
    )

    assert report.passed is True
    assert len(report.checks) == 2
    assert report.checks[0].level == "BASELINE_REGRESSION"
    assert report.checks[0].phase == "baseline"
    assert report.checks[1].level == "BASELINE_TARGET"
    assert report.checks[1].phase == "baseline"
    assert report.checks[1].subject_ids == ["AC-1", "AC-2"]
    assert report.checks[1].direct is True


def test_verify_baseline_regression_fails() -> None:
    """Test baseline verification when regression tests fail."""
    sandbox_mock = MagicMock()
    # Setup mock to return failure for regression
    sandbox_mock.run.return_value = CommandResult(
        command="python -m pytest -q -p no:cacheprovider",
        exit_code=1,
        stdout="FAILED tests/test_example.py::test_example",
        stderr="AssertionError",
        duration_seconds=2.0,
    )

    verifier = Verifier(sandbox=sandbox_mock)
    report = verifier.verify_baseline(run_id="test-baseline-3")

    assert report.passed is False
    assert len(report.checks) == 1
    assert report.checks[0].level == "BASELINE_REGRESSION"
    assert report.checks[0].phase == "baseline"
    assert report.checks[0].passed is False
    assert report.failed_level == "BASELINE_REGRESSION"
    assert report.failure_type is not None


def test_verify_post_patch_collects_all_evidence() -> None:
    """Test post-patch verification collects complete evidence despite failures."""
    sandbox_mock = MagicMock()
    # Setup mock: ruff fails, target tests fail, regression passes
    sandbox_mock.run.side_effect = [
        CommandResult(
            command="ruff check --no-cache .",
            exit_code=1,
            stdout="",
            stderr="E501 line too long",
            duration_seconds=0.5,
        ),
        CommandResult(
            command="python -m pytest tests/test_failing.py -q -p no:cacheprovider",
            exit_code=1,
            stdout="FAILED tests/test_failing.py::test_example",
            stderr="AssertionError",
            duration_seconds=1.5,
        ),
        CommandResult(
            command="python -m pytest -q -p no:cacheprovider",
            exit_code=0,
            stdout="",
            stderr="",
            duration_seconds=2.0,
        ),
    ]

    verifier = Verifier(sandbox=sandbox_mock)
    report = verifier.verify_post_patch(
        run_id="test-post-patch-1",
        target_tests=["tests/test_failing.py"],
        subject_ids=["AC-1"],
        direct_subject_ids=["AC-1"],
    )

    assert report.passed is False
    assert len(report.checks) == 3  # All checks run
    assert report.checks[0].level == "LEVEL_1_LINT"
    assert report.checks[0].phase == "post_patch"
    assert report.checks[0].passed is False
    assert report.checks[1].level == "LEVEL_2_TARGET_TESTS"
    assert report.checks[1].phase == "post_patch"
    assert report.checks[1].passed is False
    assert report.checks[2].level == "LEVEL_3_REGRESSION"
    assert report.checks[2].phase == "post_patch"
    assert report.checks[2].passed is True

    # Verify all checks were called (complete evidence collection)
    assert sandbox_mock.run.call_count == 3
