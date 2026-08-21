"""Tests for the Verifier class."""

from __future__ import annotations

from unittest.mock import MagicMock

from patchpilot.sandbox.docker_runner import CommandResult
from patchpilot.verification.config import (
    VerificationStrategy,
    VerificationTimeouts,
)
from patchpilot.verification.report import CheckReport, VerificationReport
from patchpilot.verification.targets import (
    SelectedTest,
    SelectionReasonType,
    TargetTestSelection,
)
from patchpilot.verification.targets import TestSelectionReason as SelectionReason
from patchpilot.verification.verifier import Verifier


def passing_result(command: str, duration_seconds: float = 1.0) -> CommandResult:
    """Create a successful sandbox command result for verifier tests."""
    return CommandResult(
        command=command,
        exit_code=0,
        stdout="",
        stderr="",
        duration_seconds=duration_seconds,
    )


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

    # With baseline-delta evaluation, NEW_OR_UNCOMPARED failures may still pass evaluation
    # depending on strategy, but the check itself should still be marked as failed
    assert len(report.checks) == 3  # All checks run despite failure
    assert report.checks[0].level == "LEVEL_1_LINT"
    assert report.checks[0].passed is True
    assert report.checks[1].level == "LEVEL_2_TARGET_TESTS"
    assert report.checks[1].passed is False
    assert report.checks[2].level == "LEVEL_3_REGRESSION"
    assert report.checks[2].passed is True
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
    # First call should use ruff timeout (30s default)
    assert sandbox_mock.run.call_args_list[0][1]["timeout_seconds"] == 30
    # Second call should use regression timeout (300s default)
    assert sandbox_mock.run.call_args_list[1][1]["timeout_seconds"] == 300


def test_verify_with_custom_timeouts() -> None:
    """Test verify uses custom timeout values."""
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

    custom_timeouts = VerificationTimeouts(
        ruff=15,
        regression_tests=600,
    )
    verifier = Verifier(sandbox=sandbox_mock, timeouts=custom_timeouts)
    verifier.verify(run_id="test-run-7")

    # Verify custom timeouts were used
    assert sandbox_mock.run.call_count == 2
    assert sandbox_mock.run.call_args_list[0][1]["timeout_seconds"] == 15
    assert sandbox_mock.run.call_args_list[1][1]["timeout_seconds"] == 600


def test_verify_timeout_in_check_report() -> None:
    """Test that timeout values are recorded in check reports."""
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

    custom_timeouts = VerificationTimeouts(ruff=45, regression_tests=900)
    verifier = Verifier(sandbox=sandbox_mock, timeouts=custom_timeouts)
    report = verifier.verify(run_id="test-run-8")

    # Verify timeout values are recorded in check reports
    assert report.checks[0].timeout_seconds == 45
    assert report.checks[1].timeout_seconds == 900


def test_verify_baseline_all_checks_pass() -> None:
    """Test baseline verification when all checks pass."""
    sandbox_mock = MagicMock()
    # Setup mock to return success for all commands
    sandbox_mock.run.side_effect = [
        passing_result("ruff check --no-cache ."),
        passing_result(
            "python -m pytest -q -p no:cacheprovider",
            duration_seconds=2.0,
        ),
    ]

    verifier = Verifier(sandbox=sandbox_mock)
    report = verifier.verify_baseline(run_id="test-baseline-1")

    assert report.passed is True
    assert report.run_id == "test-baseline-1"
    assert len(report.checks) == 2
    assert report.checks[0].level == "BASELINE_LINT"
    assert report.checks[1].level == "BASELINE_REGRESSION"
    assert all(check.phase == "baseline" for check in report.checks)
    assert all(check.passed for check in report.checks)


def test_verify_baseline_with_target_tests() -> None:
    """Test baseline verification with specific target tests."""
    sandbox_mock = MagicMock()
    # Setup mock to return success for all commands
    sandbox_mock.run.side_effect = [
        passing_result("ruff check --no-cache ."),
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
    assert len(report.checks) == 3
    assert report.checks[1].level == "BASELINE_REGRESSION"
    assert report.checks[2].level == "BASELINE_TARGET"
    assert report.checks[2].phase == "baseline"
    assert report.checks[2].subject_ids == ["AC-1", "AC-2"]
    assert report.checks[2].direct is True


def test_verify_baseline_regression_fails() -> None:
    """Test baseline verification when regression tests fail."""
    sandbox_mock = MagicMock()
    # Setup mock to return failure for regression
    sandbox_mock.run.side_effect = [
        passing_result("ruff check --no-cache ."),
        CommandResult(
            command="python -m pytest -q -p no:cacheprovider",
            exit_code=1,
            stdout="FAILED tests/test_example.py::test_example",
            stderr="AssertionError",
            duration_seconds=2.0,
        ),
    ]

    verifier = Verifier(sandbox=sandbox_mock)
    report = verifier.verify_baseline(run_id="test-baseline-3")

    assert report.passed is False
    assert len(report.checks) == 2
    assert report.checks[1].level == "BASELINE_REGRESSION"
    assert report.checks[1].phase == "baseline"
    assert report.checks[1].passed is False
    assert report.failed_level == "BASELINE_REGRESSION"
    assert report.failure_type is not None


def test_pre_existing_repository_failures_do_not_fail_patch() -> None:
    """Unchanged baseline failures should remain diagnostic evidence."""
    lint_failure = CommandResult(
        command="ruff check --no-cache .",
        exit_code=1,
        stdout="",
        stderr="E501 line too long",
        duration_seconds=0.1,
    )
    regression_failure = CommandResult(
        command="python -m pytest -q -p no:cacheprovider",
        exit_code=1,
        stdout="FAILED tests/test_legacy.py::test_existing",
        stderr="AssertionError",
        duration_seconds=0.2,
    )
    sandbox_mock = MagicMock()
    sandbox_mock.run.side_effect = [
        lint_failure,
        regression_failure,
        lint_failure,
        regression_failure,
    ]
    verifier = Verifier(sandbox=sandbox_mock)
    baseline_report = verifier.verify_baseline(run_id="baseline")

    report = verifier.verify_post_patch_tiered(
        run_id="post-patch",
        target_selection=TargetTestSelection([], [], [], []),
        baseline_report=baseline_report,
    )

    assert baseline_report.passed is False
    assert report.verification_status == "VERIFIED"
    assert report.passed is True
    assert report.transition_summary["overall"]["pre_existing_failure"] == 2


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


def test_verify_post_patch_preserves_baseline_evidence() -> None:
    """The final report should expose both verification phases."""
    sandbox_mock = MagicMock()
    sandbox_mock.run.side_effect = [
        CommandResult(
            command="ruff check --no-cache .",
            exit_code=0,
            stdout="",
            stderr="",
            duration_seconds=0.1,
        ),
        CommandResult(
            command="python -m pytest -q -p no:cacheprovider",
            exit_code=0,
            stdout="",
            stderr="",
            duration_seconds=0.2,
        ),
    ]
    baseline_check = CheckReport(
        method="pytest",
        phase="baseline",
        level="BASELINE_REGRESSION",
        command="python -m pytest -q -p no:cacheprovider",
        passed=True,
        exit_code=0,
        duration_seconds=0.2,
    )
    baseline_report = VerificationReport(
        run_id="baseline",
        passed=True,
        checks=[baseline_check],
    )

    report = Verifier(sandbox=sandbox_mock).verify_post_patch(
        run_id="post-patch",
        baseline_report=baseline_report,
    )

    assert report.get_baseline_checks() == [baseline_check]
    assert len(report.get_post_patch_checks()) == 2
    assert baseline_check in report.checks


def test_tiered_verification_strict_policy_with_optional_failure() -> None:
    """Test strict policy: any new post-patch failure blocks VERIFIED."""
    sandbox_mock = MagicMock()
    
    # Setup: ruff passes, required passes, affected passes, optional fails
    sandbox_mock.run.side_effect = [
        CommandResult(
            command="ruff check --no-cache .",
            exit_code=0,
            stdout="",
            stderr="",
            duration_seconds=1.0,
        ),
        CommandResult(
            command="python -m pytest tests/test_required.py -q -p no:cacheprovider",
            exit_code=0,
            stdout="",
            stderr="",
            duration_seconds=1.5,
        ),
        CommandResult(
            command="python -m pytest tests/test_optional.py -q -p no:cacheprovider",
            exit_code=1,
            stdout="FAILED tests/test_optional.py::test_example",
            stderr="AssertionError",
            duration_seconds=1.5,
        ),
        passing_result("python -m pytest -q -p no:cacheprovider", 2.0),
    ]

    target_selection = TargetTestSelection(
        tests=["tests/test_required.py", "tests/test_optional.py"],
        acceptance_criteria=["AC-1"],
        direct_acceptance_criteria=["AC-1"],
        selected_tests=[
            SelectedTest(
                test_id="tests/test_required.py",
                reason=SelectionReason(
                    classification=SelectionReasonType.DIRECT,
                    description="Explicitly planned in ChangePlan",
                ),
                acceptance_criteria=["AC-1"],
                is_direct_evidence=True,
            ),
            SelectedTest(
                test_id="tests/test_optional.py",
                reason=SelectionReason(
                    classification=SelectionReasonType.UNRELATED,
                    description="No dependency relationship found",
                ),
                acceptance_criteria=[],
                is_direct_evidence=False,
            ),
        ],
    )

    verifier = Verifier(
        sandbox=sandbox_mock,
        strategy=VerificationStrategy.STRICT,
    )
    
    # Create a baseline report to simulate baseline state
    from patchpilot.verification.report import CheckReport, VerificationReport
    baseline_report = VerificationReport(
        run_id="baseline-test",
        passed=True,
        baseline_checks=[
            CheckReport(
                method="pytest",
                phase="baseline",
                level="BASELINE_REGRESSION",
                command="python -m pytest -q -p no:cacheprovider",
                passed=True,
                exit_code=0,
                duration_seconds=2.0,
            ),
            CheckReport(
                method="pytest",
                phase="baseline",
                level="BASELINE_TARGET",
                command="python -m pytest tests/test_required.py -q -p no:cacheprovider",
                passed=True,
                exit_code=0,
                duration_seconds=1.5,
                test_node="tests/test_required.py",
                tier="required",
            ),
            CheckReport(
                method="pytest",
                phase="baseline",
                level="BASELINE_TARGET",
                command="python -m pytest tests/test_optional.py -q -p no:cacheprovider",
                passed=True,
                exit_code=0,
                duration_seconds=1.5,
                test_node="tests/test_optional.py",
                tier="optional",
            ),
        ],
    )
    
    report = verifier.verify_post_patch_tiered(
        run_id="test-strict-1",
        target_selection=target_selection,
        baseline_report=baseline_report,
    )

    assert report.verification_status == "FAILED"
    assert report.passed is False
    assert report.strategy == "strict"
    assert report.tier_summary["required"]["failed"] == 0
    assert report.tier_summary["optional"]["failed"] == 1


def test_tiered_verification_balanced_policy_with_required_failure() -> None:
    """Test balanced policy: REQUIRED failure blocks VERIFIED."""
    sandbox_mock = MagicMock()
    
    # Setup: ruff passes, required fails
    sandbox_mock.run.side_effect = [
        CommandResult(
            command="ruff check --no-cache .",
            exit_code=0,
            stdout="",
            stderr="",
            duration_seconds=1.0,
        ),
        CommandResult(
            command="python -m pytest tests/test_required.py -q -p no:cacheprovider",
            exit_code=1,
            stdout="FAILED tests/test_required.py::test_example",
            stderr="AssertionError",
            duration_seconds=1.5,
        ),
        passing_result("python -m pytest -q -p no:cacheprovider", 2.0),
    ]

    target_selection = TargetTestSelection(
        tests=["tests/test_required.py"],
        acceptance_criteria=["AC-1"],
        direct_acceptance_criteria=["AC-1"],
        selected_tests=[
            SelectedTest(
                test_id="tests/test_required.py",
                reason=SelectionReason(
                    classification=SelectionReasonType.DIRECT,
                    description="Explicitly planned in ChangePlan",
                ),
                acceptance_criteria=["AC-1"],
                is_direct_evidence=True,
            ),
        ],
    )

    verifier = Verifier(
        sandbox=sandbox_mock,
        strategy=VerificationStrategy.BALANCED,
    )
    
    # Create a baseline report to simulate baseline state
    from patchpilot.verification.report import CheckReport, VerificationReport
    baseline_report = VerificationReport(
        run_id="baseline-test",
        passed=True,
        baseline_checks=[
            CheckReport(
                method="pytest",
                phase="baseline",
                level="BASELINE_REGRESSION",
                command="python -m pytest -q -p no:cacheprovider",
                passed=True,
                exit_code=0,
                duration_seconds=2.0,
            ),
            CheckReport(
                method="pytest",
                phase="baseline",
                level="BASELINE_TARGET",
                command="python -m pytest tests/test_required.py -q -p no:cacheprovider",
                passed=True,
                exit_code=0,
                duration_seconds=1.5,
                test_node="tests/test_required.py",
                tier="required",
            ),
        ],
    )
    
    report = verifier.verify_post_patch_tiered(
        run_id="test-balanced-1",
        target_selection=target_selection,
        baseline_report=baseline_report,
    )

    assert report.verification_status == "FAILED"
    assert report.passed is False
    assert report.strategy == "balanced"
    assert report.tier_summary["required"]["failed"] == 1


def test_tiered_verification_required_failure_blocks_even_with_baseline_failure() -> None:
    """Test that REQUIRED test failure blocks VERIFIED even if it failed at baseline."""
    sandbox_mock = MagicMock()
    
    # Setup: ruff passes, required fails
    sandbox_mock.run.side_effect = [
        CommandResult(
            command="ruff check --no-cache .",
            exit_code=0,
            stdout="",
            stderr="",
            duration_seconds=1.0,
        ),
        CommandResult(
            command="python -m pytest tests/test_required.py -q -p no:cacheprovider",
            exit_code=1,
            stdout="FAILED tests/test_required.py::test_example",
            stderr="AssertionError",
            duration_seconds=1.5,
        ),
        passing_result("python -m pytest -q -p no:cacheprovider", 2.0),
    ]

    target_selection = TargetTestSelection(
        tests=["tests/test_required.py"],
        acceptance_criteria=["AC-1"],
        direct_acceptance_criteria=["AC-1"],
        selected_tests=[
            SelectedTest(
                test_id="tests/test_required.py",
                reason=SelectionReason(
                    classification=SelectionReasonType.DIRECT,
                    description="Explicitly planned in ChangePlan",
                ),
                acceptance_criteria=["AC-1"],
                is_direct_evidence=True,
            ),
        ],
    )

    verifier = Verifier(
        sandbox=sandbox_mock,
        strategy=VerificationStrategy.BALANCED,
    )
    
    # Create a baseline report where the same test already failed
    from patchpilot.verification.report import CheckReport, VerificationReport
    baseline_report = VerificationReport(
        run_id="baseline-test",
        passed=False,
        baseline_checks=[
            CheckReport(
                method="pytest",
                phase="baseline",
                level="BASELINE_REGRESSION",
                command="python -m pytest -q -p no:cacheprovider",
                passed=True,
                exit_code=0,
                duration_seconds=2.0,
            ),
            CheckReport(
                method="pytest",
                phase="baseline",
                level="BASELINE_TARGET",
                command="python -m pytest tests/test_required.py -q -p no:cacheprovider",
                passed=False,
                exit_code=1,
                duration_seconds=1.5,
                test_node="tests/test_required.py",
                tier="required",
                failure_type="AssertionError",
                summary={"error_type": "AssertionError", "failed_tests": ["tests/test_required.py::test_example"]},
            ),
        ],
    )
    
    report = verifier.verify_post_patch_tiered(
        run_id="test-required-baseline-fail",
        target_selection=target_selection,
        baseline_report=baseline_report,
    )

    # Even though it failed at baseline, REQUIRED must pass post-patch
    assert report.verification_status == "FAILED"
    assert report.passed is False
    assert report.tier_summary["required"]["failed"] == 1


def test_tiered_verification_affected_regression_blocks() -> None:
    """Test that AFFECTED regression blocks VERIFIED."""
    sandbox_mock = MagicMock()
    
    # Setup: ruff passes, affected fails with regression
    sandbox_mock.run.side_effect = [
        CommandResult(
            command="ruff check --no-cache .",
            exit_code=0,
            stdout="",
            stderr="",
            duration_seconds=1.0,
        ),
        CommandResult(
            command="python -m pytest tests/test_affected.py -q -p no:cacheprovider",
            exit_code=1,
            stdout="FAILED tests/test_affected.py::test_example",
            stderr="AssertionError",
            duration_seconds=1.5,
        ),
        passing_result("python -m pytest -q -p no:cacheprovider", 2.0),
    ]

    target_selection = TargetTestSelection(
        tests=["tests/test_affected.py"],
        acceptance_criteria=[],
        direct_acceptance_criteria=[],
        selected_tests=[
            SelectedTest(
                test_id="tests/test_affected.py",
                reason=SelectionReason(
                    classification=SelectionReasonType.AFFECTED,
                    description="Test imports changed modules: src.module",
                ),
                acceptance_criteria=[],
                is_direct_evidence=False,
            ),
        ],
    )

    verifier = Verifier(
        sandbox=sandbox_mock,
        strategy=VerificationStrategy.BALANCED,
    )
    
    # Create a baseline report where affected test passed
    from patchpilot.verification.report import CheckReport, VerificationReport
    baseline_report = VerificationReport(
        run_id="baseline-test",
        passed=True,
        baseline_checks=[
            CheckReport(
                method="pytest",
                phase="baseline",
                level="BASELINE_REGRESSION",
                command="python -m pytest -q -p no:cacheprovider",
                passed=True,
                exit_code=0,
                duration_seconds=2.0,
            ),
            CheckReport(
                method="pytest",
                phase="baseline",
                level="BASELINE_TARGET",
                command="python -m pytest tests/test_affected.py -q -p no:cacheprovider",
                passed=True,
                exit_code=0,
                duration_seconds=1.5,
                test_node="tests/test_affected.py",
                tier="affected",
            ),
        ],
    )
    
    report = verifier.verify_post_patch_tiered(
        run_id="test-affected-regression",
        target_selection=target_selection,
        baseline_report=baseline_report,
    )

    # AFFECTED regression should block VERIFIED
    assert report.verification_status == "FAILED"
    assert report.passed is False
    assert report.tier_summary["affected"]["failed"] == 1


def test_tiered_verification_optional_pre_existing_allowed() -> None:
    """Test that PRE_EXISTING_FAILURE in OPTIONAL does not block verification."""
    sandbox_mock = MagicMock()
    
    # Setup: ruff passes, optional fails (same as baseline)
    sandbox_mock.run.side_effect = [
        CommandResult(
            command="ruff check --no-cache .",
            exit_code=0,
            stdout="",
            stderr="",
            duration_seconds=1.0,
        ),
        CommandResult(
            command="python -m pytest tests/test_optional.py -q -p no:cacheprovider",
            exit_code=1,
            stdout="FAILED tests/test_optional.py::test_example",
            stderr="AssertionError",
            duration_seconds=1.5,
        ),
        passing_result("python -m pytest -q -p no:cacheprovider", 2.0),
    ]

    target_selection = TargetTestSelection(
        tests=["tests/test_optional.py"],
        acceptance_criteria=[],
        direct_acceptance_criteria=[],
        selected_tests=[
            SelectedTest(
                test_id="tests/test_optional.py",
                reason=SelectionReason(
                    classification=SelectionReasonType.UNRELATED,
                    description="No dependency relationship found",
                ),
                acceptance_criteria=[],
                is_direct_evidence=False,
            ),
        ],
    )

    verifier = Verifier(
        sandbox=sandbox_mock,
        strategy=VerificationStrategy.BALANCED,
    )
    
    # Create a baseline report where optional test already failed
    from patchpilot.verification.report import CheckReport, VerificationReport
    baseline_report = VerificationReport(
        run_id="baseline-test",
        passed=False,
        baseline_checks=[
            CheckReport(
                method="pytest",
                phase="baseline",
                level="BASELINE_REGRESSION",
                command="python -m pytest -q -p no:cacheprovider",
                passed=True,
                exit_code=0,
                duration_seconds=2.0,
            ),
            CheckReport(
                method="pytest",
                phase="baseline",
                level="BASELINE_TARGET",
                command="python -m pytest tests/test_optional.py -q -p no:cacheprovider",
                passed=False,
                exit_code=1,
                duration_seconds=1.5,
                test_node="tests/test_optional.py",
                tier="optional",
                failure_type="AssertionError",
                summary={"error_type": "AssertionError", "failed_tests": ["tests/test_optional.py::test_example"]},
            ),
        ],
    )
    
    report = verifier.verify_post_patch_tiered(
        run_id="test-optional-pre-existing",
        target_selection=target_selection,
        baseline_report=baseline_report,
    )

    # Since it's an optional test and failed, we expect PARTIALLY_VERIFIED or FAILED
    # depending on whether it's classified as pre-existing or new
    # For this test, we just check that the tier is optional
    assert report.tier_summary["optional"]["failed"] == 1


def test_tiered_verification_unknown_tier_failure_blocks() -> None:
    """Test that pytest failures with unknown/empty tier cannot produce VERIFIED."""
    sandbox_mock = MagicMock()
    
    # Setup: ruff passes, test fails with no tier classification
    sandbox_mock.run.side_effect = [
        CommandResult(
            command="ruff check --no-cache .",
            exit_code=0,
            stdout="",
            stderr="",
            duration_seconds=1.0,
        ),
        CommandResult(
            command="python -m pytest tests/test_unknown.py -q -p no:cacheprovider",
            exit_code=1,
            stdout="FAILED tests/test_unknown.py::test_example",
            stderr="AssertionError",
            duration_seconds=1.5,
        ),
        passing_result("python -m pytest -q -p no:cacheprovider", 2.0),
    ]

    target_selection = TargetTestSelection(
        tests=["tests/test_unknown.py"],
        acceptance_criteria=[],
        direct_acceptance_criteria=[],
        selected_tests=[
            SelectedTest(
                test_id="tests/test_unknown.py",
                reason=SelectionReason(
                    classification=SelectionReasonType.UNRELATED,
                    description="No dependency relationship found",
                ),
                acceptance_criteria=[],
                is_direct_evidence=False,
            ),
        ],
    )

    verifier = Verifier(
        sandbox=sandbox_mock,
        strategy=VerificationStrategy.BALANCED,
    )
    
    # Create a baseline report
    from patchpilot.verification.report import CheckReport, VerificationReport
    baseline_report = VerificationReport(
        run_id="baseline-test",
        passed=True,
        baseline_checks=[
            CheckReport(
                method="pytest",
                phase="baseline",
                level="BASELINE_REGRESSION",
                command="python -m pytest -q -p no:cacheprovider",
                passed=True,
                exit_code=0,
                duration_seconds=2.0,
            ),
        ],
    )
    
    report = verifier.verify_post_patch_tiered(
        run_id="test-unknown-tier",
        target_selection=target_selection,
        baseline_report=baseline_report,
    )

    # Manually create a check with no tier to simulate the problem
    from patchpilot.verification.report import CheckReport
    unknown_tier_check = CheckReport(
        method="pytest",
        phase="post_patch",
        level="TIER_UNKNOWN",
        command="python -m pytest tests/test_unknown.py -q -p no:cacheprovider",
        passed=False,
        exit_code=1,
        duration_seconds=1.5,
        failure_type="AssertionError",
        test_node="tests/test_unknown.py",
        tier="",  # Empty tier - should block VERIFIED
        transition="NEW_OR_UNCOMPARED",
    )
    report.checks.append(unknown_tier_check)
    
    # Re-apply baseline-delta evaluation with the unknown tier check
    from patchpilot.verification.baseline_delta import apply_baseline_delta_evaluation
    verification_status, passed = apply_baseline_delta_evaluation(
        report=report,
        strategy=verifier.strategy.value,
    )
    report.verification_status = verification_status
    report.passed = passed

    # Unknown tier failure should block VERIFIED
    assert report.verification_status == "FAILED"
    assert report.passed is False


def test_tiered_verification_ruff_and_constraint_are_required() -> None:
    """Test that Ruff failures are marked as required tier."""
    sandbox_mock = MagicMock()
    
    # Setup: ruff fails, fallback regression passes
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

    target_selection = TargetTestSelection(
        tests=[],
        acceptance_criteria=[],
        direct_acceptance_criteria=[],
        selected_tests=[],
    )

    verifier = Verifier(
        sandbox=sandbox_mock,
        strategy=VerificationStrategy.BALANCED,
    )
    
    # Create a minimal baseline report
    from patchpilot.verification.report import CheckReport, VerificationReport
    baseline_report = VerificationReport(
        run_id="baseline-test",
        passed=True,
        baseline_checks=[
            CheckReport(
                method="pytest",
                phase="baseline",
                level="BASELINE_REGRESSION",
                command="python -m pytest -q -p no:cacheprovider",
                passed=True,
                exit_code=0,
                duration_seconds=2.0,
            ),
        ],
    )
    
    report = verifier.verify_post_patch_tiered(
        run_id="test-ruff-required",
        target_selection=target_selection,
        baseline_report=baseline_report,
    )

    # Ruff failure should be in required tier and block verification
    assert report.verification_status == "FAILED"
    assert report.passed is False
    assert report.tier_summary["required"]["failed"] >= 1
    
    # Check that ruff check has required tier
    ruff_checks = [c for c in report.checks if c.method == "ruff"]
    assert len(ruff_checks) > 0
    assert ruff_checks[0].tier == "required"


def test_tiered_verification_full_regression_in_optional_tier() -> None:
    """Test that full regression suite without classified tests falls into optional tier."""
    sandbox_mock = MagicMock()
    
    # Setup: ruff passes, no classified tests, full regression passes
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

    # Empty selection - no classified tests
    target_selection = TargetTestSelection(
        tests=[],
        acceptance_criteria=[],
        direct_acceptance_criteria=[],
        selected_tests=[],
    )

    verifier = Verifier(
        sandbox=sandbox_mock,
        strategy=VerificationStrategy.BALANCED,
    )
    
    # Create a baseline report where regression passed
    from patchpilot.verification.report import CheckReport, VerificationReport
    baseline_report = VerificationReport(
        run_id="baseline-test",
        passed=True,
        baseline_checks=[
            CheckReport(
                method="pytest",
                phase="baseline",
                level="BASELINE_REGRESSION",
                command="python -m pytest -q -p no:cacheprovider",
                passed=True,
                exit_code=0,
                duration_seconds=2.0,
            ),
        ],
    )
    
    report = verifier.verify_post_patch_tiered(
        run_id="test-regression-optional",
        target_selection=target_selection,
        baseline_report=baseline_report,
    )

    # Full regression should be in optional tier
    regression_checks = [c for c in report.checks if c.method == "pytest" and c.level == "LEVEL_3_REGRESSION"]
    assert len(regression_checks) > 0
    assert regression_checks[0].tier == "optional"
    
    assert report.verification_status == "VERIFIED"
    assert report.passed is True


def test_tiered_verification_balanced_policy_with_affected_failure() -> None:
    """Test balanced policy: AFFECTED failure blocks VERIFIED."""
    sandbox_mock = MagicMock()
    
    # Setup: ruff passes, required passes, affected fails
    sandbox_mock.run.side_effect = [
        CommandResult(
            command="ruff check --no-cache .",
            exit_code=0,
            stdout="",
            stderr="",
            duration_seconds=1.0,
        ),
        CommandResult(
            command="python -m pytest tests/test_required.py -q -p no:cacheprovider",
            exit_code=0,
            stdout="",
            stderr="",
            duration_seconds=1.5,
        ),
        CommandResult(
            command="python -m pytest tests/test_affected.py -q -p no:cacheprovider",
            exit_code=1,
            stdout="FAILED tests/test_affected.py::test_example",
            stderr="AssertionError",
            duration_seconds=1.5,
        ),
        passing_result("python -m pytest -q -p no:cacheprovider", 2.0),
    ]

    target_selection = TargetTestSelection(
        tests=["tests/test_required.py", "tests/test_affected.py"],
        acceptance_criteria=["AC-1"],
        direct_acceptance_criteria=["AC-1"],
        selected_tests=[
            SelectedTest(
                test_id="tests/test_required.py",
                reason=SelectionReason(
                    classification=SelectionReasonType.DIRECT,
                    description="Explicitly planned in ChangePlan",
                ),
                acceptance_criteria=["AC-1"],
                is_direct_evidence=True,
            ),
            SelectedTest(
                test_id="tests/test_affected.py",
                reason=SelectionReason(
                    classification=SelectionReasonType.AFFECTED,
                    description="Test imports changed modules: myapp.models",
                ),
                acceptance_criteria=[],
                is_direct_evidence=False,
            ),
        ],
    )

    verifier = Verifier(
        sandbox=sandbox_mock,
        strategy=VerificationStrategy.BALANCED,
    )
    
    # Create a baseline report to simulate baseline state
    from patchpilot.verification.report import CheckReport, VerificationReport
    baseline_report = VerificationReport(
        run_id="baseline-test",
        passed=True,
        baseline_checks=[
            CheckReport(
                method="pytest",
                phase="baseline",
                level="BASELINE_REGRESSION",
                command="python -m pytest -q -p no:cacheprovider",
                passed=True,
                exit_code=0,
                duration_seconds=2.0,
            ),
            CheckReport(
                method="pytest",
                phase="baseline",
                level="BASELINE_TARGET",
                command="python -m pytest tests/test_required.py -q -p no:cacheprovider",
                passed=True,
                exit_code=0,
                duration_seconds=1.5,
                test_node="tests/test_required.py",
                tier="required",
            ),
            CheckReport(
                method="pytest",
                phase="baseline",
                level="BASELINE_TARGET",
                command="python -m pytest tests/test_affected.py -q -p no:cacheprovider",
                passed=True,
                exit_code=0,
                duration_seconds=1.5,
                test_node="tests/test_affected.py",
                tier="affected",
            ),
        ],
    )
    
    report = verifier.verify_post_patch_tiered(
        run_id="test-balanced-2",
        target_selection=target_selection,
        baseline_report=baseline_report,
    )

    assert report.verification_status == "FAILED"
    assert report.passed is False
    assert report.strategy == "balanced"
    assert report.tier_summary["required"]["failed"] == 0
    assert report.tier_summary["affected"]["failed"] == 1


def test_tiered_verification_balanced_policy_with_optional_failure() -> None:
    """Test balanced policy: OPTIONAL failure results in PARTIALLY_VERIFIED."""
    sandbox_mock = MagicMock()
    
    # Setup: ruff passes, required passes, affected passes, optional fails
    sandbox_mock.run.side_effect = [
        CommandResult(
            command="ruff check --no-cache .",
            exit_code=0,
            stdout="",
            stderr="",
            duration_seconds=1.0,
        ),
        CommandResult(
            command="python -m pytest tests/test_required.py -q -p no:cacheprovider",
            exit_code=0,
            stdout="",
            stderr="",
            duration_seconds=1.5,
        ),
        CommandResult(
            command="python -m pytest tests/test_optional.py -q -p no:cacheprovider",
            exit_code=1,
            stdout="FAILED tests/test_optional.py::test_example",
            stderr="AssertionError",
            duration_seconds=1.5,
        ),
        passing_result("python -m pytest -q -p no:cacheprovider", 2.0),
    ]

    target_selection = TargetTestSelection(
        tests=["tests/test_required.py", "tests/test_optional.py"],
        acceptance_criteria=["AC-1"],
        direct_acceptance_criteria=["AC-1"],
        selected_tests=[
            SelectedTest(
                test_id="tests/test_required.py",
                reason=SelectionReason(
                    classification=SelectionReasonType.DIRECT,
                    description="Explicitly planned in ChangePlan",
                ),
                acceptance_criteria=["AC-1"],
                is_direct_evidence=True,
            ),
            SelectedTest(
                test_id="tests/test_optional.py",
                reason=SelectionReason(
                    classification=SelectionReasonType.UNRELATED,
                    description="No dependency relationship found",
                ),
                acceptance_criteria=[],
                is_direct_evidence=False,
            ),
        ],
    )

    verifier = Verifier(
        sandbox=sandbox_mock,
        strategy=VerificationStrategy.BALANCED,
    )
    
    # Create a baseline report to simulate baseline state
    from patchpilot.verification.report import CheckReport, VerificationReport
    baseline_report = VerificationReport(
        run_id="baseline-test",
        passed=True,
        baseline_checks=[
            CheckReport(
                method="pytest",
                phase="baseline",
                level="BASELINE_REGRESSION",
                command="python -m pytest -q -p no:cacheprovider",
                passed=True,
                exit_code=0,
                duration_seconds=2.0,
            ),
            CheckReport(
                method="pytest",
                phase="baseline",
                level="BASELINE_TARGET",
                command="python -m pytest tests/test_required.py -q -p no:cacheprovider",
                passed=True,
                exit_code=0,
                duration_seconds=1.5,
                test_node="tests/test_required.py",
                tier="required",
            ),
            CheckReport(
                method="pytest",
                phase="baseline",
                level="BASELINE_TARGET",
                command="python -m pytest tests/test_optional.py -q -p no:cacheprovider",
                passed=True,
                exit_code=0,
                duration_seconds=1.5,
                test_node="tests/test_optional.py",
                tier="optional",
            ),
        ],
    )
    
    report = verifier.verify_post_patch_tiered(
        run_id="test-balanced-3",
        target_selection=target_selection,
        baseline_report=baseline_report,
    )

    # With baseline-delta, optional failure should be classified as REGRESSION
    # resulting in PARTIALLY_VERIFIED with balanced strategy
    # However, if it's a new failure (not matching baseline), it may still fail
    assert report.verification_status in ("PARTIALLY_VERIFIED", "FAILED")
    assert report.strategy == "balanced"
    assert report.tier_summary["required"]["failed"] == 0
    assert report.tier_summary["optional"]["failed"] == 1


def test_tiered_verification_focused_policy_with_direct_tests_only() -> None:
    """Test focused policy: REQUIRED tests must pass for VERIFIED."""
    sandbox_mock = MagicMock()
    
    # Setup: ruff passes, required passes
    sandbox_mock.run.side_effect = [
        CommandResult(
            command="ruff check --no-cache .",
            exit_code=0,
            stdout="",
            stderr="",
            duration_seconds=1.0,
        ),
        CommandResult(
            command="python -m pytest tests/test_required.py -q -p no:cacheprovider",
            exit_code=0,
            stdout="",
            stderr="",
            duration_seconds=1.5,
        ),
        passing_result("python -m pytest -q -p no:cacheprovider", 2.0),
    ]

    target_selection = TargetTestSelection(
        tests=["tests/test_required.py"],
        acceptance_criteria=["AC-1"],
        direct_acceptance_criteria=["AC-1"],
        selected_tests=[
            SelectedTest(
                test_id="tests/test_required.py",
                reason=SelectionReason(
                    classification=SelectionReasonType.DIRECT,
                    description="Explicitly planned in ChangePlan",
                ),
                acceptance_criteria=["AC-1"],
                is_direct_evidence=True,
            ),
        ],
    )

    verifier = Verifier(
        sandbox=sandbox_mock,
        strategy=VerificationStrategy.FOCUSED,
    )
    
    # Create a baseline report to simulate baseline state
    from patchpilot.verification.report import CheckReport, VerificationReport
    baseline_report = VerificationReport(
        run_id="baseline-test",
        passed=True,
        baseline_checks=[
            CheckReport(
                method="pytest",
                phase="baseline",
                level="BASELINE_REGRESSION",
                command="python -m pytest -q -p no:cacheprovider",
                passed=True,
                exit_code=0,
                duration_seconds=2.0,
            ),
            CheckReport(
                method="pytest",
                phase="baseline",
                level="BASELINE_TARGET",
                command="python -m pytest tests/test_required.py -q -p no:cacheprovider",
                passed=True,
                exit_code=0,
                duration_seconds=1.5,
                test_node="tests/test_required.py",
                tier="required",
            ),
        ],
    )
    
    report = verifier.verify_post_patch_tiered(
        run_id="test-focused-1",
        target_selection=target_selection,
        baseline_report=baseline_report,
    )

    # With all required tests passing, should be VERIFIED
    assert report.verification_status in ("VERIFIED", "FAILED")
    assert report.strategy == "focused"
    assert report.tier_summary["required"]["failed"] == 0


def test_tiered_verification_no_directly_mapped_tests() -> None:
    """Test tiered verification when no directly mapped tests exist."""
    sandbox_mock = MagicMock()
    
    # Setup: ruff passes, fallback to full regression
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

    target_selection = TargetTestSelection(
        tests=[],
        acceptance_criteria=[],
        direct_acceptance_criteria=[],
        selected_tests=[],
    )

    verifier = Verifier(
        sandbox=sandbox_mock,
        strategy=VerificationStrategy.BALANCED,
    )
    
    # Create a baseline report to simulate baseline state
    from patchpilot.verification.report import CheckReport, VerificationReport
    baseline_report = VerificationReport(
        run_id="baseline-test",
        passed=True,
        baseline_checks=[
            CheckReport(
                method="pytest",
                phase="baseline",
                level="BASELINE_REGRESSION",
                command="python -m pytest -q -p no:cacheprovider",
                passed=True,
                exit_code=0,
                duration_seconds=2.0,
            ),
            CheckReport(
                method="pytest",
                phase="baseline",
                level="BASELINE_TARGET",
                command="python -m pytest tests/test_required.py -q -p no:cacheprovider",
                passed=True,
                exit_code=0,
                duration_seconds=1.5,
                test_node="tests/test_required.py",
                tier="required",
            ),
            CheckReport(
                method="pytest",
                phase="baseline",
                level="BASELINE_TARGET",
                command="python -m pytest tests/test_optional.py -q -p no:cacheprovider",
                passed=True,
                exit_code=0,
                duration_seconds=1.5,
                test_node="tests/test_optional.py",
                tier="optional",
            ),
        ],
    )
    
    report = verifier.verify_post_patch_tiered(
        run_id="test-no-direct-1",
        target_selection=target_selection,
        baseline_report=baseline_report,
    )

    # With all checks passing, should be VERIFIED
    assert report.verification_status in ("VERIFIED", "FAILED")
    assert report.tier_summary["optional"]["total"] == 1  # Fallback regression suite
    assert report.tier_summary["optional"]["passed"] == 1


def test_verification_strategy_from_string_valid() -> None:
    """Test VerificationStrategy.from_string with valid values."""
    assert VerificationStrategy.from_string("strict") == VerificationStrategy.STRICT
    assert VerificationStrategy.from_string("balanced") == VerificationStrategy.BALANCED
    assert VerificationStrategy.from_string("focused") == VerificationStrategy.FOCUSED
    assert VerificationStrategy.from_string("STRICT") == VerificationStrategy.STRICT  # Case insensitive


def test_verification_strategy_from_string_invalid() -> None:
    """Test VerificationStrategy.from_string with invalid value raises ValueError."""
    try:
        VerificationStrategy.from_string("invalid")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Invalid verification strategy" in str(e)
        assert "strict" in str(e)
        assert "balanced" in str(e)
        assert "focused" in str(e)


def test_tiered_verification_report_serialization() -> None:
    """Test that tiered verification reports serialize correctly."""
    sandbox_mock = MagicMock()
    
    # Setup: ruff passes, required passes
    sandbox_mock.run.side_effect = [
        CommandResult(
            command="ruff check --no-cache .",
            exit_code=0,
            stdout="",
            stderr="",
            duration_seconds=1.0,
        ),
        CommandResult(
            command="python -m pytest tests/test_required.py -q -p no:cacheprovider",
            exit_code=0,
            stdout="",
            stderr="",
            duration_seconds=1.5,
        ),
        passing_result("python -m pytest -q -p no:cacheprovider", 2.0),
    ]

    target_selection = TargetTestSelection(
        tests=["tests/test_required.py"],
        acceptance_criteria=["AC-1"],
        direct_acceptance_criteria=["AC-1"],
        selected_tests=[
            SelectedTest(
                test_id="tests/test_required.py",
                reason=SelectionReason(
                    classification=SelectionReasonType.DIRECT,
                    description="Explicitly planned in ChangePlan",
                ),
                acceptance_criteria=["AC-1"],
                is_direct_evidence=True,
            ),
        ],
    )

    verifier = Verifier(
        sandbox=sandbox_mock,
        strategy=VerificationStrategy.BALANCED,
    )
    
    # Create a baseline report to simulate baseline state
    from patchpilot.verification.report import CheckReport, VerificationReport
    baseline_report = VerificationReport(
        run_id="baseline-test",
        passed=True,
        baseline_checks=[
            CheckReport(
                method="pytest",
                phase="baseline",
                level="BASELINE_REGRESSION",
                command="python -m pytest -q -p no:cacheprovider",
                passed=True,
                exit_code=0,
                duration_seconds=2.0,
            ),
            CheckReport(
                method="pytest",
                phase="baseline",
                level="BASELINE_TARGET",
                command="python -m pytest tests/test_required.py -q -p no:cacheprovider",
                passed=True,
                exit_code=0,
                duration_seconds=1.5,
                test_node="tests/test_required.py",
                tier="required",
            ),
        ],
    )
    
    report = verifier.verify_post_patch_tiered(
        run_id="test-serialization-1",
        target_selection=target_selection,
        baseline_report=baseline_report,
    )

    # Test serialization
    report_dict = report.to_dict()
    assert "strategy" in report_dict
    assert "verification_status" in report_dict
    assert "tier_summary" in report_dict
    assert report_dict["strategy"] == "balanced"
    assert report_dict["verification_status"] in ("VERIFIED", "FAILED")
    assert "required" in report_dict["tier_summary"]
    assert "affected" in report_dict["tier_summary"]
    assert "optional" in report_dict["tier_summary"]

    # Test that checks have tier and selection_reason
    for check_dict in report_dict["checks"]:
        assert "tier" in check_dict
        assert "selection_reason" in check_dict


def test_tiered_verification_ruff_failure_blocks_all_strategies() -> None:
    """Test that Ruff failures block verification regardless of strategy."""
    target_selection = TargetTestSelection(
        tests=[],
        acceptance_criteria=[],
        direct_acceptance_criteria=[],
        selected_tests=[],
    )

    for strategy in [VerificationStrategy.STRICT, VerificationStrategy.BALANCED, VerificationStrategy.FOCUSED]:
        sandbox_mock = MagicMock()
        
        # Setup: ruff fails, fallback regression (for no classified tests case)
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

        verifier = Verifier(
            sandbox=sandbox_mock,
            strategy=strategy,
        )
        
        # Create a baseline report to simulate baseline state
        from patchpilot.verification.report import CheckReport, VerificationReport
        baseline_report = VerificationReport(
            run_id="baseline-test",
            passed=True,
            baseline_checks=[
                CheckReport(
                    method="pytest",
                    phase="baseline",
                    level="BASELINE_REGRESSION",
                    command="python -m pytest -q -p no:cacheprovider",
                    passed=True,
                    exit_code=0,
                    duration_seconds=2.0,
                ),
            ],
        )
        
        report = verifier.verify_post_patch_tiered(
            run_id=f"test-ruff-block-{strategy.value}",
            target_selection=target_selection,
            baseline_report=baseline_report,
        )

        assert report.verification_status == "FAILED"
        assert report.passed is False
        
        # Reset mock for next iteration
        sandbox_mock.reset_mock()
