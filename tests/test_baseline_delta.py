"""Tests for baseline-delta comparison and transition classification."""

from __future__ import annotations

from patchpilot.evidence.schema import CheckTransition
from patchpilot.sandbox.docker_runner import CommandResult
from patchpilot.verification.baseline_delta import (
    apply_baseline_delta_evaluation,
    classify_transition,
    compute_check_identity,
    compute_failure_fingerprint,
    compute_transition_summary,
    match_baseline_checks,
)
from patchpilot.verification.config import VerificationStrategy
from patchpilot.verification.error_parser import parse_failure
from patchpilot.verification.report import CheckReport, VerificationReport


def test_compute_check_identity_pytest():
    """Test check identity computation for pytest checks."""
    check = CheckReport(
        method="pytest",
        phase="post_patch",
        level="LEVEL_2_TARGET_TESTS",
        command="python -m pytest tests/test_example.py -q -p no:cacheprovider",
        passed=True,
        exit_code=0,
        duration_seconds=1.0,
        test_node="tests/test_example.py::test_func",
        tier="required",
    )

    identity = compute_check_identity(check)
    assert isinstance(identity, str)
    assert len(identity) == 16  # SHA256 truncated to 16 chars

    # Same check should produce same identity
    identity2 = compute_check_identity(check)
    assert identity == identity2


def test_compute_check_identity_ruff():
    """Test check identity computation for ruff checks."""
    check = CheckReport(
        method="ruff",
        phase="post_patch",
        level="LEVEL_1_LINT",
        command="ruff check --no-cache .",
        passed=True,
        exit_code=0,
        duration_seconds=0.5,
        tier="",
    )

    identity = compute_check_identity(check)
    assert isinstance(identity, str)
    assert len(identity) == 16


def test_compute_failure_fingerprint():
    """Test failure fingerprint computation for pytest returns dict mapping."""
    check = CheckReport(
        method="pytest",
        phase="post_patch",
        level="LEVEL_2_TARGET_TESTS",
        command="python -m pytest tests/test_example.py -q -p no:cacheprovider",
        passed=False,
        exit_code=1,
        duration_seconds=1.0,
        failure_type="AssertionError",
        summary={
            "error_type": "AssertionError",
            "failed_tests": ["tests/test_example.py::test_func"],
            "relevant_output": "AssertionError: Expected 5 but got 3",
        },
        test_node="tests/test_example.py::test_func",
    )

    fingerprint = compute_failure_fingerprint(check)
    # For pytest, fingerprint is now a dict mapping test nodes to error fingerprints
    assert isinstance(fingerprint, dict)
    assert "tests/test_example.py::test_func" in fingerprint
    assert len(fingerprint["tests/test_example.py::test_func"]) == 16

    # Same failure should produce same fingerprint
    fingerprint2 = compute_failure_fingerprint(check)
    assert fingerprint == fingerprint2


def test_compute_failure_fingerprint_passed_check():
    """Test that passed checks return empty fingerprint."""
    check = CheckReport(
        method="pytest",
        phase="post_patch",
        level="LEVEL_2_TARGET_TESTS",
        command="python -m pytest tests/test_example.py -q -p no:cacheprovider",
        passed=True,
        exit_code=0,
        duration_seconds=1.0,
        test_node="tests/test_example.py::test_func",
    )

    fingerprint = compute_failure_fingerprint(check)
    assert fingerprint == ""


def test_compute_failure_fingerprint_ruff_returns_string():
    """Test that non-pytest methods still return string fingerprints for backward compatibility."""
    check = CheckReport(
        method="ruff",
        phase="post_patch",
        level="LEVEL_1_LINT",
        command="ruff check --no-cache .",
        passed=False,
        exit_code=1,
        duration_seconds=0.5,
        failure_type="LintError",
        summary={
            "error_type": "LintError",
            "failed_tests": [],
            "relevant_output": "E501 line too long",
        },
    )

    fingerprint = compute_failure_fingerprint(check)
    # For non-pytest methods, fingerprint should still be a string
    assert isinstance(fingerprint, str)
    assert len(fingerprint) == 16


def test_classify_transition_resolved():
    """Test FAIL → PASS transition classification."""
    baseline_check = CheckReport(
        method="pytest",
        phase="baseline",
        level="BASELINE_TARGET",
        command="python -m pytest tests/test_example.py -q -p no:cacheprovider",
        passed=False,
        exit_code=1,
        duration_seconds=1.0,
        failure_type="AssertionError",
        summary={"error_type": "AssertionError"},
        test_node="tests/test_example.py::test_func",
    )

    post_patch_check = CheckReport(
        method="pytest",
        phase="post_patch",
        level="LEVEL_2_TARGET_TESTS",
        command="python -m pytest tests/test_example.py -q -p no:cacheprovider",
        passed=True,
        exit_code=0,
        duration_seconds=1.0,
        test_node="tests/test_example.py::test_func",
    )

    transition, baseline_id = classify_transition(baseline_check, post_patch_check)
    assert transition == CheckTransition.RESOLVED.value
    assert baseline_id == baseline_check.verification_id


def test_classify_transition_preserved():
    """Test PASS → PASS transition classification."""
    baseline_check = CheckReport(
        method="pytest",
        phase="baseline",
        level="BASELINE_TARGET",
        command="python -m pytest tests/test_example.py -q -p no:cacheprovider",
        passed=True,
        exit_code=0,
        duration_seconds=1.0,
        test_node="tests/test_example.py::test_func",
    )

    post_patch_check = CheckReport(
        method="pytest",
        phase="post_patch",
        level="LEVEL_2_TARGET_TESTS",
        command="python -m pytest tests/test_example.py -q -p no:cacheprovider",
        passed=True,
        exit_code=0,
        duration_seconds=1.0,
        test_node="tests/test_example.py::test_func",
    )

    transition, baseline_id = classify_transition(baseline_check, post_patch_check)
    assert transition == CheckTransition.PRESERVED.value
    assert baseline_id == baseline_check.verification_id


def test_classify_transition_regression():
    """Test PASS → FAIL transition classification."""
    baseline_check = CheckReport(
        method="pytest",
        phase="baseline",
        level="BASELINE_TARGET",
        command="python -m pytest tests/test_example.py -q -p no:cacheprovider",
        passed=True,
        exit_code=0,
        duration_seconds=1.0,
        test_node="tests/test_example.py::test_func",
    )

    post_patch_check = CheckReport(
        method="pytest",
        phase="post_patch",
        level="LEVEL_2_TARGET_TESTS",
        command="python -m pytest tests/test_example.py -q -p no:cacheprovider",
        passed=False,
        exit_code=1,
        duration_seconds=1.0,
        failure_type="AssertionError",
        summary={"error_type": "AssertionError"},
        test_node="tests/test_example.py::test_func",
    )

    transition, baseline_id = classify_transition(baseline_check, post_patch_check)
    assert transition == CheckTransition.REGRESSION.value
    assert baseline_id == baseline_check.verification_id


def test_classify_transition_pre_existing_failure():
    """Test FAIL → FAIL with equivalent fingerprint using pytest per-test comparison."""
    baseline_check = CheckReport(
        method="pytest",
        phase="baseline",
        level="BASELINE_REGRESSION",
        command="python -m pytest -q -p no:cacheprovider",
        passed=False,
        exit_code=1,
        duration_seconds=1.0,
        failure_type="AssertionError",
        summary={"error_type": "AssertionError", "failed_tests": ["tests/test_example.py::test_func"], "relevant_output": "FAILED tests/test_example.py::test_func\nAssertionError: error"},
        test_node="",
    )

    post_patch_check = CheckReport(
        method="pytest",
        phase="post_patch",
        level="LEVEL_3_REGRESSION",
        command="python -m pytest -q -p no:cacheprovider",
        passed=False,
        exit_code=1,
        duration_seconds=1.0,
        failure_type="AssertionError",
        summary={"error_type": "AssertionError", "failed_tests": ["tests/test_example.py::test_func"], "relevant_output": "FAILED tests/test_example.py::test_func\nAssertionError: error"},
        test_node="",
    )

    transition, baseline_id = classify_transition(baseline_check, post_patch_check)
    assert transition == CheckTransition.PRE_EXISTING_FAILURE.value
    assert baseline_id == baseline_check.verification_id


def test_classify_transition_worsened():
    """Test FAIL → FAIL with different error fingerprint for same test."""
    baseline_check = CheckReport(
        method="pytest",
        phase="baseline",
        level="BASELINE_REGRESSION",
        command="python -m pytest -q -p no:cacheprovider",
        passed=False,
        exit_code=1,
        duration_seconds=1.0,
        failure_type="AssertionError",
        summary={"error_type": "AssertionError", "failed_tests": ["tests/test_example.py::test_func"], "relevant_output": "FAILED tests/test_example.py::test_func\nAssertionError: error a"},
        test_node="",
    )

    post_patch_check = CheckReport(
        method="pytest",
        phase="post_patch",
        level="LEVEL_3_REGRESSION",
        command="python -m pytest -q -p no:cacheprovider",
        passed=False,
        exit_code=1,
        duration_seconds=1.0,
        failure_type="TypeError",  # Different error type for same test
        summary={"error_type": "TypeError", "failed_tests": ["tests/test_example.py::test_func"], "relevant_output": "FAILED tests/test_example.py::test_func\nTypeError: error b"},
        test_node="",
    )

    transition, baseline_id = classify_transition(baseline_check, post_patch_check)
    assert transition == CheckTransition.WORSENED.value
    assert baseline_id == baseline_check.verification_id


def test_classify_transition_ruff_pre_existing_failure():
    """Test FAIL → FAIL with equivalent fingerprint for ruff (string comparison)."""
    baseline_check = CheckReport(
        method="ruff",
        phase="baseline",
        level="BASELINE_REGRESSION",
        command="ruff check --no-cache .",
        passed=False,
        exit_code=1,
        duration_seconds=1.0,
        failure_type="LintError",
        summary={"error_type": "LintError", "failed_tests": [], "relevant_output": "E501 line too long"},
    )

    post_patch_check = CheckReport(
        method="ruff",
        phase="post_patch",
        level="LEVEL_1_LINT",
        command="ruff check --no-cache .",
        passed=False,
        exit_code=1,
        duration_seconds=1.0,
        failure_type="LintError",
        summary={"error_type": "LintError", "failed_tests": [], "relevant_output": "E501 line too long"},
    )

    transition, baseline_id = classify_transition(baseline_check, post_patch_check)
    assert transition == CheckTransition.PRE_EXISTING_FAILURE.value
    assert baseline_id == baseline_check.verification_id


def test_classify_transition_new_or_uncompared():
    """Test transition when no baseline match exists."""
    post_patch_check = CheckReport(
        method="pytest",
        phase="post_patch",
        level="LEVEL_2_TARGET_TESTS",
        command="python -m pytest tests/test_example.py -q -p no:cacheprovider",
        passed=True,
        exit_code=0,
        duration_seconds=1.0,
        test_node="tests/test_example.py::test_func",
    )

    transition, baseline_id = classify_transition(None, post_patch_check)
    assert transition == CheckTransition.NEW_OR_UNCOMPARED.value
    assert baseline_id == ""


def test_match_baseline_checks():
    """Test matching post-patch checks to baseline checks."""
    baseline_checks = [
        CheckReport(
            method="pytest",
            phase="baseline",
            level="BASELINE_TARGET",
            command="python -m pytest tests/test_example.py -q -p no:cacheprovider",
            passed=False,
            exit_code=1,
            duration_seconds=1.0,
            test_node="tests/test_example.py::test_func",
        ),
        CheckReport(
            method="ruff",
            phase="baseline",
            level="BASELINE_REGRESSION",
            command="ruff check --no-cache .",
            passed=True,
            exit_code=0,
            duration_seconds=0.5,
        ),
    ]

    post_patch_checks = [
        CheckReport(
            method="pytest",
            phase="post_patch",
            level="LEVEL_2_TARGET_TESTS",
            command="python -m pytest tests/test_example.py -q -p no:cacheprovider",
            passed=True,
            exit_code=0,
            duration_seconds=1.0,
            test_node="tests/test_example.py::test_func",
        ),
        CheckReport(
            method="ruff",
            phase="post_patch",
            level="LEVEL_1_LINT",
            command="ruff check --no-cache .",
            passed=True,
            exit_code=0,
            duration_seconds=0.5,
        ),
    ]

    matches = match_baseline_checks(post_patch_checks, baseline_checks)
    assert len(matches) == 2
    assert post_patch_checks[0].verification_id in matches
    assert post_patch_checks[1].verification_id in matches


def test_compute_transition_summary():
    """Test transition summary computation."""
    post_patch_checks = [
        CheckReport(
            method="pytest",
            phase="post_patch",
            level="LEVEL_2_TARGET_TESTS",
            command="python -m pytest tests/test_required.py -q -p no:cacheprovider",
            passed=True,
            exit_code=0,
            duration_seconds=1.0,
            test_node="tests/test_required.py::test_func",
            tier="required",
            transition="RESOLVED",
        ),
        CheckReport(
            method="pytest",
            phase="post_patch",
            level="LEVEL_2_TARGET_TESTS",
            command="python -m pytest tests/test_affected.py -q -p no:cacheprovider",
            passed=False,
            exit_code=1,
            duration_seconds=1.0,
            test_node="tests/test_affected.py::test_func",
            tier="affected",
            transition="REGRESSION",
        ),
        CheckReport(
            method="pytest",
            phase="post_patch",
            level="LEVEL_3_REGRESSION",
            command="python -m pytest tests/test_optional.py -q -p no:cacheprovider",
            passed=True,
            exit_code=0,
            duration_seconds=1.0,
            test_node="tests/test_optional.py::test_func",
            tier="optional",
            transition="PRESERVED",
        ),
    ]

    summary = compute_transition_summary(post_patch_checks)

    assert "overall" in summary
    assert "by_tier" in summary
    assert summary["overall"]["resolved"] == 1
    assert summary["overall"]["regression"] == 1
    assert summary["overall"]["preserved"] == 1
    assert summary["by_tier"]["required"]["resolved"] == 1
    assert summary["by_tier"]["affected"]["regression"] == 1
    assert summary["by_tier"]["optional"]["preserved"] == 1


def test_apply_baseline_delta_evaluation_verified():
    """Test baseline-delta evaluation results in VERIFIED."""
    report = VerificationReport(
        run_id="test-run",
        passed=True,
        checks=[],
        transition_summary={
            "overall": {
                "resolved": 1,
                "preserved": 2,
                "regression": 0,
                "pre_existing_failure": 0,
                "worsened": 0,
                "new_or_uncompared": 0,
                "unverified": 0,
            },
            "by_tier": {
                "required": {
                    "resolved": 1,
                    "preserved": 0,
                    "regression": 0,
                    "pre_existing_failure": 0,
                    "worsened": 0,
                    "new_or_uncompared": 0,
                    "unverified": 0,
                },
                "affected": {
                    "resolved": 0,
                    "preserved": 1,
                    "regression": 0,
                    "pre_existing_failure": 0,
                    "worsened": 0,
                    "new_or_uncompared": 0,
                    "unverified": 0,
                },
                "optional": {
                    "resolved": 0,
                    "preserved": 1,
                    "regression": 0,
                    "pre_existing_failure": 0,
                    "worsened": 0,
                    "new_or_uncompared": 0,
                    "unverified": 0,
                },
            },
        },
    )

    status, passed = apply_baseline_delta_evaluation(report, VerificationStrategy.BALANCED.value)
    assert status == "VERIFIED"
    assert passed is True


def test_failed_specialized_check_blocks_verification() -> None:
    """A failed direct acceptance check must prevent VERIFIED."""
    check = CheckReport(
        method="acceptance_probe",
        phase="post_patch",
        level="SPECIALIZED_PROBE",
        command="probe:ac-1",
        passed=False,
        exit_code=1,
        duration_seconds=0.1,
        tier="required",
        subject_ids=["AC-1"],
        direct=True,
    )
    report = VerificationReport(
        run_id="specialized-failure",
        checks=[check],
        transition_summary=compute_transition_summary([check]),
    )

    status, passed = apply_baseline_delta_evaluation(report)

    assert status == "FAILED"
    assert passed is False


def test_apply_baseline_delta_evaluation_required_regression():
    """Test that REQUIRED regression blocks VERIFIED."""
    report = VerificationReport(
        run_id="test-run",
        passed=True,
        checks=[],
        transition_summary={
            "overall": {
                "resolved": 0,
                "preserved": 1,
                "regression": 1,
                "pre_existing_failure": 0,
                "worsened": 0,
                "new_or_uncompared": 0,
                "unverified": 0,
            },
            "by_tier": {
                "required": {
                    "resolved": 0,
                    "preserved": 0,
                    "regression": 1,
                    "pre_existing_failure": 0,
                    "worsened": 0,
                    "new_or_uncompared": 0,
                    "unverified": 0,
                },
                "affected": {
                    "resolved": 0,
                    "preserved": 1,
                    "regression": 0,
                    "pre_existing_failure": 0,
                    "worsened": 0,
                    "new_or_uncompared": 0,
                    "unverified": 0,
                },
                "optional": {
                    "resolved": 0,
                    "preserved": 0,
                    "regression": 0,
                    "pre_existing_failure": 0,
                    "worsened": 0,
                    "new_or_uncompared": 0,
                    "unverified": 0,
                },
            },
        },
    )

    status, passed = apply_baseline_delta_evaluation(report, VerificationStrategy.BALANCED.value)
    assert status == "FAILED"
    assert passed is False


def test_apply_baseline_delta_evaluation_affected_regression():
    """Test that AFFECTED regression blocks VERIFIED."""
    report = VerificationReport(
        run_id="test-run",
        passed=True,
        checks=[],
        transition_summary={
            "overall": {
                "resolved": 1,
                "preserved": 0,
                "regression": 1,
                "pre_existing_failure": 0,
                "worsened": 0,
                "new_or_uncompared": 0,
                "unverified": 0,
            },
            "by_tier": {
                "required": {
                    "resolved": 1,
                    "preserved": 0,
                    "regression": 0,
                    "pre_existing_failure": 0,
                    "worsened": 0,
                    "new_or_uncompared": 0,
                    "unverified": 0,
                },
                "affected": {
                    "resolved": 0,
                    "preserved": 0,
                    "regression": 1,
                    "pre_existing_failure": 0,
                    "worsened": 0,
                    "new_or_uncompared": 0,
                    "unverified": 0,
                },
                "optional": {
                    "resolved": 0,
                    "preserved": 0,
                    "regression": 0,
                    "pre_existing_failure": 0,
                    "worsened": 0,
                    "new_or_uncompared": 0,
                    "unverified": 0,
                },
            },
        },
    )

    status, passed = apply_baseline_delta_evaluation(report, VerificationStrategy.BALANCED.value)
    assert status == "FAILED"
    assert passed is False


def test_apply_baseline_delta_evaluation_pre_existing_failure_allowed():
    """Test that PRE_EXISTING_FAILURE in OPTIONAL does not block verification."""
    report = VerificationReport(
        run_id="test-run",
        passed=True,
        checks=[],
        transition_summary={
            "overall": {
                "resolved": 1,
                "preserved": 0,
                "regression": 0,
                "pre_existing_failure": 1,
                "worsened": 0,
                "new_or_uncompared": 0,
                "unverified": 0,
            },
            "by_tier": {
                "required": {
                    "resolved": 1,
                    "preserved": 0,
                    "regression": 0,
                    "pre_existing_failure": 0,
                    "worsened": 0,
                    "new_or_uncompared": 0,
                    "unverified": 0,
                },
                "affected": {
                    "resolved": 0,
                    "preserved": 0,
                    "regression": 0,
                    "pre_existing_failure": 0,
                    "worsened": 0,
                    "new_or_uncompared": 0,
                    "unverified": 0,
                },
                "optional": {
                    "resolved": 0,
                    "preserved": 0,
                    "regression": 0,
                    "pre_existing_failure": 1,
                    "worsened": 0,
                    "new_or_uncompared": 0,
                    "unverified": 0,
                },
            },
        },
    )

    status, passed = apply_baseline_delta_evaluation(report, VerificationStrategy.BALANCED.value)
    assert status == "VERIFIED"
    assert passed is True


def test_apply_baseline_delta_evaluation_unknown_tier_blocks():
    """Test that pytest failures with unknown tier block VERIFIED."""
    from patchpilot.verification.report import CheckReport

    report = VerificationReport(
        run_id="test-run",
        passed=True,
        checks=[
            CheckReport(
                method="pytest",
                phase="post_patch",
                level="LEVEL_2_TARGET_TESTS",
                command="python -m pytest tests/test_unknown.py -q -p no:cacheprovider",
                passed=False,
                exit_code=1,
                duration_seconds=1.0,
                failure_type="AssertionError",
                test_node="tests/test_unknown.py::test_func",
                tier="",  # Empty tier - should block
                transition="NEW_OR_UNCOMPARED",
            ),
        ],
        transition_summary={
            "overall": {
                "resolved": 0,
                "preserved": 0,
                "regression": 0,
                "pre_existing_failure": 0,
                "worsened": 0,
                "new_or_uncompared": 1,
                "unverified": 0,
            },
            "by_tier": {
                "required": {
                    "resolved": 0,
                    "preserved": 0,
                    "regression": 0,
                    "pre_existing_failure": 0,
                    "worsened": 0,
                    "new_or_uncompared": 0,
                    "unverified": 0,
                },
                "affected": {
                    "resolved": 0,
                    "preserved": 0,
                    "regression": 0,
                    "pre_existing_failure": 0,
                    "worsened": 0,
                    "new_or_uncompared": 0,
                    "unverified": 0,
                },
                "optional": {
                    "resolved": 0,
                    "preserved": 0,
                    "regression": 0,
                    "pre_existing_failure": 0,
                    "worsened": 0,
                    "new_or_uncompared": 0,
                    "unverified": 0,
                },
            },
        },
    )

    status, passed = apply_baseline_delta_evaluation(report, VerificationStrategy.BALANCED.value)
    assert status == "FAILED"
    assert passed is False


def test_apply_baseline_delta_evaluation_optional_regression_balanced():
    """Balanced verification should reject a deterministic regression."""
    report = VerificationReport(
        run_id="test-run",
        passed=True,
        checks=[],
        transition_summary={
            "overall": {
                "resolved": 1,
                "preserved": 0,
                "regression": 0,
                "pre_existing_failure": 0,
                "worsened": 0,
                "new_or_uncompared": 0,
                "unverified": 0,
            },
            "by_tier": {
                "required": {
                    "resolved": 1,
                    "preserved": 0,
                    "regression": 0,
                    "pre_existing_failure": 0,
                    "worsened": 0,
                    "new_or_uncompared": 0,
                    "unverified": 0,
                },
                "affected": {
                    "resolved": 0,
                    "preserved": 0,
                    "regression": 0,
                    "pre_existing_failure": 0,
                    "worsened": 0,
                    "new_or_uncompared": 0,
                    "unverified": 0,
                },
                "optional": {
                    "resolved": 0,
                    "preserved": 0,
                    "regression": 1,
                    "pre_existing_failure": 0,
                    "worsened": 0,
                    "new_or_uncompared": 0,
                    "unverified": 0,
                },
            },
        },
    )

    status, passed = apply_baseline_delta_evaluation(report, VerificationStrategy.BALANCED.value)
    assert status == "FAILED"
    assert passed is False


def test_apply_baseline_delta_evaluation_optional_regression_strict():
    """Test that OPTIONAL regression results in FAILED with STRICT strategy."""
    report = VerificationReport(
        run_id="test-run",
        passed=True,
        checks=[],
        transition_summary={
            "overall": {
                "resolved": 1,
                "preserved": 0,
                "regression": 0,
                "pre_existing_failure": 0,
                "worsened": 0,
                "new_or_uncompared": 0,
                "unverified": 0,
            },
            "by_tier": {
                "required": {
                    "resolved": 1,
                    "preserved": 0,
                    "regression": 0,
                    "pre_existing_failure": 0,
                    "worsened": 0,
                    "new_or_uncompared": 0,
                    "unverified": 0,
                },
                "affected": {
                    "resolved": 0,
                    "preserved": 0,
                    "regression": 0,
                    "pre_existing_failure": 0,
                    "worsened": 0,
                    "new_or_uncompared": 0,
                    "unverified": 0,
                },
                "optional": {
                    "resolved": 0,
                    "preserved": 0,
                    "regression": 1,
                    "pre_existing_failure": 0,
                    "worsened": 0,
                    "new_or_uncompared": 0,
                    "unverified": 0,
                },
            },
        },
    )

    status, passed = apply_baseline_delta_evaluation(report, VerificationStrategy.STRICT.value)
    assert status == "FAILED"
    assert passed is False


def test_apply_baseline_delta_evaluation_required_failure_blocks():
    """Test that REQUIRED failure (including pre-existing) blocks verification."""
    from patchpilot.verification.report import CheckReport

    report = VerificationReport(
        run_id="test-run",
        passed=True,
        checks=[
            CheckReport(
                method="pytest",
                phase="post_patch",
                level="LEVEL_2_TARGET_TESTS",
                command="python -m pytest tests/test_required.py -q -p no:cacheprovider",
                passed=False,
                exit_code=1,
                duration_seconds=1.0,
                failure_type="AssertionError",
                test_node="tests/test_required.py::test_func",
                tier="required",
                transition="pre_existing_failure",
            ),
        ],
        transition_summary={
            "overall": {
                "resolved": 0,
                "preserved": 0,
                "regression": 0,
                "pre_existing_failure": 1,
                "worsened": 0,
                "new_or_uncompared": 0,
                "unverified": 0,
            },
            "by_tier": {
                "required": {
                    "resolved": 0,
                    "preserved": 0,
                    "regression": 0,
                    "pre_existing_failure": 1,
                    "worsened": 0,
                    "new_or_uncompared": 0,
                    "unverified": 0,
                },
                "affected": {
                    "resolved": 0,
                    "preserved": 0,
                    "regression": 0,
                    "pre_existing_failure": 0,
                    "worsened": 0,
                    "new_or_uncompared": 0,
                    "unverified": 0,
                },
                "optional": {
                    "resolved": 0,
                    "preserved": 0,
                    "regression": 0,
                    "pre_existing_failure": 0,
                    "worsened": 0,
                    "new_or_uncompared": 0,
                    "unverified": 0,
                },
            },
        },
    )

    status, passed = apply_baseline_delta_evaluation(report, VerificationStrategy.BALANCED.value)
    assert status == "FAILED"
    assert passed is False


def test_apply_baseline_delta_evaluation_ruff_failure_blocks():
    """Test that Ruff failures always block verification."""
    report = VerificationReport(
        run_id="test-run",
        passed=False,
        checks=[
            CheckReport(
                method="ruff",
                phase="post_patch",
                level="LEVEL_1_LINT",
                command="ruff check --no-cache .",
                passed=False,
                exit_code=1,
                duration_seconds=0.5,
            ),
        ],
        transition_summary={
            "overall": {
                "resolved": 0,
                "preserved": 0,
                "regression": 0,
                "pre_existing_failure": 0,
                "worsened": 0,
                "new_or_uncompared": 0,
                "unverified": 0,
            },
            "by_tier": {
                "required": {
                    "resolved": 0,
                    "preserved": 0,
                    "regression": 0,
                    "pre_existing_failure": 0,
                    "worsened": 0,
                    "new_or_uncompared": 0,
                    "unverified": 0,
                },
                "affected": {
                    "resolved": 0,
                    "preserved": 0,
                    "regression": 0,
                    "pre_existing_failure": 0,
                    "worsened": 0,
                    "new_or_uncompared": 0,
                    "unverified": 0,
                },
                "optional": {
                    "resolved": 0,
                    "preserved": 0,
                    "regression": 0,
                    "pre_existing_failure": 0,
                    "worsened": 0,
                    "new_or_uncompared": 0,
                    "unverified": 0,
                },
            },
        },
    )

    status, passed = apply_baseline_delta_evaluation(report, VerificationStrategy.BALANCED.value)
    assert status == "FAILED"
    assert passed is False


def test_apply_baseline_delta_evaluation_no_report():
    """Test that None report results in FAILED."""
    status, passed = apply_baseline_delta_evaluation(None, VerificationStrategy.BALANCED.value)
    assert status == "FAILED"
    assert passed is False


def test_pytest_per_test_failure_mapping():
    """Test that pytest checks generate per-test failure mappings."""
    check = CheckReport(
        method="pytest",
        phase="post_patch",
        level="LEVEL_3_REGRESSION",
        command="python -m pytest -q -p no:cacheprovider",
        passed=False,
        exit_code=1,
        duration_seconds=2.0,
        failure_type="AssertionError",
        summary={
            "error_type": "AssertionError",
            "failed_tests": [
                "tests/test_a.py::test_a",
                "tests/test_b.py::test_b",
            ],
            "relevant_output": "FAILED tests/test_a.py::test_a\nAssertionError: error a\nFAILED tests/test_b.py::test_b\nValueError: error b",
        },
        test_node="",
    )

    fingerprint = compute_failure_fingerprint(check)
    assert isinstance(fingerprint, dict)
    assert len(fingerprint) == 2
    assert "tests/test_a.py::test_a" in fingerprint
    assert "tests/test_b.py::test_b" in fingerprint


def test_classify_pytest_transition_improved():
    """Test {a, b} → {b} results in IMPROVED transition."""
    baseline_check = CheckReport(
        method="pytest",
        phase="baseline",
        level="BASELINE_REGRESSION",
        command="python -m pytest -q -p no:cacheprovider",
        passed=False,
        exit_code=1,
        duration_seconds=2.0,
        failure_type="AssertionError",
        summary={
            "error_type": "AssertionError",
            "failed_tests": [
                "tests/test_a.py::test_a",
                "tests/test_b.py::test_b",
            ],
            "relevant_output": "FAILED tests/test_a.py::test_a\nAssertionError: error a\nFAILED tests/test_b.py::test_b\nAssertionError: error b",
        },
        test_node="",
    )

    post_patch_check = CheckReport(
        method="pytest",
        phase="post_patch",
        level="LEVEL_3_REGRESSION",
        command="python -m pytest -q -p no:cacheprovider",
        passed=False,
        exit_code=1,
        duration_seconds=2.0,
        failure_type="AssertionError",
        summary={
            "error_type": "AssertionError",
            "failed_tests": [
                "tests/test_b.py::test_b",
            ],
            "relevant_output": "FAILED tests/test_b.py::test_b\nAssertionError: error b",
        },
        test_node="",
    )

    transition, baseline_id = classify_transition(baseline_check, post_patch_check)
    assert transition == CheckTransition.IMPROVED.value
    assert baseline_id == baseline_check.verification_id


def test_classify_pytest_transition_pre_existing_failure():
    """Test {a, b} → {a, b} results in PRE_EXISTING_FAILURE."""
    baseline_check = CheckReport(
        method="pytest",
        phase="baseline",
        level="BASELINE_REGRESSION",
        command="python -m pytest -q -p no:cacheprovider",
        passed=False,
        exit_code=1,
        duration_seconds=2.0,
        failure_type="AssertionError",
        summary={
            "error_type": "AssertionError",
            "failed_tests": [
                "tests/test_a.py::test_a",
                "tests/test_b.py::test_b",
            ],
            "relevant_output": "FAILED tests/test_a.py::test_a\nAssertionError: error a\nFAILED tests/test_b.py::test_b\nAssertionError: error b",
        },
        test_node="",
    )

    post_patch_check = CheckReport(
        method="pytest",
        phase="post_patch",
        level="LEVEL_3_REGRESSION",
        command="python -m pytest -q -p no:cacheprovider",
        passed=False,
        exit_code=1,
        duration_seconds=2.0,
        failure_type="AssertionError",
        summary={
            "error_type": "AssertionError",
            "failed_tests": [
                "tests/test_a.py::test_a",
                "tests/test_b.py::test_b",
            ],
            "relevant_output": "FAILED tests/test_a.py::test_a\nAssertionError: error a\nFAILED tests/test_b.py::test_b\nAssertionError: error b",
        },
        test_node="",
    )

    transition, baseline_id = classify_transition(baseline_check, post_patch_check)
    assert transition == CheckTransition.PRE_EXISTING_FAILURE.value
    assert baseline_id == baseline_check.verification_id


def test_classify_pytest_transition_resolved_all():
    """Test {a, b} → {} results in RESOLVED."""
    baseline_check = CheckReport(
        method="pytest",
        phase="baseline",
        level="BASELINE_REGRESSION",
        command="python -m pytest -q -p no:cacheprovider",
        passed=False,
        exit_code=1,
        duration_seconds=2.0,
        failure_type="AssertionError",
        summary={
            "error_type": "AssertionError",
            "failed_tests": [
                "tests/test_a.py::test_a",
                "tests/test_b.py::test_b",
            ],
            "relevant_output": "FAILED tests/test_a.py::test_a\nAssertionError: error a\nFAILED tests/test_b.py::test_b\nAssertionError: error b",
        },
        test_node="",
    )

    post_patch_check = CheckReport(
        method="pytest",
        phase="post_patch",
        level="LEVEL_3_REGRESSION",
        command="python -m pytest -q -p no:cacheprovider",
        passed=True,
        exit_code=0,
        duration_seconds=2.0,
        test_node="",
    )

    transition, baseline_id = classify_transition(baseline_check, post_patch_check)
    assert transition == CheckTransition.RESOLVED.value
    assert baseline_id == baseline_check.verification_id


def test_classify_pytest_transition_regression_with_improvement():
    """Test {a, b} → {b, c} results in REGRESSION (new failure outweighs improvement)."""
    baseline_check = CheckReport(
        method="pytest",
        phase="baseline",
        level="BASELINE_REGRESSION",
        command="python -m pytest -q -p no:cacheprovider",
        passed=False,
        exit_code=1,
        duration_seconds=2.0,
        failure_type="AssertionError",
        summary={
            "error_type": "AssertionError",
            "failed_tests": [
                "tests/test_a.py::test_a",
                "tests/test_b.py::test_b",
            ],
            "relevant_output": "FAILED tests/test_a.py::test_a\nAssertionError: error a\nFAILED tests/test_b.py::test_b\nAssertionError: error b",
        },
        test_node="",
    )

    post_patch_check = CheckReport(
        method="pytest",
        phase="post_patch",
        level="LEVEL_3_REGRESSION",
        command="python -m pytest -q -p no:cacheprovider",
        passed=False,
        exit_code=1,
        duration_seconds=2.0,
        failure_type="AssertionError",
        summary={
            "error_type": "AssertionError",
            "failed_tests": [
                "tests/test_b.py::test_b",
                "tests/test_c.py::test_c",
            ],
            "relevant_output": "FAILED tests/test_b.py::test_b\nAssertionError: error b\nFAILED tests/test_c.py::test_c\nAssertionError: error c",
        },
        test_node="",
    )

    transition, baseline_id = classify_transition(baseline_check, post_patch_check)
    assert transition == CheckTransition.REGRESSION.value
    assert baseline_id == baseline_check.verification_id


def test_classify_pytest_transition_worsened_error_type():
    """Test {a} → {a} with error type change results in WORSENED."""
    baseline_check = CheckReport(
        method="pytest",
        phase="baseline",
        level="BASELINE_REGRESSION",
        command="python -m pytest -q -p no:cacheprovider",
        passed=False,
        exit_code=1,
        duration_seconds=2.0,
        failure_type="AssertionError",
        summary={
            "error_type": "AssertionError",
            "failed_tests": [
                "tests/test_a.py::test_a",
            ],
            "relevant_output": "FAILED tests/test_a.py::test_a\nAssertionError: error a",
        },
        test_node="",
    )

    post_patch_check = CheckReport(
        method="pytest",
        phase="post_patch",
        level="LEVEL_3_REGRESSION",
        command="python -m pytest -q -p no:cacheprovider",
        passed=False,
        exit_code=1,
        duration_seconds=2.0,
        failure_type="TypeError",  # Different error type
        summary={
            "error_type": "TypeError",
            "failed_tests": [
                "tests/test_a.py::test_a",
            ],
            "relevant_output": "FAILED tests/test_a.py::test_a\nTypeError: error a",
        },
        test_node="",
    )

    transition, baseline_id = classify_transition(baseline_check, post_patch_check)
    assert transition == CheckTransition.WORSENED.value
    assert baseline_id == baseline_check.verification_id


def test_classify_pytest_transition_timeout_unverified():
    """Test timeout results in UNVERIFIED."""
    baseline_check = CheckReport(
        method="pytest",
        phase="baseline",
        level="BASELINE_REGRESSION",
        command="python -m pytest -q -p no:cacheprovider",
        passed=False,
        exit_code=1,
        duration_seconds=2.0,
        failure_type="AssertionError",
        summary={
            "error_type": "AssertionError",
            "failed_tests": [
                "tests/test_a.py::test_a",
            ],
            "relevant_output": "FAILED tests/test_a.py::test_a\nAssertionError: error a",
        },
        test_node="",
    )

    post_patch_check = CheckReport(
        method="pytest",
        phase="post_patch",
        level="LEVEL_3_REGRESSION",
        command="python -m pytest -q -p no:cacheprovider",
        passed=False,
        exit_code=1,
        duration_seconds=2.0,
        failure_type="TIMEOUT",
        summary={
            "error_type": "Timeout",
            "failed_tests": [],
            "relevant_output": "Timeout",
        },
        test_node="",
    )

    transition, baseline_id = classify_transition(baseline_check, post_patch_check)
    assert transition == CheckTransition.UNVERIFIED.value
    assert baseline_id == baseline_check.verification_id


def test_transition_summary_includes_improved():
    """Test that transition summary includes improved counts."""
    post_patch_checks = [
        CheckReport(
            method="pytest",
            phase="post_patch",
            level="LEVEL_3_REGRESSION",
            command="python -m pytest -q -p no:cacheprovider",
            passed=False,
            exit_code=1,
            duration_seconds=2.0,
            test_node="",
            tier="optional",
            transition="IMPROVED",
        ),
    ]

    summary = compute_transition_summary(post_patch_checks)

    assert "overall" in summary
    assert "by_tier" in summary
    assert summary["overall"]["improved"] == 1
    assert summary["by_tier"]["optional"]["improved"] == 1


def test_improved_transition_non_blocking_optional():
    """Test that IMPROVED in OPTIONAL tier does not block VERIFIED."""
    report = VerificationReport(
        run_id="test-run",
        passed=True,
        checks=[],
        transition_summary={
            "overall": {
                "resolved": 0,
                "preserved": 0,
                "regression": 0,
                "pre_existing_failure": 0,
                "worsened": 0,
                "improved": 1,
                "new_or_uncompared": 0,
                "unverified": 0,
            },
            "by_tier": {
                "required": {
                    "resolved": 0,
                    "preserved": 0,
                    "regression": 0,
                    "pre_existing_failure": 0,
                    "worsened": 0,
                    "improved": 0,
                    "new_or_uncompared": 0,
                    "unverified": 0,
                },
                "affected": {
                    "resolved": 0,
                    "preserved": 0,
                    "regression": 0,
                    "pre_existing_failure": 0,
                    "worsened": 0,
                    "improved": 0,
                    "new_or_uncompared": 0,
                    "unverified": 0,
                },
                "optional": {
                    "resolved": 0,
                    "preserved": 0,
                    "regression": 0,
                    "pre_existing_failure": 0,
                    "worsened": 0,
                    "improved": 1,
                    "new_or_uncompared": 0,
                    "unverified": 0,
                },
            },
        },
    )

    status, passed = apply_baseline_delta_evaluation(report, VerificationStrategy.BALANCED.value)
    assert status == "VERIFIED"
    assert passed is True


def test_improved_transition_non_blocking_affected():
    """Test that IMPROVED in AFFECTED tier does not block VERIFIED."""
    report = VerificationReport(
        run_id="test-run",
        passed=True,
        checks=[],
        transition_summary={
            "overall": {
                "resolved": 0,
                "preserved": 0,
                "regression": 0,
                "pre_existing_failure": 0,
                "worsened": 0,
                "improved": 1,
                "new_or_uncompared": 0,
                "unverified": 0,
            },
            "by_tier": {
                "required": {
                    "resolved": 0,
                    "preserved": 0,
                    "regression": 0,
                    "pre_existing_failure": 0,
                    "worsened": 0,
                    "improved": 0,
                    "new_or_uncompared": 0,
                    "unverified": 0,
                },
                "affected": {
                    "resolved": 0,
                    "preserved": 0,
                    "regression": 0,
                    "pre_existing_failure": 0,
                    "worsened": 0,
                    "improved": 1,
                    "new_or_uncompared": 0,
                    "unverified": 0,
                },
                "optional": {
                    "resolved": 0,
                    "preserved": 0,
                    "regression": 0,
                    "pre_existing_failure": 0,
                    "worsened": 0,
                    "improved": 0,
                    "new_or_uncompared": 0,
                    "unverified": 0,
                },
            },
        },
    )

    status, passed = apply_baseline_delta_evaluation(report, VerificationStrategy.BALANCED.value)
    assert status == "VERIFIED"
    assert passed is True


def test_required_test_improved_still_blocks():
    """Test that REQUIRED tests with remaining failures still block verification."""
    from patchpilot.verification.report import CheckReport

    report = VerificationReport(
        run_id="test-run",
        passed=True,
        checks=[
            CheckReport(
                method="pytest",
                phase="post_patch",
                level="LEVEL_2_TARGET_TESTS",
                command="python -m pytest tests/test_required.py -q -p no:cacheprovider",
                passed=False,
                exit_code=1,
                duration_seconds=1.0,
                failure_type="AssertionError",
                test_node="tests/test_required.py::test_func",
                tier="required",
                transition="IMPROVED",  # Even with IMPROVED, required must pass
            ),
        ],
        transition_summary={
            "overall": {
                "resolved": 0,
                "preserved": 0,
                "regression": 0,
                "pre_existing_failure": 0,
                "worsened": 0,
                "improved": 1,
                "new_or_uncompared": 0,
                "unverified": 0,
            },
            "by_tier": {
                "required": {
                    "resolved": 0,
                    "preserved": 0,
                    "regression": 0,
                    "pre_existing_failure": 0,
                    "worsened": 0,
                    "improved": 1,
                    "new_or_uncompared": 0,
                    "unverified": 0,
                },
                "affected": {
                    "resolved": 0,
                    "preserved": 0,
                    "regression": 0,
                    "pre_existing_failure": 0,
                    "worsened": 0,
                    "improved": 0,
                    "new_or_uncompared": 0,
                    "unverified": 0,
                },
                "optional": {
                    "resolved": 0,
                    "preserved": 0,
                    "regression": 0,
                    "pre_existing_failure": 0,
                    "worsened": 0,
                    "improved": 0,
                    "new_or_uncompared": 0,
                    "unverified": 0,
                },
            },
        },
    )

    status, passed = apply_baseline_delta_evaluation(report, VerificationStrategy.BALANCED.value)
    assert status == "FAILED"
    assert passed is False


def test_empty_failure_maps_different_errors_return_worsened():
    """Test that empty failure maps with different errors return WORSENED."""
    baseline_check = CheckReport(
        method="pytest",
        phase="baseline",
        level="BASELINE_REGRESSION",
        command="python -m pytest -q -p no:cacheprovider",
        passed=False,
        exit_code=1,
        duration_seconds=2.0,
        failure_type="CollectionError",
        summary={"error_type": "CollectionError", "failed_tests": [], "relevant_output": "ERROR collecting tests", "full_output": "ERROR collecting tests"},
        test_node="",
    )

    post_patch_check = CheckReport(
        method="pytest",
        phase="post_patch",
        level="LEVEL_3_REGRESSION",
        command="python -m pytest -q -p no:cacheprovider",
        passed=False,
        exit_code=1,
        duration_seconds=2.0,
        failure_type="ImportError",
        summary={"error_type": "ImportError", "failed_tests": [], "relevant_output": "ERROR importing module", "full_output": "ERROR importing module"},
        test_node="",
    )

    transition, baseline_id = classify_transition(baseline_check, post_patch_check)
    # Different errors should return WORSENED
    assert transition == CheckTransition.WORSENED.value
    assert baseline_id == baseline_check.verification_id


def test_import_error_both_sides_empty_maps_same_error():
    """Test import errors with empty failure maps and same error on both sides."""
    baseline_check = CheckReport(
        method="pytest",
        phase="baseline",
        level="BASELINE_REGRESSION",
        command="python -m pytest -q -p no:cacheprovider",
        passed=False,
        exit_code=1,
        duration_seconds=2.0,
        failure_type="ImportError",
        summary={"error_type": "ImportError", "failed_tests": [], "relevant_output": "ImportError: No module named 'xyz'", "full_output": "ImportError: No module named 'xyz'"},
        test_node="",
    )

    post_patch_check = CheckReport(
        method="pytest",
        phase="post_patch",
        level="LEVEL_3_REGRESSION",
        command="python -m pytest -q -p no:cacheprovider",
        passed=False,
        exit_code=1,
        duration_seconds=2.0,
        failure_type="ImportError",
        summary={"error_type": "ImportError", "failed_tests": [], "relevant_output": "ImportError: No module named 'xyz'", "full_output": "ImportError: No module named 'xyz'"},
        test_node="",
    )

    transition, baseline_id = classify_transition(baseline_check, post_patch_check)
    # Same error with empty maps should return PRE_EXISTING_FAILURE
    assert transition == CheckTransition.PRE_EXISTING_FAILURE.value
    assert baseline_id == baseline_check.verification_id


def test_empty_failure_maps_no_fingerprint_returns_unverified():
    """Test that empty failure maps without reliable fingerprint return UNVERIFIED."""
    baseline_check = CheckReport(
        method="pytest",
        phase="baseline",
        level="BASELINE_REGRESSION",
        command="python -m pytest -q -p no:cacheprovider",
        passed=False,
        exit_code=1,
        duration_seconds=2.0,
        failure_type=None,  # No failure type
        summary={},  # Empty summary - no information
        test_node="",
    )

    post_patch_check = CheckReport(
        method="pytest",
        phase="post_patch",
        level="LEVEL_3_REGRESSION",
        command="python -m pytest -q -p no:cacheprovider",
        passed=False,
        exit_code=1,
        duration_seconds=2.0,
        failure_type=None,  # No failure type
        summary={},  # Empty summary - no information
        test_node="",
    )

    transition, baseline_id = classify_transition(baseline_check, post_patch_check)
    # No reliable fingerprint should return UNVERIFIED
    assert transition == CheckTransition.UNVERIFIED.value
    assert baseline_id == baseline_check.verification_id


def test_optional_new_or_uncompared_blocks_balanced():
    """Test that NEW_OR_UNCOMPARED in optional tier blocks VERIFIED for balanced strategy."""
    report = VerificationReport(
        run_id="test-run",
        passed=True,
        checks=[
            CheckReport(
                method="pytest",
                phase="post_patch",
                level="LEVEL_3_REGRESSION",
                command="python -m pytest -q -p no:cacheprovider",
                passed=False,
                exit_code=1,
                duration_seconds=2.0,
                failure_type="CollectionError",
                summary={"error_type": "CollectionError", "failed_tests": [], "relevant_output": "ERROR", "full_output": "ERROR"},
                test_node="",
                tier="optional",
                transition="NEW_OR_UNCOMPARED",
            ),
        ],
        transition_summary={
            "overall": {
                "resolved": 0,
                "preserved": 0,
                "regression": 0,
                "pre_existing_failure": 0,
                "worsened": 0,
                "improved": 0,
                "new_or_uncompared": 1,
                "unverified": 0,
            },
            "by_tier": {
                "required": {
                    "resolved": 0,
                    "preserved": 0,
                    "regression": 0,
                    "pre_existing_failure": 0,
                    "worsened": 0,
                    "improved": 0,
                    "new_or_uncompared": 0,
                    "unverified": 0,
                },
                "affected": {
                    "resolved": 0,
                    "preserved": 0,
                    "regression": 0,
                    "pre_existing_failure": 0,
                    "worsened": 0,
                    "improved": 0,
                    "new_or_uncompared": 0,
                    "unverified": 0,
                },
                "optional": {
                    "resolved": 0,
                    "preserved": 0,
                    "regression": 0,
                    "pre_existing_failure": 0,
                    "worsened": 0,
                    "improved": 0,
                    "new_or_uncompared": 1,
                    "unverified": 0,
                },
            },
        },
    )

    status, passed = apply_baseline_delta_evaluation(report, VerificationStrategy.BALANCED.value)
    assert status == "FAILED"
    assert passed is False


def test_optional_new_or_uncompared_partial_focused():
    """Test that NEW_OR_UNCOMPARED in optional tier returns PARTIALLY_VERIFIED for focused strategy."""
    report = VerificationReport(
        run_id="test-run",
        passed=True,
        checks=[
            CheckReport(
                method="pytest",
                phase="post_patch",
                level="LEVEL_3_REGRESSION",
                command="python -m pytest -q -p no:cacheprovider",
                passed=False,
                exit_code=1,
                duration_seconds=2.0,
                failure_type="CollectionError",
                summary={"error_type": "CollectionError", "failed_tests": [], "relevant_output": "ERROR", "full_output": "ERROR"},
                test_node="",
                tier="optional",
                transition="NEW_OR_UNCOMPARED",
            ),
        ],
        transition_summary={
            "overall": {
                "resolved": 0,
                "preserved": 0,
                "regression": 0,
                "pre_existing_failure": 0,
                "worsened": 0,
                "improved": 0,
                "new_or_uncompared": 1,
                "unverified": 0,
            },
            "by_tier": {
                "required": {
                    "resolved": 0,
                    "preserved": 0,
                    "regression": 0,
                    "pre_existing_failure": 0,
                    "worsened": 0,
                    "improved": 0,
                    "new_or_uncompared": 0,
                    "unverified": 0,
                },
                "affected": {
                    "resolved": 0,
                    "preserved": 0,
                    "regression": 0,
                    "pre_existing_failure": 0,
                    "worsened": 0,
                    "improved": 0,
                    "new_or_uncompared": 0,
                    "unverified": 0,
                },
                "optional": {
                    "resolved": 0,
                    "preserved": 0,
                    "regression": 0,
                    "pre_existing_failure": 0,
                    "worsened": 0,
                    "improved": 0,
                    "new_or_uncompared": 1,
                    "unverified": 0,
                },
            },
        },
    )

    status, passed = apply_baseline_delta_evaluation(report, VerificationStrategy.FOCUSED.value)
    assert status == "PARTIALLY_VERIFIED"
    assert passed is True


def test_assertion_error_content_change_detected_as_worsened():
    """Test that AssertionError content changes are detected as WORSENED."""
    baseline_check = CheckReport(
        method="pytest",
        phase="baseline",
        level="BASELINE_REGRESSION",
        command="python -m pytest -q -p no:cacheprovider",
        passed=False,
        exit_code=1,
        duration_seconds=2.0,
        failure_type="AssertionError",
        summary={
            "error_type": "AssertionError",
            "failed_tests": ["tests/test_example.py::test_func"],
            "relevant_output": "AssertionError: expected 5 but got 3",
            "full_output": "FAILED tests/test_example.py::test_func\nAssertionError: expected 5 but got 3",
        },
        test_node="",
    )

    post_patch_check = CheckReport(
        method="pytest",
        phase="post_patch",
        level="LEVEL_3_REGRESSION",
        command="python -m pytest -q -p no:cacheprovider",
        passed=False,
        exit_code=1,
        duration_seconds=2.0,
        failure_type="AssertionError",
        summary={
            "error_type": "AssertionError",
            "failed_tests": ["tests/test_example.py::test_func"],
            "relevant_output": "AssertionError: expected 10 but got 3",
            "full_output": "FAILED tests/test_example.py::test_func\nAssertionError: expected 10 but got 3",
        },
        test_node="",
    )

    transition, baseline_id = classify_transition(baseline_check, post_patch_check)
    # Same test, same error type but different assertion content should be WORSENED
    assert transition == CheckTransition.WORSENED.value
    assert baseline_id == baseline_check.verification_id


def test_value_error_content_change_detected_as_worsened():
    """Test that ValueError content changes are detected as WORSENED."""
    baseline_check = CheckReport(
        method="pytest",
        phase="baseline",
        level="BASELINE_REGRESSION",
        command="python -m pytest -q -p no:cacheprovider",
        passed=False,
        exit_code=1,
        duration_seconds=2.0,
        failure_type="ValueError",
        summary={
            "error_type": "ValueError",
            "failed_tests": ["tests/test_example.py::test_b"],
            "relevant_output": "ValueError: old-b",
            "full_output": "FAILED tests/test_example.py::test_b\nValueError: old-b",
        },
        test_node="",
    )

    post_patch_check = CheckReport(
        method="pytest",
        phase="post_patch",
        level="LEVEL_3_REGRESSION",
        command="python -m pytest -q -p no:cacheprovider",
        passed=False,
        exit_code=1,
        duration_seconds=2.0,
        failure_type="ValueError",
        summary={
            "error_type": "ValueError",
            "failed_tests": ["tests/test_example.py::test_b"],
            "relevant_output": "ValueError: changed-b",
            "full_output": "FAILED tests/test_example.py::test_b\nValueError: changed-b",
        },
        test_node="",
    )

    transition, baseline_id = classify_transition(baseline_check, post_patch_check)
    # Same test, same error type but different error content should be WORSENED
    assert transition == CheckTransition.WORSENED.value
    assert baseline_id == baseline_check.verification_id


def test_end_to_end_collection_error_pipeline():
    """Test complete pipeline from CommandResult to final verification status for collection errors."""

    # Simulate baseline collection error
    baseline_result = CommandResult(
        command="python -m pytest -q -p no:cacheprovider",
        exit_code=1,
        stdout="",
        stderr="ERROR collecting tests collection error",
        duration_seconds=2.0,
        timed_out=False,
    )

    # Simulate post-patch collection error (different error)
    post_patch_result = CommandResult(
        command="python -m pytest -q -p no:cacheprovider",
        exit_code=1,
        stdout="",
        stderr="ERROR importing module import error",
        duration_seconds=2.0,
        timed_out=False,
    )

    # Parse failures
    baseline_summary = parse_failure(baseline_result)
    post_patch_summary = parse_failure(post_patch_result)

    # Create check reports with different error types to ensure WORSENED classification
    baseline_check = CheckReport(
        method="pytest",
        phase="baseline",
        level="BASELINE_REGRESSION",
        command=baseline_result.command,
        passed=False,
        exit_code=baseline_result.exit_code,
        duration_seconds=baseline_result.duration_seconds,
        failure_type="CollectionError",  # Explicitly set different error type
        summary={
            "error_type": "CollectionError",
            "failed_tests": baseline_summary.failed_tests,
            "relevant_output": baseline_summary.relevant_output,
            "full_output": baseline_summary.full_output,
        },
        test_node="",
    )

    post_patch_check = CheckReport(
        method="pytest",
        phase="post_patch",
        level="LEVEL_3_REGRESSION",
        command=post_patch_result.command,
        passed=False,
        exit_code=post_patch_result.exit_code,
        duration_seconds=post_patch_result.duration_seconds,
        failure_type="ImportError",  # Explicitly set different error type
        summary={
            "error_type": "ImportError",
            "failed_tests": post_patch_summary.failed_tests,
            "relevant_output": post_patch_summary.relevant_output,
            "full_output": post_patch_summary.full_output,
        },
        test_node="",
        tier="optional",
    )

    # Classify transition
    transition, baseline_id = classify_transition(baseline_check, post_patch_check)
    post_patch_check.transition = transition
    post_patch_check.baseline_check_id = baseline_id

    # Different collection errors should be WORSENED
    assert transition == CheckTransition.WORSENED.value

    # Create verification report with manual transition summary to match actual transition
    report = VerificationReport(
        run_id="test-e2e",
        passed=True,
        checks=[post_patch_check],
        transition_summary={
            "overall": {
                "resolved": 0,
                "preserved": 0,
                "regression": 0,
                "pre_existing_failure": 0,
                "worsened": 1,
                "improved": 0,
                "new_or_uncompared": 0,
                "unverified": 0,
            },
            "by_tier": {
                "required": {
                    "resolved": 0,
                    "preserved": 0,
                    "regression": 0,
                    "pre_existing_failure": 0,
                    "worsened": 0,
                    "improved": 0,
                    "new_or_uncompared": 0,
                    "unverified": 0,
                },
                "affected": {
                    "resolved": 0,
                    "preserved": 0,
                    "regression": 0,
                    "pre_existing_failure": 0,
                    "worsened": 0,
                    "improved": 0,
                    "new_or_uncompared": 0,
                    "unverified": 0,
                },
                "optional": {
                    "resolved": 0,
                    "preserved": 0,
                    "regression": 0,
                    "pre_existing_failure": 0,
                    "worsened": 1,
                    "improved": 0,
                    "new_or_uncompared": 0,
                    "unverified": 0,
                },
            },
        },
    )

    # Apply baseline-delta evaluation
    status, passed = apply_baseline_delta_evaluation(report, VerificationStrategy.BALANCED.value)

    # WORSENED in optional tier should block VERIFIED for balanced strategy
    assert status == "FAILED"
    assert passed is False


def test_end_to_end_improved_pipeline():
    """Test complete pipeline for IMPROVED transition scenario."""

    # Simulate baseline with two failing tests
    baseline_result = CommandResult(
        command="python -m pytest -q -p no:cacheprovider",
        exit_code=1,
        stdout="FAILED tests/test_a.py::test_a\nFAILED tests/test_b.py::test_b",
        stderr="",
        duration_seconds=2.0,
        timed_out=False,
    )

    # Simulate post-patch with only one failing test (test_a resolved)
    post_patch_result = CommandResult(
        command="python -m pytest -q -p no:cacheprovider",
        exit_code=1,
        stdout="FAILED tests/test_b.py::test_b",
        stderr="",
        duration_seconds=2.0,
        timed_out=False,
    )

    # Parse failures
    baseline_summary = parse_failure(baseline_result)
    post_patch_summary = parse_failure(post_patch_result)

    # Create check reports
    baseline_check = CheckReport(
        method="pytest",
        phase="baseline",
        level="BASELINE_REGRESSION",
        command=baseline_result.command,
        passed=False,
        exit_code=baseline_result.exit_code,
        duration_seconds=baseline_result.duration_seconds,
        failure_type=baseline_summary.error_type,
        summary={
            "error_type": baseline_summary.error_type,
            "failed_tests": baseline_summary.failed_tests,
            "relevant_output": baseline_summary.relevant_output,
            "full_output": baseline_summary.full_output,
        },
        test_node="",
    )

    post_patch_check = CheckReport(
        method="pytest",
        phase="post_patch",
        level="LEVEL_3_REGRESSION",
        command=post_patch_result.command,
        passed=False,
        exit_code=post_patch_result.exit_code,
        duration_seconds=post_patch_result.duration_seconds,
        failure_type=post_patch_summary.error_type,
        summary={
            "error_type": post_patch_summary.error_type,
            "failed_tests": post_patch_summary.failed_tests,
            "relevant_output": post_patch_summary.relevant_output,
            "full_output": post_patch_summary.full_output,
        },
        test_node="",
        tier="optional",
    )

    # Classify transition
    transition, baseline_id = classify_transition(baseline_check, post_patch_check)
    post_patch_check.transition = transition
    post_patch_check.baseline_check_id = baseline_id

    # Should be IMPROVED since test_a was resolved
    assert transition == CheckTransition.IMPROVED.value

    # Create verification report
    report = VerificationReport(
        run_id="test-e2e-improved",
        passed=True,
        checks=[post_patch_check],
        transition_summary=compute_transition_summary([post_patch_check]),
    )

    # Apply baseline-delta evaluation
    status, passed = apply_baseline_delta_evaluation(report, VerificationStrategy.BALANCED.value)

    # IMPROVED in optional tier should allow VERIFIED
    assert status == "VERIFIED"
    assert passed is True


def test_full_output_parsing_for_error_sections():
    """Test that full output parsing provides better error section extraction."""
    check = CheckReport(
        method="pytest",
        phase="post_patch",
        level="LEVEL_3_REGRESSION",
        command="python -m pytest -q -p no:cacheprovider",
        passed=False,
        exit_code=1,
        duration_seconds=2.0,
        failure_type="AssertionError",
        summary={
            "error_type": "AssertionError",
            "failed_tests": [
                "tests/test_a.py::test_a",
                "tests/test_b.py::test_b",
            ],
            "relevant_output": "FAILED tests/test_a.py::test_a\nAssertionError: error a",  # Truncated
            "full_output": "FAILED tests/test_a.py::test_a\nAssertionError: error a\nFAILED tests/test_b.py::test_b\nAssertionError: error b",  # Full
        },
        test_node="",
    )

    fingerprint = compute_failure_fingerprint(check)
    assert isinstance(fingerprint, dict)
    # With full output, both tests should be in the mapping
    assert len(fingerprint) == 2
    assert "tests/test_a.py::test_a" in fingerprint
    assert "tests/test_b.py::test_b" in fingerprint
