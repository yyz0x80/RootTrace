"""Tests for baseline-delta comparison and transition classification."""

from __future__ import annotations

from patchpilot.evidence.schema import CheckTransition
from patchpilot.verification.baseline_delta import (
    apply_baseline_delta_evaluation,
    classify_transition,
    compute_check_identity,
    compute_failure_fingerprint,
    compute_transition_summary,
    match_baseline_checks,
)
from patchpilot.verification.config import VerificationStrategy
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
    """Test failure fingerprint computation."""
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
    assert isinstance(fingerprint, str)
    assert len(fingerprint) == 16
    
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
    """Test FAIL → FAIL with equivalent fingerprint."""
    baseline_check = CheckReport(
        method="pytest",
        phase="baseline",
        level="BASELINE_TARGET",
        command="python -m pytest tests/test_example.py -q -p no:cacheprovider",
        passed=False,
        exit_code=1,
        duration_seconds=1.0,
        failure_type="AssertionError",
        summary={"error_type": "AssertionError", "failed_tests": ["tests/test_example.py::test_func"]},
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
        summary={"error_type": "AssertionError", "failed_tests": ["tests/test_example.py::test_func"]},
        test_node="tests/test_example.py::test_func",
    )
    
    transition, baseline_id = classify_transition(baseline_check, post_patch_check)
    assert transition == CheckTransition.PRE_EXISTING_FAILURE.value
    assert baseline_id == baseline_check.verification_id


def test_classify_transition_worsened():
    """Test FAIL → FAIL with different fingerprint."""
    baseline_check = CheckReport(
        method="pytest",
        phase="baseline",
        level="BASELINE_TARGET",
        command="python -m pytest tests/test_example.py -q -p no:cacheprovider",
        passed=False,
        exit_code=1,
        duration_seconds=1.0,
        failure_type="AssertionError",
        summary={"error_type": "AssertionError", "failed_tests": ["tests/test_example.py::test_func"]},
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
        failure_type="TypeError",  # Different error type
        summary={"error_type": "TypeError", "failed_tests": ["tests/test_example.py::test_func"]},
        test_node="tests/test_example.py::test_func",
    )
    
    transition, baseline_id = classify_transition(baseline_check, post_patch_check)
    assert transition == CheckTransition.WORSENED.value
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
