"""Tests for the repair candidate selector for relevance-aware repair loop."""


from patchpilot.evidence.schema import CheckTransition
from patchpilot.planning.schema import ChangePlan, PlannedChange
from patchpilot.verification.report import CheckReport, VerificationReport
from patchpilot.workflow.failure_classifier import FailureType
from patchpilot.workflow.repair_selector import (
    ExcludedFailure,
    RepairCandidate,
    RepairSelection,
    RepairSelector,
)


class TestRepairSelectorInit:
    """Tests for RepairSelector initialization."""

    def test_init_with_default_parameters(self):
        """Test initialization with default parameters."""
        selector = RepairSelector()

        assert selector.strategy == "balanced"
        assert selector.changed_files == []
        assert selector.approved_files == set()

    def test_init_with_custom_parameters(self):
        """Test initialization with custom parameters."""
        selector = RepairSelector(
            strategy="strict",
            changed_files=["src/file.py", "tests/test_file.py"],
            approved_files={"src/file.py", "src/other.py"},
        )

        assert selector.strategy == "strict"
        assert selector.changed_files == ["src/file.py", "tests/test_file.py"]
        assert selector.approved_files == {"src/file.py", "src/other.py"}


class TestRepairSelectorSelectCandidates:
    """Tests for repair candidate selection logic."""

    def test_select_with_no_failures(self):
        """Test selection when there are no failures."""
        report = VerificationReport(passed=True)
        selector = RepairSelector()

        selection = selector.select_repair_candidates(report)

        assert selection.should_repair is False
        assert selection.should_stop is True
        assert selection.stop_reason == "No failures to repair"
        assert selection.completion_hint == "VERIFIED"
        assert len(selection.repair_candidates) == 0
        assert len(selection.excluded_failures) == 0

    def test_baseline_failures_are_not_repair_candidates(self):
        """Merged baseline diagnostics must not be sent to the repair agent."""
        report = VerificationReport(passed=False)
        report.add_check(
            CheckReport(
                method="structural_check",
                phase="baseline",
                level="SPECIALIZED_STRUCTURAL",
                command="structural:method_parameter:.",
                passed=False,
                exit_code=1,
                duration_seconds=0.0,
                failure_type="structural_failure",
                tier="required",
                failure_fingerprint="baseline-check",
            )
        )

        selection = RepairSelector(strategy="strict").select_repair_candidates(
            report
        )

        assert selection.should_stop
        assert not selection.repair_candidates

    def test_scratch_failure_is_never_repairable(self):
        """Supplemental agent tests remain non-blocking under strict strategy."""
        report = VerificationReport(passed=False)
        report.add_check(
            CheckReport(
                method="agent_scratch_pytest",
                phase="post_patch",
                level="LEVEL_2_AGENT_SCRATCH",
                command="python -m pytest .patchpilot_checks -q",
                passed=False,
                exit_code=1,
                duration_seconds=0.1,
                failure_type="TEST_FAILURE",
                tier="optional",
                transition=CheckTransition.NEW_OR_UNCOMPARED.value,
                failure_fingerprint="scratch-check",
            )
        )

        selection = RepairSelector(strategy="strict").select_repair_candidates(
            report
        )

        assert selection.should_stop
        assert not selection.repair_candidates
        assert "supplemental" in selection.excluded_failures[0].reason

    def test_required_target_failure_is_repairable(self):
        """Test that REQUIRED target failures are selected for repair."""
        report = VerificationReport(passed=False)
        report.add_check(
            CheckReport(
                method="pytest",
                phase="post_patch",
                level="LEVEL_2_TARGET_TESTS",
                command="pytest tests/test_target.py",
                passed=False,
                exit_code=1,
                duration_seconds=1.0,
                failure_type="AssertionError",
                tier="required",
                transition=CheckTransition.NEW_OR_UNCOMPARED.value,
                failure_fingerprint="abc123",
                summary={"error_type": "AssertionError", "failed_tests": ["test_target"]},
            )
        )
        selector = RepairSelector()

        selection = selector.select_repair_candidates(report)

        assert selection.should_repair is True
        assert selection.should_stop is False
        assert len(selection.repair_candidates) == 1
        assert selection.repair_candidates[0].tier == "required"
        assert selection.repair_candidates[0].reason == "REQUIRED test failure (new or worsened)"
        assert len(selection.excluded_failures) == 0

    def test_affected_regression_is_repairable(self):
        """Test that AFFECTED PASS → FAIL regression is selected for repair."""
        report = VerificationReport(passed=False)
        report.add_check(
            CheckReport(
                method="pytest",
                phase="post_patch",
                level="LEVEL_2_TARGET_TESTS",
                command="pytest tests/test_affected.py",
                passed=False,
                exit_code=1,
                duration_seconds=1.0,
                failure_type="AssertionError",
                tier="affected",
                transition=CheckTransition.REGRESSION.value,
                failure_fingerprint="def456",
                summary={"error_type": "AssertionError", "failed_tests": ["test_affected"]},
            )
        )
        selector = RepairSelector()

        selection = selector.select_repair_candidates(report)

        assert selection.should_repair is True
        assert len(selection.repair_candidates) == 1
        assert selection.repair_candidates[0].tier == "affected"
        assert selection.repair_candidates[0].reason == "AFFECTED regression (PASS → FAIL)"

    def test_unchanged_baseline_failure_is_excluded(self):
        """Test that unchanged pre-existing failures are excluded."""
        report = VerificationReport(passed=False)
        report.add_check(
            CheckReport(
                method="pytest",
                phase="post_patch",
                level="LEVEL_3_REGRESSION",
                command="pytest tests/",
                passed=False,
                exit_code=1,
                duration_seconds=1.0,
                failure_type="AssertionError",
                tier="affected",
                transition=CheckTransition.PRE_EXISTING_FAILURE.value,
                failure_fingerprint="preexisting123",
                summary={"error_type": "AssertionError", "failed_tests": ["test_old"]},
            )
        )
        selector = RepairSelector()

        selection = selector.select_repair_candidates(report)

        assert selection.should_repair is False
        assert selection.should_stop is True
        assert len(selection.repair_candidates) == 0
        assert len(selection.excluded_failures) == 1
        assert selection.excluded_failures[0].reason == "Pre-existing unchanged AFFECTED failure (unrelated)"
        assert selection.excluded_failures[0].is_blocking is False

    def test_environment_failure_is_excluded_as_blocking(self):
        """Test that environment failures are excluded as blocking."""
        report = VerificationReport(passed=False)
        report.add_check(
            CheckReport(
                method="pytest",
                phase="post_patch",
                level="LEVEL_2_TARGET_TESTS",
                command="pytest tests/test_target.py",
                passed=False,
                exit_code=1,
                duration_seconds=1.0,
                failure_type=FailureType.ENVIRONMENT_FAILURE.value,
                tier="required",
                transition=CheckTransition.NEW_OR_UNCOMPARED.value,
                failure_fingerprint="env789",
                summary={"error_type": "ModuleNotFoundError", "relevant_output": "command not found"},
            )
        )
        selector = RepairSelector()

        selection = selector.select_repair_candidates(report)

        assert selection.should_repair is False
        assert selection.should_stop is True
        assert len(selection.repair_candidates) == 0
        assert len(selection.excluded_failures) == 1
        assert selection.excluded_failures[0].reason == "Non-repairable failure type (environment, permission, or timeout)"
        assert selection.excluded_failures[0].is_blocking is True

    def test_permission_failure_is_excluded_as_blocking(self):
        """Test that permission failures are excluded as blocking."""
        report = VerificationReport(passed=False)
        report.add_check(
            CheckReport(
                method="pytest",
                phase="post_patch",
                level="LEVEL_2_TARGET_TESTS",
                command="pytest tests/test_target.py",
                passed=False,
                exit_code=1,
                duration_seconds=1.0,
                failure_type=FailureType.PERMISSION_FAILURE.value,
                tier="required",
                transition=CheckTransition.NEW_OR_UNCOMPARED.value,
                failure_fingerprint="perm789",
                summary={"error_type": "PermissionError", "relevant_output": "permission denied"},
            )
        )
        selector = RepairSelector()

        selection = selector.select_repair_candidates(report)

        assert selection.should_repair is False
        assert selection.should_stop is True
        assert len(selection.excluded_failures) == 1
        assert selection.excluded_failures[0].is_blocking is True

    def test_timeout_failure_is_excluded_as_blocking(self):
        """Test that timeout failures are excluded as blocking."""
        report = VerificationReport(passed=False)
        report.add_check(
            CheckReport(
                method="pytest",
                phase="post_patch",
                level="LEVEL_2_TARGET_TESTS",
                command="pytest tests/test_target.py",
                passed=False,
                exit_code=1,
                duration_seconds=1.0,
                failure_type=FailureType.TIMEOUT.value,
                tier="required",
                transition=CheckTransition.NEW_OR_UNCOMPARED.value,
                failure_fingerprint="timeout789",
                summary={"timed_out": True},
            )
        )
        selector = RepairSelector()

        selection = selector.select_repair_candidates(report)

        assert selection.should_repair is False
        assert selection.should_stop is True
        assert len(selection.excluded_failures) == 1
        assert selection.excluded_failures[0].is_blocking is True

    def test_ruff_violation_is_repairable(self):
        """Test that patch-introduced Ruff violations are repairable."""
        report = VerificationReport(passed=False)
        report.add_check(
            CheckReport(
                method="ruff",
                phase="post_patch",
                level="LEVEL_1_LINT",
                command="ruff check --no-cache .",
                passed=False,
                exit_code=1,
                duration_seconds=0.5,
                failure_type="CODE_FAILURE",
                tier="required",
                transition=CheckTransition.NEW_OR_UNCOMPARED.value,
                failure_fingerprint="ruff123",
                summary={"error": "F821 undefined name 'value'"},
            )
        )
        selector = RepairSelector()

        selection = selector.select_repair_candidates(report)

        assert selection.should_repair is True
        assert len(selection.repair_candidates) == 1
        assert selection.repair_candidates[0].reason == "Patch-introduced Ruff violation"

    def test_optional_failure_balanced_strategy_excluded(self):
        """Test that OPTIONAL failures are excluded with balanced strategy."""
        report = VerificationReport(passed=False)
        report.add_check(
            CheckReport(
                method="pytest",
                phase="post_patch",
                level="LEVEL_3_REGRESSION",
                command="pytest tests/test_optional.py",
                passed=False,
                exit_code=1,
                duration_seconds=1.0,
                failure_type="AssertionError",
                tier="optional",
                transition=CheckTransition.REGRESSION.value,
                failure_fingerprint="opt123",
                summary={"error_type": "AssertionError", "failed_tests": ["test_optional"]},
            )
        )
        selector = RepairSelector(strategy="balanced")

        selection = selector.select_repair_candidates(report)

        assert selection.should_repair is False
        assert selection.should_stop is True
        assert len(selection.repair_candidates) == 0
        assert len(selection.excluded_failures) == 1
        assert selection.excluded_failures[0].reason == "OPTIONAL failure (balanced strategy - non-blocking)"
        assert selection.excluded_failures[0].is_blocking is False

    def test_optional_failure_strict_strategy_repairable(self):
        """Test that OPTIONAL failures are repairable with strict strategy."""
        report = VerificationReport(passed=False)
        report.add_check(
            CheckReport(
                method="pytest",
                phase="post_patch",
                level="LEVEL_3_REGRESSION",
                command="pytest tests/test_optional.py",
                passed=False,
                exit_code=1,
                duration_seconds=1.0,
                failure_type="AssertionError",
                tier="optional",
                transition=CheckTransition.REGRESSION.value,
                failure_fingerprint="opt123",
                summary={"error_type": "AssertionError", "failed_tests": ["test_optional"]},
            )
        )
        selector = RepairSelector(strategy="strict")

        selection = selector.select_repair_candidates(report)

        assert selection.should_repair is True
        assert len(selection.repair_candidates) == 1
        assert selection.repair_candidates[0].reason == "OPTIONAL failure (strict strategy)"

    def test_mixed_relevant_and_irrelevant_failures(self):
        """Test selection with mixed relevant and irrelevant failures."""
        report = VerificationReport(passed=False)
        # Required failure (repairable)
        report.add_check(
            CheckReport(
                method="pytest",
                phase="post_patch",
                level="LEVEL_2_TARGET_TESTS",
                command="pytest tests/test_required.py",
                passed=False,
                exit_code=1,
                duration_seconds=1.0,
                failure_type="AssertionError",
                tier="required",
                transition=CheckTransition.NEW_OR_UNCOMPARED.value,
                failure_fingerprint="req123",
                summary={"error_type": "AssertionError", "failed_tests": ["test_required"]},
            )
        )
        # Pre-existing failure (exclude)
        report.add_check(
            CheckReport(
                method="pytest",
                phase="post_patch",
                level="LEVEL_3_REGRESSION",
                command="pytest tests/test_old.py",
                passed=False,
                exit_code=1,
                duration_seconds=1.0,
                failure_type="AssertionError",
                tier="affected",
                transition=CheckTransition.PRE_EXISTING_FAILURE.value,
                failure_fingerprint="old123",
                summary={"error_type": "AssertionError", "failed_tests": ["test_old"]},
            )
        )
        selector = RepairSelector()

        selection = selector.select_repair_candidates(report)

        assert selection.should_repair is True
        assert len(selection.repair_candidates) == 1
        assert len(selection.excluded_failures) == 1
        assert selection.repair_candidates[0].tier == "required"
        assert selection.excluded_failures[0].tier == "affected"

    def test_worsened_failure_is_repairable(self):
        """Test that worsened failures are repairable."""
        report = VerificationReport(passed=False)
        report.add_check(
            CheckReport(
                method="pytest",
                phase="post_patch",
                level="LEVEL_2_TARGET_TESTS",
                command="pytest tests/test_target.py",
                passed=False,
                exit_code=1,
                duration_seconds=1.0,
                failure_type="AssertionError",
                tier="required",
                transition=CheckTransition.WORSENED.value,
                failure_fingerprint="worsened123",
                summary={"error_type": "AssertionError", "failed_tests": ["test_target"]},
            )
        )
        selector = RepairSelector()

        selection = selector.select_repair_candidates(report)

        assert selection.should_repair is True
        assert len(selection.repair_candidates) == 1
        assert selection.repair_candidates[0].reason == "REQUIRED test failure (new or worsened)"

    def test_out_of_scope_files_excluded(self):
        """Test that failures requiring out-of-scope files are excluded."""
        from patchpilot.planning.schema import ChangeAction
        change_plan = ChangePlan(
            planned_changes=[
                PlannedChange(
                    path="src/approved.py",
                    action=ChangeAction.MODIFY,
                    description="Modify approved file",
                    acceptance_criteria=["AC-1"],
                )
            ],
            risk_level="low",
        )
        report = VerificationReport(passed=False)
        report.add_check(
            CheckReport(
                method="pytest",
                phase="post_patch",
                level="LEVEL_2_TARGET_TESTS",
                command="pytest tests/test_target.py",
                passed=False,
                exit_code=1,
                duration_seconds=1.0,
                failure_type="AssertionError",
                tier="required",
                transition=CheckTransition.NEW_OR_UNCOMPARED.value,
                failure_fingerprint="scope123",
                summary={"required_files": ["src/unapproved_file.py"]},
            )
        )
        selector = RepairSelector(approved_files={"src/approved.py"})

        selection = selector.select_repair_candidates(report, change_plan)

        assert selection.should_repair is False
        assert selection.should_stop is True
        assert len(selection.repair_candidates) == 0
        assert len(selection.excluded_failures) == 1
        assert selection.excluded_failures[0].reason == "Failure requires files outside approved change plan"
        assert selection.excluded_failures[0].is_blocking is True

    def test_unstructured_probe_error_remains_repairable(self):
        """Do not infer out-of-scope requirements from missing file names."""
        change_plan = ChangePlan(
            planned_changes=[
                PlannedChange(
                    path="src/approved.py",
                    action="modify",
                    description="Fix behavior",
                )
            ],
            risk_level="low",
        )
        report = VerificationReport(passed=False)
        report.add_check(
            CheckReport(
                method="acceptance_probe",
                phase="post_patch",
                level="SPECIALIZED_PROBE",
                command="probe:behavior",
                passed=False,
                exit_code=1,
                duration_seconds=0.1,
                failure_type="probe_failure",
                tier="required",
                transition=CheckTransition.WORSENED.value,
                summary={"error": "name 'Dependency' is not defined"},
            )
        )
        selector = RepairSelector(approved_files={"src/approved.py"})

        selection = selector.select_repair_candidates(report, change_plan)

        assert selection.should_repair is True
        assert selection.excluded_failures == []


class TestRepairCandidate:
    """Tests for RepairCandidate dataclass."""

    def test_repair_candidate_creation(self):
        """Test creating a RepairCandidate."""
        check = CheckReport(
            method="pytest",
            phase="post_patch",
            level="LEVEL_2_TARGET_TESTS",
            command="pytest tests/test.py",
            passed=False,
            exit_code=1,
            duration_seconds=1.0,
        )

        candidate = RepairCandidate(
            check=check,
            reason="Test failure",
            tier="required",
            transition="REGRESSION",
            fingerprint="abc123",
            bounded_output="Error output",
        )

        assert candidate.check == check
        assert candidate.reason == "Test failure"
        assert candidate.tier == "required"
        assert candidate.transition == "REGRESSION"
        assert candidate.fingerprint == "abc123"
        assert candidate.bounded_output == "Error output"


class TestExcludedFailure:
    """Tests for ExcludedFailure dataclass."""

    def test_excluded_failure_creation(self):
        """Test creating an ExcludedFailure."""
        check = CheckReport(
            method="pytest",
            phase="post_patch",
            level="LEVEL_3_REGRESSION",
            command="pytest tests/",
            passed=False,
            exit_code=1,
            duration_seconds=1.0,
        )

        excluded = ExcludedFailure(
            check=check,
            reason="Pre-existing failure",
            tier="affected",
            transition="PRE_EXISTING_FAILURE",
            is_blocking=False,
        )

        assert excluded.check == check
        assert excluded.reason == "Pre-existing failure"
        assert excluded.tier == "affected"
        assert excluded.transition == "PRE_EXISTING_FAILURE"
        assert excluded.is_blocking is False


class TestRepairSelection:
    """Tests for RepairSelection dataclass."""

    def test_repair_selection_creation(self):
        """Test creating a RepairSelection."""
        selection = RepairSelection(
            repair_candidates=[],
            excluded_failures=[],
            should_repair=False,
            should_stop=True,
            stop_reason="No repairable failures",
            completion_hint="PARTIALLY_VERIFIED",
        )

        assert selection.repair_candidates == []
        assert selection.excluded_failures == []
        assert selection.should_repair is False
        assert selection.should_stop is True
        assert selection.stop_reason == "No repairable failures"
        assert selection.completion_hint == "PARTIALLY_VERIFIED"

    def test_repair_selection_defaults(self):
        """Test RepairSelection with default values."""
        selection = RepairSelection()

        assert selection.repair_candidates == []
        assert selection.excluded_failures == []
        assert selection.should_repair is False
        assert selection.should_stop is False
        assert selection.stop_reason == ""
        assert selection.completion_hint == ""


class TestRepairSelectorBoundedOutput:
    """Tests for bounded output generation."""

    def test_bounded_output_truncation(self):
        """Test that long output is truncated to 2000 characters."""
        report = VerificationReport(passed=False)
        long_output = "x" * 3000
        report.add_check(
            CheckReport(
                method="pytest",
                phase="post_patch",
                level="LEVEL_2_TARGET_TESTS",
                command="pytest tests/test.py",
                passed=False,
                exit_code=1,
                duration_seconds=1.0,
                failure_type="AssertionError",
                tier="required",
                transition=CheckTransition.NEW_OR_UNCOMPARED.value,
                failure_fingerprint="abc123",
                summary={"relevant_output": long_output},
            )
        )
        selector = RepairSelector()

        selection = selector.select_repair_candidates(report)

        assert len(selection.repair_candidates[0].bounded_output) <= 2100  # Allow some margin for truncation marker
        assert "truncated" in selection.repair_candidates[0].bounded_output

    def test_bounded_output_short(self):
        """Test that short output is not truncated."""
        report = VerificationReport(passed=False)
        short_output = "Short error message"
        report.add_check(
            CheckReport(
                method="pytest",
                phase="post_patch",
                level="LEVEL_2_TARGET_TESTS",
                command="pytest tests/test.py",
                passed=False,
                exit_code=1,
                duration_seconds=1.0,
                failure_type="AssertionError",
                tier="required",
                transition=CheckTransition.NEW_OR_UNCOMPARED.value,
                failure_fingerprint="abc123",
                summary={"relevant_output": short_output},
            )
        )
        selector = RepairSelector()

        selection = selector.select_repair_candidates(report)

        assert selection.repair_candidates[0].bounded_output == short_output
        assert "truncated" not in selection.repair_candidates[0].bounded_output
