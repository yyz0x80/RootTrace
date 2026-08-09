"""Tests for verification report generation and management."""

import json
import tempfile
from pathlib import Path

from patchpilot.verification.report import (
    CheckReport,
    VerificationReport,
    failure_fingerprint,
)


def test_check_report_creation():
    """Test creating a basic CheckReport."""
    report = CheckReport(
        level="standard",
        command="pytest tests/",
        passed=True,
        exit_code=0,
        duration_seconds=5.2,
    )

    assert report.level == "standard"
    assert report.command == "pytest tests/"
    assert report.passed is True
    assert report.exit_code == 0
    assert report.duration_seconds == 5.2
    assert report.failure_type is None
    assert report.summary is None


def test_check_report_with_failure():
    """Test creating a CheckReport with failure information."""
    report = CheckReport(
        level="quick",
        command="ruff check patchpilot/",
        passed=False,
        exit_code=1,
        duration_seconds=1.5,
        failure_type="LintError",
        summary={"errors": 5, "warnings": 2},
    )

    assert report.level == "quick"
    assert report.passed is False
    assert report.exit_code == 1
    assert report.failure_type == "LintError"
    assert report.summary == {"errors": 5, "warnings": 2}


def test_check_report_to_dict():
    """Test converting CheckReport to dictionary."""
    report = CheckReport(
        level="comprehensive",
        command="pytest tests/ -v",
        passed=True,
        exit_code=0,
        duration_seconds=10.0,
        failure_type=None,
        summary={"tests_run": 42},
    )

    data = report.to_dict()

    assert data["level"] == "comprehensive"
    assert data["command"] == "pytest tests/ -v"
    assert data["passed"] is True
    assert data["exit_code"] == 0
    assert data["duration_seconds"] == 10.0
    assert data["failure_type"] is None
    assert data["summary"] == {"tests_run": 42}


def test_verification_report_creation():
    """Test creating a basic VerificationReport."""
    report = VerificationReport()

    assert report.passed is True
    assert len(report.checks) == 0
    assert report.retry_count == 0
    assert report.failed_level is None
    assert report.failure_type is None
    assert isinstance(report.run_id, str)
    assert len(report.run_id) > 0


def test_verification_report_custom_id():
    """Test creating a VerificationReport with custom run_id."""
    report = VerificationReport(
        run_id="custom-run-123",
        passed=False,
        retry_count=2,
    )

    assert report.run_id == "custom-run-123"
    assert report.passed is False
    assert report.retry_count == 2


def test_verification_report_add_check_passed():
    """Test adding a passed check to VerificationReport."""
    report = VerificationReport()
    check = CheckReport(
        level="quick",
        command="pytest tests/",
        passed=True,
        exit_code=0,
        duration_seconds=3.0,
    )

    report.add_check(check)

    assert len(report.checks) == 1
    assert report.passed is True
    assert report.failed_level is None
    assert report.failure_type is None


def test_verification_report_add_check_failed():
    """Test adding a failed check to VerificationReport."""
    report = VerificationReport()
    check = CheckReport(
        level="standard",
        command="pytest tests/",
        passed=False,
        exit_code=1,
        duration_seconds=4.0,
        failure_type="AssertionError",
    )

    report.add_check(check)

    assert len(report.checks) == 1
    assert report.passed is False
    assert report.failed_level == "standard"
    assert report.failure_type == "AssertionError"


def test_verification_report_multiple_checks():
    """Test adding multiple checks to VerificationReport."""
    report = VerificationReport()

    check1 = CheckReport(
        level="quick",
        command="ruff check patchpilot/",
        passed=True,
        exit_code=0,
        duration_seconds=1.0,
    )

    check2 = CheckReport(
        level="standard",
        command="pytest tests/",
        passed=False,
        exit_code=1,
        duration_seconds=5.0,
        failure_type="TestFailure",
    )

    report.add_check(check1)
    report.add_check(check2)

    assert len(report.checks) == 2
    assert report.passed is False
    assert report.failed_level == "standard"
    assert report.failure_type == "TestFailure"


def test_verification_report_get_failed_checks():
    """Test retrieving failed checks from VerificationReport."""
    report = VerificationReport()

    report.add_check(
        CheckReport(
            level="quick",
            command="ruff check",
            passed=True,
            exit_code=0,
            duration_seconds=1.0,
        )
    )

    report.add_check(
        CheckReport(
            level="standard",
            command="pytest tests/",
            passed=False,
            exit_code=1,
            duration_seconds=2.0,
            failure_type="AssertionError",
        )
    )

    report.add_check(
        CheckReport(
            level="comprehensive",
            command="pytest tests/ -v",
            passed=False,
            exit_code=1,
            duration_seconds=3.0,
            failure_type="Timeout",
        )
    )

    failed = report.get_failed_checks()

    assert len(failed) == 2
    assert all(not check.passed for check in failed)
    assert failed[0].failure_type == "AssertionError"
    assert failed[1].failure_type == "Timeout"


def test_verification_report_get_passed_checks():
    """Test retrieving passed checks from VerificationReport."""
    report = VerificationReport()

    report.add_check(
        CheckReport(
            level="quick",
            command="ruff check",
            passed=True,
            exit_code=0,
            duration_seconds=1.0,
        )
    )

    report.add_check(
        CheckReport(
            level="standard",
            command="pytest tests/",
            passed=False,
            exit_code=1,
            duration_seconds=2.0,
        )
    )

    passed = report.get_passed_checks()

    assert len(passed) == 1
    assert passed[0].passed is True
    assert passed[0].level == "quick"


def test_verification_report_get_checks_by_level():
    """Test retrieving checks by verification level."""
    report = VerificationReport()

    report.add_check(
        CheckReport(
            level="quick",
            command="ruff check",
            passed=True,
            exit_code=0,
            duration_seconds=1.0,
        )
    )

    report.add_check(
        CheckReport(
            level="standard",
            command="pytest tests/",
            passed=True,
            exit_code=0,
            duration_seconds=2.0,
        )
    )

    report.add_check(
        CheckReport(
            level="quick",
            command="python -m pytest tests/unit/",
            passed=True,
            exit_code=0,
            duration_seconds=1.5,
        )
    )

    quick_checks = report.get_checks_by_level("quick")
    standard_checks = report.get_checks_by_level("standard")
    comprehensive_checks = report.get_checks_by_level("comprehensive")

    assert len(quick_checks) == 2
    assert len(standard_checks) == 1
    assert len(comprehensive_checks) == 0


def test_verification_report_total_duration():
    """Test calculating total duration across all checks."""
    report = VerificationReport()

    report.add_check(
        CheckReport(
            level="quick",
            command="ruff check",
            passed=True,
            exit_code=0,
            duration_seconds=1.5,
        )
    )

    report.add_check(
        CheckReport(
            level="standard",
            command="pytest tests/",
            passed=True,
            exit_code=0,
            duration_seconds=3.7,
        )
    )

    report.add_check(
        CheckReport(
            level="comprehensive",
            command="pytest tests/ -v",
            passed=True,
            exit_code=0,
            duration_seconds=5.2,
        )
    )

    total = report.total_duration()

    assert total == 1.5 + 3.7 + 5.2


def test_verification_report_to_dict():
    """Test converting VerificationReport to dictionary."""
    report = VerificationReport(run_id="test-run-123")

    report.add_check(
        CheckReport(
            level="quick",
            command="ruff check",
            passed=True,
            exit_code=0,
            duration_seconds=1.0,
        )
    )

    data = report.to_dict()

    assert data["run_id"] == "test-run-123"
    assert data["passed"] is True
    assert data["retry_count"] == 0
    assert len(data["checks"]) == 1
    assert data["checks"][0]["level"] == "quick"
    assert data["checks"][0]["command"] == "ruff check"


def test_verification_report_save():
    """Test saving VerificationReport to JSON file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        report = VerificationReport(run_id="save-test-456")

        report.add_check(
            CheckReport(
                level="standard",
                command="pytest tests/",
                passed=False,
                exit_code=1,
                duration_seconds=2.5,
                failure_type="AssertionError",
            )
        )

        save_path = Path(tmpdir) / "reports" / "verification.json"
        report.save(save_path)

        assert save_path.exists()
        assert save_path.parent.exists()

        content = save_path.read_text(encoding="utf-8")
        data = json.loads(content)

        assert data["run_id"] == "save-test-456"
        assert data["passed"] is False
        assert data["failed_level"] == "standard"
        assert data["failure_type"] == "AssertionError"
        assert len(data["checks"]) == 1


def test_verification_report_load():
    """Test loading VerificationReport from JSON file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # First create and save a report
        original_report = VerificationReport(run_id="load-test-789")

        original_report.add_check(
            CheckReport(
                level="quick",
                command="ruff check",
                passed=True,
                exit_code=0,
                duration_seconds=1.0,
            )
        )

        original_report.add_check(
            CheckReport(
                level="standard",
                command="pytest tests/",
                passed=False,
                exit_code=1,
                duration_seconds=3.0,
                failure_type="TestFailure",
                summary={"failed": 2},
            )
        )

        save_path = Path(tmpdir) / "verification.json"
        original_report.save(save_path)

        # Load the report back
        loaded_report = VerificationReport.load(save_path)

        assert loaded_report.run_id == "load-test-789"
        assert loaded_report.passed is False
        assert loaded_report.retry_count == 0
        assert loaded_report.failed_level == "standard"
        assert loaded_report.failure_type == "TestFailure"
        assert len(loaded_report.checks) == 2

        # Verify check details
        assert loaded_report.checks[0].level == "quick"
        assert loaded_report.checks[0].passed is True
        assert loaded_report.checks[1].level == "standard"
        assert loaded_report.checks[1].passed is False
        assert loaded_report.checks[1].summary == {"failed": 2}


def test_verification_report_load_creates_directories():
    """Test that save creates parent directories when they don't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        report = VerificationReport()

        report.add_check(
            CheckReport(
                level="quick",
                command="ruff check",
                passed=True,
                exit_code=0,
                duration_seconds=1.0,
            )
        )

        # Use a nested path that doesn't exist
        save_path = (
            Path(tmpdir) / "deep" / "nested" / "path" / "verification.json"
        )
        report.save(save_path)

        assert save_path.exists()
        assert save_path.parent.exists()


def test_verification_report_save_utf8_encoding():
    """Test that saved files use UTF-8 encoding with unicode content."""
    with tempfile.TemporaryDirectory() as tmpdir:
        report = VerificationReport()

        report.add_check(
            CheckReport(
                level="standard",
                command="pytest tests/",
                passed=False,
                exit_code=1,
                duration_seconds=1.0,
                failure_type="TestFailure",
                summary={"message": "Test failed with unicode: ñ, 中文, 🚀"},
            )
        )

        save_path = Path(tmpdir) / "verification.json"
        report.save(save_path)

        # Read with UTF-8 encoding
        content = save_path.read_text(encoding="utf-8")
        data = json.loads(content)

        assert "ñ" in data["checks"][0]["summary"]["message"]
        assert "中文" in data["checks"][0]["summary"]["message"]
        assert "🚀" in data["checks"][0]["summary"]["message"]


def test_verification_report_load_missing_file():
    """Test loading from a non-existent file raises FileNotFoundError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "nonexistent.json"

        try:
            VerificationReport.load(save_path)
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            pass  # Expected


def test_verification_report_load_invalid_json():
    """Test loading from a file with invalid JSON raises JSONDecodeError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "invalid.json"
        save_path.write_text("not valid json", encoding="utf-8")

        try:
            VerificationReport.load(save_path)
            assert False, "Should have raised json.JSONDecodeError"
        except json.JSONDecodeError:
            pass  # Expected


def test_verification_report_load_partial_data():
    """Test loading a report with missing optional fields uses defaults."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a minimal JSON structure
        minimal_data = {
            "run_id": "minimal-test",
            "passed": True,
            "checks": [],
        }

        save_path = Path(tmpdir) / "minimal.json"
        save_path.write_text(
            json.dumps(minimal_data, indent=2), encoding="utf-8"
        )

        loaded = VerificationReport.load(save_path)

        assert loaded.run_id == "minimal-test"
        assert loaded.passed is True
        assert loaded.retry_count == 0
        assert loaded.failed_level is None
        assert loaded.failure_type is None
        assert len(loaded.checks) == 0


def test_verification_report_id_generation():
    """Test that run_id is generated automatically when not provided."""
    report1 = VerificationReport()
    report2 = VerificationReport()

    assert report1.run_id != report2.run_id
    assert len(report1.run_id) > 0
    assert len(report2.run_id) > 0


def test_check_report_immutable_behavior():
    """Test that CheckReport fields can be accessed but dataclass pattern is maintained."""
    report = CheckReport(
        level="standard",
        command="pytest tests/",
        passed=True,
        exit_code=0,
        duration_seconds=2.0,
    )

    # Dataclasses are mutable by default, so we can modify
    report.passed = False
    assert report.passed is False

    # But the original structure is preserved
    assert report.level == "standard"
    assert report.command == "pytest tests/"


def test_failure_fingerprint_passed_report():
    """Test that passed reports return empty tuple."""
    report = VerificationReport(passed=True)
    report.add_check(
        CheckReport(
            level="standard",
            command="pytest tests/",
            passed=True,
            exit_code=0,
            duration_seconds=1.0,
        )
    )

    fingerprint = failure_fingerprint(report)
    assert fingerprint == tuple()


def test_failure_fingerprint_empty_report():
    """Test that empty reports return empty tuple."""
    report = VerificationReport()
    fingerprint = failure_fingerprint(report)
    assert fingerprint == tuple()


def test_failure_fingerprint_with_failure():
    """Test fingerprint generation for failed report."""
    report = VerificationReport(passed=False)
    report.add_check(
        CheckReport(
            level="standard",
            command="pytest tests/",
            passed=False,
            exit_code=1,
            duration_seconds=1.0,
            failure_type="AssertionError",
            summary={
                "failed_tests": ["test_create_task", "test_update_task"],
                "error_type": "AssertionError",
                "relevant_output": "expected 'high', got None",
            },
        )
    )

    fingerprint = failure_fingerprint(report)
    assert fingerprint == (
        "AssertionError",
        ("test_create_task", "test_update_task"),
        "AssertionError",
        "expected 'high', got None",
    )


def test_failure_fingerprint_relevant_output_truncation():
    """Test that relevant output is truncated to 500 characters."""
    long_output = "x" * 1000
    report = VerificationReport(passed=False)
    report.add_check(
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
                "relevant_output": long_output,
            },
        )
    )

    fingerprint = failure_fingerprint(report)
    assert len(fingerprint[3]) == 500


def test_failure_fingerprint_missing_summary_fields():
    """Test fingerprint with missing summary fields."""
    report = VerificationReport(passed=False)
    report.add_check(
        CheckReport(
            level="standard",
            command="pytest tests/",
            passed=False,
            exit_code=1,
            duration_seconds=1.0,
            failure_type="SyntaxError",
            summary=None,
        )
    )

    fingerprint = failure_fingerprint(report)
    assert fingerprint == ("SyntaxError", tuple(), None, "")


def test_failure_fingerprint_uses_latest_failure():
    """Test that fingerprint uses the most recent failure."""
    report = VerificationReport(passed=False)
    
    # First failure
    report.add_check(
        CheckReport(
            level="standard",
            command="pytest tests/",
            passed=False,
            exit_code=1,
            duration_seconds=1.0,
            failure_type="AssertionError",
            summary={
                "failed_tests": ["test_first"],
                "error_type": "AssertionError",
                "relevant_output": "first error",
            },
        )
    )
    
    # Second failure (more recent)
    report.add_check(
        CheckReport(
            level="standard",
            command="pytest tests/",
            passed=False,
            exit_code=1,
            duration_seconds=1.0,
            failure_type="TypeError",
            summary={
                "failed_tests": ["test_second"],
                "error_type": "TypeError",
                "relevant_output": "second error",
            },
        )
    )

    fingerprint = failure_fingerprint(report)
    assert fingerprint == (
        "TypeError",
        ("test_second",),
        "TypeError",
        "second error",
    )


def test_failure_fingerprint_consistency():
    """Test that identical failures produce identical fingerprints."""
    report1 = VerificationReport(passed=False)
    report1.add_check(
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

    report2 = VerificationReport(passed=False)
    report2.add_check(
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

    fingerprint1 = failure_fingerprint(report1)
    fingerprint2 = failure_fingerprint(report2)
    assert fingerprint1 == fingerprint2
