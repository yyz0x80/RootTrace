"""Tests for error parsing utilities."""

from patchpilot.sandbox.docker_runner import CommandResult
from patchpilot.verification.error_parser import (
    FailureSummary,
    _extract_relevant_output,
    _find_error_type,
    _find_failed_tests,
    parse_failure,
)


def test_find_failed_tests_with_failures():
    """Test extracting failed test names from pytest output."""
    output = """
    FAILED tests/test_example.py::test_function_one
    FAILED tests/test_example.py::test_function_two
    PASSED tests/test_example.py::test_function_three
    """
    
    failed = _find_failed_tests(output)
    
    assert "tests/test_example.py::test_function_one" in failed
    assert "tests/test_example.py::test_function_two" in failed
    assert len(failed) == 2


def test_find_failed_tests_no_failures():
    """Test extracting failed tests when none exist."""
    output = """
    tests/test_example.py::test_function_one PASSED
    tests/test_example.py::test_function_two PASSED
    """
    
    failed = _find_failed_tests(output)
    
    assert len(failed) == 0


def test_find_error_type_assertion():
    """Test identifying AssertionError type."""
    output = "AssertionError: Expected 5 but got 3"
    
    error_type = _find_error_type(output)
    
    assert error_type == "AssertionError"


def test_find_error_type_syntax():
    """Test identifying SyntaxError type."""
    output = "SyntaxError: invalid syntax"
    
    error_type = _find_error_type(output)
    
    assert error_type == "SyntaxError"


def test_find_error_type_import():
    """Test identifying ImportError type."""
    output = "ImportError: cannot import name 'missing_module'"
    
    error_type = _find_error_type(output)
    
    assert error_type == "ImportError"


def test_find_error_type_unknown():
    """Test returning UnknownError when no error type is found."""
    output = "Some generic error message without specific exception type"
    
    error_type = _find_error_type(output)
    
    assert error_type == "UnknownError"


def test_find_error_type_priority():
    """Test that error type detection follows priority order."""
    output = "ValueError: invalid value\nAssertionError: assertion failed"
    
    error_type = _find_error_type(output)
    
    # Should return the first match in priority order
    assert error_type == "AssertionError"


def test_extract_relevant_output_with_markers():
    """Test extracting output lines containing error markers."""
    output = """
    Running tests...
    Some informational line
    FAILED tests/test_example.py::test_function
    ERROR during collection
    Another info line
    AssertionError: Expected 5 but got 3
    """
    
    relevant = _extract_relevant_output(output)
    
    assert "FAILED" in relevant
    assert "ERROR" in relevant
    assert "AssertionError" in relevant
    assert "Some informational line" not in relevant


def test_extract_relevant_output_without_markers():
    """Test falling back to last lines when no markers found."""
    output = "\n".join([f"Line {i}" for i in range(30)])
    
    relevant = _extract_relevant_output(output, max_lines=10)
    
    # Should contain the last 10 lines
    assert "Line 20" in relevant
    assert "Line 29" in relevant
    assert "Line 0" not in relevant


def test_extract_relevant_output_truncation():
    """Test that output is truncated to max character limit."""
    long_line = "A" * 100
    output = "\n".join([long_line for _ in range(50)])
    
    relevant = _extract_relevant_output(output)
    
    # Should be truncated to 3000 characters
    assert len(relevant) <= 3000


def test_extract_relevant_output_empty_lines_filtered():
    """Test that empty lines are filtered out."""
    output = """
    Line 1
    
    Line 3
    
    Line 5
    """
    
    relevant = _extract_relevant_output(output)
    
    assert "\n\n" not in relevant
    assert "Line 1" in relevant


def test_parse_failure_timeout():
    """Test parsing a timeout failure."""
    result = CommandResult(
        command="pytest tests/",
        exit_code=124,
        stdout="",
        stderr="",
        duration_seconds=30.0,
        timed_out=True,
    )
    
    summary = parse_failure(result)
    
    assert summary.command == "pytest tests/"
    assert summary.exit_code == 124
    assert summary.error_type == "Timeout"
    assert summary.timed_out is True


def test_parse_failure_ruff():
    """Test parsing a ruff lint error."""
    result = CommandResult(
        command="ruff check patchpilot/",
        exit_code=1,
        stdout="patchpilot/module.py:1:1: F401 Unused import",
        stderr="",
        duration_seconds=2.0,
        timed_out=False,
    )
    
    summary = parse_failure(result)
    
    assert summary.command == "ruff check patchpilot/"
    assert summary.exit_code == 1
    assert summary.error_type == "LintError"
    assert summary.timed_out is False


def test_parse_failure_pytest():
    """Test parsing a pytest failure."""
    result = CommandResult(
        command="pytest tests/",
        exit_code=1,
        stdout="""
        FAILED tests/test_example.py::test_two
        FAILED tests/test_example.py::test_three
        PASSED tests/test_example.py::test_one
        
        ============== FAILURES ==============
        """,
        stderr="AssertionError: Expected 5 but got 3",
        duration_seconds=5.0,
        timed_out=False,
    )
    
    summary = parse_failure(result)
    
    assert summary.command == "pytest tests/"
    assert summary.exit_code == 1
    assert summary.error_type == "AssertionError"
    assert len(summary.failed_tests) == 2
    assert "tests/test_example.py::test_two" in summary.failed_tests
    assert summary.timed_out is False


def test_parse_failure_combined_output():
    """Test that both stdout and stderr are analyzed."""
    result = CommandResult(
        command="pytest tests/",
        exit_code=1,
        stdout="FAILED tests/test.py::test_func",
        stderr="AssertionError: test failed",
        duration_seconds=1.0,
        timed_out=False,
    )
    
    summary = parse_failure(result)
    
    assert len(summary.failed_tests) == 1
    assert summary.error_type == "AssertionError"
    assert "FAILED" in summary.relevant_output or "AssertionError" in summary.relevant_output


def test_parse_failure_unknown_error():
    """Test parsing failure with unknown error type."""
    result = CommandResult(
        command="pytest tests/",
        exit_code=1,
        stdout="Some unexpected error occurred",
        stderr="Process finished with exit code 1",
        duration_seconds=1.0,
        timed_out=False,
    )
    
    summary = parse_failure(result)
    
    assert summary.error_type == "UnknownError"


def test_failure_summary_dataclass():
    """Test FailureSummary dataclass structure."""
    summary = FailureSummary(
        command="pytest tests/",
        exit_code=1,
        failed_tests=["test_one", "test_two"],
        error_type="AssertionError",
        relevant_output="Test failed",
        timed_out=False,
    )
    
    assert summary.command == "pytest tests/"
    assert summary.exit_code == 1
    assert len(summary.failed_tests) == 2
    assert summary.error_type == "AssertionError"
    assert summary.timed_out is False


def test_parse_failure_whitespace_command():
    """Test parsing with leading/trailing whitespace in command."""
    result = CommandResult(
        command="  pytest tests/  ",
        exit_code=1,
        stdout="",
        stderr="",
        duration_seconds=1.0,
        timed_out=False,
    )
    
    summary = parse_failure(result)
    
    # Should handle whitespace in command detection
    assert summary.error_type == "UnknownError"
