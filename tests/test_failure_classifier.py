"""Tests for failure classification utilities."""

from patchpilot.verification.error_parser import FailureSummary
from patchpilot.workflow.failure_classifier import (
    FailureType,
    classify_failure,
)


def test_classify_timeout():
    """Test classification of timeout failures."""
    summary = FailureSummary(
        command="pytest tests/",
        exit_code=124,
        failed_tests=[],
        error_type="Timeout",
        relevant_output="Command timed out after 30 seconds",
        timed_out=True,
    )

    failure_type = classify_failure(summary)

    assert failure_type == FailureType.TIMEOUT


def test_classify_permission_denied():
    """Test classification of permission failures."""
    summary = FailureSummary(
        command="cat /etc/passwd",
        exit_code=1,
        failed_tests=[],
        error_type="PermissionError",
        relevant_output="permission denied: /etc/passwd",
        timed_out=False,
    )

    failure_type = classify_failure(summary)

    assert failure_type == FailureType.PERMISSION_FAILURE


def test_classify_operation_not_permitted():
    """Test classification of operation not permitted failures."""
    summary = FailureSummary(
        command="mount -o loop image.iso",
        exit_code=1,
        failed_tests=[],
        error_type="PermissionError",
        relevant_output="operation not permitted",
        timed_out=False,
    )

    failure_type = classify_failure(summary)

    assert failure_type == FailureType.PERMISSION_FAILURE


def test_classify_command_not_found():
    """Test classification of command not found failures."""
    summary = FailureSummary(
        command="nonexistent_command",
        exit_code=127,
        failed_tests=[],
        error_type="UnknownError",
        relevant_output="command not found: nonexistent_command",
        timed_out=False,
    )

    failure_type = classify_failure(summary)

    assert failure_type == FailureType.ENVIRONMENT_FAILURE


def test_classify_module_not_found():
    """Test classification of module not found failures."""
    summary = FailureSummary(
        command="python -m pytest",
        exit_code=1,
        failed_tests=[],
        error_type="ModuleNotFoundError",
        relevant_output="ModuleNotFoundError: No module named 'pytest'",
        timed_out=False,
    )

    failure_type = classify_failure(summary)

    assert failure_type == FailureType.ENVIRONMENT_FAILURE


def test_classify_dns_failure():
    """Test classification of DNS resolution failures."""
    summary = FailureSummary(
        command="pip install package",
        exit_code=1,
        failed_tests=[],
        error_type="UnknownError",
        relevant_output="could not resolve host: pypi.org",
        timed_out=False,
    )

    failure_type = classify_failure(summary)

    assert failure_type == FailureType.ENVIRONMENT_FAILURE


def test_classify_connection_refused():
    """Test classification of connection refused failures."""
    summary = FailureSummary(
        command="curl http://localhost:8080",
        exit_code=7,
        failed_tests=[],
        error_type="UnknownError",
        relevant_output="connection refused",
        timed_out=False,
    )

    failure_type = classify_failure(summary)

    assert failure_type == FailureType.ENVIRONMENT_FAILURE


def test_classify_ruff_failure():
    """Test classification of ruff lint failures."""
    summary = FailureSummary(
        command="ruff check patchpilot/",
        exit_code=1,
        failed_tests=[],
        error_type="LintError",
        relevant_output="patchpilot/module.py:1:1: F401 Unused import",
        timed_out=False,
    )

    failure_type = classify_failure(summary)

    assert failure_type == FailureType.CODE_FAILURE


def test_classify_test_failure_with_failed_tests():
    """Test classification when failed tests are present."""
    summary = FailureSummary(
        command="pytest tests/",
        exit_code=1,
        failed_tests=["tests/test_example.py::test_one", "tests/test_example.py::test_two"],
        error_type="AssertionError",
        relevant_output="FAILED tests/test_example.py::test_one",
        timed_out=False,
    )

    failure_type = classify_failure(summary)

    assert failure_type == FailureType.TEST_FAILURE


def test_classify_syntax_error():
    """Test classification of syntax errors."""
    summary = FailureSummary(
        command="python -m pytest",
        exit_code=1,
        failed_tests=[],
        error_type="SyntaxError",
        relevant_output="SyntaxError: invalid syntax",
        timed_out=False,
    )

    failure_type = classify_failure(summary)

    assert failure_type == FailureType.CODE_FAILURE


def test_classify_indentation_error():
    """Test classification of indentation errors."""
    summary = FailureSummary(
        command="python -m pytest",
        exit_code=1,
        failed_tests=[],
        error_type="IndentationError",
        relevant_output="IndentationError: unexpected indent",
        timed_out=False,
    )

    failure_type = classify_failure(summary)

    assert failure_type == FailureType.CODE_FAILURE


def test_classify_name_error():
    """Test classification of name errors."""
    summary = FailureSummary(
        command="python -m pytest",
        exit_code=1,
        failed_tests=[],
        error_type="NameError",
        relevant_output="NameError: name 'undefined_var' is not defined",
        timed_out=False,
    )

    failure_type = classify_failure(summary)

    assert failure_type == FailureType.CODE_FAILURE


def test_classify_type_error():
    """Test classification of type errors."""
    summary = FailureSummary(
        command="python -m pytest",
        exit_code=1,
        failed_tests=[],
        error_type="TypeError",
        relevant_output="TypeError: unsupported operand type(s)",
        timed_out=False,
    )

    failure_type = classify_failure(summary)

    assert failure_type == FailureType.CODE_FAILURE


def test_classify_import_error():
    """Test classification of import errors."""
    summary = FailureSummary(
        command="python -m pytest",
        exit_code=1,
        failed_tests=[],
        error_type="ImportError",
        relevant_output="ImportError: cannot import name 'missing'",
        timed_out=False,
    )

    failure_type = classify_failure(summary)

    assert failure_type == FailureType.CODE_FAILURE


def test_classify_pytest_command():
    """Test classification of pytest commands without explicit failures."""
    summary = FailureSummary(
        command="pytest tests/",
        exit_code=1,
        failed_tests=[],
        error_type="AssertionError",
        relevant_output="Some pytest error",
        timed_out=False,
    )

    failure_type = classify_failure(summary)

    assert failure_type == FailureType.TEST_FAILURE


def test_classify_python_module_pytest_command():
    """Test classification of pytest invoked as a Python module."""
    summary = FailureSummary(
        command="python -m pytest tests/",
        exit_code=1,
        failed_tests=[],
        error_type="AssertionError",
        relevant_output="Some pytest error",
        timed_out=False,
    )

    failure_type = classify_failure(summary)

    assert failure_type == FailureType.TEST_FAILURE


def test_classify_default_code_failure():
    """Test default classification for unknown errors."""
    summary = FailureSummary(
        command="python script.py",
        exit_code=1,
        failed_tests=[],
        error_type="UnknownError",
        relevant_output="Some unknown error occurred",
        timed_out=False,
    )

    failure_type = classify_failure(summary)

    assert failure_type == FailureType.CODE_FAILURE


def test_classify_case_insensitive():
    """Test that classification is case-insensitive."""
    summary = FailureSummary(
        command="cat file",
        exit_code=1,
        failed_tests=[],
        error_type="UnknownError",
        relevant_output="PERMISSION DENIED: file",
        timed_out=False,
    )

    failure_type = classify_failure(summary)

    assert failure_type == FailureType.PERMISSION_FAILURE


def test_classify_whitespace_command():
    """Test classification with whitespace in command."""
    summary = FailureSummary(
        command="  pytest tests/  ",
        exit_code=1,
        failed_tests=[],
        error_type="AssertionError",
        relevant_output="Test error",
        timed_out=False,
    )

    failure_type = classify_failure(summary)

    assert failure_type == FailureType.TEST_FAILURE


def test_failure_type_enum_values():
    """Test that FailureType enum has expected values."""
    assert FailureType.CODE_FAILURE == "CODE_FAILURE"
    assert FailureType.TEST_FAILURE == "TEST_FAILURE"
    assert FailureType.ENVIRONMENT_FAILURE == "ENVIRONMENT_FAILURE"
    assert FailureType.PERMISSION_FAILURE == "PERMISSION_FAILURE"
    assert FailureType.TIMEOUT == "TIMEOUT"
    assert FailureType.MODEL_FAILURE == "MODEL_FAILURE"
    assert FailureType.REQUIREMENT_AMBIGUITY == "REQUIREMENT_AMBIGUITY"
    assert FailureType.SCOPE_VIOLATION == "SCOPE_VIOLATION"
    assert FailureType.NO_SOURCE_CHANGES == "NO_SOURCE_CHANGES"
