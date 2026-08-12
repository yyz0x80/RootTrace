"""Tests for the validation module."""

import tempfile
from pathlib import Path

import pytest

from patchpilot.validation import (
    SyntaxValidationError,
    ValidationError,
    run_intermediate_validation,
    validate_file_integrity,
    validate_python_imports,
    validate_python_syntax,
)


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for testing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestValidatePythonSyntax:
    """Tests for validate_python_syntax function."""

    def test_valid_python_syntax(self, temp_workspace):
        """Test validation of valid Python file"""
        test_file = temp_workspace / "valid.py"
        test_file.write_text("def hello():\n    return 'world'\n")

        is_valid, error = validate_python_syntax(test_file)
        assert is_valid
        assert error == ""

    def test_invalid_python_syntax(self, temp_workspace):
        """Test validation of invalid Python file"""
        test_file = temp_workspace / "invalid.py"
        test_file.write_text("def hello(\n    return 'world'\n")

        is_valid, error = validate_python_syntax(test_file)
        assert not is_valid
        assert "Syntax error" in error

    def test_non_python_file(self, temp_workspace):
        """Test validation of non-Python file"""
        test_file = temp_workspace / "test.txt"
        test_file.write_text("Some text content")

        is_valid, error = validate_python_syntax(test_file)
        assert is_valid
        assert error == ""

    def test_nonexistent_file(self, temp_workspace):
        """Test validation of nonexistent file"""
        test_file = temp_workspace / "nonexistent.py"

        is_valid, error = validate_python_syntax(test_file)
        assert not is_valid
        assert "not found" in error


class TestValidateFileIntegrity:
    """Tests for validate_file_integrity function."""

    def test_valid_file(self, temp_workspace):
        """Test validation of valid file"""
        test_file = temp_workspace / "test.txt"
        test_file.write_text("Valid content")

        is_valid, error = validate_file_integrity(test_file)
        assert is_valid
        assert error == ""

    def test_empty_python_file(self, temp_workspace):
        """Test validation of empty Python file"""
        test_file = temp_workspace / "empty.py"
        test_file.write_text("")

        # Empty files are now considered valid for integrity check
        # The syntax validation will handle them appropriately
        is_valid, error = validate_file_integrity(test_file)
        assert is_valid
        assert error == ""

    def test_corrupted_file(self, temp_workspace):
        """Test validation of corrupted file with null bytes"""
        test_file = temp_workspace / "corrupted.txt"
        test_file.write_bytes(b"Valid\x00content")

        is_valid, error = validate_file_integrity(test_file)
        assert not is_valid
        assert "null bytes" in error

    def test_nonexistent_file_integrity(self, temp_workspace):
        """Test integrity check of nonexistent file"""
        test_file = temp_workspace / "nonexistent.txt"

        is_valid, error = validate_file_integrity(test_file)
        assert not is_valid
        assert "not found" in error


class TestValidatePythonImports:
    """Tests for validate_python_imports function."""

    def test_valid_imports(self, temp_workspace):
        """Test validation of valid Python imports"""
        test_file = temp_workspace / "valid_imports.py"
        test_file.write_text("import os\nfrom pathlib import Path\n\ndef hello():\n    pass\n")

        is_valid, error = validate_python_imports(test_file)
        assert is_valid
        assert error == ""

    def test_invalid_import_syntax(self, temp_workspace):
        """Test validation of invalid import syntax"""
        test_file = temp_workspace / "invalid_imports.py"
        test_file.write_text("import os\nfrom\n")

        is_valid, error = validate_python_imports(test_file)
        assert not is_valid
        assert "error" in error

    def test_non_python_file_imports(self, temp_workspace):
        """Test import validation of non-Python file"""
        test_file = temp_workspace / "test.txt"
        test_file.write_text("Some text")

        is_valid, error = validate_python_imports(test_file)
        assert is_valid
        assert error == ""


class TestRunIntermediateValidation:
    """Tests for run_intermediate_validation function."""

    def test_all_validations_pass(self, temp_workspace):
        """Test when all validations pass"""
        test_file = temp_workspace / "valid.py"
        test_file.write_text("def hello():\n    return 'world'\n")

        all_passed, errors = run_intermediate_validation(test_file)
        assert all_passed
        assert len(errors) == 0

    def test_syntax_validation_fails(self, temp_workspace):
        """Test when syntax validation fails"""
        test_file = temp_workspace / "invalid.py"
        test_file.write_text("def hello(\n    return 'world'\n")

        all_passed, errors = run_intermediate_validation(test_file)
        assert not all_passed
        assert len(errors) > 0
        assert any("Syntax" in error for error in errors)

    def test_integrity_validation_fails(self, temp_workspace):
        """Test when integrity validation fails"""
        test_file = temp_workspace / "corrupted.py"
        test_file.write_bytes(b"Valid\x00content")

        all_passed, errors = run_intermediate_validation(test_file)
        assert not all_passed
        assert len(errors) > 0
        assert any("Integrity" in error for error in errors)

    def test_multiple_validation_failures(self, temp_workspace):
        """Test when multiple validations fail"""
        test_file = temp_workspace / "multiple_issues.py"
        test_file.write_bytes(b"def hello(\n\x00return 'world'\n")

        all_passed, errors = run_intermediate_validation(test_file)
        assert not all_passed
        assert len(errors) >= 2

    def test_empty_python_file_skips_syntax_validation(self, temp_workspace):
        """Test that empty Python files skip syntax validation"""
        test_file = temp_workspace / "empty.py"
        test_file.write_text("")

        all_passed, errors = run_intermediate_validation(test_file)
        assert all_passed
        assert len(errors) == 0
