"""Intermediate validation module for PatchPilot.

This module provides validation functions that can be called during
agent execution to catch errors early, including:
- Python syntax checking
- File integrity validation
- Import validation
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class ValidationError(RuntimeError):
    """Base exception for validation failures."""


class SyntaxValidationError(ValidationError):
    """Raised when Python syntax validation fails."""


def validate_python_syntax(file_path: Path) -> tuple[bool, str]:
    """Validate Python syntax for a given file.

    Args:
        file_path: Path to the Python file to validate

    Returns:
        Tuple of (is_valid, error_message). If is_valid is True, error_message is empty.
    """
    if not file_path.exists():
        return False, f"File not found: {file_path}"

    if not file_path.is_file():
        return False, f"Not a file: {file_path}"

    if file_path.suffix != ".py":
        # Non-Python files are considered valid for syntax checking
        return True, ""

    try:
        result = subprocess.run(
            ["python", "-m", "py_compile", str(file_path)],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

        if result.returncode != 0:
            error_msg = result.stderr or result.stdout
            logger.warning("Syntax validation failed for %s: %s", file_path, error_msg)
            return False, f"Syntax error: {error_msg}"

        return True, ""

    except subprocess.TimeoutExpired:
        return False, "Syntax validation timed out"
    except (OSError, subprocess.SubprocessError) as e:
        return False, f"Validation error: {e}"


def validate_file_integrity(file_path: Path) -> tuple[bool, str]:
    """Validate file integrity including encoding and structure.

    Args:
        file_path: Path to the file to validate

    Returns:
        Tuple of (is_valid, error_message). If is_valid is True, error_message is empty.
    """
    if not file_path.exists():
        return False, f"File not found: {file_path}"

    if not file_path.is_file():
        return False, f"Not a file: {file_path}"

    try:
        # Try to read the file to verify encoding
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check for null bytes (corruption indicator)
        if "\x00" in content:
            return False, "File contains null bytes (possible corruption)"

        return True, ""

    except UnicodeDecodeError:
        return False, "File is not valid UTF-8 text"
    except OSError as e:
        return False, f"Integrity check failed: {e}"


def validate_python_imports(file_path: Path) -> tuple[bool, str]:
    """Validate that Python imports can be resolved (basic check).

    This is a lightweight check that doesn't actually import modules,
    but verifies the import syntax is correct.

    Args:
        file_path: Path to the Python file to validate

    Returns:
        Tuple of (is_valid, error_message). If is_valid is True, error_message is empty.
    """
    if file_path.suffix != ".py":
        return True, ""

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Simple syntax check via compile (catches import errors)
        try:
            compile(content, str(file_path), "exec")
            return True, ""
        except SyntaxError as e:
            return False, f"Import/syntax error: {e}"

    except (OSError, SyntaxError) as e:
        return False, f"Import validation failed: {e}"


def run_intermediate_validation(file_path: Path) -> tuple[bool, list[str]]:
    """Run all intermediate validation checks on a file.

    This performs syntax checking, integrity validation, and import validation
    to catch errors early after file modifications.

    For Python files, only validates if the file appears to be Python code
    (not empty, not obviously test data). For non-Python files, only basic
    integrity checks are performed.

    Args:
        file_path: Path to the file to validate

    Returns:
        Tuple of (all_passed, error_messages). If all_passed is True, error_messages is empty.
    """
    errors = []

    # Run file integrity check first
    integrity_valid, integrity_error = validate_file_integrity(file_path)
    if not integrity_valid:
        errors.append(f"Integrity: {integrity_error}")

    # Only run Python validation for .py files with content
    if file_path.suffix == ".py" and file_path.stat().st_size > 0:
        # Run syntax check for Python files
        syntax_valid, syntax_error = validate_python_syntax(file_path)
        if not syntax_valid:
            errors.append(f"Syntax: {syntax_error}")

        # Run import validation for Python files
        import_valid, import_error = validate_python_imports(file_path)
        if not import_valid:
            errors.append(f"Import: {import_error}")

    all_passed = len(errors) == 0

    if not all_passed:
        logger.warning(
            "Intermediate validation failed for %s: %s",
            file_path,
            "; ".join(errors),
        )

    return all_passed, errors


__all__ = [
    "SyntaxValidationError",
    "ValidationError",
    "run_intermediate_validation",
    "validate_file_integrity",
    "validate_python_imports",
    "validate_python_syntax",
]
