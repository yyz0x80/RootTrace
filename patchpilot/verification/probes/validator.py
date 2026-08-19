"""Validator for acceptance probe safety and correctness.

This module performs AST-based validation of generated probes to ensure
they only use whitelisted operations and are safe to execute in temporary
directories.
"""

from __future__ import annotations

import ast
from typing import ClassVar


class ProbeValidationError(Exception):
    """Exception raised when probe validation fails."""

    def __init__(self, message: str, node: ast.AST | None = None) -> None:
        """Initialize the validation error.

        Args:
            message: Error message describing the validation failure
            node: Optional AST node that caused the failure
        """
        self.message = message
        self.node = node
        super().__init__(message)


class ProbeValidator:
    """Validate acceptance probes using AST whitelist checking.

    The validator ensures that generated probes:
    - Only use whitelisted Python operations
    - Do not contain dangerous operations (file I/O, network, etc.)
    - Are safe to execute in temporary directories
    - Follow structural constraints
    """

    # No specific AST node types are forbidden - we use function call blacklist instead
    # Python 3.12 removed ast.Exec and ast.Eval, so we rely on call name checking

    # Whitelist of allowed import modules
    ALLOWED_IMPORTS: ClassVar[set[str]] = {
        # Standard library safe modules
        "math",
        "random",
        "datetime",
        "collections",
        "itertools",
        "functools",
        "typing",
        "dataclasses",
        "enum",
        "json",
        "re",
        "string",
        "copy",
    }

    # Blacklist of dangerous function calls
    FORBIDDEN_CALLS: ClassVar[set[str]] = {
        "open",
        "exec",
        "eval",
        "compile",
        "__import__",
        "exit",
        "quit",
        "input",
        # File operations
        "read",
        "write",
        "remove",
        "rmdir",
        "mkdir",
        # Network operations
        "urlopen",
        "requests",
        "socket",
        # System operations
        "system",
        "popen",
        "subprocess",
        "os.system",
        "os.popen",
    }

    def validate(self, code: str) -> list[str]:
        """Validate probe code against blacklist constraints.

        Args:
            code: Python code to validate

        Returns:
            List of validation error messages (empty if valid)

        Raises:
            ProbeValidationError: If code cannot be parsed as valid Python
        """
        errors: list[str] = []

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise ProbeValidationError(f"Syntax error in probe code: {e}")

        # Check for dangerous function calls
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                call_name = self._get_call_name(node)
                if call_name and any(
                    forbidden in call_name for forbidden in self.FORBIDDEN_CALLS
                ):
                    errors.append(
                        f"Forbidden function call: {call_name} at line {node.lineno if hasattr(node, 'lineno') else 'unknown'}"
                    )

            # Check imports - only allow whitelisted modules
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in self.ALLOWED_IMPORTS:
                        errors.append(
                            f"Forbidden import: {alias.name} at line {node.lineno if hasattr(node, 'lineno') else 'unknown'}"
                        )

            if isinstance(node, ast.ImportFrom) and node.module and node.module not in self.ALLOWED_IMPORTS:
                errors.append(
                    f"Forbidden import from: {node.module} at line {node.lineno if hasattr(node, 'lineno') else 'unknown'}"
                )

        return errors

    def _get_call_name(self, node: ast.Call) -> str | None:
        """Extract the name of a function call.

        Args:
            node: AST Call node

        Returns:
            Function name as string, or None if not extractable
        """
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return f"{self._get_attribute_name(node.func)}"
        return None

    def _get_attribute_name(self, node: ast.Attribute) -> str:
        """Recursively build attribute name.

        Args:
            node: AST Attribute node

        Returns:
            Full attribute name as string
        """
        if isinstance(node.value, ast.Name):
            return f"{node.value.id}.{node.attr}"
        elif isinstance(node.value, ast.Attribute):
            return f"{self._get_attribute_name(node.value)}.{node.attr}"
        return node.attr

    def validate_probe(self, probe_code: str) -> bool:
        """Validate a complete probe (all code sections).

        Args:
            probe_code: Complete probe code including setup, steps, and teardown

        Returns:
            True if probe is valid, False otherwise
        """
        errors = self.validate(probe_code)
        return len(errors) == 0
