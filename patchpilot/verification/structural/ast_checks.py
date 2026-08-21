"""AST-based structural checks for code validation.

This module provides structural verification using AST analysis to ensure
code changes meet structural requirements without executing the code.

Supported checks:
- Function or method existence
- Signature preservation
- Call relationship verification
- Import restriction enforcement
- Class method/decorator existence
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


class CheckType(StrEnum):
    """Classification of structural check types."""

    FUNCTION_EXISTS = "function_exists"
    SIGNATURE_PRESERVED = "signature_preserved"
    CALL_RELATIONSHIP = "call_relationship"
    NO_NEW_IMPORTS = "no_new_imports"
    METHOD_EXISTS = "method_exists"
    DECORATOR_EXISTS = "decorator_exists"
    DATACLASS_FIELD = "dataclass_field"
    METHOD_PARAMETER = "method_parameter"


@dataclass
class StructuralCheck:
    """Single structural check to perform.

    Attributes:
        check_type: Type of structural check
        target: Target element (function name, class name, etc.)
        parameters: Additional parameters for the check
        description: Human-readable description of the check
    """

    check_type: CheckType
    target: str
    parameters: dict[str, Any]
    description: str


@dataclass
class CheckResult:
    """Result of a structural check.

    Attributes:
        check: The structural check that was performed
        passed: Whether the check passed
        message: Detailed message about the result
        location: File location where the check was performed
    """

    check: StructuralCheck
    passed: bool
    message: str
    location: str


class ASTChecker:
    """Perform AST-based structural checks on Python code.

    The checker analyzes code structure without execution to verify:
    - Function and method existence
    - Signature preservation
    - Call relationships
    - Import restrictions
    - Decorator presence
    """

    def __init__(self, file_path: Path) -> None:
        """Initialize the AST checker.

        Args:
            file_path: Path to the Python file to check
        """
        self.file_path = file_path
        self.tree = self._parse_file()

    def _parse_file(self) -> ast.Module:
        """Parse the Python file into an AST.

        Returns:
            AST module node

        Raises:
            SyntaxError: If the file contains invalid Python syntax
        """
        code = self.file_path.read_text(encoding="utf-8")
        return ast.parse(code, filename=str(self.file_path))

    def check_function_exists(self, function_name: str) -> CheckResult:
        """Check if a function exists in the file.

        Args:
            function_name: Name of the function to check

        Returns:
            CheckResult indicating if the function exists
        """
        check = StructuralCheck(
            check_type=CheckType.FUNCTION_EXISTS,
            target=function_name,
            parameters={},
            description=f"Check if function '{function_name}' exists",
        )

        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                return CheckResult(
                    check=check,
                    passed=True,
                    message=f"Function '{function_name}' found at line {node.lineno}",
                    location=str(self.file_path),
                )

        return CheckResult(
            check=check,
            passed=False,
            message=f"Function '{function_name}' not found",
            location=str(self.file_path),
        )

    def check_signature_preserved(
        self,
        function_name: str,
        expected_params: list[str],
    ) -> CheckResult:
        """Check if a function's signature is preserved.

        Args:
            function_name: Name of the function to check
            expected_params: Expected parameter names

        Returns:
            CheckResult indicating if signature is preserved
        """
        check = StructuralCheck(
            check_type=CheckType.SIGNATURE_PRESERVED,
            target=function_name,
            parameters={"expected_params": expected_params},
            description=f"Check if function '{function_name}' signature is preserved",
        )

        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                actual_params = [arg.arg for arg in node.args.args]

                if actual_params == expected_params:
                    return CheckResult(
                        check=check,
                        passed=True,
                        message=f"Function '{function_name}' signature preserved: {actual_params}",
                        location=str(self.file_path),
                    )
                else:
                    return CheckResult(
                        check=check,
                        passed=False,
                        message=f"Function '{function_name}' signature changed: expected {expected_params}, got {actual_params}",
                        location=str(self.file_path),
                    )

        return CheckResult(
            check=check,
            passed=False,
            message=f"Function '{function_name}' not found for signature check",
            location=str(self.file_path),
        )

    def check_call_relationship(
        self,
        caller_function: str,
        callee_function: str,
    ) -> CheckResult:
        """Check if a function calls another function.

        Args:
            caller_function: Name of the calling function
            callee_function: Name of the function being called

        Returns:
            CheckResult indicating if call relationship exists
        """
        check = StructuralCheck(
            check_type=CheckType.CALL_RELATIONSHIP,
            target=caller_function,
            parameters={"callee": callee_function},
            description=f"Check if '{caller_function}' calls '{callee_function}'",
        )

        # Find the caller function
        caller_node = None
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and node.name == caller_function:
                caller_node = node
                break

        if not caller_node:
            return CheckResult(
                check=check,
                passed=False,
                message=f"Caller function '{caller_function}' not found",
                location=str(self.file_path),
            )

        # Check if it calls the callee
        for node in ast.walk(caller_node):
            if isinstance(node, ast.Call):
                call_name = self._get_call_name(node)
                if call_name == callee_function:
                    return CheckResult(
                        check=check,
                        passed=True,
                        message=f"Function '{caller_function}' calls '{callee_function}'",
                        location=str(self.file_path),
                    )

        return CheckResult(
            check=check,
            passed=False,
            message=f"Function '{caller_function}' does not call '{callee_function}'",
            location=str(self.file_path),
        )

    def check_no_new_imports(
        self,
        allowed_imports: set[str],
    ) -> CheckResult:
        """Check that no new imports beyond allowed set are present.

        Args:
            allowed_imports: Set of allowed import module names

        Returns:
            CheckResult indicating if only allowed imports are present
        """
        check = StructuralCheck(
            check_type=CheckType.NO_NEW_IMPORTS,
            target="imports",
            parameters={"allowed_imports": list(allowed_imports)},
            description="Check that no new imports are added",
        )

        found_imports: set[str] = set()

        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    found_imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                found_imports.add(node.module)

        new_imports = found_imports - allowed_imports

        if not new_imports:
            return CheckResult(
                check=check,
                passed=True,
                message=f"All imports are allowed: {found_imports}",
                location=str(self.file_path),
            )

        return CheckResult(
            check=check,
            passed=False,
            message=f"Found new imports not in allowed set: {new_imports}",
            location=str(self.file_path),
        )

    def check_method_exists(
        self,
        class_name: str,
        method_name: str,
    ) -> CheckResult:
        """Check if a method exists in a class.

        Args:
            class_name: Name of the class
            method_name: Name of the method

        Returns:
            CheckResult indicating if method exists
        """
        check = StructuralCheck(
            check_type=CheckType.METHOD_EXISTS,
            target=f"{class_name}.{method_name}",
            parameters={"class": class_name, "method": method_name},
            description=f"Check if method '{method_name}' exists in class '{class_name}'",
        )

        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == method_name:
                        return CheckResult(
                            check=check,
                            passed=True,
                            message=f"Method '{method_name}' found in class '{class_name}' at line {item.lineno}",
                            location=str(self.file_path),
                        )

        return CheckResult(
            check=check,
            passed=False,
            message=f"Method '{method_name}' not found in class '{class_name}'",
            location=str(self.file_path),
        )

    def check_decorator_exists(
        self,
        function_name: str,
        decorator_name: str,
    ) -> CheckResult:
        """Check if a function has a specific decorator.

        Args:
            function_name: Name of the function
            decorator_name: Name of the decorator

        Returns:
            CheckResult indicating if decorator exists
        """
        check = StructuralCheck(
            check_type=CheckType.DECORATOR_EXISTS,
            target=function_name,
            parameters={"decorator": decorator_name},
            description=f"Check if function '{function_name}' has decorator '{decorator_name}'",
        )

        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                for decorator in node.decorator_list:
                    decorator_str = self._get_decorator_name(decorator)
                    if decorator_str == decorator_name:
                        return CheckResult(
                            check=check,
                            passed=True,
                            message=f"Function '{function_name}' has decorator '{decorator_name}'",
                            location=str(self.file_path),
                        )

        return CheckResult(
            check=check,
            passed=False,
            message=f"Function '{function_name}' does not have decorator '{decorator_name}'",
            location=str(self.file_path),
        )

    def check_dataclass_field(
        self,
        class_name: str,
        field_name: str,
        annotation: str = "",
        expected_default: Any = None,
        require_default: bool = False,
    ) -> CheckResult:
        """Check a dataclass field annotation and optional default value."""
        check = StructuralCheck(
            check_type=CheckType.DATACLASS_FIELD,
            target=class_name,
            parameters={
                "field": field_name,
                "annotation": annotation,
                "expected_default": expected_default,
                "require_default": require_default,
            },
            description=f"Check dataclass field '{class_name}.{field_name}'",
        )
        for node in self.tree.body:
            if not isinstance(node, ast.ClassDef) or node.name != class_name:
                continue
            decorators = {self._get_decorator_name(item) for item in node.decorator_list}
            if "dataclass" not in decorators:
                return CheckResult(
                    check=check,
                    passed=False,
                    message=f"Class '{class_name}' is not a dataclass",
                    location=str(self.file_path),
                )
            for item in node.body:
                if not isinstance(item, ast.AnnAssign):
                    continue
                if not isinstance(item.target, ast.Name) or item.target.id != field_name:
                    continue
                actual_annotation = ast.unparse(item.annotation)
                if annotation and actual_annotation != annotation:
                    return CheckResult(
                        check=check,
                        passed=False,
                        message=(
                            f"Field '{field_name}' annotation is {actual_annotation}, "
                            f"expected {annotation}"
                        ),
                        location=str(self.file_path),
                    )
                if require_default:
                    if item.value is None:
                        return CheckResult(
                            check=check,
                            passed=False,
                            message=f"Field '{field_name}' has no default",
                            location=str(self.file_path),
                        )
                    try:
                        actual_default = ast.literal_eval(item.value)
                    except (ValueError, TypeError):
                        actual_default = object()
                    if actual_default != expected_default:
                        return CheckResult(
                            check=check,
                            passed=False,
                            message=(
                                f"Field '{field_name}' default does not match "
                                f"{expected_default!r}"
                            ),
                            location=str(self.file_path),
                        )
                return CheckResult(
                    check=check,
                    passed=True,
                    message=f"Dataclass field '{class_name}.{field_name}' matches",
                    location=str(self.file_path),
                )
        return CheckResult(
            check=check,
            passed=False,
            message=f"Dataclass field '{class_name}.{field_name}' not found",
            location=str(self.file_path),
        )

    def check_method_parameter(
        self,
        class_name: str,
        method_name: str,
        parameter_name: str,
        annotation: str = "",
        expected_default: Any = None,
        require_default: bool = False,
    ) -> CheckResult:
        """Check a method parameter annotation and optional default value."""
        check = StructuralCheck(
            check_type=CheckType.METHOD_PARAMETER,
            target=f"{class_name}.{method_name}",
            parameters={
                "class": class_name,
                "method": method_name,
                "parameter": parameter_name,
                "annotation": annotation,
                "expected_default": expected_default,
                "require_default": require_default,
            },
            description=f"Check method parameter '{class_name}.{method_name}'",
        )
        method = self._find_method(class_name, method_name)
        if method is None:
            return CheckResult(
                check=check,
                passed=False,
                message=f"Method '{class_name}.{method_name}' not found",
                location=str(self.file_path),
            )
        arguments = list(method.args.args)
        defaults: list[ast.expr | None] = [None] * (
            len(arguments) - len(method.args.defaults)
        ) + list(method.args.defaults)
        for argument, default in zip(arguments, defaults, strict=True):
            if argument.arg != parameter_name:
                continue
            actual_annotation = ast.unparse(argument.annotation) if argument.annotation else ""
            if annotation and actual_annotation != annotation:
                return CheckResult(
                    check=check,
                    passed=False,
                    message=(
                        f"Parameter '{parameter_name}' annotation is "
                        f"{actual_annotation or 'missing'}, expected {annotation}"
                    ),
                    location=str(self.file_path),
                )
            if require_default:
                try:
                    actual_default = ast.literal_eval(default) if default is not None else object()
                except (ValueError, TypeError):
                    actual_default = object()
                if actual_default != expected_default:
                    return CheckResult(
                        check=check,
                        passed=False,
                        message=(
                            f"Parameter '{parameter_name}' default does not match "
                            f"{expected_default!r}"
                        ),
                        location=str(self.file_path),
                    )
            return CheckResult(
                check=check,
                passed=True,
                message=f"Method parameter '{class_name}.{method_name}.{parameter_name}' matches",
                location=str(self.file_path),
            )
        return CheckResult(
            check=check,
            passed=False,
            message=f"Parameter '{parameter_name}' not found",
            location=str(self.file_path),
        )

    def _find_method(
        self,
        class_name: str,
        method_name: str,
    ) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
        """Find a method directly declared by a class."""
        for node in self.tree.body:
            if not isinstance(node, ast.ClassDef) or node.name != class_name:
                continue
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == method_name:
                    return item
        return None

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
            return self._get_attribute_name(node.func)
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

    def _get_decorator_name(self, node: ast.expr) -> str:
        """Extract decorator name from AST node.

        Args:
            node: AST expression node representing a decorator

        Returns:
            Decorator name as string
        """
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return self._get_attribute_name(node)
        elif isinstance(node, ast.Call):
            return self._get_decorator_name(node.func)
        return str(node)
