"""Runner for executing structural checks.

This module handles the execution of structural checks across files,
aggregating results and providing comprehensive verification reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from patchpilot.verification.structural.ast_checks import (
    ASTChecker,
    CheckResult,
    CheckType,
    StructuralCheck,
)


@dataclass
class StructuralReport:
    """Report for structural check execution.

    Attributes:
        checks: List of structural checks performed
        results: List of check results
        passed: Overall pass status
        total_checks: Total number of checks performed
        passed_checks: Number of checks that passed
        failed_checks: Number of checks that failed
    """

    checks: list[StructuralCheck] = field(default_factory=list)
    results: list[CheckResult] = field(default_factory=list)
    passed: bool = True
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0

    def add_result(self, result: CheckResult) -> None:
        """Add a check result to the report.

        Args:
            result: CheckResult to add
        """
        self.results.append(result)
        self.checks.append(result.check)
        self.total_checks += 1

        if result.passed:
            self.passed_checks += 1
        else:
            self.failed_checks += 1
            self.passed = False

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary for serialization.

        Returns:
            Dictionary representation of the report
        """
        from dataclasses import asdict

        return asdict(self)


class StructuralRunner:
    """Execute structural checks across files.

    The runner performs AST-based structural verification to ensure
    code changes meet structural requirements without execution.
    """

    def __init__(self, workspace_root: Path) -> None:
        """Initialize the structural runner.

        Args:
            workspace_root: Root directory of the target workspace
        """
        self.workspace_root = workspace_root

    def run_checks(
        self,
        checks: list[StructuralCheck],
        file_path: Path,
    ) -> StructuralReport:
        """Run structural checks on a single file.

        Args:
            checks: List of StructuralCheck to perform
            file_path: Path to the file to check

        Returns:
            StructuralReport with all check results
        """
        report = StructuralReport()

        if not file_path.exists():
            # Create failure results for all checks if file doesn't exist
            for check in checks:
                result = CheckResult(
                    check=check,
                    passed=False,
                    message=f"File not found: {file_path}",
                    location=str(file_path),
                )
                report.add_result(result)
            return report

        checker = ASTChecker(file_path)

        for check in checks:
            result = self._execute_check(check, checker)
            report.add_result(result)

        return report

    def _execute_check(
        self,
        check: StructuralCheck,
        checker: ASTChecker,
    ) -> CheckResult:
        """Execute a single structural check.

        Args:
            check: StructuralCheck to execute
            checker: ASTChecker instance for the file

        Returns:
            CheckResult with execution information
        """
        try:
            match check.check_type:
                case CheckType.FUNCTION_EXISTS:
                    return checker.check_function_exists(check.target)
                case CheckType.SIGNATURE_PRESERVED:
                    return checker.check_signature_preserved(
                        check.target,
                        check.parameters.get("expected_params", []),
                    )
                case CheckType.CALL_RELATIONSHIP:
                    return checker.check_call_relationship(
                        check.target,
                        check.parameters.get("callee", ""),
                    )
                case CheckType.NO_NEW_IMPORTS:
                    return checker.check_no_new_imports(
                        set(check.parameters.get("allowed_imports", [])),
                    )
                case CheckType.METHOD_EXISTS:
                    return checker.check_method_exists(
                        check.parameters.get("class", ""),
                        check.parameters.get("method", ""),
                    )
                case CheckType.DECORATOR_EXISTS:
                    return checker.check_decorator_exists(
                        check.target,
                        check.parameters.get("decorator", ""),
                    )
                case _:
                    return CheckResult(
                        check=check,
                        passed=False,
                        message=f"Unknown check type: {check.check_type}",
                        location=str(checker.file_path),
                    )
        except Exception as e:  # noqa: BLE001 - Catch all exceptions for check execution
            return CheckResult(
                check=check,
                passed=False,
                message=f"Check execution failed: {e!s}",
                location=str(checker.file_path),
            )

    def run_baseline_checks(
        self,
        checks: list[StructuralCheck],
        file_path: Path,
    ) -> StructuralReport:
        """Run structural checks in baseline phase (before changes).

        Args:
            checks: List of StructuralCheck to perform
            file_path: Path to the file to check

        Returns:
            StructuralReport from baseline execution
        """
        return self.run_checks(checks, file_path)

    def run_post_patch_checks(
        self,
        checks: list[StructuralCheck],
        file_path: Path,
    ) -> StructuralReport:
        """Run structural checks in post-patch phase (after changes).

        Args:
            checks: List of StructuralCheck to perform
            file_path: Path to the file to check

        Returns:
            StructuralReport from post-patch execution
        """
        return self.run_checks(checks, file_path)

    def create_function_exists_check(
        self,
        function_name: str,
    ) -> StructuralCheck:
        """Create a function existence check.

        Args:
            function_name: Name of the function to check

        Returns:
            StructuralCheck for function existence
        """
        return StructuralCheck(
            check_type=CheckType.FUNCTION_EXISTS,
            target=function_name,
            parameters={},
            description=f"Check if function '{function_name}' exists",
        )

    def create_signature_preserved_check(
        self,
        function_name: str,
        expected_params: list[str],
    ) -> StructuralCheck:
        """Create a signature preservation check.

        Args:
            function_name: Name of the function to check
            expected_params: Expected parameter names

        Returns:
            StructuralCheck for signature preservation
        """
        return StructuralCheck(
            check_type=CheckType.SIGNATURE_PRESERVED,
            target=function_name,
            parameters={"expected_params": expected_params},
            description=f"Check if function '{function_name}' signature is preserved",
        )

    def create_call_relationship_check(
        self,
        caller_function: str,
        callee_function: str,
    ) -> StructuralCheck:
        """Create a call relationship check.

        Args:
            caller_function: Name of the calling function
            callee_function: Name of the function being called

        Returns:
            StructuralCheck for call relationship
        """
        return StructuralCheck(
            check_type=CheckType.CALL_RELATIONSHIP,
            target=caller_function,
            parameters={"callee": callee_function},
            description=f"Check if '{caller_function}' calls '{callee_function}'",
        )

    def create_no_new_imports_check(
        self,
        allowed_imports: set[str],
    ) -> StructuralCheck:
        """Create a no-new-imports check.

        Args:
            allowed_imports: Set of allowed import module names

        Returns:
            StructuralCheck for import restriction
        """
        return StructuralCheck(
            check_type=CheckType.NO_NEW_IMPORTS,
            target="imports",
            parameters={"allowed_imports": list(allowed_imports)},
            description="Check that no new imports are added",
        )

    def create_method_exists_check(
        self,
        class_name: str,
        method_name: str,
    ) -> StructuralCheck:
        """Create a method existence check.

        Args:
            class_name: Name of the class
            method_name: Name of the method

        Returns:
            StructuralCheck for method existence
        """
        return StructuralCheck(
            check_type=CheckType.METHOD_EXISTS,
            target=f"{class_name}.{method_name}",
            parameters={"class": class_name, "method": method_name},
            description=f"Check if method '{method_name}' exists in class '{class_name}'",
        )

    def create_decorator_exists_check(
        self,
        function_name: str,
        decorator_name: str,
    ) -> StructuralCheck:
        """Create a decorator existence check.

        Args:
            function_name: Name of the function
            decorator_name: Name of the decorator

        Returns:
            StructuralCheck for decorator existence
        """
        return StructuralCheck(
            check_type=CheckType.DECORATOR_EXISTS,
            target=function_name,
            parameters={"decorator": decorator_name},
            description=f"Check if function '{function_name}' has decorator '{decorator_name}'",
        )
