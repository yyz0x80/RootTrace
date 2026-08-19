"""Tests for structural checker functionality."""

import tempfile
from pathlib import Path

import pytest

from patchpilot.verification.structural.ast_checks import (
    ASTChecker,
    CheckResult,
    CheckType,
    StructuralCheck,
)
from patchpilot.verification.structural.runner import (
    StructuralReport,
    StructuralRunner,
)


class TestStructuralCheck:
    """Test structural check definitions."""

    def test_structural_check_creation(self):
        """Test creating a structural check."""
        check = StructuralCheck(
            check_type=CheckType.FUNCTION_EXISTS,
            target="test_function",
            parameters={},
            description="Check if test_function exists",
        )
        assert check.check_type == CheckType.FUNCTION_EXISTS
        assert check.target == "test_function"
        assert check.description == "Check if test_function exists"

    def test_check_result_creation(self):
        """Test creating a check result."""
        check = StructuralCheck(
            check_type=CheckType.FUNCTION_EXISTS,
            target="test_function",
            parameters={},
            description="Check if test_function exists",
        )
        result = CheckResult(
            check=check,
            passed=True,
            message="Function found",
            location="test.py",
        )
        assert result.passed is True
        assert result.message == "Function found"
        assert result.location == "test.py"


class TestASTChecker:
    """Test AST checker functionality."""

    def test_checker_with_valid_file(self):
        """Test checker with a valid Python file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write("def test_function():\n    pass\n")
            f.flush()
            file_path = Path(f.name)

        try:
            checker = ASTChecker(file_path)
            assert checker.tree is not None
        finally:
            file_path.unlink()

    def test_checker_with_invalid_syntax(self):
        """Test checker with invalid Python syntax."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write("def test_function(\n")  # Invalid syntax
            f.flush()
            file_path = Path(f.name)

        try:
            with pytest.raises(SyntaxError):
                ASTChecker(file_path)
        finally:
            file_path.unlink()

    def test_check_function_exists_success(self):
        """Test function existence check when function exists."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write("def test_function():\n    pass\n")
            f.flush()
            file_path = Path(f.name)

        try:
            checker = ASTChecker(file_path)
            result = checker.check_function_exists("test_function")
            assert result.passed is True
            assert "found" in result.message.lower()
        finally:
            file_path.unlink()

    def test_check_function_exists_failure(self):
        """Test function existence check when function doesn't exist."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write("def other_function():\n    pass\n")
            f.flush()
            file_path = Path(f.name)

        try:
            checker = ASTChecker(file_path)
            result = checker.check_function_exists("test_function")
            assert result.passed is False
            assert "not found" in result.message.lower()
        finally:
            file_path.unlink()

    def test_check_signature_preserved_success(self):
        """Test signature preservation check when signature matches."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write("def test_function(a, b, c):\n    pass\n")
            f.flush()
            file_path = Path(f.name)

        try:
            checker = ASTChecker(file_path)
            result = checker.check_signature_preserved(
                "test_function", ["a", "b", "c"]
            )
            assert result.passed is True
        finally:
            file_path.unlink()

    def test_check_signature_preserved_failure(self):
        """Test signature preservation check when signature doesn't match."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write("def test_function(x, y):\n    pass\n")
            f.flush()
            file_path = Path(f.name)

        try:
            checker = ASTChecker(file_path)
            result = checker.check_signature_preserved(
                "test_function", ["a", "b", "c"]
            )
            assert result.passed is False
            assert "changed" in result.message.lower()
        finally:
            file_path.unlink()

    def test_check_call_relationship_success(self):
        """Test call relationship check when call exists."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(
                "def caller():\n    callee()\n\ndef callee():\n    pass\n"
            )
            f.flush()
            file_path = Path(f.name)

        try:
            checker = ASTChecker(file_path)
            result = checker.check_call_relationship("caller", "callee")
            assert result.passed is True
        finally:
            file_path.unlink()

    def test_check_call_relationship_failure(self):
        """Test call relationship check when call doesn't exist."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(
                "def caller():\n    pass\n\ndef callee():\n    pass\n"
            )
            f.flush()
            file_path = Path(f.name)

        try:
            checker = ASTChecker(file_path)
            result = checker.check_call_relationship("caller", "callee")
            assert result.passed is False
            assert "does not call" in result.message.lower()
        finally:
            file_path.unlink()

    def test_check_no_new_imports_success(self):
        """Test no new imports check when only allowed imports present."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write("import math\nimport random\n")
            f.flush()
            file_path = Path(f.name)

        try:
            checker = ASTChecker(file_path)
            result = checker.check_no_new_imports({"math", "random"})
            assert result.passed is True
        finally:
            file_path.unlink()

    def test_check_no_new_imports_failure(self):
        """Test no new imports check when new imports present."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write("import math\nimport os\n")
            f.flush()
            file_path = Path(f.name)

        try:
            checker = ASTChecker(file_path)
            result = checker.check_no_new_imports({"math"})
            assert result.passed is False
            assert "new imports" in result.message.lower()
        finally:
            file_path.unlink()

    def test_check_method_exists_success(self):
        """Test method existence check when method exists."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(
                "class TestClass:\n    def test_method(self):\n        pass\n"
            )
            f.flush()
            file_path = Path(f.name)

        try:
            checker = ASTChecker(file_path)
            result = checker.check_method_exists("TestClass", "test_method")
            assert result.passed is True
        finally:
            file_path.unlink()

    def test_check_method_exists_failure(self):
        """Test method existence check when method doesn't exist."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write("class TestClass:\n    pass\n")
            f.flush()
            file_path = Path(f.name)

        try:
            checker = ASTChecker(file_path)
            result = checker.check_method_exists("TestClass", "test_method")
            assert result.passed is False
            assert "not found" in result.message.lower()
        finally:
            file_path.unlink()

    def test_check_decorator_exists_success(self):
        """Test decorator existence check when decorator exists."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(
                "@property\ndef test_function():\n    pass\n"
            )
            f.flush()
            file_path = Path(f.name)

        try:
            checker = ASTChecker(file_path)
            result = checker.check_decorator_exists(
                "test_function", "property"
            )
            assert result.passed is True
        finally:
            file_path.unlink()

    def test_check_decorator_exists_failure(self):
        """Test decorator existence check when decorator doesn't exist."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write("def test_function():\n    pass\n")
            f.flush()
            file_path = Path(f.name)

        try:
            checker = ASTChecker(file_path)
            result = checker.check_decorator_exists(
                "test_function", "property"
            )
            assert result.passed is False
            assert "does not have" in result.message.lower()
        finally:
            file_path.unlink()


class TestStructuralRunner:
    """Test structural runner functionality."""

    def test_runner_initialization(self):
        """Test runner initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = StructuralRunner(Path(temp_dir))
            assert runner.workspace_root == Path(temp_dir)

    def test_run_checks_with_valid_file(self):
        """Test running checks on a valid file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test.py"
            file_path.write_text("def test_function():\n    pass\n")

            runner = StructuralRunner(Path(temp_dir))
            checks = [
                StructuralCheck(
                    check_type=CheckType.FUNCTION_EXISTS,
                    target="test_function",
                    parameters={},
                    description="Check if test_function exists",
                )
            ]

            report = runner.run_checks(checks, file_path)
            assert report.total_checks == 1
            assert report.passed_checks == 1
            assert report.failed_checks == 0
            assert report.passed is True

    def test_run_checks_with_missing_file(self):
        """Test running checks on a missing file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "nonexistent.py"

            runner = StructuralRunner(Path(temp_dir))
            checks = [
                StructuralCheck(
                    check_type=CheckType.FUNCTION_EXISTS,
                    target="test_function",
                    parameters={},
                    description="Check if test_function exists",
                )
            ]

            report = runner.run_checks(checks, file_path)
            assert report.total_checks == 1
            assert report.failed_checks == 1
            assert report.passed is False

    def test_run_checks_with_mixed_results(self):
        """Test running checks with mixed pass/fail results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test.py"
            file_path.write_text("def test_function():\n    pass\n")

            runner = StructuralRunner(Path(temp_dir))
            checks = [
                StructuralCheck(
                    check_type=CheckType.FUNCTION_EXISTS,
                    target="test_function",
                    parameters={},
                    description="Check if test_function exists",
                ),
                StructuralCheck(
                    check_type=CheckType.FUNCTION_EXISTS,
                    target="nonexistent_function",
                    parameters={},
                    description="Check if nonexistent_function exists",
                ),
            ]

            report = runner.run_checks(checks, file_path)
            assert report.total_checks == 2
            assert report.passed_checks == 1
            assert report.failed_checks == 1
            assert report.passed is False

    def test_create_function_exists_check(self):
        """Test creating a function exists check."""
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = StructuralRunner(Path(temp_dir))
            check = runner.create_function_exists_check("test_function")
            assert check.check_type == CheckType.FUNCTION_EXISTS
            assert check.target == "test_function"

    def test_create_signature_preserved_check(self):
        """Test creating a signature preserved check."""
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = StructuralRunner(Path(temp_dir))
            check = runner.create_signature_preserved_check(
                "test_function", ["a", "b"]
            )
            assert check.check_type == CheckType.SIGNATURE_PRESERVED
            assert check.target == "test_function"
            assert check.parameters["expected_params"] == ["a", "b"]

    def test_create_call_relationship_check(self):
        """Test creating a call relationship check."""
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = StructuralRunner(Path(temp_dir))
            check = runner.create_call_relationship_check(
                "caller", "callee"
            )
            assert check.check_type == CheckType.CALL_RELATIONSHIP
            assert check.target == "caller"
            assert check.parameters["callee"] == "callee"

    def test_create_no_new_imports_check(self):
        """Test creating a no new imports check."""
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = StructuralRunner(Path(temp_dir))
            check = runner.create_no_new_imports_check({"math", "random"})
            assert check.check_type == CheckType.NO_NEW_IMPORTS
            assert check.parameters["allowed_imports"] == ["math", "random"]

    def test_create_method_exists_check(self):
        """Test creating a method exists check."""
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = StructuralRunner(Path(temp_dir))
            check = runner.create_method_exists_check("TestClass", "test_method")
            assert check.check_type == CheckType.METHOD_EXISTS
            assert check.target == "TestClass.test_method"

    def test_create_decorator_exists_check(self):
        """Test creating a decorator exists check."""
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = StructuralRunner(Path(temp_dir))
            check = runner.create_decorator_exists_check(
                "test_function", "property"
            )
            assert check.check_type == CheckType.DECORATOR_EXISTS
            assert check.target == "test_function"


class TestStructuralReport:
    """Test structural report functionality."""

    def test_report_initialization(self):
        """Test report initialization."""
        report = StructuralReport()
        assert report.total_checks == 0
        assert report.passed_checks == 0
        assert report.failed_checks == 0
        assert report.passed is True

    def test_add_result_pass(self):
        """Test adding a passing result."""
        report = StructuralReport()
        check = StructuralCheck(
            check_type=CheckType.FUNCTION_EXISTS,
            target="test_function",
            parameters={},
            description="Check if test_function exists",
        )
        result = CheckResult(
            check=check,
            passed=True,
            message="Function found",
            location="test.py",
        )

        report.add_result(result)
        assert report.total_checks == 1
        assert report.passed_checks == 1
        assert report.failed_checks == 0
        assert report.passed is True

    def test_add_result_fail(self):
        """Test adding a failing result."""
        report = StructuralReport()
        check = StructuralCheck(
            check_type=CheckType.FUNCTION_EXISTS,
            target="test_function",
            parameters={},
            description="Check if test_function exists",
        )
        result = CheckResult(
            check=check,
            passed=False,
            message="Function not found",
            location="test.py",
        )

        report.add_result(result)
        assert report.total_checks == 1
        assert report.passed_checks == 0
        assert report.failed_checks == 1
        assert report.passed is False

    def test_report_to_dict(self):
        """Test converting report to dictionary."""
        report = StructuralReport()
        report_dict = report.to_dict()
        assert isinstance(report_dict, dict)
        assert "total_checks" in report_dict
        assert "passed_checks" in report_dict
        assert "failed_checks" in report_dict


class TestCheckType:
    """Test check type enumeration."""

    def test_check_type_values(self):
        """Test that check type has expected values."""
        assert CheckType.FUNCTION_EXISTS == "function_exists"
        assert CheckType.SIGNATURE_PRESERVED == "signature_preserved"
        assert CheckType.CALL_RELATIONSHIP == "call_relationship"
        assert CheckType.NO_NEW_IMPORTS == "no_new_imports"
        assert CheckType.METHOD_EXISTS == "method_exists"
        assert CheckType.DECORATOR_EXISTS == "decorator_exists"

    def test_check_type_is_string_enum(self):
        """Test that check type is a string enum."""
        assert isinstance(CheckType.FUNCTION_EXISTS, str)
