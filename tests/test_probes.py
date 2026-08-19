"""Tests for acceptance probe functionality."""

import pytest

from patchpilot.verification.probes.schema import (
    AcceptanceProbe,
    ProbeExecutionResult,
    ProbeStep,
    ProbeType,
    StepResult,
)
from patchpilot.verification.probes.validator import (
    ProbeValidationError,
    ProbeValidator,
)


class TestProbeSchema:
    """Test probe schema definitions."""

    def test_probe_step_creation(self):
        """Test creating a probe step."""
        step = ProbeStep(
            description="Test step",
            code="x = 1 + 1",
            expected_outcome="no_exception",
        )
        assert step.description == "Test step"
        assert step.code == "x = 1 + 1"
        assert step.expected_outcome == "no_exception"
        assert step.tolerance is None

    def test_probe_step_with_tolerance(self):
        """Test creating a probe step with tolerance."""
        step = ProbeStep(
            description="Numeric comparison",
            code="result = 0.1 + 0.2",
            expected_outcome="approximate",
            tolerance=0.001,
        )
        assert step.tolerance == 0.001

    def test_acceptance_probe_creation(self):
        """Test creating an acceptance probe."""
        probe = AcceptanceProbe(
            id="test-probe-1",
            name="Test Probe",
            description="A test probe",
            probe_type=ProbeType.FUNCTION_IO,
            target_function="test_function",
            steps=[
                ProbeStep(
                    description="Step 1",
                    code="x = 1",
                    expected_outcome="no_exception",
                )
            ],
        )
        assert probe.id == "test-probe-1"
        assert probe.name == "Test Probe"
        assert probe.probe_type == ProbeType.FUNCTION_IO
        assert len(probe.steps) == 1

    def test_probe_to_dict(self):
        """Test converting probe to dictionary."""
        probe = AcceptanceProbe(
            id="test-probe-1",
            name="Test Probe",
            description="A test probe",
            probe_type=ProbeType.FUNCTION_IO,
            target_function="test_function",
            steps=[
                ProbeStep(
                    description="Step 1",
                    code="x = 1",
                    expected_outcome="no_exception",
                )
            ],
        )
        probe_dict = probe.to_dict()
        assert isinstance(probe_dict, dict)
        assert probe_dict["id"] == "test-probe-1"
        assert probe_dict["name"] == "Test Probe"

    def test_step_result_creation(self):
        """Test creating a step result."""
        result = StepResult(
            step_index=0,
            description="Test step",
            passed=True,
            expected_outcome="no_exception",
            actual_outcome="no_exception",
        )
        assert result.step_index == 0
        assert result.passed is True
        assert result.error is None

    def test_step_result_failure(self):
        """Test creating a failed step result."""
        result = StepResult(
            step_index=0,
            description="Test step",
            passed=False,
            expected_outcome="no_exception",
            actual_outcome="exception",
            error="Test error",
        )
        assert result.passed is False
        assert result.error == "Test error"

    def test_probe_execution_result_creation(self):
        """Test creating a probe execution result."""
        result = ProbeExecutionResult(
            probe_id="test-probe-1",
            passed=True,
            step_results=[],
            execution_time_seconds=1.0,
            output="Test output",
        )
        assert result.probe_id == "test-probe-1"
        assert result.passed is True
        assert result.execution_time_seconds == 1.0


class TestProbeValidator:
    """Test probe validation functionality."""

    def test_validator_accepts_safe_code(self):
        """Test that validator accepts safe code (with allowed imports)."""
        validator = ProbeValidator()
        safe_code = """
import math
x = 1 + 1
y = [1, 2, 3]
for item in y:
    print(item)
"""
        errors = validator.validate(safe_code)
        assert len(errors) == 0

    def test_validator_rejects_file_operations(self):
        """Test that validator rejects file operations."""
        validator = ProbeValidator()
        unsafe_code = """
f = open('test.txt', 'w')
f.write('test')
f.close()
"""
        errors = validator.validate(unsafe_code)
        assert len(errors) > 0
        assert any("open" in error for error in errors)

    def test_validator_rejects_dangerous_functions(self):
        """Test that validator rejects dangerous function calls."""
        validator = ProbeValidator()
        unsafe_code = """
exec('print("dangerous")')
eval('1 + 1')
"""
        errors = validator.validate(unsafe_code)
        assert len(errors) > 0
        assert any("exec" in error or "eval" in error for error in errors)

    def test_validator_rejects_forbidden_imports(self):
        """Test that validator rejects forbidden imports."""
        validator = ProbeValidator()
        unsafe_code = """
import os
import subprocess
"""
        errors = validator.validate(unsafe_code)
        assert len(errors) > 0
        assert any("os" in error or "subprocess" in error for error in errors)

    def test_validator_accepts_whitelisted_imports(self):
        """Test that validator accepts whitelisted imports."""
        validator = ProbeValidator()
        safe_code = """
import math
import random
from collections import defaultdict
"""
        errors = validator.validate(safe_code)
        assert len(errors) == 0

    def test_validator_rejects_syntax_errors(self):
        """Test that validator rejects syntax errors."""
        validator = ProbeValidator()
        invalid_code = """
x = 1 +
"""
        with pytest.raises(ProbeValidationError):
            validator.validate(invalid_code)

    def test_validate_probe_method(self):
        """Test the validate_probe convenience method."""
        validator = ProbeValidator()
        safe_code = "import math\nx = 1 + 1"
        assert validator.validate_probe(safe_code) is True

        unsafe_code = "open('test.txt')"
        assert validator.validate_probe(unsafe_code) is False


class TestProbeType:
    """Test probe type enumeration."""

    def test_probe_type_values(self):
        """Test that probe type has expected values."""
        assert ProbeType.FUNCTION_IO == "function_io"
        assert ProbeType.EXCEPTION == "exception"
        assert ProbeType.STATE_CHANGE == "state_change"
        assert ProbeType.INVARIANT == "invariant"
        assert ProbeType.RETURN_STRUCTURE == "return_structure"

    def test_probe_type_is_string_enum(self):
        """Test that probe type is a string enum."""
        assert isinstance(ProbeType.FUNCTION_IO, str)
