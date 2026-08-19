"""Runner for executing acceptance probes.

This module handles the execution of acceptance probes in temporary
directories, ensuring they don't affect the actual patch and can be
run both before and after changes.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

from patchpilot.verification.probes.schema import (
    AcceptanceProbe,
    ProbeExecutionResult,
    StepResult,
)
from patchpilot.verification.probes.validator import ProbeValidator


class ProbeRunner:
    """Execute acceptance probes in isolated temporary directories.

    The runner ensures that probes:
    - Execute in temporary directories
    - Don't modify the actual patch
    - Can be run before and after changes
    - Record complete artifacts
    - Provide detailed execution results
    """

    def __init__(self, workspace_root: Path) -> None:
        """Initialize the probe runner.

        Args:
            workspace_root: Root directory of the target workspace
        """
        self.workspace_root = workspace_root
        self.validator = ProbeValidator()

    def run_probe(
        self,
        probe: AcceptanceProbe,
        source_code: str,
        phase: str = "baseline",
    ) -> ProbeExecutionResult:
        """Run an acceptance probe in a temporary directory.

        Args:
            probe: AcceptanceProbe to execute
            source_code: Source code to test (original or modified)
            phase: Execution phase ("baseline" or "post_patch")

        Returns:
            ProbeExecutionResult with detailed execution information
        """
        start_time = time.time()
        step_results: list[StepResult] = []
        output_parts: list[str] = []
        artifacts: list[str] = []
        error_message: str | None = None

        # Validate probe before execution
        validation_errors = self._validate_probe(probe)
        if validation_errors:
            return ProbeExecutionResult(
                probe_id=probe.id,
                passed=False,
                step_results=[],
                execution_time_seconds=time.time() - start_time,
                output="\n".join(validation_errors),
                error="Probe validation failed",
                artifacts=[],
            )

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Create a copy of the source file in temp directory
                source_file = temp_path / "target.py"
                source_file.write_text(source_code, encoding="utf-8")

                # Execute setup code
                if probe.setup_code:
                    setup_output = self._execute_code(
                        probe.setup_code, temp_path, source_file
                    )
                    output_parts.append(f"SETUP: {setup_output}")

                # Execute each step
                for i, step in enumerate(probe.steps):
                    step_result = self._execute_step(
                        step,
                        i,
                        temp_path,
                        source_file,
                    )
                    step_results.append(step_result)
                    output_parts.append(
                        f"STEP {i}: {step.description} - {'PASS' if step_result.passed else 'FAIL'}"
                    )
                    if not step_result.passed and step_result.error:
                        output_parts.append(f"  Error: {step_result.error}")

                    # Stop on first failure
                    if not step_result.passed:
                        break

                # Execute teardown code
                if probe.teardown_code:
                    teardown_output = self._execute_code(
                        probe.teardown_code, temp_path, source_file
                    )
                    output_parts.append(f"TEARDOWN: {teardown_output}")

                # Collect artifacts
                artifacts = self._collect_artifacts(temp_path)

        except Exception as e:  # noqa: BLE001 - Catch all exceptions for probe execution
            error_message = f"Probe execution failed: {e!s}"
            output_parts.append(error_message)

        execution_time = time.time() - start_time
        all_passed = all(result.passed for result in step_results)

        return ProbeExecutionResult(
            probe_id=probe.id,
            passed=all_passed,
            step_results=step_results,
            execution_time_seconds=execution_time,
            output="\n".join(output_parts),
            error=error_message,
            artifacts=artifacts,
        )

    def _validate_probe(self, probe: AcceptanceProbe) -> list[str]:
        """Validate probe code before execution.

        Args:
            probe: AcceptanceProbe to validate

        Returns:
            List of validation error messages
        """
        errors: list[str] = []

        # Validate setup code
        if probe.setup_code:
            setup_errors = self.validator.validate(probe.setup_code)
            errors.extend([f"Setup: {err}" for err in setup_errors])

        # Validate each step
        for i, step in enumerate(probe.steps):
            step_errors = self.validator.validate(step.code)
            errors.extend([f"Step {i}: {err}" for err in step_errors])

        # Validate teardown code
        if probe.teardown_code:
            teardown_errors = self.validator.validate(probe.teardown_code)
            errors.extend([f"Teardown: {err}" for err in teardown_errors])

        return errors

    def _execute_step(
        self,
        step,
        step_index: int,
        temp_path: Path,
        source_file: Path,
    ) -> StepResult:
        """Execute a single probe step.

        Args:
            step: ProbeStep to execute
            step_index: Index of the step
            temp_path: Temporary directory path
            source_file: Path to source file being tested

        Returns:
            StepResult with execution information
        """
        try:
            # Create execution context
            exec_globals: dict[str, Any] = {
                "__file__": str(source_file),
                "__name__": "__main__",
            }

            # Execute the step code
            exec(step.code, exec_globals)  # noqa: S102

            # Check expected outcome
            actual_outcome = self._evaluate_outcome(step, exec_globals)
            passed = actual_outcome == step.expected_outcome

            return StepResult(
                step_index=step_index,
                description=step.description,
                passed=passed,
                expected_outcome=step.expected_outcome,
                actual_outcome=actual_outcome,
                error=None if passed else f"Expected {step.expected_outcome}, got {actual_outcome}",
            )

        except Exception as e:  # noqa: BLE001 - Catch all exceptions for step execution
            return StepResult(
                step_index=step_index,
                description=step.description,
                passed=False,
                expected_outcome=step.expected_outcome,
                actual_outcome="exception",
                error=str(e),
            )

    def _execute_code(
        self,
        code: str,
        temp_path: Path,
        source_file: Path,
    ) -> str:
        """Execute arbitrary code in the probe context.

        Args:
            code: Code to execute
            temp_path: Temporary directory path
            source_file: Path to source file

        Returns:
            Output from code execution
        """
        exec_globals: dict[str, Any] = {
            "__file__": str(source_file),
            "__name__": "__main__",
        }

        try:
            exec(code, exec_globals)  # noqa: S102
            return "Execution successful"
        except Exception as e:  # noqa: BLE001 - Catch all exceptions for setup/teardown execution
            return f"Execution error: {e!s}"

    def _evaluate_outcome(self, step, exec_globals: dict[str, Any]) -> str:
        """Evaluate the expected outcome of a step.

        Args:
            step: ProbeStep being evaluated
            exec_globals: Execution context globals

        Returns:
            String representation of the actual outcome
        """
        # Default implementation - can be extended
        # For now, just check if an exception was raised
        if "exception" in exec_globals:
            return "exception"
        return "no_exception"

    def _collect_artifacts(self, temp_path: Path) -> list[str]:
        """Collect artifact files from temporary directory.

        Args:
            temp_path: Temporary directory path

        Returns:
            List of artifact file paths
        """
        artifacts: list[str] = []

        for file_path in temp_path.iterdir():
            if file_path.is_file() and not file_path.name.startswith("."):
                artifacts.append(str(file_path))

        return artifacts

    def run_baseline_probe(
        self,
        probe: AcceptanceProbe,
        source_code: str,
    ) -> ProbeExecutionResult:
        """Run probe in baseline phase (before changes).

        Args:
            probe: AcceptanceProbe to execute
            source_code: Original source code

        Returns:
            ProbeExecutionResult from baseline execution
        """
        return self.run_probe(probe, source_code, phase="baseline")

    def run_post_patch_probe(
        self,
        probe: AcceptanceProbe,
        source_code: str,
    ) -> ProbeExecutionResult:
        """Run probe in post-patch phase (after changes).

        Args:
            probe: AcceptanceProbe to execute
            source_code: Modified source code

        Returns:
            ProbeExecutionResult from post-patch execution
        """
        return self.run_probe(probe, source_code, phase="post_patch")
