"""Specialized verification coordinator for probes and structural checks.

This module provides the SpecializedVerifier class which coordinates the execution
of acceptance probes and structural checks, converting their results into CheckReport
format for integration with the main verification workflow.

The specialized verifier:
- Validates that ChangePlan contains sufficient information for specialized checks
- Executes acceptance probes in isolated temporary directories
- Executes structural checks using AST analysis
- Converts all results to CheckReport format with proper metadata
- Handles both baseline and post-patch phases for comparison
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from patchpilot.planning.schema import (
    AcceptanceProbeSpec,
    ChangePlan,
    StructuralCheckSpec,
)
from patchpilot.verification.config import VerificationTimeouts
from patchpilot.verification.probes.runner import ProbeRunner
from patchpilot.verification.report import CheckReport
from patchpilot.verification.structural.ast_checks import CheckType, StructuralCheck
from patchpilot.verification.structural.runner import StructuralRunner

if TYPE_CHECKING:
    from patchpilot.sandbox.docker_runner import DockerSandbox
    from patchpilot.verification.probes.schema import ProbeExecutionResult
    from patchpilot.verification.structural.runner import StructuralReport


class SpecializedVerifier:
    """Coordinate execution of specialized verification checks.

    The SpecializedVerifier bridges the gap between ChangePlan specifications
    and the existing ProbeRunner and StructuralRunner implementations, ensuring
    that specialized checks are only executed when sufficient validated information
    is available in the approved ChangePlan.

    Attributes:
        workspace_root: Root directory of the target workspace
        probe_runner: ProbeRunner instance for acceptance probe execution
        structural_runner: StructuralRunner instance for structural check execution
        timeouts: VerificationTimeouts configuration for specialized checks
    """

    def __init__(
        self,
        workspace_root: Path,
        sandbox: DockerSandbox,
        timeouts: VerificationTimeouts | None = None,
    ) -> None:
        """Initialize the specialized verifier.

        Args:
            workspace_root: Root directory of the target workspace
            timeouts: Optional VerificationTimeouts configuration (uses defaults if None)
        """
        self.workspace_root = workspace_root
        self.probe_runner = ProbeRunner(sandbox)
        self.structural_runner = StructuralRunner(workspace_root)
        self.timeouts = timeouts or VerificationTimeouts()

    def has_specialized_checks(self, change_plan: ChangePlan) -> bool:
        """Check if the change plan contains any specialized verification specs.

        Args:
            change_plan: The approved ChangePlan to check

        Returns:
            True if the plan contains acceptance probes or structural checks
        """
        return bool(
            change_plan.acceptance_probes or change_plan.structural_checks
        )

    def execute_specialized_checks(
        self,
        change_plan: ChangePlan,
        phase: str = "post_patch",
    ) -> list[CheckReport]:
        """Execute all specialized checks from the change plan.

        This method validates that the change plan contains sufficient information
        for specialized checks, then executes them and converts results to CheckReport format.

        Args:
            change_plan: The approved ChangePlan containing verification specs
            phase: Verification phase ("baseline" or "post_patch")

        Returns:
            List of CheckReport objects from specialized check execution
        """
        check_reports: list[CheckReport] = []

        # Execute acceptance probes if available
        if change_plan.acceptance_probes:
            probe_reports = self._execute_probes(
                change_plan.acceptance_probes,
                phase,
            )
            check_reports.extend(probe_reports)

        # Execute structural checks if available
        if change_plan.structural_checks:
            structural_reports = self._execute_structural_checks(
                change_plan.structural_checks,
                phase,
            )
            check_reports.extend(structural_reports)

        return check_reports

    def _execute_probes(
        self,
        probe_specs: list[AcceptanceProbeSpec],
        phase: str,
    ) -> list[CheckReport]:
        """Execute acceptance probes and convert to CheckReport format.

        Args:
            probe_specs: List of AcceptanceProbeSpec from ChangePlan
            phase: Verification phase

        Returns:
            List of CheckReport objects from probe execution
        """
        check_reports: list[CheckReport] = []

        for spec in probe_specs:
            # Validate spec has required information
            if not spec.probe_id or not spec.module or not spec.target:
                # Skip invalid specs - they should not result in invented PASS
                continue

            # Execute probe in the appropriate phase
            if phase == "baseline":
                result: ProbeExecutionResult = (
                    self.probe_runner.run_baseline_probe(spec)
                )
            else:
                result = self.probe_runner.run_post_patch_probe(spec)

            # Convert to CheckReport
            check_report = self._probe_result_to_check_report(
                result,
                phase,
                spec.criterion_ids,
            )
            check_reports.append(check_report)

        return check_reports

    def _execute_structural_checks(
        self,
        check_specs: list[StructuralCheckSpec],
        phase: str,
    ) -> list[CheckReport]:
        """Execute structural checks and convert to CheckReport format.

        Args:
            check_specs: List of StructuralCheckSpec from ChangePlan
            phase: Verification phase

        Returns:
            List of CheckReport objects from structural check execution
        """
        check_reports: list[CheckReport] = []

        # Group checks by file path for efficient execution
        checks_by_file: dict[str, list[StructuralCheckSpec]] = {}
        for spec in check_specs:
            if not spec.file_path:
                continue
            if spec.file_path not in checks_by_file:
                checks_by_file[spec.file_path] = []
            checks_by_file[spec.file_path].append(spec)

        # Execute checks for each file
        for file_path, specs in checks_by_file.items():
            # Convert specs to StructuralCheck objects
            structural_checks = [
                self._spec_to_structural_check(spec) for spec in specs
            ]

            # Execute checks
            file_path_obj = self.workspace_root / file_path
            if phase == "baseline":
                report: StructuralReport = (
                    self.structural_runner.run_baseline_checks(
                        structural_checks,
                        file_path_obj,
                    )
                )
            else:
                report = self.structural_runner.run_post_patch_checks(
                    structural_checks,
                    file_path_obj,
                )

            # Convert each result to CheckReport
            for i, result in enumerate(report.results):
                spec = specs[i]
                check_report = self._structural_result_to_check_report(
                    result,
                    phase,
                    spec.criterion_ids,
                )
                check_reports.append(check_report)

        return check_reports

    def _spec_to_structural_check(
        self,
        spec: StructuralCheckSpec,
    ) -> StructuralCheck:
        """Convert StructuralCheckSpec to StructuralCheck.

        Args:
            spec: StructuralCheckSpec from ChangePlan

        Returns:
            StructuralCheck object for execution
        """
        return StructuralCheck(
            check_type=CheckType(spec.check_type),
            target=spec.target,
            parameters=spec.parameters,
            description=f"Structural check: {spec.check_type} on {spec.target}",
        )

    def _probe_result_to_check_report(
        self,
        result: ProbeExecutionResult,
        phase: str,
        subject_ids: list[str],
    ) -> CheckReport:
        """Convert ProbeExecutionResult to CheckReport.

        Args:
            result: ProbeExecutionResult from ProbeRunner
            phase: Verification phase
            subject_ids: Acceptance criteria IDs

        Returns:
            CheckReport with proper metadata
        """
        # Bound failure output to prevent excessive size
        failure_output = None
        if not result.passed and result.output:
            failure_output = result.output[:2000]

        return CheckReport(
            method="acceptance_probe",
            phase=phase,
            level="SPECIALIZED_PROBE",
            command=f"probe:{result.probe_id}",
            passed=result.passed,
            exit_code=0 if result.passed else 1,
            duration_seconds=result.execution_time_seconds,
            timeout_seconds=self.timeouts.specialized,
            failure_type="probe_failure" if not result.passed else None,
            summary={
                "probe_id": result.probe_id,
                "step_count": len(result.step_results),
                "passed_steps": sum(1 for s in result.step_results if s.passed),
                "error": result.error,
                "output": failure_output,
            } if not result.passed else None,
            subject_ids=subject_ids,
            direct=True,  # Probes provide direct evidence
        )

    def _structural_result_to_check_report(
        self,
        result,
        phase: str,
        subject_ids: list[str],
    ) -> CheckReport:
        """Convert structural CheckResult to CheckReport.

        Args:
            result: CheckResult from StructuralRunner
            phase: Verification phase
            subject_ids: Acceptance criteria IDs

        Returns:
            CheckReport with proper metadata
        """
        # Bound failure output to prevent excessive size
        failure_output = None
        if not result.passed and result.message:
            failure_output = result.message[:2000]

        return CheckReport(
            method="structural_check",
            phase=phase,
            level="SPECIALIZED_STRUCTURAL",
            command=f"structural:{result.check.check_type.value}:{result.check.target}",
            passed=result.passed,
            exit_code=0 if result.passed else 1,
            duration_seconds=0.0,  # Structural checks are fast, no precise timing
            timeout_seconds=self.timeouts.specialized,
            failure_type="structural_failure" if not result.passed else None,
            summary={
                "check_type": result.check.check_type.value,
                "target": result.check.target,
                "message": failure_output,
                "location": result.location,
            } if not result.passed else None,
            subject_ids=subject_ids,
            direct=True,  # Structural checks provide direct evidence
        )
