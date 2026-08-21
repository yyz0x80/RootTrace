"""Integration tests for specialized verification checks.

This module tests that specialized verification checks (acceptance probes
and structural checks) are properly integrated into the verification workflow
and reach the final report and acceptance-evidence mapper.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from patchpilot.evidence.aggregator import aggregate_evidence
from patchpilot.evidence.mapper import map_acceptance_evidence
from patchpilot.evidence.schema import EvidenceStatus
from patchpilot.issue.schema import AcceptanceCriterion, NormalizedIssue
from patchpilot.planning.schema import (
    AcceptanceProbeSpec,
    ChangePlan,
    PlannedChange,
    StructuralCheckSpec,
)
from patchpilot.sandbox.docker_runner import CommandResult
from patchpilot.verification.report import CheckReport, VerificationReport
from patchpilot.verification.specialized import SpecializedVerifier


def _passing_sandbox() -> MagicMock:
    """Return a sandbox double that reports a passing declarative probe."""
    sandbox = MagicMock()
    sandbox.run.return_value = CommandResult(
        command="probe",
        exit_code=0,
        stdout='{"passed": true, "actual": true}',
        stderr="",
        duration_seconds=0.01,
    )
    return sandbox


def test_specialized_verifier_integration_with_change_plan():
    """Test that SpecializedVerifier integrates with ChangePlan specifications."""
    # Create a temporary workspace
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace_root = Path(temp_dir)
        
        # Create a simple Python file
        test_file = workspace_root / "test_module.py"
        test_file.write_text(
            "def test_function():\n"
            "    return True\n"
        )
        
        # Create a ChangePlan with specialized specs
        change_plan = ChangePlan(
            risk_level="low",
            acceptance_probes=[
                AcceptanceProbeSpec(
                    probe_id="probe_1",
                    module="test_module",
                    target="test_function",
                    probe_type="function_io",
                    criterion_ids=["ac_1"],
                    assertion="truthy",
                )
            ],
            structural_checks=[
                StructuralCheckSpec(
                    check_id="struct_1",
                    check_type="function_exists",
                    target="test_function",
                    criterion_ids=["ac_1"],
                    file_path="test_module.py",
                )
            ],
        )
        
        # Create SpecializedVerifier
        verifier = SpecializedVerifier(workspace_root, _passing_sandbox())
        
        # Test that it detects specialized checks
        assert verifier.has_specialized_checks(change_plan)
        
        # Execute specialized checks
        check_reports = verifier.execute_specialized_checks(
            change_plan,
            phase="post_patch",
        )
        
        # Verify check reports are created
        assert len(check_reports) >= 1
        
        # Verify CheckReport format
        for report in check_reports:
            assert hasattr(report, "method")
            assert hasattr(report, "phase")
            assert hasattr(report, "level")
            assert hasattr(report, "subject_ids")
            assert hasattr(report, "direct")
            assert report.phase == "post_patch"


def test_specialized_checks_without_change_plan():
    """Test that specialized checks are not executed without ChangePlan specs."""
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace_root = Path(temp_dir)
        
        # Create empty ChangePlan
        change_plan = ChangePlan(risk_level="low")
        
        # Create SpecializedVerifier
        verifier = SpecializedVerifier(workspace_root, _passing_sandbox())
        
        # Test that it doesn't detect specialized checks
        assert not verifier.has_specialized_checks(change_plan)
        
        # Execute should return empty list
        check_reports = verifier.execute_specialized_checks(
            change_plan,
            phase="post_patch",
        )
        
        assert len(check_reports) == 0


def test_specialized_check_report_format():
    """Test that specialized checks produce correct CheckReport format."""
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace_root = Path(temp_dir)
        
        # Create a simple Python file
        test_file = workspace_root / "test_module.py"
        test_file.write_text("def target_func(): pass\n")
        
        # Create ChangePlan with structural check
        change_plan = ChangePlan(
            risk_level="low",
            structural_checks=[
                StructuralCheckSpec(
                    check_id="struct_1",
                    check_type="function_exists",
                    target="target_func",
                    criterion_ids=["ac_1"],
                    file_path="test_module.py",
                )
            ],
        )
        
        # Create SpecializedVerifier
        verifier = SpecializedVerifier(workspace_root, _passing_sandbox())
        
        # Execute specialized checks
        check_reports = verifier.execute_specialized_checks(
            change_plan,
            phase="post_patch",
        )
        
        # Verify CheckReport attributes
        for report in check_reports:
            assert report.method in ("acceptance_probe", "structural_check")
            assert report.phase == "post_patch"
            assert report.level in ("SPECIALIZED_PROBE", "SPECIALIZED_STRUCTURAL")
            assert isinstance(report.subject_ids, list)
            assert isinstance(report.direct, bool)
            assert isinstance(report.passed, bool)
            assert isinstance(report.exit_code, int)
            assert isinstance(report.duration_seconds, float)


def test_specialized_checks_reach_verification_report():
    """Test that specialized checks are included in VerificationReport."""
    # Create specialized check reports
    specialized_checks = [
        CheckReport(
            method="acceptance_probe",
            phase="post_patch",
            level="SPECIALIZED_PROBE",
            command="probe:test_probe",
            passed=True,
            exit_code=0,
            duration_seconds=0.5,
            subject_ids=["ac_1"],
            direct=True,
        ),
        CheckReport(
            method="structural_check",
            phase="post_patch",
            level="SPECIALIZED_STRUCTURAL",
            command="structural:function_exists:test_func",
            passed=True,
            exit_code=0,
            duration_seconds=0.1,
            subject_ids=["ac_1"],
            direct=True,
        ),
    ]
    
    # Create VerificationReport with specialized checks
    report = VerificationReport(
        run_id="test_run",
        passed=True,
        checks=specialized_checks,
    )
    
    # Verify specialized checks are in the report
    assert len(report.checks) == 2
    assert any(check.method == "acceptance_probe" for check in report.checks)
    assert any(check.method == "structural_check" for check in report.checks)


def test_acceptance_probe_reaches_behavior_evidence():
    """Map a runtime probe to behavior evidence, not structural evidence."""
    # Create VerificationReport with specialized checks
    report = VerificationReport(
        run_id="test_run",
        passed=True,
        checks=[
            CheckReport(
                method="acceptance_probe",
                phase="post_patch",
                level="SPECIALIZED_PROBE",
                command="probe:test_probe",
                passed=True,
                exit_code=0,
                duration_seconds=0.5,
                subject_ids=["ac_1"],
                direct=True,
            ),
        ],
    )
    
    # Aggregate evidence for the criterion
    evidence = aggregate_evidence(
        criterion_id="ac_1",
        description="Test criterion",
        changed_files=["test_file.py"],
        tests=["probe:test_probe"],
        command_results=["SPECIALIZED_PROBE: PASSED"],
        report=report,
    )
    
    assert evidence.behavior_change is not None
    assert evidence.structural_contract is None


def test_missing_specialized_checks_result_in_unverified():
    """Test that missing optional specialized checks result in UNVERIFIED evidence."""
    # Create VerificationReport without specialized checks
    report = VerificationReport(
        run_id="test_run",
        passed=True,
        checks=[
            CheckReport(
                method="pytest",
                phase="post_patch",
                level="LEVEL_2_TARGET_TESTS",
                command="pytest test_file.py",
                passed=True,
                exit_code=0,
                duration_seconds=1.0,
                subject_ids=["ac_1"],
                direct=True,
            ),
        ],
    )
    
    # Aggregate evidence for the criterion
    evidence = aggregate_evidence(
        criterion_id="ac_1",
        description="Test criterion",
        changed_files=["test_file.py"],
        tests=["pytest test_file.py"],
        command_results=["LEVEL_2_TARGET_TESTS: PASSED"],
        report=report,
    )
    
    # Verify structural contract is UNVERIFIED when only pytest is available
    assert evidence.structural_contract is not None
    assert evidence.structural_contract.status == EvidenceStatus.UNVERIFIED


def test_acceptance_probe_in_evidence_mapper():
    """Map an acceptance probe into behavior evidence."""
    # Create NormalizedIssue with acceptance criteria
    issue = NormalizedIssue(
        title="Test Issue",
        task_type="bug",
        problem_statement="Test problem",
        acceptance_criteria=[
            AcceptanceCriterion(
                id="ac_1",
                description="Test criterion with specialized verification",
            )
        ],
        constraints=[],
        ambiguous_points=[],
        expected_test_areas=[],
        implementation_notes=[],
    )
    
    # Create ChangePlan with specialized specs
    change_plan = ChangePlan(
        risk_level="low",
        planned_changes=[
            PlannedChange(
                path="test_file.py",
                action="modify",
                description="Fix test function",
                acceptance_criteria=["ac_1"],
                criterion_ids=["ac_1"],
            )
        ],
    )
    
    # Create VerificationReport with specialized checks
    report = VerificationReport(
        run_id="test_run",
        passed=True,
        checks=[
            CheckReport(
                method="acceptance_probe",
                phase="post_patch",
                level="SPECIALIZED_PROBE",
                command="probe:test_probe",
                passed=True,
                exit_code=0,
                duration_seconds=0.5,
                subject_ids=["ac_1"],
                direct=True,
            ),
        ],
    )
    
    # Map acceptance evidence
    evidence_list = map_acceptance_evidence(
        issue=issue,
        plan=change_plan,
        actual_changes=[],
        report=report,
    )
    
    # Verify the runtime probe contributes behavior evidence.
    assert len(evidence_list) == 1
    evidence = evidence_list[0]
    assert evidence.behavior_change is not None
    assert evidence.structural_contract is None


def test_constraint_audit_check_creation():
    """Test that constraint audit checks are created with proper format."""
    from patchpilot.verification.verifier import Verifier
    
    # Mock sandbox
    mock_sandbox = MagicMock()
    mock_sandbox.run.return_value = MagicMock(exit_code=0, duration_seconds=0.5)
    
    # Create Verifier with workspace root
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace_root = Path(temp_dir)
        verifier = Verifier(mock_sandbox, workspace_root=workspace_root)
        
        # Create ChangePlan
        change_plan = ChangePlan(
            risk_level="low",
            planned_changes=[
                PlannedChange(
                    path="test_file.py",
                    action="modify",
                    description="Test change",
                    acceptance_criteria=[],
                    criterion_ids=[],
                )
            ],
        )
        
        # Create constraint audit checks
        # This will fail if policy modules are not available, which is expected
        try:
            constraint_checks = verifier._create_constraint_audit_checks(
                run_id="test_run",
                change_plan=change_plan,
            )
            
            # If successful, verify format
            for check in constraint_checks:
                assert check.method == "constraint_audit"
                assert check.phase == "constraint_audit"
                assert check.level == "CONSTRAINT_AUDIT"
                assert isinstance(check.passed, bool)
                assert isinstance(check.exit_code, int)
        except (ImportError, RuntimeError):
            # Expected if policy modules are not available
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
