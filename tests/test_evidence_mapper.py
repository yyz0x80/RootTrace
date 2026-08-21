"""Tests for evidence mapper functionality."""

from patchpilot.evidence import (
    BehaviorChangeStatus,
    BehaviorPreservationStatus,
    ConstraintStatus,
    EvidenceStatus,
    StructuralContractStatus,
    map_acceptance_evidence,
)
from patchpilot.issue.schema import AcceptanceCriterion, NormalizedIssue
from patchpilot.planning.schema import ChangeAction, ChangePlan, PlannedChange
from patchpilot.tools import WorkspaceChange
from patchpilot.verification.report import CheckReport, VerificationReport


def test_map_acceptance_evidence_pass_case():
    """Test evidence mapping with passing verification and actual changes."""
    issue = NormalizedIssue(
        title="Test issue",
        task_type="bug",
        problem_statement="Test problem",
        acceptance_criteria=[
            AcceptanceCriterion(id="AC-1", description="First criterion"),
            AcceptanceCriterion(id="AC-2", description="Second criterion"),
        ],
    )

    plan = ChangePlan(
        planned_changes=[
            PlannedChange(
                path="src/module.py",
                action=ChangeAction.MODIFY,
                description="Fix bug",
                acceptance_criteria=["AC-1"],
            ),
            PlannedChange(
                path="src/other.py",
                action=ChangeAction.MODIFY,
                description="Add feature",
                acceptance_criteria=["AC-2"],
            ),
        ],
        risk_level="low",
    )

    actual_changes = [
        WorkspaceChange(path="src/module.py", action="modify"),
        WorkspaceChange(path="src/other.py", action="modify"),
    ]

    report = VerificationReport(
        run_id="test-run",
        passed=True,
        checks=[
            # Baseline checks (FAIL to simulate bug)
            CheckReport(
                method="pytest",
                phase="baseline",
                level="LEVEL_2_TARGET_TESTS",
                command="pytest tests/test_module.py",
                passed=False,
                exit_code=1,
                duration_seconds=1.0,
                subject_ids=["AC-1"],
                direct=True,
            ),
            CheckReport(
                method="pytest",
                phase="baseline",
                level="LEVEL_2_TARGET_TESTS",
                command="pytest tests/test_other.py",
                passed=False,
                exit_code=1,
                duration_seconds=1.0,
                subject_ids=["AC-2"],
                direct=True,
            ),
            # Post-patch checks (PASS to simulate fix)
            CheckReport(
                method="pytest",
                phase="post_patch",
                level="LEVEL_2_TARGET_TESTS",
                command="pytest tests/test_module.py",
                passed=True,
                exit_code=0,
                duration_seconds=1.0,
                subject_ids=["AC-1"],
                direct=True,
            ),
            CheckReport(
                method="pytest",
                phase="post_patch",
                level="LEVEL_2_TARGET_TESTS",
                command="pytest tests/test_other.py",
                passed=True,
                exit_code=0,
                duration_seconds=1.0,
                subject_ids=["AC-2"],
                direct=True,
            ),
        ],
    )

    evidence = map_acceptance_evidence(issue, plan, actual_changes, report)

    assert len(evidence) == 2
    assert evidence[0].criterion_id == "AC-1"
    assert evidence[0].status == EvidenceStatus.PASS
    assert "src/module.py" in evidence[0].changed_files
    assert evidence[0].behavior_change is not None
    assert evidence[0].behavior_change.status == BehaviorChangeStatus.PASS
    assert evidence[1].criterion_id == "AC-2"
    assert evidence[1].status == EvidenceStatus.PASS
    assert "src/other.py" in evidence[1].changed_files
    assert evidence[1].behavior_change is not None
    assert evidence[1].behavior_change.status == BehaviorChangeStatus.PASS


def test_broad_mapped_test_remains_unverified_without_direct_evidence():
    """A passing file-level suite must not prove every mapped behavior."""
    issue = NormalizedIssue(
        title="Median behavior",
        task_type="bug",
        problem_statement="Even-length inputs return the wrong median.",
        acceptance_criteria=[
            AcceptanceCriterion(
                id="AC-1",
                description="Even-length inputs use both middle values.",
            )
        ],
    )
    plan = ChangePlan(
        planned_changes=[
            PlannedChange(
                path="src/statistics.py",
                action=ChangeAction.MODIFY,
                description="Fix median",
                acceptance_criteria=["AC-1"],
            )
        ],
        risk_level="low",
    )
    report = VerificationReport(
        passed=True,
        checks=[
            CheckReport(
                method="pytest",
                phase="post_patch",
                level="LEVEL_2_TARGET_TESTS",
                command="pytest tests/test_statistics.py",
                passed=True,
                exit_code=0,
                duration_seconds=1.0,
                subject_ids=["AC-1"],
                direct=False,
            )
        ],
    )

    evidence = map_acceptance_evidence(
        issue,
        plan,
        [WorkspaceChange(path="src/statistics.py", action="modify")],
        report,
    )

    # Without direct checks, behavior_change should be None (no evidence)
    assert evidence[0].behavior_change is None
    # Overall status should be UNVERIFIED since no direct evidence
    assert evidence[0].status == EvidenceStatus.UNVERIFIED


def test_map_acceptance_evidence_fail_case():
    """Test evidence mapping with failed verification."""
    issue = NormalizedIssue(
        title="Test issue",
        task_type="bug",
        problem_statement="Test problem",
        acceptance_criteria=[
            AcceptanceCriterion(id="AC-1", description="First criterion"),
        ],
    )

    plan = ChangePlan(
        planned_changes=[
            PlannedChange(
                path="src/module.py",
                action=ChangeAction.MODIFY,
                description="Fix bug",
                acceptance_criteria=["AC-1"],
            ),
        ],
        risk_level="low",
    )

    actual_changes = [
        WorkspaceChange(path="src/module.py", action="modify"),
    ]

    report = VerificationReport(
        run_id="test-run",
        passed=False,
        checks=[
            # Baseline check (FAIL)
            CheckReport(
                method="pytest",
                phase="baseline",
                level="LEVEL_2_TARGET_TESTS",
                command="pytest tests/test_module.py",
                passed=False,
                exit_code=1,
                duration_seconds=1.0,
                failure_type="AssertionError",
                subject_ids=["AC-1"],
                direct=True,
            ),
            # Post-patch check (still FAIL - fix ineffective)
            CheckReport(
                method="pytest",
                phase="post_patch",
                level="LEVEL_2_TARGET_TESTS",
                command="pytest tests/test_module.py",
                passed=False,
                exit_code=1,
                duration_seconds=1.0,
                failure_type="AssertionError",
                subject_ids=["AC-1"],
                direct=True,
            ),
        ],
    )

    evidence = map_acceptance_evidence(issue, plan, actual_changes, report)

    assert len(evidence) == 1
    assert evidence[0].criterion_id == "AC-1"
    assert evidence[0].status == EvidenceStatus.FAIL
    assert evidence[0].behavior_change is not None
    assert evidence[0].behavior_change.status == BehaviorChangeStatus.FAIL


def test_map_acceptance_evidence_unverified_case():
    """Test evidence mapping with no actual changes."""
    issue = NormalizedIssue(
        title="Test issue",
        task_type="bug",
        problem_statement="Test problem",
        acceptance_criteria=[
            AcceptanceCriterion(id="AC-1", description="First criterion"),
        ],
    )

    plan = ChangePlan(
        planned_changes=[
            PlannedChange(
                path="src/module.py",
                action=ChangeAction.MODIFY,
                description="Fix bug",
                acceptance_criteria=["AC-1"],
            ),
        ],
        risk_level="low",
    )

    actual_changes = []  # No actual changes

    report = VerificationReport(
        run_id="test-run",
        passed=True,
        checks=[
            CheckReport(
                method="pytest",
                phase="post_patch",
                level="LEVEL_2_TARGET_TESTS",
                command="pytest tests/test_module.py",
                passed=True,
                exit_code=0,
                duration_seconds=1.0,
                subject_ids=["AC-1"],
                direct=False,
            ),
        ],
    )

    evidence = map_acceptance_evidence(issue, plan, actual_changes, report)

    assert len(evidence) == 1
    assert evidence[0].criterion_id == "AC-1"
    # No direct checks means no behavior change evidence
    assert evidence[0].behavior_change is None
    # Overall status should be UNVERIFIED since no direct evidence
    assert evidence[0].status == EvidenceStatus.UNVERIFIED


def test_map_acceptance_evidence_no_mapped_checks():
    """Test evidence mapping when no checks are mapped to criterion."""
    issue = NormalizedIssue(
        title="Test issue",
        task_type="bug",
        problem_statement="Test problem",
        acceptance_criteria=[
            AcceptanceCriterion(id="AC-1", description="First criterion"),
        ],
    )

    plan = ChangePlan(
        planned_changes=[
            PlannedChange(
                path="src/module.py",
                action=ChangeAction.MODIFY,
                description="Fix bug",
                acceptance_criteria=["AC-1"],
            ),
        ],
        risk_level="low",
    )

    actual_changes = [
        WorkspaceChange(path="src/module.py", action="modify"),
    ]

    report = VerificationReport(
        run_id="test-run",
        passed=True,
        checks=[
            CheckReport(
                method="ruff",
                phase="post_patch",
                level="LEVEL_1_LINT",
                command="ruff check",
                passed=True,
                exit_code=0,
                duration_seconds=1.0,
                subject_ids=[],  # No AC mapping
                direct=False,
            ),
        ],
    )

    evidence = map_acceptance_evidence(issue, plan, actual_changes, report)

    assert len(evidence) == 1
    assert evidence[0].criterion_id == "AC-1"
    # No direct checks means no behavior change evidence
    assert evidence[0].behavior_change is None
    # Overall status should be UNVERIFIED since no direct evidence
    assert evidence[0].status == EvidenceStatus.UNVERIFIED


def test_map_acceptance_evidence_unrelated_regression_tests_pass():
    """Test that unrelated regression tests passing does not mark AC as PASS.

    This test ensures that when general regression tests pass but are not
    specifically mapped to the acceptance criterion, the criterion should
    remain UNVERIFIED rather than incorrectly marked as PASS.
    """
    issue = NormalizedIssue(
        title="Test issue",
        task_type="bug",
        problem_statement="Test problem",
        acceptance_criteria=[
            AcceptanceCriterion(id="AC-1", description="First criterion"),
        ],
    )

    plan = ChangePlan(
        planned_changes=[
            PlannedChange(
                path="src/module.py",
                action=ChangeAction.MODIFY,
                description="Fix bug",
                acceptance_criteria=["AC-1"],
            ),
        ],
        risk_level="low",
    )

    actual_changes = [
        WorkspaceChange(path="src/module.py", action="modify"),
    ]

    report = VerificationReport(
        run_id="test-run",
        passed=True,
        checks=[
            # Unrelated regression tests that pass but are not mapped to AC-1
            CheckReport(
                method="pytest",
                phase="post_patch",
                level="LEVEL_3_REGRESSION",
                command="pytest tests/test_regression.py",
                passed=True,
                exit_code=0,
                duration_seconds=1.0,
                subject_ids=[],  # Not mapped to AC-1
                direct=False,
            ),
            CheckReport(
                method="ruff",
                phase="post_patch",
                level="LEVEL_1_LINT",
                command="ruff check",
                passed=True,
                exit_code=0,
                duration_seconds=1.0,
                subject_ids=[],  # Not mapped to AC-1
                direct=False,
            ),
        ],
    )

    evidence = map_acceptance_evidence(issue, plan, actual_changes, report)

    assert len(evidence) == 1
    assert evidence[0].criterion_id == "AC-1"
    # No direct checks means no behavior change evidence
    assert evidence[0].behavior_change is None
    # Overall status should be UNVERIFIED since no direct evidence
    assert evidence[0].status == EvidenceStatus.UNVERIFIED


def test_behavior_change_already_satisfied():
    """Test behavior change when baseline already passes (ALREADY_SATISFIED)."""
    issue = NormalizedIssue(
        title="Test issue",
        task_type="bug",
        problem_statement="Test problem",
        acceptance_criteria=[
            AcceptanceCriterion(id="AC-1", description="First criterion"),
        ],
    )

    plan = ChangePlan(
        planned_changes=[
            PlannedChange(
                path="src/module.py",
                action=ChangeAction.MODIFY,
                description="Fix bug",
                acceptance_criteria=["AC-1"],
            ),
        ],
        risk_level="low",
    )

    actual_changes = [
        WorkspaceChange(path="src/module.py", action="modify"),
    ]

    report = VerificationReport(
        run_id="test-run",
        passed=True,
        checks=[
            # Baseline check (PASS - already satisfied)
            CheckReport(
                method="pytest",
                phase="baseline",
                level="LEVEL_2_TARGET_TESTS",
                command="pytest tests/test_module.py",
                passed=True,
                exit_code=0,
                duration_seconds=1.0,
                subject_ids=["AC-1"],
                direct=True,
            ),
            # Post-patch check (PASS - still satisfied)
            CheckReport(
                method="pytest",
                phase="post_patch",
                level="LEVEL_2_TARGET_TESTS",
                command="pytest tests/test_module.py",
                passed=True,
                exit_code=0,
                duration_seconds=1.0,
                subject_ids=["AC-1"],
                direct=True,
            ),
        ],
    )

    evidence = map_acceptance_evidence(issue, plan, actual_changes, report)

    assert len(evidence) == 1
    assert evidence[0].criterion_id == "AC-1"
    assert evidence[0].behavior_change is not None
    assert evidence[0].behavior_change.status == BehaviorChangeStatus.ALREADY_SATISFIED
    # Overall status should be PASS since behavior preservation is PASS
    assert evidence[0].status == EvidenceStatus.PASS


def test_behavior_change_regression():
    """Test behavior change when baseline passes but post-patch fails (regression)."""
    issue = NormalizedIssue(
        title="Test issue",
        task_type="bug",
        problem_statement="Test problem",
        acceptance_criteria=[
            AcceptanceCriterion(id="AC-1", description="First criterion"),
        ],
    )

    plan = ChangePlan(
        planned_changes=[
            PlannedChange(
                path="src/module.py",
                action=ChangeAction.MODIFY,
                description="Fix bug",
                acceptance_criteria=["AC-1"],
            ),
        ],
        risk_level="low",
    )

    actual_changes = [
        WorkspaceChange(path="src/module.py", action="modify"),
    ]

    report = VerificationReport(
        run_id="test-run",
        passed=False,
        checks=[
            # Baseline check (PASS)
            CheckReport(
                method="pytest",
                phase="baseline",
                level="LEVEL_2_TARGET_TESTS",
                command="pytest tests/test_module.py",
                passed=True,
                exit_code=0,
                duration_seconds=1.0,
                subject_ids=["AC-1"],
                direct=True,
            ),
            # Post-patch check (FAIL - regression)
            CheckReport(
                method="pytest",
                phase="post_patch",
                level="LEVEL_2_TARGET_TESTS",
                command="pytest tests/test_module.py",
                passed=False,
                exit_code=1,
                duration_seconds=1.0,
                subject_ids=["AC-1"],
                direct=True,
            ),
        ],
    )

    evidence = map_acceptance_evidence(issue, plan, actual_changes, report)

    assert len(evidence) == 1
    assert evidence[0].criterion_id == "AC-1"
    assert evidence[0].status == EvidenceStatus.FAIL
    assert evidence[0].behavior_change is not None
    assert evidence[0].behavior_change.status == BehaviorChangeStatus.FAIL
    assert "regression" in evidence[0].behavior_change.explanation.lower()


def test_behavior_preservation_pass():
    """Test behavior preservation when baseline and post-patch both pass."""
    issue = NormalizedIssue(
        title="Test issue",
        task_type="bug",
        problem_statement="Test problem",
        acceptance_criteria=[
            AcceptanceCriterion(id="AC-1", description="First criterion"),
        ],
    )

    plan = ChangePlan(
        planned_changes=[
            PlannedChange(
                path="src/module.py",
                action=ChangeAction.MODIFY,
                description="Fix bug",
                acceptance_criteria=["AC-1"],
            ),
        ],
        risk_level="low",
    )

    actual_changes = [
        WorkspaceChange(path="src/module.py", action="modify"),
    ]

    report = VerificationReport(
        run_id="test-run",
        passed=True,
        checks=[
            CheckReport(
                method="pytest",
                phase="baseline",
                level="LEVEL_2_TARGET_TESTS",
                command="pytest tests/test_module.py",
                passed=True,
                exit_code=0,
                duration_seconds=1.0,
                subject_ids=["AC-1"],
                direct=True,
            ),
            CheckReport(
                method="pytest",
                phase="post_patch",
                level="LEVEL_2_TARGET_TESTS",
                command="pytest tests/test_module.py",
                passed=True,
                exit_code=0,
                duration_seconds=1.0,
                subject_ids=["AC-1"],
                direct=True,
            ),
        ],
    )

    evidence = map_acceptance_evidence(issue, plan, actual_changes, report)

    assert len(evidence) == 1
    assert evidence[0].criterion_id == "AC-1"
    assert evidence[0].behavior_preservation is not None
    assert evidence[0].behavior_preservation.status == BehaviorPreservationStatus.PASS


def test_structural_contract_with_specialized_check():
    """Test structural contract with specialized AST/mock check."""
    issue = NormalizedIssue(
        title="Test issue",
        task_type="bug",
        problem_statement="Test problem",
        acceptance_criteria=[
            AcceptanceCriterion(id="AC-1", description="First criterion"),
        ],
    )

    plan = ChangePlan(
        planned_changes=[
            PlannedChange(
                path="src/module.py",
                action=ChangeAction.MODIFY,
                description="Fix bug",
                acceptance_criteria=["AC-1"],
            ),
        ],
        risk_level="low",
    )

    actual_changes = [
        WorkspaceChange(path="src/module.py", action="modify"),
    ]

    report = VerificationReport(
        run_id="test-run",
        passed=True,
        checks=[
            CheckReport(
                method="ast_check",
                phase="post_patch",
                level="LEVEL_2_TARGET_TESTS",
                command="ast_check --verify-interface",
                passed=True,
                exit_code=0,
                duration_seconds=1.0,
                subject_ids=["AC-1"],
                direct=True,
            ),
        ],
    )

    evidence = map_acceptance_evidence(issue, plan, actual_changes, report)

    assert len(evidence) == 1
    assert evidence[0].criterion_id == "AC-1"
    assert evidence[0].structural_contract is not None
    assert evidence[0].structural_contract.status == StructuralContractStatus.PASS
    assert evidence[0].structural_contract.has_specialized_check is True


def test_structural_contract_pytest_only():
    """Test structural contract with only pytest (UNVERIFIED)."""
    issue = NormalizedIssue(
        title="Test issue",
        task_type="bug",
        problem_statement="Test problem",
        acceptance_criteria=[
            AcceptanceCriterion(id="AC-1", description="First criterion"),
        ],
    )

    plan = ChangePlan(
        planned_changes=[
            PlannedChange(
                path="src/module.py",
                action=ChangeAction.MODIFY,
                description="Fix bug",
                acceptance_criteria=["AC-1"],
            ),
        ],
        risk_level="low",
    )

    actual_changes = [
        WorkspaceChange(path="src/module.py", action="modify"),
    ]

    report = VerificationReport(
        run_id="test-run",
        passed=True,
        checks=[
            CheckReport(
                method="pytest",
                phase="post_patch",
                level="LEVEL_2_TARGET_TESTS",
                command="pytest tests/test_module.py",
                passed=True,
                exit_code=0,
                duration_seconds=1.0,
                subject_ids=["AC-1"],
                direct=True,
            ),
        ],
    )

    evidence = map_acceptance_evidence(issue, plan, actual_changes, report)

    assert len(evidence) == 1
    assert evidence[0].criterion_id == "AC-1"
    assert evidence[0].structural_contract is not None
    assert evidence[0].structural_contract.status == StructuralContractStatus.UNVERIFIED
    assert evidence[0].structural_contract.has_pytest_only is True


def test_constraint_compliant():
    """Test constraint compliance when no violations occur."""
    issue = NormalizedIssue(
        title="Test issue",
        task_type="bug",
        problem_statement="Test problem",
        acceptance_criteria=[
            AcceptanceCriterion(id="AC-1", description="First criterion"),
        ],
    )

    plan = ChangePlan(
        planned_changes=[
            PlannedChange(
                path="src/module.py",
                action=ChangeAction.MODIFY,
                description="Fix bug",
                acceptance_criteria=["AC-1"],
            ),
        ],
        risk_level="low",
    )

    actual_changes = [
        WorkspaceChange(path="src/module.py", action="modify"),
    ]

    report = VerificationReport(
        run_id="test-run",
        passed=True,
        checks=[
            CheckReport(
                method="constraint_audit",
                phase="constraint_audit",
                level="LEVEL_4_CONSTRAINT",
                command="constraint-audit",
                passed=True,
                exit_code=0,
                duration_seconds=1.0,
                subject_ids=["AC-1"],
                direct=True,
                summary={"violation_type": None},
            ),
        ],
    )

    evidence = map_acceptance_evidence(issue, plan, actual_changes, report)

    assert len(evidence) == 1
    assert evidence[0].criterion_id == "AC-1"
    assert evidence[0].status == EvidenceStatus.UNVERIFIED
    assert evidence[0].constraint is not None
    assert evidence[0].constraint.status == ConstraintStatus.COMPLIANT


def test_constraint_violated():
    """Test constraint when hard policy is violated."""
    issue = NormalizedIssue(
        title="Test issue",
        task_type="bug",
        problem_statement="Test problem",
        acceptance_criteria=[
            AcceptanceCriterion(id="AC-1", description="First criterion"),
        ],
    )

    plan = ChangePlan(
        planned_changes=[
            PlannedChange(
                path="src/module.py",
                action=ChangeAction.MODIFY,
                description="Fix bug",
                acceptance_criteria=["AC-1"],
            ),
        ],
        risk_level="low",
    )

    actual_changes = [
        WorkspaceChange(path="src/module.py", action="modify"),
    ]

    report = VerificationReport(
        run_id="test-run",
        passed=False,
        checks=[
            CheckReport(
                method="constraint_audit",
                phase="constraint_audit",
                level="LEVEL_4_CONSTRAINT",
                command="constraint-audit",
                passed=False,
                exit_code=1,
                duration_seconds=1.0,
                subject_ids=["AC-1"],
                direct=True,
                summary={"violation_type": "hard_policy"},
            ),
        ],
    )

    evidence = map_acceptance_evidence(issue, plan, actual_changes, report)

    assert len(evidence) == 1
    assert evidence[0].criterion_id == "AC-1"
    assert evidence[0].status == EvidenceStatus.FAIL
    assert evidence[0].constraint is not None
    assert evidence[0].constraint.status == ConstraintStatus.VIOLATED
