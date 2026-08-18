"""Tests for evidence mapper functionality."""

from patchpilot.evidence import EvidenceStatus, map_acceptance_evidence
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
            CheckReport(
                level="LEVEL_2_TARGET_TESTS",
                command="pytest tests/test_module.py",
                passed=True,
                exit_code=0,
                duration_seconds=1.0,
                acceptance_criteria=["AC-1"],
                direct_acceptance_criteria=["AC-1"],
            ),
            CheckReport(
                level="LEVEL_2_TARGET_TESTS",
                command="pytest tests/test_other.py",
                passed=True,
                exit_code=0,
                duration_seconds=1.0,
                acceptance_criteria=["AC-2"],
                direct_acceptance_criteria=["AC-2"],
            ),
        ],
    )

    evidence = map_acceptance_evidence(issue, plan, actual_changes, report)

    assert len(evidence) == 2
    assert evidence[0].criterion_id == "AC-1"
    assert evidence[0].status == EvidenceStatus.PASS
    assert "src/module.py" in evidence[0].changed_files
    assert evidence[1].criterion_id == "AC-2"
    assert evidence[1].status == EvidenceStatus.PASS
    assert "src/other.py" in evidence[1].changed_files


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
                level="LEVEL_2_TARGET_TESTS",
                command="pytest tests/test_statistics.py",
                passed=True,
                exit_code=0,
                duration_seconds=1.0,
                acceptance_criteria=["AC-1"],
            )
        ],
    )

    evidence = map_acceptance_evidence(
        issue,
        plan,
        [WorkspaceChange(path="src/statistics.py", action="modify")],
        report,
    )

    assert evidence[0].status == EvidenceStatus.UNVERIFIED
    assert "direct behavioral evidence" in evidence[0].explanation


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
            CheckReport(
                level="LEVEL_2_TARGET_TESTS",
                command="pytest tests/test_module.py",
                passed=False,
                exit_code=1,
                duration_seconds=1.0,
                failure_type="AssertionError",
                acceptance_criteria=["AC-1"],
            ),
        ],
    )

    evidence = map_acceptance_evidence(issue, plan, actual_changes, report)

    assert len(evidence) == 1
    assert evidence[0].criterion_id == "AC-1"
    assert evidence[0].status == EvidenceStatus.FAIL
    assert "failed" in evidence[0].explanation.lower()


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
                level="LEVEL_2_TARGET_TESTS",
                command="pytest tests/test_module.py",
                passed=True,
                exit_code=0,
                duration_seconds=1.0,
                acceptance_criteria=["AC-1"],
            ),
        ],
    )

    evidence = map_acceptance_evidence(issue, plan, actual_changes, report)

    assert len(evidence) == 1
    assert evidence[0].criterion_id == "AC-1"
    assert evidence[0].status == EvidenceStatus.UNVERIFIED
    assert "lacks" in evidence[0].explanation.lower()


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
                level="LEVEL_1_LINT",
                command="ruff check",
                passed=True,
                exit_code=0,
                duration_seconds=1.0,
                acceptance_criteria=[],  # No AC mapping
            ),
        ],
    )

    evidence = map_acceptance_evidence(issue, plan, actual_changes, report)

    assert len(evidence) == 1
    assert evidence[0].criterion_id == "AC-1"
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
                level="LEVEL_3_REGRESSION",
                command="pytest tests/test_regression.py",
                passed=True,
                exit_code=0,
                duration_seconds=1.0,
                acceptance_criteria=[],  # Not mapped to AC-1
            ),
            CheckReport(
                level="LEVEL_1_LINT",
                command="ruff check",
                passed=True,
                exit_code=0,
                duration_seconds=1.0,
                acceptance_criteria=[],  # Not mapped to AC-1
            ),
        ],
    )

    evidence = map_acceptance_evidence(issue, plan, actual_changes, report)

    assert len(evidence) == 1
    assert evidence[0].criterion_id == "AC-1"
    # Should be UNVERIFIED since no tests are specifically mapped to AC-1
    assert evidence[0].status == EvidenceStatus.UNVERIFIED
    assert "lacks" in evidence[0].explanation.lower()
