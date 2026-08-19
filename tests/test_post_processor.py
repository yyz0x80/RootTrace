"""Tests for plan post-processor module."""

from __future__ import annotations

import pytest

from patchpilot.issue.schema import AcceptanceCriterion, NormalizedIssue
from patchpilot.planning.post_processor import (
    PlanPostProcessError,
    _check_ambiguity,
    _classify_ambiguities,
    _complete_ac_mapping,
    _is_test_file,
    _merge_duplicate_file_changes,
    _migrate_test_files_to_tests,
    _validate_pytest_targets,
    post_process_plan,
)
from patchpilot.planning.schema import (
    ChangeAction,
    ChangePlan,
    CriterionPlan,
    CriterionPlanDetail,
    PlannedChange,
    PlannedTest,
)
from patchpilot.repository.schema import RepositoryContext


def test_is_test_file() -> None:
    """Test test file detection logic."""
    # Files in tests/ directory
    assert _is_test_file("tests/test_example.py")
    assert _is_test_file("tests/unit/test_utils.py")
    assert _is_test_file("src/tests/test_integration.py")

    # Files starting with test_
    assert _is_test_file("test_example.py")
    assert _is_test_file("test_utils.py")

    # Non-test files
    assert not _is_test_file("src/main.py")
    assert not _is_test_file("utils.py")
    assert not _is_test_file("example.py")


def test_migrate_test_files_to_tests() -> None:
    """Test automatic migration of test files from planned_changes to planned_tests."""
    plan = ChangePlan(
        base_commit="abc123",
        repository_match=True,
        relevant_files=["src/main.py", "tests/test_main.py"],
        planned_changes=[
            PlannedChange(
                path="src/main.py",
                action=ChangeAction.MODIFY,
                description="Fix bug in main",
                acceptance_criteria=["AC-1"],
                criterion_ids=["AC-1"],
            ),
            PlannedChange(
                path="tests/test_main.py",
                action=ChangeAction.MODIFY,
                description="Update test",
                acceptance_criteria=["AC-1"],
                criterion_ids=["AC-1"],
            ),
        ],
        planned_tests=[],
        out_of_scope=[],
        risk_level="low",
    )

    processed = _migrate_test_files_to_tests(plan)

    # Test file should be removed from planned_changes
    assert len(processed.planned_changes) == 1
    assert processed.planned_changes[0].path == "src/main.py"

    # Test file should be added to planned_tests
    assert len(processed.planned_tests) == 1
    assert processed.planned_tests[0].command == "pytest tests/test_main.py -q"
    assert processed.planned_tests[0].acceptance_criteria == ["AC-1"]


def test_migrate_test_files_to_tests_with_test_prefix() -> None:
    """Test migration of test files starting with test_ prefix."""
    plan = ChangePlan(
        base_commit="abc123",
        repository_match=True,
        relevant_files=["test_main.py"],
        planned_changes=[
            PlannedChange(
                path="test_main.py",
                action=ChangeAction.MODIFY,
                description="Update test",
                acceptance_criteria=["AC-1"],
                criterion_ids=["AC-1"],
            ),
        ],
        planned_tests=[],
        out_of_scope=[],
        risk_level="low",
    )

    processed = _migrate_test_files_to_tests(plan)

    assert len(processed.planned_changes) == 0
    assert len(processed.planned_tests) == 1
    assert processed.planned_tests[0].command == "pytest test_main.py -q"


def test_merge_duplicate_file_changes() -> None:
    """Test merging of multiple planned changes for the same file."""
    plan = ChangePlan(
        base_commit="abc123",
        repository_match=True,
        relevant_files=["src/main.py"],
        planned_changes=[
            PlannedChange(
                path="src/main.py",
                action=ChangeAction.MODIFY,
                description="Fix bug",
                acceptance_criteria=["AC-1"],
                criterion_ids=["AC-1"],
            ),
            PlannedChange(
                path="src/main.py",
                action=ChangeAction.MODIFY,
                description="Add feature",
                acceptance_criteria=["AC-2"],
                criterion_ids=["AC-2"],
            ),
        ],
        planned_tests=[],
        out_of_scope=[],
        risk_level="low",
    )

    processed = _merge_duplicate_file_changes(plan)

    # Should be merged into one change
    assert len(processed.planned_changes) == 1
    assert processed.planned_changes[0].path == "src/main.py"
    assert "Fix bug" in processed.planned_changes[0].description
    assert "Add feature" in processed.planned_changes[0].description
    assert set(processed.planned_changes[0].acceptance_criteria) == {"AC-1", "AC-2"}


def test_merge_duplicate_file_changes_dedup_ac() -> None:
    """Test that duplicate acceptance criteria are deduplicated."""
    plan = ChangePlan(
        base_commit="abc123",
        repository_match=True,
        relevant_files=["src/main.py"],
        planned_changes=[
            PlannedChange(
                path="src/main.py",
                action=ChangeAction.MODIFY,
                description="Fix bug",
                acceptance_criteria=["AC-1", "AC-2"],
                criterion_ids=["AC-1", "AC-2"],
            ),
            PlannedChange(
                path="src/main.py",
                action=ChangeAction.MODIFY,
                description="Add feature",
                acceptance_criteria=["AC-2", "AC-3"],
                criterion_ids=["AC-2", "AC-3"],
            ),
        ],
        planned_tests=[],
        out_of_scope=[],
        risk_level="low",
    )

    processed = _merge_duplicate_file_changes(plan)

    assert len(processed.planned_changes) == 1
    assert set(processed.planned_changes[0].acceptance_criteria) == {
        "AC-1",
        "AC-2",
        "AC-3",
    }


def test_validate_pytest_targets_success() -> None:
    """Test successful validation of pytest targets."""
    plan = ChangePlan(
        base_commit="abc123",
        repository_match=True,
        relevant_files=["tests/test_main.py"],
        planned_changes=[],
        planned_tests=[
            PlannedTest(
                command="pytest tests/test_main.py -q",
                purpose="Verify main",
                acceptance_criteria=["AC-1"],
                criterion_ids=["AC-1"],
            ),
        ],
        out_of_scope=[],
        risk_level="low",
    )

    repository_context = RepositoryContext(
        base_commit="abc123",
        tracked_files=["tests/test_main.py", "src/main.py"],
        python_files=["tests/test_main.py", "src/main.py"],
        test_files=["tests/test_main.py"],
        config_files=[],
        keyword_matches=[],
    )

    # Should not raise
    processed = _validate_pytest_targets(plan, repository_context)
    assert processed == plan


def test_validate_pytest_targets_nonexistent() -> None:
    """Test validation fails for non-existent pytest targets."""
    plan = ChangePlan(
        base_commit="abc123",
        repository_match=True,
        relevant_files=["tests/test_main.py"],
        planned_changes=[],
        planned_tests=[
            PlannedTest(
                command="pytest tests/nonexistent.py -q",
                purpose="Verify nonexistent",
                acceptance_criteria=["AC-1"],
                criterion_ids=["AC-1"],
            ),
        ],
        out_of_scope=[],
        risk_level="low",
    )

    repository_context = RepositoryContext(
        base_commit="abc123",
        tracked_files=["tests/test_main.py", "src/main.py"],
        python_files=["tests/test_main.py", "src/main.py"],
        test_files=["tests/test_main.py"],
        config_files=[],
        keyword_matches=[],
    )

    with pytest.raises(PlanPostProcessError, match="does not exist"):
        _validate_pytest_targets(plan, repository_context)


def test_validate_pytest_targets_non_test_file() -> None:
    """Test validation fails for non-test file targets."""
    plan = ChangePlan(
        base_commit="abc123",
        repository_match=True,
        relevant_files=["src/main.py"],
        planned_changes=[],
        planned_tests=[
            PlannedTest(
                command="pytest src/main.py -q",
                purpose="Verify main",
                acceptance_criteria=["AC-1"],
                criterion_ids=["AC-1"],
            ),
        ],
        out_of_scope=[],
        risk_level="low",
    )

    repository_context = RepositoryContext(
        base_commit="abc123",
        tracked_files=["src/main.py"],
        python_files=["src/main.py"],
        test_files=[],
        config_files=[],
        keyword_matches=[],
    )

    with pytest.raises(PlanPostProcessError, match="must be a test file"):
        _validate_pytest_targets(plan, repository_context)


def test_check_ambiguity_with_ambiguous_points() -> None:
    """Test that non-empty ambiguous_points raises error."""
    plan = ChangePlan(
        base_commit="abc123",
        repository_match=True,
        relevant_files=[],
        planned_changes=[],
        planned_tests=[],
        out_of_scope=[],
        risk_level="low",
    )

    issue = NormalizedIssue(
        title="Test",
        task_type="feature",
        problem_statement="Test issue",
        acceptance_criteria=[],
        constraints=[],
        ambiguous_points=["What should happen when X?"],
        expected_test_areas=[],
        implementation_notes=[],
    )

    with pytest.raises(PlanPostProcessError, match="ambiguous points"):
        _check_ambiguity(plan, issue)


def test_check_ambiguity_no_ambiguous_points() -> None:
    """Test that empty ambiguous_points passes."""
    plan = ChangePlan(
        base_commit="abc123",
        repository_match=True,
        relevant_files=[],
        planned_changes=[],
        planned_tests=[],
        out_of_scope=[],
        risk_level="low",
    )

    issue = NormalizedIssue(
        title="Test",
        task_type="feature",
        problem_statement="Test issue",
        acceptance_criteria=[],
        constraints=[],
        ambiguous_points=[],
        expected_test_areas=[],
        implementation_notes=[],
    )

    # Should not raise
    processed = _check_ambiguity(plan, issue)
    assert processed == plan


def test_classify_ambiguities_ordering_without_rules() -> None:
    """Test that ordering without explicit rules now passes (keyword check removed)."""
    plan = ChangePlan(
        base_commit="abc123",
        repository_match=True,
        relevant_files=[],
        planned_changes=[],
        planned_tests=[],
        out_of_scope=[],
        risk_level="low",
    )

    issue = NormalizedIssue(
        title="Test",
        task_type="feature",
        problem_statement="Sort items by priority",
        acceptance_criteria=[],
        constraints=[],
        ambiguous_points=[],
        expected_test_areas=[],
        implementation_notes=[],
    )

    # Should not raise - keyword-based classification removed
    processed = _classify_ambiguities(plan, issue)
    assert processed == plan


def test_classify_ambiguities_ordering_with_rules() -> None:
    """Test that ordering with explicit rules passes (no-op after keyword removal)."""
    plan = ChangePlan(
        base_commit="abc123",
        repository_match=True,
        relevant_files=[],
        planned_changes=[],
        planned_tests=[],
        out_of_scope=[],
        risk_level="low",
    )

    issue = NormalizedIssue(
        title="Test",
        task_type="feature",
        problem_statement="Sort items in order of creation date",
        acceptance_criteria=[],
        constraints=[],
        ambiguous_points=[],
        expected_test_areas=[],
        implementation_notes=[],
    )

    # Should not raise - function is now a no-op
    processed = _classify_ambiguities(plan, issue)
    assert processed == plan


def test_classify_ambiguities_no_ordering() -> None:
    """Test that issues without ordering references pass."""
    plan = ChangePlan(
        base_commit="abc123",
        repository_match=True,
        relevant_files=[],
        planned_changes=[],
        planned_tests=[],
        out_of_scope=[],
        risk_level="low",
    )

    issue = NormalizedIssue(
        title="Test",
        task_type="feature",
        problem_statement="Add a new feature",
        acceptance_criteria=[],
        constraints=[],
        ambiguous_points=[],
        expected_test_areas=[],
        implementation_notes=[],
    )

    # Should not raise
    processed = _classify_ambiguities(plan, issue)
    assert processed == plan


def test_complete_ac_mapping_unknown_ac_failure() -> None:
    """Test that unknown AC references now hard fail instead of auto-correction."""
    plan = ChangePlan(
        base_commit="abc123",
        repository_match=True,
        relevant_files=["src/main.py"],
        planned_changes=[
            PlannedChange(
                path="src/main.py",
                action=ChangeAction.MODIFY,
                description="Fix bug",
                acceptance_criteria=["ac-1"],  # Typo: lowercase
                criterion_ids=["ac-1"],
            ),
        ],
        planned_tests=[
            PlannedTest(
                command="pytest tests/test_main.py -q",
                purpose="Verify",
                acceptance_criteria=["ac-1"],  # Typo: lowercase
                criterion_ids=["ac-1"],
            ),
        ],
        out_of_scope=[],
        risk_level="low",
    )

    issue = NormalizedIssue(
        title="Test",
        task_type="feature",
        problem_statement="Test issue",
        acceptance_criteria=[
            AcceptanceCriterion(id="AC-1", description="Test criterion"),
        ],
        constraints=[],
        ambiguous_points=[],
        expected_test_areas=[],
        implementation_notes=[],
    )

    # Should now hard fail instead of auto-correcting
    with pytest.raises(PlanPostProcessError, match="unknown acceptance criteria"):
        _complete_ac_mapping(plan, issue)


def test_complete_ac_mapping_missing_test_allowed() -> None:
    """Test that missing test mappings no longer hard fail (GLM median case)."""
    plan = ChangePlan(
        base_commit="abc123",
        repository_match=True,
        relevant_files=["src/main.py"],
        planned_changes=[
            PlannedChange(
                path="src/main.py",
                action=ChangeAction.MODIFY,
                description="Fix bug",
                acceptance_criteria=["AC-1"],
                criterion_ids=["AC-1"],
            ),
        ],
        planned_tests=[],  # Missing test for AC-1
        out_of_scope=[],
        risk_level="low",
        criterion_plans=[
            CriterionPlan(
                criterion_id="AC-1",
                disposition=CriterionPlanDetail.TO_IMPLEMENT,
                relevant_source_files=["src/main.py"],
            ),
        ],
    )

    issue = NormalizedIssue(
        title="Test",
        task_type="feature",
        problem_statement="Test issue",
        acceptance_criteria=[
            AcceptanceCriterion(id="AC-1", description="Test criterion"),
        ],
        constraints=[],
        ambiguous_points=[],
        expected_test_areas=[],
        implementation_notes=[],
    )

    # Should now pass (validator will generate warning instead)
    processed = _complete_ac_mapping(plan, issue)
    assert processed == plan


def test_post_process_plan_integration() -> None:
    """Test complete post-processing pipeline."""
    plan = ChangePlan(
        base_commit="abc123",
        repository_match=True,
        relevant_files=["src/main.py", "tests/test_main.py"],
        planned_changes=[
            PlannedChange(
                path="src/main.py",
                action=ChangeAction.MODIFY,
                description="Fix bug",
                acceptance_criteria=["AC-1"],
                criterion_ids=["AC-1"],
            ),
            PlannedChange(
                path="tests/test_main.py",
                action=ChangeAction.MODIFY,
                description="Update test",
                acceptance_criteria=["AC-1"],
                criterion_ids=["AC-1"],
            ),
        ],
        planned_tests=[],
        out_of_scope=[],
        risk_level="low",
        criterion_plans=[
            CriterionPlan(
                criterion_id="AC-1",
                disposition=CriterionPlanDetail.TO_IMPLEMENT,
                relevant_source_files=["src/main.py"],
            ),
        ],
    )

    issue = NormalizedIssue(
        title="Test",
        task_type="feature",
        problem_statement="Add a new feature",
        acceptance_criteria=[
            AcceptanceCriterion(id="AC-1", description="Test criterion"),
        ],
        constraints=[],
        ambiguous_points=[],
        expected_test_areas=[],
        implementation_notes=[],
    )

    repository_context = RepositoryContext(
        base_commit="abc123",
        tracked_files=["src/main.py", "tests/test_main.py"],
        python_files=["src/main.py", "tests/test_main.py"],
        test_files=["tests/test_main.py"],
        config_files=[],
        keyword_matches=[],
    )

    processed = post_process_plan(plan, issue, repository_context)

    # Test file should be migrated
    assert len(processed.planned_changes) == 1
    assert processed.planned_changes[0].path == "src/main.py"

    # Test should be added
    assert len(processed.planned_tests) == 1
    assert "pytest tests/test_main.py" in processed.planned_tests[0].command


def test_post_process_plan_with_ambiguity() -> None:
    """Test that post-processing fails with ambiguous points."""
    plan = ChangePlan(
        base_commit="abc123",
        repository_match=True,
        relevant_files=[],
        planned_changes=[],
        planned_tests=[],
        out_of_scope=[],
        risk_level="low",
    )

    issue = NormalizedIssue(
        title="Test",
        task_type="feature",
        problem_statement="Test issue",
        acceptance_criteria=[],
        constraints=[],
        ambiguous_points=["What should happen?"],
        expected_test_areas=[],
        implementation_notes=[],
    )

    repository_context = RepositoryContext(
        base_commit="abc123",
        tracked_files=[],
        python_files=[],
        test_files=[],
        config_files=[],
        keyword_matches=[],
    )

    with pytest.raises(PlanPostProcessError, match="ambiguous points"):
        post_process_plan(plan, issue, repository_context)
