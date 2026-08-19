"""Tests for the plan validator module."""

import pytest

from patchpilot.issue.schema import AcceptanceCriterion, NormalizedIssue
from patchpilot.planning.schema import (
    ChangePlan,
    CriterionPlan,
    CriterionPlanDetail,
    PlannedChange,
    PlannedTest,
)
from patchpilot.planning.validator import (
    validate_acceptance_coverage,
    validate_command_support,
    validate_plan,
    validate_plan_against_repository,
    validate_planned_changes,
    validate_test_targets,
)
from patchpilot.repository.schema import RepositoryContext


def test_validate_plan_success():
    """Test validation passes for a valid plan with proper repository context."""
    plan = ChangePlan(
        repository_match=True,
        relevant_files=["src/main.py", "src/utils.py"],
        planned_changes=[
            PlannedChange(
                path="src/main.py",
                action="modify",
                description="Fix bug",
                criterion_ids=["AC-1"],
            ),
            PlannedChange(
                path="src/utils.py",
                action="modify",
                description="Update utility function",
                criterion_ids=["AC-2"],
            )
        ],
        risk_level="low"
    )

    repo_context = RepositoryContext(
        base_commit="abc123",
        tracked_files=["src/main.py", "src/utils.py", "tests/test_main.py"],
        python_files=["src/main.py", "src/utils.py"],
        test_files=["tests/test_main.py"],
        config_files=[],
        keyword_matches=[]
    )

    result = validate_plan(plan, repo_context)

    assert result.allowed is True
    assert len(result.violations) == 0
    assert len(result.warnings) == 0


def test_validate_plan_with_scope_violation():
    """Test validation catches scope violations."""
    plan = ChangePlan(
        repository_match=True,
        relevant_files=[".env"],
        planned_changes=[
            PlannedChange(
                path=".env",
                action="modify",
                description="Update config",
                criterion_ids=[],
            )
        ],
        risk_level="low"
    )

    repo_context = RepositoryContext(
        base_commit="abc123",
        tracked_files=[".env", "src/main.py"],
        python_files=["src/main.py"],
        test_files=[],
        config_files=[],
        keyword_matches=[]
    )

    result = validate_plan(plan, repo_context)

    assert result.allowed is False
    assert len(result.violations) > 0
    assert any(".env" in v for v in result.violations)


def test_validate_plan_reports_scope_before_missing_coverage():
    """Unsafe plans should be blocked without raising a coverage error."""
    issue = NormalizedIssue(
        title="Disable CI",
        task_type="feature",
        problem_statement="Disable quality checks.",
        acceptance_criteria=[
            AcceptanceCriterion(id="AC-1", description="Disable CI"),
        ],
    )
    plan = ChangePlan(
        repository_match=True,
        relevant_files=[".github/workflows/ci.yml"],
        planned_changes=[
            PlannedChange(
                path=".github/workflows/ci.yml",
                action="modify",
                description="Disable CI",
                criterion_ids=["AC-1"],
            ),
        ],
        risk_level="low",
        criterion_plans=[
            CriterionPlan(
                criterion_id="AC-1",
                disposition=CriterionPlanDetail.TO_IMPLEMENT,
                relevant_source_files=[".github/workflows/ci.yml"],
            ),
        ],
    )
    repository_context = RepositoryContext(
        base_commit="abc123",
        tracked_files=[".github/workflows/ci.yml"],
        python_files=[],
        test_files=[],
        config_files=[".github/workflows/ci.yml"],
        keyword_matches=[],
    )

    result = validate_plan(plan, repository_context, issue=issue)

    assert not result.allowed
    assert any("CI/CD" in violation for violation in result.violations)


def test_validate_plan_with_repository_mismatch():
    """Test validation raises error for repository mismatch."""
    plan = ChangePlan(
        repository_match=False,
        repository_mismatch_reason="No existing Task model found",
        planned_changes=[],
        risk_level="low"
    )

    repo_context = RepositoryContext(
        base_commit="abc123",
        tracked_files=["src/main.py"],
        python_files=["src/main.py"],
        test_files=[],
        config_files=[],
        keyword_matches=[]
    )

    with pytest.raises(ValueError) as exc_info:
        validate_plan(plan, repo_context)

    assert "Issue does not match repository" in str(exc_info.value)
    assert "No existing Task model found" in str(exc_info.value)


def test_validate_plan_with_inconsistent_files():
    """Test validation raises error for inconsistent file references."""
    plan = ChangePlan(
        repository_match=True,
        relevant_files=["src/missing.py"],
        planned_changes=[
            PlannedChange(
                path="src/missing.py",
                action="modify",
                description="Fix bug",
                criterion_ids=[],
            )
        ],
        risk_level="low"
    )

    repo_context = RepositoryContext(
        base_commit="abc123",
        tracked_files=["src/main.py"],
        python_files=["src/main.py"],
        test_files=[],
        config_files=[],
        keyword_matches=[]
    )

    with pytest.raises(ValueError) as exc_info:
        validate_plan(plan, repo_context)

    assert "non-existing file" in str(exc_info.value)
    assert "modify" in str(exc_info.value)


def test_validate_plan_against_repository_success():
    """Test validation passes for a valid plan."""
    plan = ChangePlan(
        repository_match=True,
        planned_changes=[
            PlannedChange(
                path="src/main.py",
                action="modify",
                description="Fix bug",
                criterion_ids=[],
            ),
            PlannedChange(
                path="tests/test_new.py",
                action="create",
                description="Add new test",
                criterion_ids=[],
            )
        ],
        risk_level="low"
    )

    repo_context = RepositoryContext(
        base_commit="abc123",
        tracked_files=["src/main.py", "src/utils.py", "tests/test_main.py"],
        python_files=["src/main.py", "src/utils.py"],
        test_files=["tests/test_main.py"],
        config_files=[],
        keyword_matches=[]
    )

    # Should not raise
    validate_plan_against_repository(plan, repo_context)


def test_validate_plan_against_repository_mismatch():
    """Test validation fails when repository_match is False."""
    plan = ChangePlan(
        repository_match=False,
        repository_mismatch_reason="No existing Task model found",
        planned_changes=[],
        risk_level="low"
    )

    repo_context = RepositoryContext(
        base_commit="abc123",
        tracked_files=["src/main.py"],
        python_files=["src/main.py"],
        test_files=[],
        config_files=[],
        keyword_matches=[]
    )

    with pytest.raises(ValueError) as exc_info:
        validate_plan_against_repository(plan, repo_context)

    assert "Issue does not match repository" in str(exc_info.value)
    assert "No existing Task model found" in str(exc_info.value)


def test_validate_plan_against_repository_modify_nonexistent():
    """Test validation fails when modifying a non-existent file."""
    plan = ChangePlan(
        repository_match=True,
        planned_changes=[
            PlannedChange(
                path="src/missing.py",
                action="modify",
                description="Fix bug",
                criterion_ids=[],
            )
        ],
        risk_level="low"
    )

    repo_context = RepositoryContext(
        base_commit="abc123",
        tracked_files=["src/main.py"],
        python_files=["src/main.py"],
        test_files=[],
        config_files=[],
        keyword_matches=[]
    )

    with pytest.raises(ValueError) as exc_info:
        validate_plan_against_repository(plan, repo_context)

    assert "Plan references a non-existing file" in str(exc_info.value)
    assert "modify" in str(exc_info.value)
    assert "src/missing.py" in str(exc_info.value)


def test_validate_plan_against_repository_delete_nonexistent():
    """Test validation fails when deleting a non-existent file."""
    plan = ChangePlan(
        repository_match=True,
        planned_changes=[
            PlannedChange(
                path="src/missing.py",
                action="delete",
                description="Remove file",
                criterion_ids=[],
            )
        ],
        risk_level="low"
    )

    repo_context = RepositoryContext(
        base_commit="abc123",
        tracked_files=["src/main.py"],
        python_files=["src/main.py"],
        test_files=[],
        config_files=[],
        keyword_matches=[]
    )

    with pytest.raises(ValueError) as exc_info:
        validate_plan_against_repository(plan, repo_context)

    assert "Plan references a non-existing file" in str(exc_info.value)
    assert "delete" in str(exc_info.value)
    assert "src/missing.py" in str(exc_info.value)


def test_validate_plan_against_repository_create_existing():
    """Test validation fails when creating a file that already exists."""
    plan = ChangePlan(
        repository_match=True,
        planned_changes=[
            PlannedChange(
                path="src/main.py",
                action="create",
                description="Add new file",
                criterion_ids=[],
            )
        ],
        risk_level="low"
    )

    repo_context = RepositoryContext(
        base_commit="abc123",
        tracked_files=["src/main.py"],
        python_files=["src/main.py"],
        test_files=[],
        config_files=[],
        keyword_matches=[]
    )

    with pytest.raises(ValueError) as exc_info:
        validate_plan_against_repository(plan, repo_context)

    assert "Plan attempts to create an existing file" in str(exc_info.value)
    assert "src/main.py" in str(exc_info.value)


def test_validate_plan_against_repository_mixed_actions():
    """Test validation with mixed create, modify, and delete actions."""
    plan = ChangePlan(
        repository_match=True,
        planned_changes=[
            PlannedChange(
                path="src/main.py",
                action="modify",
                description="Update existing file",
                criterion_ids=[],
            ),
            PlannedChange(
                path="src/deprecated.py",
                action="delete",
                description="Remove deprecated file",
                criterion_ids=[],
            ),
            PlannedChange(
                path="src/new_feature.py",
                action="create",
                description="Add new feature",
                criterion_ids=[],
            )
        ],
        risk_level="low"
    )

    repo_context = RepositoryContext(
        base_commit="abc123",
        tracked_files=["src/main.py", "src/deprecated.py", "src/utils.py"],
        python_files=["src/main.py", "src/deprecated.py", "src/utils.py"],
        test_files=[],
        config_files=[],
        keyword_matches=[]
    )

    # Should not raise
    validate_plan_against_repository(plan, repo_context)


def _make_issue(
    criterion_ids: list[str] | None = None,
) -> NormalizedIssue:
    """Create a normalized issue for AC coverage validation tests."""
    ids = criterion_ids or ["AC-1", "AC-2"]
    return NormalizedIssue(
        title="Test issue",
        task_type="feature",
        problem_statement="Implement observable behavior.",
        acceptance_criteria=[
            AcceptanceCriterion(
                id=criterion_id,
                description=f"Requirement {criterion_id}",
            )
            for criterion_id in ids
        ],
    )


def _make_covered_plan() -> ChangePlan:
    """Create a plan with complete change and verification mappings."""
    return ChangePlan(
        relevant_files=["src/main.py", "tests/test_main.py"],
        planned_changes=[
            PlannedChange(
                path="src/main.py",
                action="modify",
                description="Implement requirements",
                acceptance_criteria=["AC-1", "AC-2"],
                criterion_ids=["AC-1", "AC-2"],
            )
        ],
        planned_tests=[
            PlannedTest(
                command="pytest tests/test_main.py -q",
                purpose="Verify requirements",
                acceptance_criteria=["AC-1", "AC-2"],
                criterion_ids=["AC-1", "AC-2"],
            )
        ],
        risk_level="low",
        criterion_plans=[
            CriterionPlan(
                criterion_id="AC-1",
                disposition=CriterionPlanDetail.TO_IMPLEMENT,
                relevant_source_files=["src/main.py"],
            ),
            CriterionPlan(
                criterion_id="AC-2",
                disposition=CriterionPlanDetail.TO_IMPLEMENT,
                relevant_source_files=["src/main.py"],
            ),
        ],
    )


def test_validate_acceptance_coverage_success():
    """Accept plans that map every AC to changes and verification."""
    validate_acceptance_coverage(
        _make_covered_plan(),
        _make_issue(),
    )


def test_validate_acceptance_coverage_rejects_unknown_id():
    """Reject plan mappings that reference an AC absent from the issue."""
    plan = _make_covered_plan()
    plan.planned_tests[0].acceptance_criteria.append("AC-99")
    plan.planned_tests[0].criterion_ids.append("AC-99")

    with pytest.raises(
        ValueError,
        match="unknown acceptance criterion IDs: AC-99",
    ):
        validate_acceptance_coverage(plan, _make_issue())


def test_validate_acceptance_coverage_requires_source_change():
    """No longer reject ACs without planned source implementation (warning only)."""
    plan = _make_covered_plan()
    plan.planned_changes[0].acceptance_criteria.remove("AC-2")
    plan.planned_changes[0].criterion_ids.remove("AC-2")

    # Should now generate warning instead of hard failure
    warnings = validate_acceptance_coverage(plan, _make_issue())
    assert len(warnings) > 0
    assert any("AC-2" in warning and "source changes" in warning for warning in warnings)


def test_validate_acceptance_coverage_requires_verification():
    """No longer reject ACs without planned verification (warning only)."""
    plan = _make_covered_plan()
    plan.planned_tests[0].acceptance_criteria.remove("AC-2")
    plan.planned_tests[0].criterion_ids.remove("AC-2")

    # Should now generate warning instead of hard failure
    warnings = validate_acceptance_coverage(plan, _make_issue())
    assert len(warnings) > 0
    assert any("AC-2" in warning for warning in warnings)


def test_validate_acceptance_coverage_rejects_duplicate_issue_ids():
    """Reject duplicate AC IDs in the authoritative normalized issue."""
    with pytest.raises(
        ValueError,
        match="duplicate acceptance criterion IDs",
    ):
        validate_acceptance_coverage(
            _make_covered_plan(),
            _make_issue(["AC-1", "AC-1"]),
        )


def test_validate_plan_checks_acceptance_coverage_when_issue_is_given():
    """Run AC coverage validation through the public validator entry point."""
    plan = _make_covered_plan()
    repo_context = RepositoryContext(
        base_commit="abc123",
        tracked_files=["src/main.py", "tests/test_main.py"],
        python_files=["src/main.py"],
        test_files=["tests/test_main.py"],
        config_files=[],
        keyword_matches=[],
    )

    result = validate_plan(
        plan,
        repo_context,
        issue=_make_issue(),
    )

    assert result.allowed is True


def test_validate_acceptance_coverage_behavior_change_without_paths():
    """Hard failure: behavior change claims IMPLEMENT but has no planned paths."""
    plan = ChangePlan(
        relevant_files=["src/main.py"],
        planned_changes=[
            PlannedChange(
                path="src/main.py",
                action="modify",
                description="Implement requirements",
                acceptance_criteria=["AC-1"],
                criterion_ids=["AC-1"],
            )
        ],
        planned_tests=[
            PlannedTest(
                command="pytest tests/test_main.py -q",
                purpose="Verify requirements",
                acceptance_criteria=["AC-1"],
                criterion_ids=["AC-1"],
            )
        ],
        risk_level="low",
        criterion_plans=[
            CriterionPlan(
                criterion_id="AC-1",
                disposition=CriterionPlanDetail.TO_IMPLEMENT,
                relevant_source_files=[],  # Empty - should hard fail
            ),
        ],
    )

    issue = NormalizedIssue(
        title="Test issue",
        task_type="feature",
        problem_statement="Implement observable behavior.",
        acceptance_criteria=[
            AcceptanceCriterion(
                id="AC-1",
                description="Behavior requirement",
                kind="behavior",  # Behavior kind
            ),
        ],
    )

    with pytest.raises(ValueError, match="claims IMPLEMENT but has no planned source paths"):
        validate_acceptance_coverage(plan, issue)


def test_validate_acceptance_coverage_structural_without_paths():
    """Hard failure: structural contract has no relevant planned paths."""
    plan = ChangePlan(
        relevant_files=["src/main.py"],
        planned_changes=[
            PlannedChange(
                path="src/main.py",
                action="modify",
                description="Implement requirements",
                acceptance_criteria=["AC-1"],
                criterion_ids=["AC-1"],
            )
        ],
        planned_tests=[
            PlannedTest(
                command="pytest tests/test_main.py -q",
                purpose="Verify requirements",
                acceptance_criteria=["AC-1"],
                criterion_ids=["AC-1"],
            )
        ],
        risk_level="low",
        criterion_plans=[
            CriterionPlan(
                criterion_id="AC-1",
                disposition=CriterionPlanDetail.TO_IMPLEMENT,
                relevant_source_files=[],  # Empty - should hard fail
            ),
        ],
    )

    issue = NormalizedIssue(
        title="Test issue",
        task_type="feature",
        problem_statement="Implement structural requirement.",
        acceptance_criteria=[
            AcceptanceCriterion(
                id="AC-1",
                description="Structural requirement",
                kind="structural",  # Structural kind
            ),
        ],
    )

    with pytest.raises(ValueError, match="has no relevant planned paths"):
        validate_acceptance_coverage(plan, issue)


def test_validate_acceptance_coverage_unknown_criterion_in_plans():
    """Hard failure: unknown criterion ID in criterion_plans."""
    plan = ChangePlan(
        relevant_files=["src/main.py"],
        planned_changes=[
            PlannedChange(
                path="src/main.py",
                action="modify",
                description="Implement requirements",
                acceptance_criteria=["AC-1"],
                criterion_ids=["AC-1"],
            )
        ],
        planned_tests=[
            PlannedTest(
                command="pytest tests/test_main.py -q",
                purpose="Verify requirements",
                acceptance_criteria=["AC-1"],
                criterion_ids=["AC-1"],
            )
        ],
        risk_level="low",
        criterion_plans=[
            CriterionPlan(
                criterion_id="AC-FAKE",  # Unknown ID - should hard fail
                disposition=CriterionPlanDetail.TO_IMPLEMENT,
                relevant_source_files=["src/main.py"],
            ),
        ],
    )

    issue = NormalizedIssue(
        title="Test issue",
        task_type="feature",
        problem_statement="Implement requirements.",
        acceptance_criteria=[
            AcceptanceCriterion(
                id="AC-1",
                description="Real requirement",
            ),
        ],
    )

    with pytest.raises(ValueError, match="not found in criterion_plans"):
        validate_acceptance_coverage(plan, issue)


def test_validate_planned_changes_success():
    """Test validation passes for valid planned changes."""
    plan = ChangePlan(
        repository_match=True,
        planned_changes=[
            PlannedChange(
                path="src/main.py",
                action="modify",
                description="Fix bug",
                criterion_ids=["AC-1"],
            ),
            PlannedChange(
                path="tests/test_new.py",
                action="create",
                description="Add new test",
                criterion_ids=["AC-2"],
            ),
        ],
        risk_level="low",
    )

    repo_context = RepositoryContext(
        base_commit="abc123",
        tracked_files=["src/main.py", "src/utils.py"],
        python_files=["src/main.py"],
        test_files=[],
        config_files=[],
        keyword_matches=[],
    )

    # Should not raise
    validate_planned_changes(plan, repo_context)


def test_validate_planned_changes_nonexistent_file():
    """Hard failure: modification path does not exist."""
    plan = ChangePlan(
        repository_match=True,
        planned_changes=[
            PlannedChange(
                path="src/missing.py",
                action="modify",
                description="Fix bug",
                criterion_ids=["AC-1"],
            ),
        ],
        risk_level="low",
    )

    repo_context = RepositoryContext(
        base_commit="abc123",
        tracked_files=["src/main.py"],
        python_files=["src/main.py"],
        test_files=[],
        config_files=[],
        keyword_matches=[],
    )

    with pytest.raises(ValueError, match="does not exist for modify"):
        validate_planned_changes(plan, repo_context)


def test_validate_planned_changes_create_existing():
    """Hard failure: create action on existing file."""
    plan = ChangePlan(
        repository_match=True,
        planned_changes=[
            PlannedChange(
                path="src/main.py",
                action="create",
                description="Add new file",
                criterion_ids=["AC-1"],
            ),
        ],
        risk_level="low",
    )

    repo_context = RepositoryContext(
        base_commit="abc123",
        tracked_files=["src/main.py"],
        python_files=["src/main.py"],
        test_files=[],
        config_files=[],
        keyword_matches=[],
    )

    with pytest.raises(ValueError, match="create action on existing file"):
        validate_planned_changes(plan, repo_context)


def test_validate_test_targets_success():
    """Test validation passes for valid test targets."""
    plan = ChangePlan(
        repository_match=True,
        planned_tests=[
            PlannedTest(
                command="pytest tests/test_main.py -q",
                purpose="Verify main",
                criterion_ids=["AC-1"],
            ),
        ],
        risk_level="low",
    )

    repo_context = RepositoryContext(
        base_commit="abc123",
        tracked_files=["tests/test_main.py"],
        python_files=["tests/test_main.py"],
        test_files=["tests/test_main.py"],
        config_files=[],
        keyword_matches=[],
    )

    # Should not raise
    validate_test_targets(plan, repo_context)


def test_validate_test_targets_nonexistent():
    """Hard failure: test target does not exist."""
    plan = ChangePlan(
        repository_match=True,
        planned_tests=[
            PlannedTest(
                command="pytest tests/missing.py -q",
                purpose="Verify missing",
                criterion_ids=["AC-1"],
            ),
        ],
        risk_level="low",
    )

    repo_context = RepositoryContext(
        base_commit="abc123",
        tracked_files=["tests/test_main.py"],
        python_files=["tests/test_main.py"],
        test_files=["tests/test_main.py"],
        config_files=[],
        keyword_matches=[],
    )

    with pytest.raises(ValueError, match="does not exist in repository"):
        validate_test_targets(plan, repo_context)


def test_validate_test_targets_non_test_file():
    """Hard failure: test target is not a valid test file."""
    plan = ChangePlan(
        repository_match=True,
        planned_tests=[
            PlannedTest(
                command="pytest src/main.py -q",
                purpose="Verify main",
                criterion_ids=["AC-1"],
            ),
        ],
        risk_level="low",
    )

    repo_context = RepositoryContext(
        base_commit="abc123",
        tracked_files=["src/main.py"],
        python_files=["src/main.py"],
        test_files=[],
        config_files=[],
        keyword_matches=[],
    )

    with pytest.raises(ValueError, match="must be a test file or test directory"):
        validate_test_targets(plan, repo_context)


def test_validate_command_support_success():
    """Test validation passes for supported commands."""
    plan = ChangePlan(
        repository_match=True,
        planned_tests=[
            PlannedTest(
                command="pytest tests/test_main.py -q",
                purpose="Verify",
                criterion_ids=["AC-1"],
            ),
            PlannedTest(
                command="ruff check src/",
                purpose="Lint",
                criterion_ids=["AC-2"],
            ),
        ],
        risk_level="low",
    )

    # Should not raise
    validate_command_support(plan)


def test_validate_command_support_unsupported():
    """Hard failure: unsupported command."""
    plan = ChangePlan(
        repository_match=True,
        planned_tests=[
            PlannedTest(
                command="npm test",
                purpose="Verify",
                criterion_ids=["AC-1"],
            ),
        ],
        risk_level="low",
    )

    with pytest.raises(ValueError, match="Unsupported command"):
        validate_command_support(plan)
