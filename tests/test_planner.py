"""Tests for the planning module."""

from unittest.mock import Mock

import pytest

from patchpilot.issue.schema import AcceptanceCriterion, NormalizedIssue
from patchpilot.planning.planner import (
    IGNORED_DIRS,
    PlanGenerationError,
    _extract_json,
    create_plan,
    create_plan_with_path,
    get_repository_files,
)
from patchpilot.planning.schema import ChangePlan, PlannedChange
from patchpilot.planning.scope_gate import (
    DATABASE_MIGRATION_HINTS,
    ScopeGateResult,
    check_scope,
)
from patchpilot.policy.builtins import get_builtin_policies
from patchpilot.repository.schema import RepositoryContext


def test_get_repository_files_basic(tmp_path):
    """Test getting files from a simple repository."""
    # Create some test files
    (tmp_path / "main.py").write_text("print('hello')")
    (tmp_path / "utils.py").write_text("def helper(): pass")
    (tmp_path / "README.md").write_text("# Test")

    files = get_repository_files(str(tmp_path))

    assert len(files) == 3
    assert "main.py" in files
    assert "utils.py" in files
    assert "README.md" in files
    assert files == sorted(files)  # Should be sorted


def test_get_repository_files_ignores_directories(tmp_path):
    """Test that ignored directories are excluded."""
    # Create files in various directories
    (tmp_path / "main.py").write_text("print('hello')")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("git config")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "main.pyc").write_text("compiled")
    (tmp_path / "venv").mkdir()
    (tmp_path / "venv" / "lib").mkdir()
    (tmp_path / "venv" / "lib" / "python.py").write_text("venv file")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "package").mkdir()
    (tmp_path / "node_modules" / "package" / "index.js").write_text("js")

    files = get_repository_files(str(tmp_path))

    assert "main.py" in files
    assert not any(".git" in f for f in files)
    assert not any("__pycache__" in f for f in files)
    assert not any("venv" in f for f in files)
    assert not any("node_modules" in f for f in files)


def test_get_repository_files_nested_structure(tmp_path):
    """Test getting files from nested directory structure."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("main")
    (tmp_path / "src" / "utils").mkdir()
    (tmp_path / "src" / "utils" / "helper.py").write_text("helper")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_main.py").write_text("test")
    (tmp_path / "tests" / "__pycache__").mkdir()
    (tmp_path / "tests" / "__pycache__" / "test.pyc").write_text("cache")

    files = get_repository_files(str(tmp_path))

    assert "src/main.py" in files
    assert "src/utils/helper.py" in files
    assert "tests/test_main.py" in files
    assert not any("__pycache__" in f for f in files)


def test_get_repository_files_empty_directory(tmp_path):
    """Test getting files from an empty directory."""
    files = get_repository_files(str(tmp_path))
    assert files == []


def test_get_repository_files_nonexistent_path():
    """Test getting files from a nonexistent path."""
    files = get_repository_files("/nonexistent/path")
    assert files == []


def test_extract_json_plain():
    """Test extracting plain JSON."""
    text = '{"key": "value"}'
    result = _extract_json(text)
    assert result == {"key": "value"}


def test_extract_json_with_markdown():
    """Test extracting JSON from markdown code block."""
    text = '```json\n{"key": "value"}\n```'
    result = _extract_json(text)
    assert result == {"key": "value"}


def test_extract_json_with_markdown_no_language():
    """Test extracting JSON from markdown without language specifier."""
    text = '```\n{"key": "value"}\n```'
    result = _extract_json(text)
    assert result == {"key": "value"}


def test_extract_json_with_surrounding_text():
    """Test extracting JSON from text with surrounding content."""
    text = 'Here is the plan:\n{"key": "value"}\nThat is all.'
    result = _extract_json(text)
    assert result == {"key": "value"}


def test_extract_json_no_json_raises():
    """Test that ValueError is raised when no JSON is found."""
    with pytest.raises(ValueError, match="Planner did not return JSON"):
        _extract_json("No JSON here")


def test_extract_json_invalid_json_raises():
    """Test that ValueError is raised for invalid JSON."""
    with pytest.raises(ValueError):
        _extract_json("{invalid json}")


def test_create_plan_basic():
    """Test creating a plan with a mock generator."""
    issue = NormalizedIssue(
        title="Fix bug",
        task_type="bug",
        problem_statement="Something is broken",
    )

    mock_generate = Mock(return_value='{"repository_match": true, "relevant_files": [], "planned_changes": [], "planned_tests": [], "out_of_scope": [], "risk_level": "low"}')

    plan = create_plan_with_path(issue, "/fake/path", mock_generate)

    assert isinstance(plan, ChangePlan)
    assert plan.risk_level == "low"
    mock_generate.assert_called_once()


def test_create_plan_includes_repository_files(tmp_path):
    """Test that repository files are included in the prompt."""
    (tmp_path / "main.py").write_text("print('hello')")

    issue = NormalizedIssue(
        title="Add feature",
        task_type="feature",
        problem_statement="Need a new feature",
    )

    mock_generate = Mock(return_value='{"repository_match": true, "relevant_files": ["main.py"], "planned_changes": [], "planned_tests": [], "out_of_scope": [], "risk_level": "low"}')

    create_plan_with_path(issue, str(tmp_path), mock_generate)

    prompt = mock_generate.call_args[0][0]
    assert "main.py" in prompt
    assert "Repository context:" in prompt


def test_create_plan_parses_complex_response():
    """Test creating a plan with a complex LLM response."""
    issue = NormalizedIssue(
        title="Refactor",
        task_type="refactor",
        problem_statement="Code needs refactoring",
        acceptance_criteria=[
            AcceptanceCriterion(id="AC-1", description="Code is cleaner")
        ],
    )

    response = """
Here is the plan:

```json
{
  "repository_match": true,
  "relevant_files": ["src/main.py"],
  "planned_changes": [
    {
      "path": "src/main.py",
      "action": "modify",
      "description": "Refactor function",
      "acceptance_criteria": ["AC-1"],
      "criterion_ids": ["AC-1"]
    }
  ],
  "planned_tests": [
    {
      "command": "pytest tests/",
      "purpose": "Verify refactoring",
      "acceptance_criteria": [],
      "criterion_ids": []
    }
  ],
  "criterion_plans": [{
    "criterion_id": "AC-1",
    "disposition": "to_implement",
    "relevant_source_files": ["src/main.py"],
    "baseline_evidence": ""
  }],
  "acceptance_probes": [{
    "probe_id": "probe-ac-1",
    "module": "src.main",
    "target": "run",
    "probe_type": "function_io",
    "criterion_ids": ["AC-1"],
    "assertion": "truthy"
  }],
  "out_of_scope": ["UI changes"],
  "risk_level": "medium"
}
```
"""

    mock_generate = Mock(return_value=response)
    plan = create_plan_with_path(issue, "/fake/path", mock_generate)

    assert len(plan.relevant_files) == 1
    assert plan.relevant_files[0] == "src/main.py"
    assert len(plan.planned_changes) == 1
    assert plan.planned_changes[0].path == "src/main.py"
    assert plan.planned_changes[0].action == "modify"
    assert len(plan.planned_tests) == 1
    assert plan.risk_level == "medium"


def test_create_plan_with_base_commit():
    """Test that base_commit is set from repository context."""
    issue = NormalizedIssue(
        title="Fix bug",
        task_type="bug",
        problem_statement="Something is broken",
    )

    mock_generate = Mock(return_value='{"repository_match": true, "relevant_files": [], "planned_changes": [], "planned_tests": [], "out_of_scope": [], "risk_level": "low"}')

    plan = create_plan_with_path(issue, "/fake/path", mock_generate, base_commit="abc123def456")

    assert plan.base_commit == "abc123def456"
    mock_generate.assert_called_once()


def test_create_plan_with_repository_mismatch():
    """Test that repository mismatch is handled correctly."""
    issue = NormalizedIssue(
        title="Add Task feature",
        task_type="feature",
        problem_statement="Need Task model",
    )

    mock_generate = Mock(return_value='{"repository_match": false, "repository_mismatch_reason": "No existing Task model found in repository", "relevant_files": [], "planned_changes": [], "planned_tests": [], "out_of_scope": [], "risk_level": "low"}')

    plan = create_plan_with_path(issue, "/fake/path", mock_generate)

    assert plan.repository_match is False
    assert plan.repository_mismatch_reason == "No existing Task model found in repository"


def test_create_plan_with_create_action():
    """Test that create action is accepted for new files."""
    issue = NormalizedIssue(
        title="Add feature",
        task_type="feature",
        problem_statement="Need new feature",
    )

    response = """
```json
{
  "repository_match": true,
  "relevant_files": ["src/main.py"],
  "planned_changes": [
    {
      "path": "src/new_feature.py",
      "action": "create",
      "description": "Add new feature",
      "acceptance_criteria": []
    }
  ],
  "planned_tests": [],
  "out_of_scope": [],
  "risk_level": "low"
}
```
"""

    mock_generate = Mock(return_value=response)
    plan = create_plan_with_path(issue, "/fake/path", mock_generate)

    assert len(plan.planned_changes) == 1
    assert plan.planned_changes[0].path == "src/new_feature.py"
    assert plan.planned_changes[0].action == "create"


def test_create_plan_with_delete_action():
    """Test that delete action is accepted."""
    issue = NormalizedIssue(
        title="Remove deprecated",
        task_type="refactor",
        problem_statement="Remove deprecated code",
    )

    response = """
```json
{
  "repository_match": true,
  "relevant_files": ["src/deprecated.py"],
  "planned_changes": [
    {
      "path": "src/deprecated.py",
      "action": "delete",
      "description": "Remove deprecated module",
      "acceptance_criteria": []
    }
  ],
  "planned_tests": [],
  "out_of_scope": [],
  "risk_level": "low"
}
```
"""

    mock_generate = Mock(return_value=response)
    plan = create_plan_with_path(issue, "/fake/path", mock_generate)

    assert len(plan.planned_changes) == 1
    assert plan.planned_changes[0].path == "src/deprecated.py"
    assert plan.planned_changes[0].action == "delete"


def test_ignored_dirs_constant():
    """Test that IGNORED_DIRS contains expected directories."""
    assert ".git" in IGNORED_DIRS
    assert "__pycache__" in IGNORED_DIRS
    assert "venv" in IGNORED_DIRS
    assert ".venv" in IGNORED_DIRS


# Scope Gate Tests


def test_check_scope_allowed_plan():
    """Test that a valid plan passes scope checks."""
    plan = ChangePlan(
        relevant_files=["src/main.py", "src/utils.py"],
        planned_changes=[
            PlannedChange(
                path="src/main.py",
                action="modify",
                description="Fix bug",
                acceptance_criteria=["AC-1"],
            )
        ],
        planned_tests=[],
        out_of_scope=[],
        risk_level="low",
    )

    policy_set = get_builtin_policies()
    result = check_scope(plan, policy_set)

    assert result.allowed is True
    assert len(result.violations) == 0
    assert len(result.warnings) == 0


def test_check_scope_too_many_files():
    """Test that modifying too many files is rejected."""
    plan = ChangePlan(
        relevant_files=[f"file{i}.py" for i in range(10)],
        planned_changes=[
            PlannedChange(
                path=f"file{i}.py",
                action="modify",
                description="Change",
                acceptance_criteria=[],
            )
            for i in range(10)
        ],
        planned_tests=[],
        out_of_scope=[],
        risk_level="low",
    )

    policy_set = get_builtin_policies()
    result = check_scope(plan, policy_set, max_modified_files=6)

    assert result.allowed is False
    assert len(result.violations) == 1
    assert "maximum allowed is 6" in result.violations[0]


def test_check_scope_forbidden_env_file():
    """Test that .env file modification is forbidden."""
    plan = ChangePlan(
        relevant_files=[".env"],
        planned_changes=[
            PlannedChange(
                path=".env",
                action="modify",
                description="Update config",
                acceptance_criteria=[],
            )
        ],
        planned_tests=[],
        out_of_scope=[],
        risk_level="low",
    )

    policy_set = get_builtin_policies()
    result = check_scope(plan, policy_set)

    assert result.allowed is False
    assert len(result.violations) == 1
    assert ".env" in result.violations[0]
    assert "denied" in result.violations[0]


def test_check_scope_forbidden_cicd_files():
    """Test that CI/CD configuration modification is forbidden."""
    plan = ChangePlan(
        relevant_files=[".github/workflows/test.yml"],
        planned_changes=[
            PlannedChange(
                path=".github/workflows/test.yml",
                action="modify",
                description="Update workflow",
                acceptance_criteria=[],
            )
        ],
        planned_tests=[],
        out_of_scope=[],
        risk_level="low",
    )

    policy_set = get_builtin_policies()
    result = check_scope(plan, policy_set)

    assert result.allowed is False
    assert len(result.violations) == 1
    assert "CI/CD" in result.violations[0]


def test_check_scope_database_migration():
    """Test that database migrations require manual handling."""
    plan = ChangePlan(
        relevant_files=["migrations/001_initial.py"],
        planned_changes=[
            PlannedChange(
                path="migrations/001_initial.py",
                action="modify",
                description="Add migration",
                acceptance_criteria=[],
            )
        ],
        planned_tests=[],
        out_of_scope=[],
        risk_level="low",
    )

    policy_set = get_builtin_policies()
    result = check_scope(plan, policy_set)

    assert result.allowed is False
    assert len(result.violations) == 1
    assert "Database migration requires manual handling" in result.violations[0]


def test_check_scope_alembic_migration():
    """Test that alembic migrations require manual handling."""
    plan = ChangePlan(
        relevant_files=["alembic/versions/1234_migration.py"],
        planned_changes=[
            PlannedChange(
                path="alembic/versions/1234_migration.py",
                action="modify",
                description="Add alembic migration",
                acceptance_criteria=[],
            )
        ],
        planned_tests=[],
        out_of_scope=[],
        risk_level="low",
    )

    policy_set = get_builtin_policies()
    result = check_scope(plan, policy_set)

    assert result.allowed is False
    assert len(result.violations) == 1
    assert "Database migration requires manual handling" in result.violations[0]


def test_check_scope_file_not_in_relevant():
    """Test that modifying files not in relevant_files generates warnings."""
    plan = ChangePlan(
        relevant_files=["src/main.py"],
        planned_changes=[
            PlannedChange(
                path="src/main.py",
                action="modify",
                description="Fix bug",
                acceptance_criteria=[],
            ),
            PlannedChange(
                path="src/unrelated.py",
                action="modify",
                description="Change unrelated file",
                acceptance_criteria=[],
            ),
        ],
        planned_tests=[],
        out_of_scope=[],
        risk_level="low",
    )

    policy_set = get_builtin_policies()
    result = check_scope(plan, policy_set)

    assert result.allowed is True  # Warnings don't block execution
    assert len(result.violations) == 0
    assert len(result.warnings) == 1
    assert "src/unrelated.py" in result.warnings[0]
    assert "not listed in relevant_files" in result.warnings[0]


def test_check_scope_high_risk_blocked():
    """Test that high-risk plans cannot be automatically executed."""
    plan = ChangePlan(
        relevant_files=["src/main.py"],
        planned_changes=[
            PlannedChange(
                path="src/main.py",
                action="modify",
                description="Risky change",
                acceptance_criteria=[],
            )
        ],
        planned_tests=[],
        out_of_scope=[],
        risk_level="high",
    )

    policy_set = get_builtin_policies()
    result = check_scope(plan, policy_set)

    assert result.allowed is False
    assert len(result.violations) == 1
    assert "High-risk plan cannot be automatically executed" in result.violations[0]


def test_check_scope_multiple_violations():
    """Test that multiple violations are all reported."""
    plan = ChangePlan(
        relevant_files=[".env", ".github/workflows/test.yml"],
        planned_changes=[
            PlannedChange(
                path=".env",
                action="modify",
                description="Change env",
                acceptance_criteria=[],
            ),
            PlannedChange(
                path=".github/workflows/test.yml",
                action="modify",
                description="Change CI",
                acceptance_criteria=[],
            ),
        ],
        planned_tests=[],
        out_of_scope=[],
        risk_level="high",
    )

    policy_set = get_builtin_policies()
    result = check_scope(plan, policy_set)

    assert result.allowed is False
    assert len(result.violations) == 3  # .env, CI/CD, and high risk
    assert len(result.warnings) == 0


def test_check_scope_duplicate_files():
    """Test that duplicate file changes are counted correctly."""
    plan = ChangePlan(
        relevant_files=["src/main.py"],
        planned_changes=[
            PlannedChange(
                path="src/main.py",
                action="modify",
                description="First change",
                acceptance_criteria=[],
            ),
            PlannedChange(
                path="src/main.py",
                action="modify",
                description="Second change",
                acceptance_criteria=[],
            ),
        ],
        planned_tests=[],
        out_of_scope=[],
        risk_level="low",
    )

    policy_set = get_builtin_policies()
    result = check_scope(plan, policy_set)

    assert result.allowed is True
    assert len(result.violations) == 0
    # Duplicate changes to the same file should only count as one file


def test_check_scope_custom_max_files():
    """Test that custom max_modified_files parameter is respected."""
    plan = ChangePlan(
        relevant_files=[f"file{i}.py" for i in range(3)],
        planned_changes=[
            PlannedChange(
                path=f"file{i}.py",
                action="modify",
                description="Change",
                acceptance_criteria=[],
            )
            for i in range(3)
        ],
        planned_tests=[],
        out_of_scope=[],
        risk_level="low",
    )

    policy_set = get_builtin_policies()
    result = check_scope(plan, policy_set, max_modified_files=2)

    assert result.allowed is False
    assert "maximum allowed is 2" in result.violations[0]


def test_check_scope_empty_plan():
    """Test that an empty plan passes scope checks."""
    plan = ChangePlan(
        relevant_files=[],
        planned_changes=[],
        planned_tests=[],
        out_of_scope=[],
        risk_level="low",
    )

    policy_set = get_builtin_policies()
    result = check_scope(plan, policy_set)

    assert result.allowed is True
    assert len(result.violations) == 0
    assert len(result.warnings) == 0


def test_builtin_policies_contain_expected_restrictions():
    """Test that builtin policies contain expected security restrictions."""
    policy_set = get_builtin_policies()

    # Check that .env is forbidden for both read and write
    write_denied = set()
    for policy in policy_set.write_policies:
        write_denied.update(policy.denied_paths)
    assert ".env" in write_denied

    # Check that .git is forbidden for both read and write
    assert ".git" in write_denied

    # Check that CI/CD workflows are forbidden
    assert ".github/workflows" in write_denied

    # Check that test files are forbidden
    assert "tests" in write_denied
    assert "test_" in write_denied


def test_database_migration_hints_constant():
    """Test that DATABASE_MIGRATION_HINTS contains expected entries."""
    assert "migrations/" in DATABASE_MIGRATION_HINTS
    assert "alembic/versions/" in DATABASE_MIGRATION_HINTS


def test_scope_gate_result_model():
    """Test that ScopeGateResult model works correctly."""
    result = ScopeGateResult(
        allowed=False,
        violations=["Violation 1", "Violation 2"],
        warnings=["Warning 1"],
    )

    assert result.allowed is False
    assert len(result.violations) == 2
    assert len(result.warnings) == 1


def test_scope_gate_result_defaults():
    """Test that ScopeGateResult default values work."""
    result = ScopeGateResult(allowed=True)

    assert result.allowed is True
    assert result.violations == []
    assert result.warnings == []


def test_create_plan_with_repository_context():
    """Test creating a plan with RepositoryContext."""
    issue = NormalizedIssue(
        title="Fix bug",
        task_type="bug",
        problem_statement="Something is broken",
    )

    repository_context = RepositoryContext(
        base_commit="abc123",
        tracked_files=["src/main.py", "src/utils.py"],
        python_files=["src/main.py", "src/utils.py"],
        test_files=["tests/test_main.py"],
        config_files=["pyproject.toml"],
        keyword_matches=["src/main.py"],
    )

    mock_generate = Mock(return_value='{"repository_match": true, "relevant_files": ["src/main.py"], "planned_changes": [], "planned_tests": [], "out_of_scope": [], "risk_level": "low"}')

    plan = create_plan(issue, repository_context, mock_generate)

    assert isinstance(plan, ChangePlan)
    assert plan.risk_level == "low"
    assert plan.base_commit == "abc123"
    mock_generate.assert_called_once()

    # Verify that repository context was included in the prompt
    prompt = mock_generate.call_args[0][0]
    assert "src/main.py" in prompt
    assert "tracked_files" in prompt


def test_create_plan_retries_missing_acceptance_coverage_once():
    """Retry an incomplete plan with the exact missing AC diagnostic."""
    issue = NormalizedIssue(
        title="Fix behavior",
        task_type="bug",
        problem_statement="The behavior is incorrect.",
        acceptance_criteria=[
            AcceptanceCriterion(id="AC-1", description="First behavior"),
            AcceptanceCriterion(id="AC-2", description="Second behavior"),
        ],
    )
    repository_context = RepositoryContext(
        base_commit="abc123",
        tracked_files=["src/main.py", "tests/test_main.py"],
        python_files=["src/main.py"],
        test_files=["tests/test_main.py"],
        config_files=[],
        keyword_matches=["src/main.py"],
    )
    incomplete = """{
  "repository_match": true,
  "relevant_files": ["src/main.py"],
  "planned_changes": [{
    "path": "src/main.py",
    "action": "modify",
    "description": "Fix first behavior",
    "acceptance_criteria": ["AC-1"],
    "criterion_ids": ["AC-1"]
  }],
  "planned_tests": [{
    "command": "pytest tests/test_main.py",
    "purpose": "Verify behavior",
    "acceptance_criteria": [],
    "criterion_ids": []
  }],
  "criterion_plans": [
    {
      "criterion_id": "AC-1",
      "disposition": "to_implement",
      "relevant_source_files": ["src/main.py"]
    },
    {
      "criterion_id": "AC-2",
      "disposition": "to_implement",
      "relevant_source_files": ["src/main.py"]
    }
  ],
  "acceptance_probes": [
    {
      "probe_id": "probe-ac-1",
      "module": "src.main",
      "target": "run",
      "probe_type": "function_io",
      "criterion_ids": ["AC-1"],
      "assertion": "truthy"
    },
    {
      "probe_id": "probe-ac-2",
      "module": "src.main",
      "target": "run",
      "probe_type": "function_io",
      "criterion_ids": ["AC-2"],
      "assertion": "truthy"
    }
  ],
  "out_of_scope": [],
  "risk_level": "low"
}"""
    repaired = incomplete.replace(
        '"criterion_ids": ["AC-1"]',
        '"criterion_ids": ["AC-1", "AC-2"]',
        1,
    )
    prompts: list[str] = []
    responses = iter([incomplete, repaired])

    def mock_generate(prompt: str) -> str:
        prompts.append(prompt)
        return next(responses)

    plan = create_plan(issue, repository_context, mock_generate)

    assert plan.base_commit == "abc123"
    assert plan.planned_changes[0].criterion_ids == ["AC-1", "AC-2"]
    assert len(prompts) == 2
    assert "AC-2 has no planned source change" in prompts[1]
    assert "read-only" in prompts[1]


def test_create_plan_does_not_retry_scope_violation():
    """Unsafe plans should reach the scope gate without a coverage retry."""
    issue = NormalizedIssue(
        title="Disable CI",
        task_type="feature",
        problem_statement="Disable quality checks.",
        acceptance_criteria=[
            AcceptanceCriterion(id="AC-1", description="Disable CI"),
        ],
    )
    repository_context = RepositoryContext(
        base_commit="abc123",
        tracked_files=[".github/workflows/ci.yml", "tests/test_ci.py"],
        python_files=[],
        test_files=["tests/test_ci.py"],
        config_files=[".github/workflows/ci.yml"],
        keyword_matches=[],
    )
    response = """{
  "repository_match": true,
  "relevant_files": [".github/workflows/ci.yml"],
  "planned_changes": [{
    "path": ".github/workflows/ci.yml",
    "action": "modify",
    "description": "Disable CI",
    "acceptance_criteria": ["AC-1"]
  }],
  "planned_tests": [{
    "command": "pytest tests/test_ci.py -q",
    "purpose": "Verify CI",
    "acceptance_criteria": ["AC-1"]
  }],
  "out_of_scope": [],
  "risk_level": "low"
}"""
    generate = Mock(return_value=response)

    plan = create_plan(issue, repository_context, generate)

    # The plan should have proper AC mapping to avoid post-processing errors
    assert plan.planned_changes[0].path == ".github/workflows/ci.yml"
    assert plan.planned_changes[0].action == "modify"
    assert "AC-1" in plan.planned_changes[0].acceptance_criteria
    generate.assert_called_once()


def test_create_plan_raises_plan_generation_error_after_failed_retry():
    """A plan that remains incomplete should exhaust bounded repairs."""
    issue = NormalizedIssue(
        title="Fix behavior",
        task_type="bug",
        problem_statement="The behavior is incorrect.",
        acceptance_criteria=[
            AcceptanceCriterion(id="AC-1", description="Fix behavior"),
        ],
    )
    repository_context = RepositoryContext(
        base_commit="abc123",
        tracked_files=["src/main.py"],
        python_files=["src/main.py"],
        test_files=[],
        config_files=[],
        keyword_matches=["src/main.py"],
    )
    incomplete = """{
  "repository_match": true,
  "relevant_files": ["src/main.py"],
  "planned_changes": [],
  "planned_tests": [],
  "out_of_scope": [],
  "risk_level": "low"
}"""
    generate = Mock(return_value=incomplete)

    with pytest.raises(PlanGenerationError, match="after 2 repairs"):
        create_plan(issue, repository_context, generate)

    assert generate.call_count == 3


def test_create_plan_repairs_sequential_schema_errors():
    """Give independent schema errors separate bounded repair attempts."""
    issue = NormalizedIssue(
        title="Fix behavior",
        task_type="bug",
        problem_statement="The behavior is incorrect.",
    )
    repository_context = RepositoryContext(
        base_commit="abc123",
        tracked_files=["src/main.py"],
        python_files=["src/main.py"],
        test_files=[],
        config_files=[],
        keyword_matches=["src/main.py"],
    )
    invalid_risk = """{
  "repository_match": true,
  "relevant_files": [],
  "planned_changes": [],
  "planned_tests": [],
  "out_of_scope": [],
  "risk_level": "tiny"
}"""
    invalid_assertion = """{
  "repository_match": true,
  "relevant_files": [],
  "planned_changes": [],
  "planned_tests": [],
  "acceptance_probes": [{
    "probe_id": "probe-1",
    "module": "src.main",
    "target": "run",
    "probe_type": "function_io",
    "criterion_ids": [],
    "assertion": "call_relationship"
  }],
  "out_of_scope": [],
  "risk_level": "low"
}"""
    valid = """{
  "repository_match": true,
  "relevant_files": [],
  "planned_changes": [],
  "planned_tests": [],
  "out_of_scope": [],
  "risk_level": "low"
}"""
    prompts: list[str] = []
    responses = iter([invalid_risk, invalid_assertion, valid])

    def generate(prompt: str) -> str:
        prompts.append(prompt)
        return next(responses)

    plan = create_plan(issue, repository_context, generate)

    assert plan.risk_level == "low"
    assert len(prompts) == 3
    assert "acceptance_probes.assertion field must be exactly one of" in prompts[2]
    assert "call_relationship" in prompts[2]
