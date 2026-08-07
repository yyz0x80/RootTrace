"""Tests for the planning module."""

from unittest.mock import Mock

import pytest

from patchpilot.issue.schema import AcceptanceCriterion, NormalizedIssue
from patchpilot.planning.planner import (
    IGNORED_DIRS,
    _extract_json,
    create_plan,
    get_repository_files,
)
from patchpilot.planning.schema import ChangePlan, PlannedChange
from patchpilot.planning.scope_gate import (
    DATABASE_MIGRATION_HINTS,
    FORBIDDEN_FILES,
    FORBIDDEN_PREFIXES,
    ScopeGateResult,
    check_scope,
)


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

    mock_generate = Mock(return_value='{"relevant_files": [], "planned_changes": [], "planned_tests": [], "out_of_scope": [], "risk_level": "low"}')

    plan = create_plan(issue, "/fake/path", mock_generate)

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

    mock_generate = Mock(return_value='{"relevant_files": ["main.py"], "planned_changes": [], "planned_tests": [], "out_of_scope": [], "risk_level": "low"}')

    create_plan(issue, str(tmp_path), mock_generate)

    prompt = mock_generate.call_args[0][0]
    assert "main.py" in prompt
    assert "Repository files:" in prompt


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
  "relevant_files": ["src/main.py"],
  "planned_changes": [
    {
      "file": "src/main.py",
      "description": "Refactor function",
      "acceptance_criteria": ["AC-1"]
    }
  ],
  "planned_tests": [
    {
      "command": "pytest tests/",
      "purpose": "Verify refactoring",
      "acceptance_criteria": ["AC-1"]
    }
  ],
  "out_of_scope": ["UI changes"],
  "risk_level": "medium"
}
```
"""

    mock_generate = Mock(return_value=response)
    plan = create_plan(issue, "/fake/path", mock_generate)

    assert len(plan.relevant_files) == 1
    assert plan.relevant_files[0] == "src/main.py"
    assert len(plan.planned_changes) == 1
    assert plan.planned_changes[0].file == "src/main.py"
    assert len(plan.planned_tests) == 1
    assert plan.risk_level == "medium"


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
                file="src/main.py",
                description="Fix bug",
                acceptance_criteria=["AC-1"],
            )
        ],
        planned_tests=[],
        out_of_scope=[],
        risk_level="low",
    )

    result = check_scope(plan)

    assert result.allowed is True
    assert len(result.violations) == 0
    assert len(result.warnings) == 0


def test_check_scope_too_many_files():
    """Test that modifying too many files is rejected."""
    plan = ChangePlan(
        relevant_files=[f"file{i}.py" for i in range(10)],
        planned_changes=[
            PlannedChange(
                file=f"file{i}.py",
                description="Change",
                acceptance_criteria=[],
            )
            for i in range(10)
        ],
        planned_tests=[],
        out_of_scope=[],
        risk_level="low",
    )

    result = check_scope(plan, max_modified_files=6)

    assert result.allowed is False
    assert len(result.violations) == 1
    assert "maximum allowed is 6" in result.violations[0]


def test_check_scope_forbidden_env_file():
    """Test that .env file modification is forbidden."""
    plan = ChangePlan(
        relevant_files=[".env"],
        planned_changes=[
            PlannedChange(
                file=".env",
                description="Update config",
                acceptance_criteria=[],
            )
        ],
        planned_tests=[],
        out_of_scope=[],
        risk_level="low",
    )

    result = check_scope(plan)

    assert result.allowed is False
    assert len(result.violations) == 1
    assert ".env" in result.violations[0]
    assert "forbidden" in result.violations[0]


def test_check_scope_forbidden_cicd_files():
    """Test that CI/CD configuration modification is forbidden."""
    plan = ChangePlan(
        relevant_files=[".github/workflows/test.yml"],
        planned_changes=[
            PlannedChange(
                file=".github/workflows/test.yml",
                description="Update workflow",
                acceptance_criteria=[],
            )
        ],
        planned_tests=[],
        out_of_scope=[],
        risk_level="low",
    )

    result = check_scope(plan)

    assert result.allowed is False
    assert len(result.violations) == 1
    assert "CI/CD modification is forbidden" in result.violations[0]


def test_check_scope_database_migration():
    """Test that database migrations require manual handling."""
    plan = ChangePlan(
        relevant_files=["migrations/001_initial.py"],
        planned_changes=[
            PlannedChange(
                file="migrations/001_initial.py",
                description="Add migration",
                acceptance_criteria=[],
            )
        ],
        planned_tests=[],
        out_of_scope=[],
        risk_level="low",
    )

    result = check_scope(plan)

    assert result.allowed is False
    assert len(result.violations) == 1
    assert "Database migration requires manual handling" in result.violations[0]


def test_check_scope_alembic_migration():
    """Test that alembic migrations require manual handling."""
    plan = ChangePlan(
        relevant_files=["alembic/versions/1234_migration.py"],
        planned_changes=[
            PlannedChange(
                file="alembic/versions/1234_migration.py",
                description="Add alembic migration",
                acceptance_criteria=[],
            )
        ],
        planned_tests=[],
        out_of_scope=[],
        risk_level="low",
    )

    result = check_scope(plan)

    assert result.allowed is False
    assert len(result.violations) == 1
    assert "Database migration requires manual handling" in result.violations[0]


def test_check_scope_file_not_in_relevant():
    """Test that modifying files not in relevant_files generates warnings."""
    plan = ChangePlan(
        relevant_files=["src/main.py"],
        planned_changes=[
            PlannedChange(
                file="src/main.py",
                description="Fix bug",
                acceptance_criteria=[],
            ),
            PlannedChange(
                file="src/unrelated.py",
                description="Change unrelated file",
                acceptance_criteria=[],
            ),
        ],
        planned_tests=[],
        out_of_scope=[],
        risk_level="low",
    )

    result = check_scope(plan)

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
                file="src/main.py",
                description="Risky change",
                acceptance_criteria=[],
            )
        ],
        planned_tests=[],
        out_of_scope=[],
        risk_level="high",
    )

    result = check_scope(plan)

    assert result.allowed is False
    assert len(result.violations) == 1
    assert "High-risk plan cannot be automatically executed" in result.violations[0]


def test_check_scope_multiple_violations():
    """Test that multiple violations are all reported."""
    plan = ChangePlan(
        relevant_files=[".env", ".github/workflows/test.yml"],
        planned_changes=[
            PlannedChange(
                file=".env",
                description="Change env",
                acceptance_criteria=[],
            ),
            PlannedChange(
                file=".github/workflows/test.yml",
                description="Change CI",
                acceptance_criteria=[],
            ),
        ],
        planned_tests=[],
        out_of_scope=[],
        risk_level="high",
    )

    result = check_scope(plan)

    assert result.allowed is False
    assert len(result.violations) == 3  # .env, CI/CD, and high risk
    assert len(result.warnings) == 0


def test_check_scope_duplicate_files():
    """Test that duplicate file changes are counted correctly."""
    plan = ChangePlan(
        relevant_files=["src/main.py"],
        planned_changes=[
            PlannedChange(
                file="src/main.py",
                description="First change",
                acceptance_criteria=[],
            ),
            PlannedChange(
                file="src/main.py",
                description="Second change",
                acceptance_criteria=[],
            ),
        ],
        planned_tests=[],
        out_of_scope=[],
        risk_level="low",
    )

    result = check_scope(plan)

    assert result.allowed is True
    assert len(result.violations) == 0
    # Duplicate changes to the same file should only count as one file


def test_check_scope_custom_max_files():
    """Test that custom max_modified_files parameter is respected."""
    plan = ChangePlan(
        relevant_files=[f"file{i}.py" for i in range(3)],
        planned_changes=[
            PlannedChange(
                file=f"file{i}.py",
                description="Change",
                acceptance_criteria=[],
            )
            for i in range(3)
        ],
        planned_tests=[],
        out_of_scope=[],
        risk_level="low",
    )

    result = check_scope(plan, max_modified_files=2)

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

    result = check_scope(plan)

    assert result.allowed is True
    assert len(result.violations) == 0
    assert len(result.warnings) == 0


def test_forbidden_files_constant():
    """Test that FORBIDDEN_FILES contains expected entries."""
    assert ".env" in FORBIDDEN_FILES


def test_forbidden_prefixes_constant():
    """Test that FORBIDDEN_PREFIXES contains expected entries."""
    assert ".github/workflows/" in FORBIDDEN_PREFIXES


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
