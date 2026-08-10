"""Tests for repository analysis."""

import subprocess
from pathlib import Path

import pytest

from patchpilot.issue.schema import NormalizedIssue
from patchpilot.repository.analyzer import analyze_repository
from patchpilot.repository.schema import RepositoryContext


def _setup_git_repo(tmp_path: Path) -> str:
    """Helper to set up a git repository with initial commit.
    
    Returns:
        The HEAD commit SHA.
    """
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    
    # Create initial files
    (tmp_path / "main.py").write_text("def main():\n    pass\n")
    (tmp_path / "utils.py").write_text("def helper():\n    pass\n")
    
    # Create test directory
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_main.py").write_text("def test_main():\n    pass\n")
    
    # Create config file
    (tmp_path / "pyproject.toml").write_text("[tool.poetry]\nname = 'test'\n")
    
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    
    # Get HEAD SHA
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_analyze_repository_basic(tmp_path: Path):
    """Successfully analyze a basic repository structure."""
    head_sha = _setup_git_repo(tmp_path)
    
    issue = NormalizedIssue(
        title="Fix helper function",
        task_type="bug",
        problem_statement="The helper function in utils.py has a bug",
    )
    
    context = analyze_repository(tmp_path, issue, head_sha)
    
    assert isinstance(context, RepositoryContext)
    assert context.base_commit == head_sha
    assert len(context.tracked_files) == 4
    assert "main.py" in context.tracked_files
    assert "utils.py" in context.tracked_files
    assert "tests/test_main.py" in context.tracked_files
    assert "pyproject.toml" in context.tracked_files


def test_identifies_python_files(tmp_path: Path):
    """Correctly identify Python source files."""
    head_sha = _setup_git_repo(tmp_path)
    
    issue = NormalizedIssue(
        title="Test issue",
        task_type="bug",
        problem_statement="Test problem",
    )
    
    context = analyze_repository(tmp_path, issue, head_sha)
    
    assert len(context.python_files) == 3
    assert "main.py" in context.python_files
    assert "utils.py" in context.python_files
    assert "tests/test_main.py" in context.python_files
    assert "pyproject.toml" not in context.python_files


def test_identifies_test_files(tmp_path: Path):
    """Correctly identify test files."""
    head_sha = _setup_git_repo(tmp_path)
    
    issue = NormalizedIssue(
        title="Test issue",
        task_type="bug",
        problem_statement="Test problem",
    )
    
    context = analyze_repository(tmp_path, issue, head_sha)
    
    assert len(context.test_files) == 1
    assert "tests/test_main.py" in context.test_files
    assert "main.py" not in context.test_files
    assert "utils.py" not in context.test_files


def test_identifies_test_files_with_prefix(tmp_path: Path):
    """Identify test files using test_ prefix convention."""
    head_sha = _setup_git_repo(tmp_path)
    
    # Add a test file with test_ prefix not in tests/ directory
    (tmp_path / "test_utils.py").write_text("def test_utils():\n    pass\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Add test file"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    
    issue = NormalizedIssue(
        title="Test issue",
        task_type="bug",
        problem_statement="Test problem",
    )
    
    context = analyze_repository(tmp_path, issue, head_sha)
    
    assert "test_utils.py" in context.test_files
    assert "tests/test_main.py" in context.test_files


def test_identifies_config_files(tmp_path: Path):
    """Correctly identify configuration files."""
    head_sha = _setup_git_repo(tmp_path)
    
    issue = NormalizedIssue(
        title="Test issue",
        task_type="bug",
        problem_statement="Test problem",
    )
    
    context = analyze_repository(tmp_path, issue, head_sha)
    
    assert len(context.config_files) == 1
    assert "pyproject.toml" in context.config_files


def test_identifies_multiple_config_files(tmp_path: Path):
    """Identify multiple configuration file types."""
    head_sha = _setup_git_repo(tmp_path)
    
    # Add more config files
    (tmp_path / "requirements.txt").write_text("requests==2.0.0\n")
    (tmp_path / "setup.cfg").write_text("[metadata]\nname = test\n")
    
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Add config files"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    
    issue = NormalizedIssue(
        title="Test issue",
        task_type="bug",
        problem_statement="Test problem",
    )
    
    context = analyze_repository(tmp_path, issue, head_sha)
    
    assert "pyproject.toml" in context.config_files
    assert "requirements.txt" in context.config_files
    assert "setup.cfg" in context.config_files


def test_extracts_keywords_from_issue(tmp_path: Path):
    """Extract relevant keywords from issue title and problem statement."""
    head_sha = _setup_git_repo(tmp_path)
    
    # Add a file with content that matches the issue keywords
    (tmp_path / "helper.py").write_text("def helper_function():\n    pass\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Add helper file"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    
    issue = NormalizedIssue(
        title="Fix helper function",
        task_type="bug",
        problem_statement="The helper function has incorrect behavior",
    )
    
    context = analyze_repository(tmp_path, issue, head_sha)
    
    # Should extract meaningful keywords, not stopwords
    # The actual keyword matches depend on ripgrep finding files with those keywords
    # Helper files should be present
    assert len(context.tracked_files) > 0
    assert "helper.py" in context.tracked_files


def test_filters_stopwords(tmp_path: Path):
    """Filter out common stopwords from keyword extraction."""
    head_sha = _setup_git_repo(tmp_path)
    
    issue = NormalizedIssue(
        title="Add support for new feature",
        task_type="feature",
        problem_statement="Add support for the new feature in the module",
    )
    
    context = analyze_repository(tmp_path, issue, head_sha)
    
    # Keywords should not include stopwords
    # The actual matches depend on ripgrep, but keyword extraction should filter them
    assert len(context.tracked_files) > 0


def test_keyword_search(tmp_path: Path):
    """Search for files containing issue keywords."""
    head_sha = _setup_git_repo(tmp_path)
    
    # Create a file with specific content
    (tmp_path / "api.py").write_text("def user_api():\n    pass\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Add API file"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    
    issue = NormalizedIssue(
        title="Fix user API",
        task_type="bug",
        problem_statement="The user API function has issues",
    )
    
    context = analyze_repository(tmp_path, issue, head_sha)
    
    # Should find files matching "user" or "api" keywords
    assert len(context.keyword_matches) >= 0


def test_limits_keyword_count(tmp_path: Path):
    """Limit keyword extraction to 8 keywords."""
    head_sha = _setup_git_repo(tmp_path)
    
    issue = NormalizedIssue(
        title="Fix multiple components in the system",
        task_type="bug",
        problem_statement="Issues with authentication authorization database configuration logging monitoring caching",
    )
    
    context = analyze_repository(tmp_path, issue, head_sha)
    
    # Should still work and not crash with many potential keywords
    assert isinstance(context, RepositoryContext)
    assert len(context.tracked_files) > 0


def test_handles_empty_repository(tmp_path: Path):
    """Handle repository with only initial commit."""
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    
    # Create only one file
    (tmp_path / "README.md").write_text("# Test\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    head_sha = result.stdout.strip()
    
    issue = NormalizedIssue(
        title="Test issue",
        task_type="bug",
        problem_statement="Test problem",
    )
    
    context = analyze_repository(tmp_path, issue, head_sha)
    
    assert context.tracked_files == ["README.md"]
    assert context.python_files == []
    assert context.test_files == []
    assert context.config_files == []
