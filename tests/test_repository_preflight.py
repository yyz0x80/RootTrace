"""Tests for repository preflight validation."""

import subprocess
from pathlib import Path

import pytest

from patchpilot.repository.preflight import (
    RepositoryPreflightError,
    validate_repository,
)
from patchpilot.repository.schema import RepositoryPreflightResult


def test_validates_git_repository(tmp_path: Path):
    """Successfully validate a clean Git repository with commits."""
    # Initialize a git repository
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    
    # Create a file and commit it
    (tmp_path / "test.py").write_text("print('hello')")
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
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    # Validate the repository
    result = validate_repository(tmp_path)
    
    assert isinstance(result, RepositoryPreflightResult)
    assert result.repo_path == tmp_path.resolve()
    assert len(result.head_sha) == 40  # SHA-1 hash length


def test_rejects_nonexistent_path(tmp_path: Path):
    """Reject validation when path does not exist."""
    nonexistent = tmp_path / "nonexistent"
    
    with pytest.raises(RepositoryPreflightError, match="Repository does not exist"):
        validate_repository(nonexistent)


def test_rejects_file_instead_of_directory(tmp_path: Path):
    """Reject validation when path is a file, not a directory."""
    file_path = tmp_path / "file.txt"
    file_path.write_text("content")
    
    with pytest.raises(RepositoryPreflightError, match="not a directory"):
        validate_repository(file_path)


def test_rejects_non_git_repository(tmp_path: Path):
    """Reject validation when directory is not a Git repository."""
    # Create a directory without git initialization
    (tmp_path / "test.py").write_text("print('hello')")
    
    with pytest.raises(
        RepositoryPreflightError,
        match="not a Git repository"
    ):
        validate_repository(tmp_path)


def test_rejects_repository_without_commits(tmp_path: Path):
    """Reject validation when Git repository has no commits."""
    # Initialize git but don't create any commits
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    
    with pytest.raises(
        RepositoryPreflightError,
        match="no baseline commit"
    ):
        validate_repository(tmp_path)


def test_rejects_repository_with_uncommitted_changes(tmp_path: Path):
    """Reject validation when working tree has uncommitted changes."""
    # Initialize git and create initial commit
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
    (tmp_path / "test.py").write_text("print('hello')")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    
    # Create uncommitted changes
    (tmp_path / "test.py").write_text("print('changed')")
    
    with pytest.raises(
        RepositoryPreflightError,
        match="uncommitted changes"
    ):
        validate_repository(tmp_path)


def test_rejects_subdirectory_as_repo_path(tmp_path: Path):
    """Reject validation when path is a subdirectory, not the repository root."""
    # Initialize git repository
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
    (tmp_path / "test.py").write_text("print('hello')")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    
    # Create a subdirectory
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    
    # Try to validate the subdirectory as the repository root
    with pytest.raises(
        RepositoryPreflightError,
        match="must point to the Git repository root"
    ):
        validate_repository(subdir)


def test_resolves_relative_path(tmp_path: Path):
    """Successfully validate repository with relative path."""
    # Initialize git repository
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
    (tmp_path / "test.py").write_text("print('hello')")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    # Change to parent directory and use relative path
    parent = tmp_path.parent
    relative_path = tmp_path.name
    
    # Save current directory
    import os
    original_cwd = os.getcwd()
    try:
        os.chdir(parent)
        result = validate_repository(Path(relative_path))
        
        assert result.repo_path == tmp_path.resolve()
        assert len(result.head_sha) == 40
    finally:
        os.chdir(original_cwd)
