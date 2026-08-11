"""Data schemas for repository operations."""

from pathlib import Path

from pydantic import BaseModel


class RepositoryPreflightResult(BaseModel):
    """Result of repository preflight validation.

    Attributes:
        repo_path: Absolute path to the validated repository root.
        head_sha: Current HEAD commit SHA.
    """
    repo_path: Path
    head_sha: str


class RepositoryContext(BaseModel):
    """Analysis context for a target repository.

    Provides structured information about the repository structure
    and files relevant to the current issue.

    Attributes:
        base_commit: Git commit SHA being used as the baseline.
        tracked_files: All files tracked by Git in the repository.
        python_files: Python source files (*.py) in the repository.
        test_files: Test files identified in the repository.
        config_files: Configuration files (pyproject.toml, requirements.txt, etc.).
        keyword_matches: Files matching keywords extracted from the issue.
    """
    base_commit: str
    tracked_files: list[str]
    python_files: list[str]
    test_files: list[str]
    config_files: list[str]
    keyword_matches: list[str]
