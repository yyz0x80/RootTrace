"""Repository preflight validation for PatchPilot.

This module provides validation functions to ensure a target repository
meets the requirements for PatchPilot operations:
- Is a valid Git repository
- Has a baseline commit
- Has a clean working tree
- Can provide current HEAD information
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from patchpilot.repository.schema import RepositoryPreflightResult


class RepositoryPreflightError(RuntimeError):
    """Error raised when repository preflight validation fails."""


def _run_git(repo: Path, *args: str) -> str:
    """Run a git command in the repository directory.
    
    Args:
        repo: Path to the repository.
        *args: Git command arguments.
        
    Returns:
        Stripped stdout from the git command.
        
    Raises:
        RepositoryPreflightError: If the git command fails.
    """
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        error_message = result.stderr.strip() or result.stdout.strip()
        raise RepositoryPreflightError(error_message)

    return result.stdout.strip()


def validate_repository(repo: Path) -> RepositoryPreflightResult:
    """Validate that a repository meets PatchPilot requirements.
    
    This function performs the following checks:
    1. Path exists and is a directory
    2. Is a Git repository
    3. Has at least one commit (baseline)
    4. Working tree is clean (no uncommitted changes)
    
    Args:
        repo: Path to the repository to validate.
        
    Returns:
        RepositoryPreflightResult containing the validated path and HEAD SHA.
        
    Raises:
        RepositoryPreflightError: If any validation check fails.
    """
    repo = repo.resolve()

    # Check 1: Path exists and is a directory
    if not repo.exists():
        raise RepositoryPreflightError(
            f"Repository does not exist: {repo}"
        )

    if not repo.is_dir():
        raise RepositoryPreflightError(
            f"Repository path is not a directory: {repo}"
        )

    # Check 2: Must be a Git repository
    try:
        git_root = _run_git(repo, "rev-parse", "--show-toplevel")
    except RepositoryPreflightError:
        raise RepositoryPreflightError(
            "Target directory is not a Git repository. "
            "Initialize Git and create a baseline commit first."
        )

    # Ensure the provided path is the repository root
    if Path(git_root).resolve() != repo:
        raise RepositoryPreflightError(
            "--repo must point to the Git repository root."
        )

    # Check 3: Must have at least one commit
    try:
        head_sha = _run_git(repo, "rev-parse", "HEAD")
    except RepositoryPreflightError:
        raise RepositoryPreflightError(
            "Repository has no baseline commit. "
            "Create an initial commit before running PatchPilot."
        )

    # Check 4: Working tree must be clean
    status = _run_git(repo, "status", "--porcelain")
    if status:
        raise RepositoryPreflightError(
            "Repository contains uncommitted changes. "
            "Commit or stash them before running PatchPilot."
        )

    return RepositoryPreflightResult(
        repo_path=repo,
        head_sha=head_sha,
    )
