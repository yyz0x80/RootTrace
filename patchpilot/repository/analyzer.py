"""Repository analysis for PatchPilot.

This module provides functionality to analyze a target repository
and identify files relevant to the current issue, including:
- Python source files
- Test files
- Configuration files
- Files matching issue keywords
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from patchpilot.issue.schema import NormalizedIssue
from patchpilot.repository.schema import RepositoryContext

# Configuration file names to detect
_CONFIG_FILES = {
    "pyproject.toml",
    "requirements.txt",
    "setup.py",
    "setup.cfg",
}


# Common stopwords to filter out from keyword extraction
_STOPWORDS = {
    "add",
    "support",
    "with",
    "from",
    "into",
    "the",
    "and",
    "for",
    "to",
    "of",
    "in",
    "is",
    "a",
    "an",
    "that",
    "this",
    "it",
    "on",
    "by",
    "or",
    "be",
    "are",
    "as",
    "when",
    "not",
}


def _git_ls_files(repo: Path) -> list[str]:
    """Get all files tracked by Git in the repository.
    
    Args:
        repo: Path to the repository root.
        
    Returns:
        List of relative file paths tracked by Git.
        
    Raises:
        subprocess.CalledProcessError: If git command fails.
    """
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )

    return [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip()
    ]


def _extract_keywords(issue: NormalizedIssue) -> list[str]:
    """Extract relevant keywords from an issue for code search.
    
    Combines title and problem statement, filters stopwords,
    and returns the most relevant keywords.
    
    Args:
        issue: Normalized issue to extract keywords from.
        
    Returns:
        List of unique keywords (max 8) for searching code.
    """
    text = f"{issue.title} {issue.problem_statement}"

    # Extract words that start with a letter and contain alphanumeric characters
    words = re.findall(
        r"[A-Za-z_][A-Za-z0-9_]{2,}",
        text,
    )

    result = []

    for word in words:
        lower = word.lower()

        # Skip stopwords and duplicates
        if lower in _STOPWORDS:
            continue

        if lower not in result:
            result.append(lower)

    # Return at most 8 keywords
    return result[:8]


def _search_keyword(repo: Path, keyword: str) -> list[str]:
    """Search for files containing a keyword using ripgrep.
    
    Args:
        repo: Path to the repository root.
        keyword: Keyword to search for (case-insensitive).
        
    Returns:
        List of relative file paths containing the keyword.
    """
    result = subprocess.run(
        ["rg", "-l", "-i", keyword, "."],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )

    # ripgrep returns 0 for matches, 1 for no matches, >1 for errors
    if result.returncode not in (0, 1):
        return []

    matches = []
    for line in result.stdout.splitlines():
        # Remove "./" prefix if present
        path = line.removeprefix("./")
        if path:
            matches.append(path)

    return matches


def analyze_repository(
    repo: Path,
    issue: NormalizedIssue,
    base_commit: str,
) -> RepositoryContext:
    """Analyze a repository and identify files relevant to the issue.
    
    Performs comprehensive repository analysis:
    1. Lists all Git-tracked files
    2. Identifies Python source files
    3. Identifies test files
    4. Identifies configuration files
    5. Extracts keywords from the issue
    6. Searches for files matching those keywords
    
    Args:
        repo: Path to the repository root.
        issue: Normalized issue to analyze against.
        base_commit: Git commit SHA being used as baseline.
        
    Returns:
        RepositoryContext with categorized file lists and keyword matches.
    """
    # Get all tracked files
    tracked_files = _git_ls_files(repo)

    # Identify Python files
    python_files = [
        path
        for path in tracked_files
        if path.endswith(".py")
    ]

    # Identify test files (in tests/ directory or starting with test_)
    test_files = [
        path
        for path in python_files
        if (
            path.startswith("tests/")
            or Path(path).name.startswith("test_")
        )
    ]

    # Identify configuration files
    config_files = [
        path
        for path in tracked_files
        if Path(path).name in _CONFIG_FILES
    ]

    # Extract keywords from issue
    keywords = _extract_keywords(issue)

    # Search for files matching keywords
    keyword_matches = []
    for keyword in keywords:
        matches = _search_keyword(repo, keyword)
        for match in matches:
            if match not in keyword_matches:
                keyword_matches.append(match)

    return RepositoryContext(
        base_commit=base_commit,
        tracked_files=tracked_files,
        python_files=python_files,
        test_files=test_files,
        config_files=config_files,
        keyword_matches=keyword_matches,
    )
