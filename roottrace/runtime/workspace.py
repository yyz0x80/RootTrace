"""Workspace policy and path resolution for RCA repository access.

This module provides the Workspace class which enforces fundamental security boundaries
for file system operations within a target repository. It ensures that:

- All paths are resolved relative to the repository root
- Absolute paths are rejected
- Path traversal attacks (..) are prevented
- Sensitive files (.env, .git) are protected from read/write access

The Workspace class is the authoritative security boundary for path resolution
and fundamental file system protections. Additional caller-specific policies are
enforced by the corresponding tool or runtime registry.
"""

import subprocess
from pathlib import Path

from pydantic import BaseModel, Field

MAX_FINGERPRINT_CHARS = 8_000


class RepositoryFingerprint(BaseModel):
    """Read-only snapshot proving the target repository is unchanged."""

    head_sha: str = Field(max_length=64)
    status_porcelain: str = Field(max_length=MAX_FINGERPRINT_CHARS)
    diff_stat: str = Field(max_length=MAX_FINGERPRINT_CHARS)


def _run_git_capture(
    repo: Path,
    *args: str,
    limit: int = MAX_FINGERPRINT_CHARS,
) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    text = result.stdout
    if len(text) > limit:
        return text[:limit] + "\n...[truncated]"
    return text.rstrip("\n")


def capture_repository_fingerprint(repo: Path) -> RepositoryFingerprint:
    """Capture HEAD, porcelain status, and diff stat of the target repo."""
    return RepositoryFingerprint(
        head_sha=_run_git_capture(repo, "rev-parse", "HEAD", limit=64).strip(),
        status_porcelain=_run_git_capture(repo, "status", "--porcelain"),
        diff_stat=_run_git_capture(repo, "diff", "--stat"),
    )


def assert_fingerprint_unchanged(
    before: RepositoryFingerprint,
    after: RepositoryFingerprint,
) -> None:
    """Raise if the target repository changed between two fingerprints."""
    if before.model_dump(mode="json") != after.model_dump(mode="json"):
        raise RuntimeError("target repository changed during RCA context collection")


class Workspace:
    """Workspace manager for secure path resolution and fundamental file system protections."""

    def __init__(self, root: Path):
        """Initialize the workspace with a repository root.

        Args:
            root: Path to the repository root directory
        """
        self.root = root.resolve()

    def resolve(self, relative_path: str) -> Path:
        """Resolve relative path to absolute path under repository root.

        Args:
            relative_path: Relative path to resolve

        Returns:
            Resolved absolute path under repository root

        Raises:
            ValueError: If path is absolute or attempts repository escape
        """
        if Path(relative_path).is_absolute():
            raise ValueError(f"Absolute path rejected: {relative_path}")

        resolved = (self.root / relative_path).resolve()

        # Check if path is outside repository root
        try:
            resolved.relative_to(self.root)
        except ValueError:
            raise ValueError(f"Path escapes repository: {relative_path}")

        return resolved

    def assert_read_allowed(self, relative_path: str) -> Path:
        """Check if file read is allowed based on fundamental security boundaries.

        Args:
            relative_path: Relative path to the file to read

        Returns:
            Resolved absolute path

        Raises:
            ValueError: If path resolution fails
            PermissionError: If reading sensitive files is attempted
        """
        resolved = self.resolve(relative_path)

        # Reject reading sensitive files (fundamental security boundary)
        if resolved.name == ".env":
            raise PermissionError(f"Reading .env file rejected: {relative_path}")

        if resolved.name == ".git" or ".git" in resolved.parts:
            raise PermissionError(f"Reading .git directory rejected: {relative_path}")

        return resolved

    def assert_write_allowed(self, relative_path: str) -> Path:
        """Check if file write is allowed based on fundamental security boundaries.

        Args:
            relative_path: Relative path to the file to write

        Returns:
            Resolved absolute path

        Raises:
            ValueError: If path resolution fails
            PermissionError: If writing sensitive files is attempted
        """
        resolved = self.resolve(relative_path)

        # Reject writing sensitive files (fundamental security boundary)
        if resolved.name == ".env":
            raise PermissionError(f"Writing .env file rejected: {relative_path}")

        if resolved.name == ".git" or ".git" in resolved.parts:
            raise PermissionError(f"Writing .git directory rejected: {relative_path}")

        return resolved
