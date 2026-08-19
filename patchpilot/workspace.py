"""Workspace policy and path resolution for target repository access.

This module provides the Workspace class which enforces fundamental security boundaries
for file system operations within a target repository. It ensures that:

- All paths are resolved relative to the repository root
- Absolute paths are rejected
- Path traversal attacks (..) are prevented
- Sensitive files (.env, .git) are protected from read/write access

The Workspace class is the authoritative security boundary for path resolution
and fundamental file system protections. Project-specific policies (e.g., test file
restrictions, CI/CD restrictions) are enforced through the PolicySet system.
"""

from pathlib import Path


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