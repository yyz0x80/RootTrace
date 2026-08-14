"""Workspace policy and path resolution for target repository access.

This module provides the Workspace class which enforces security boundaries
for file system operations within a target repository. It ensures that:

- All paths are resolved relative to the repository root
- Absolute paths are rejected
- Path traversal attacks (..) are prevented
- Sensitive files (.env, .git) are protected from read/write access
- Test files are protected from modification

The Workspace class is the authoritative security boundary for all
file system operations performed by the PatchPilot agent.
"""

from pathlib import Path


class Workspace:
    def __init__(self, root: Path):
        self.root = root.resolve()

    def resolve(self, relative_path: str) -> Path:
        """Resolve relative path to absolute path under repository root"""
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
        """Check if file read is allowed"""
        resolved = self.resolve(relative_path)

        # Reject reading sensitive files
        if resolved.name == ".env":
            raise PermissionError(f"Reading .env file rejected: {relative_path}")

        if resolved.name == ".git" or ".git" in resolved.parts:
            raise PermissionError(f"Reading .git directory rejected: {relative_path}")

        return resolved

    def assert_write_allowed(self, relative_path: str) -> Path:
        """Check if file write is allowed"""
        resolved = self.resolve(relative_path)

        # Reject writing sensitive files
        if resolved.name == ".env":
            raise PermissionError(f"Writing .env file rejected: {relative_path}")

        if resolved.name == ".git" or ".git" in resolved.parts:
            raise PermissionError(f"Writing .git directory rejected: {relative_path}")

        # Reject modifying CI/CD configuration files
        if ".github" in resolved.parts and "workflows" in resolved.parts:
            raise PermissionError(f"Modifying CI/CD workflows is not allowed: {relative_path}")

        # Reject modifying test files (Day 1 restriction)
        if self._is_test_file(relative_path):
            raise PermissionError(
                f"Modifying test files is not allowed: {relative_path}. "
                "Test files must remain read-only. Only modify source code implementation."
            )

        return resolved

    def _is_test_file(self, relative_path: str) -> bool:
        """Check if a path refers to a test file.

        Args:
            relative_path: Relative path to check

        Returns:
            True if the path is a test file, False otherwise
        """
        # Check if path contains tests/ directory
        if "tests" in relative_path.split("/"):
            return True

        # Check if filename starts with test_
        parts = relative_path.split("/")
        return bool(parts and parts[-1].startswith("test_"))