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

        # Reject modifying tests/ directory
        if "tests" in resolved.parts:
            raise PermissionError(f"Modifying tests directory rejected: {relative_path}")

        return resolved