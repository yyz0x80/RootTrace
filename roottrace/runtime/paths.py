"""Repository-relative path validation shared by low-level capabilities."""

from pathlib import PurePosixPath


def validate_relative_path(value: str) -> str:
    """Validate and normalize a repository-relative path."""
    if not value or not value.strip():
        raise ValueError("repository-relative path must not be empty")
    if "\\" in value:
        raise ValueError("repository-relative path must use forward slashes")
    if value.startswith("~"):
        raise ValueError("repository-relative path must not start with '~'")
    try:
        path = PurePosixPath(value)
    except ValueError as exc:
        raise ValueError("invalid repository-relative path") from exc
    if path.is_absolute():
        raise ValueError("repository-relative path must not be absolute")
    if not path.parts or any(part in (".", "..") for part in path.parts):
        raise ValueError(
            "repository-relative path must not contain '.' or '..' segments"
        )
    return path.as_posix()
