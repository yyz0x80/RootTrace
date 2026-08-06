from pathlib import Path

import pytest

from patchpilot.workspace import Workspace


def test_allows_source_file():
    """Allow reading and writing source files"""
    root = Path("/tmp/test_repo")
    workspace = Workspace(root)

    resolved = workspace.assert_read_allowed("src/main.py")
    assert resolved == (root / "src/main.py").resolve()

    resolved = workspace.assert_write_allowed("src/main.py")
    assert resolved == (root / "src/main.py").resolve()


def test_rejects_parent_directory_escape():
    """Reject escaping repository using ../"""
    root = Path("/tmp/test_repo")
    workspace = Workspace(root)

    with pytest.raises(ValueError, match="Path escapes repository"):
        workspace.resolve("../etc/passwd")

    with pytest.raises(ValueError, match="Path escapes repository"):
        workspace.resolve("src/../../../etc/passwd")


def test_rejects_absolute_path():
    """Reject absolute paths"""
    root = Path("/tmp/test_repo")
    workspace = Workspace(root)

    with pytest.raises(ValueError, match="Absolute path rejected"):
        workspace.resolve("/etc/passwd")

    with pytest.raises(ValueError, match="Absolute path rejected"):
        workspace.resolve("/tmp/test_repo/src/main.py")


def test_rejects_env_file():
    """Reject reading and writing .env files"""
    root = Path("/tmp/test_repo")
    workspace = Workspace(root)

    with pytest.raises(PermissionError, match="Reading .env file rejected"):
        workspace.assert_read_allowed(".env")

    with pytest.raises(PermissionError, match="Writing .env file rejected"):
        workspace.assert_write_allowed(".env")

    with pytest.raises(PermissionError, match="Reading .env file rejected"):
        workspace.assert_read_allowed("config/.env")


def test_rejects_test_modification():
    """Reject modifying tests directory"""
    root = Path("/tmp/test_repo")
    workspace = Workspace(root)

    # Allow reading tests directory
    resolved = workspace.assert_read_allowed("tests/test_main.py")
    assert resolved == (root / "tests/test_main.py").resolve()

    # Reject writing to tests directory
    with pytest.raises(PermissionError, match="Modifying tests directory rejected"):
        workspace.assert_write_allowed("tests/test_main.py")

    with pytest.raises(PermissionError, match="Modifying tests directory rejected"):
        workspace.assert_write_allowed("tests/unit/test_utils.py")


def test_rejects_git_directory():
    """Reject accessing .git directory"""
    root = Path("/tmp/test_repo")
    workspace = Workspace(root)

    with pytest.raises(PermissionError, match="Reading .git directory rejected"):
        workspace.assert_read_allowed(".git")

    with pytest.raises(PermissionError, match="Writing .git directory rejected"):
        workspace.assert_write_allowed(".git")

    with pytest.raises(PermissionError, match="Reading .git directory rejected"):
        workspace.assert_read_allowed(".git/config")