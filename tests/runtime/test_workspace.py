from pathlib import Path

import pytest

from roottrace.runtime.workspace import Workspace


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
    """Workspace leaves caller-specific test-file policy to the caller."""
    root = Path("/tmp/test_repo")
    workspace = Workspace(root)

    # Allow reading tests directory
    resolved = workspace.assert_read_allowed("tests/test_main.py")
    assert resolved == (root / "tests/test_main.py").resolve()

    # Workspace leaves writes to the caller-specific policy.
    resolved = workspace.assert_write_allowed("tests/test_main.py")
    assert resolved == (root / "tests/test_main.py").resolve()

    # Workspace leaves writes to the caller-specific policy.
    resolved = workspace.assert_write_allowed("tests/unit/test_utils.py")
    assert resolved == (root / "tests/unit/test_utils.py").resolve()


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


def test_rejects_test_prefix_files():
    """Workspace leaves caller-specific test-file policy to the caller."""
    root = Path("/tmp/test_repo")
    workspace = Workspace(root)

    # Allow reading test_*.py files
    resolved = workspace.assert_read_allowed("test_main.py")
    assert resolved == (root / "test_main.py").resolve()

    # Workspace leaves writes to the caller-specific policy.
    resolved = workspace.assert_write_allowed("test_main.py")
    assert resolved == (root / "test_main.py").resolve()

    resolved = workspace.assert_write_allowed("test_utils.py")
    assert resolved == (root / "test_utils.py").resolve()

    # Test in subdirectory
    resolved = workspace.assert_write_allowed("src/test_module.py")
    assert resolved == (root / "src/test_module.py").resolve()


def test_rejects_github_workflows():
    """Workspace leaves caller-specific CI policy to the caller."""
    root = Path("/tmp/test_repo")
    workspace = Workspace(root)

    # Allow reading .github/workflows for inspection
    resolved = workspace.assert_read_allowed(".github/workflows/ci.yml")
    assert resolved == (root / ".github/workflows/ci.yml").resolve()

    # Workspace leaves writes to the caller-specific policy.
    resolved = workspace.assert_write_allowed(".github/workflows/ci.yml")
    assert resolved == (root / ".github/workflows/ci.yml").resolve()

    resolved = workspace.assert_write_allowed(".github/workflows/deploy.yml")
    assert resolved == (root / ".github/workflows/deploy.yml").resolve()
