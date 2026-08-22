"""Tests for the RCA-safe read-only tool registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from patchpilot.rca.tools import RcaToolRegistry
from patchpilot.workspace import Workspace


@pytest.fixture
def registry(git_repo, tmp_path: Path) -> RcaToolRegistry:
    external_root = tmp_path / "external_logs"
    external_root.mkdir()
    (external_root / "ci.log").write_text(
        "Traceback (most recent call last):\n"
        "  File 'pkg/calc.py', line 5, in multiply\n"
        "ValueError: boom\n",
        encoding="utf-8",
    )
    return RcaToolRegistry(
        Workspace(git_repo.repo),
        external_root=external_root,
    )


def test_registry_exposes_only_read_only_tools(registry: RcaToolRegistry) -> None:
    assert sorted(registry.get_available_tools()) == [
        "git_blame",
        "git_history",
        "git_show",
        "inspect_symbols",
        "read_external_log",
        "read_file",
        "search_code",
    ]
    for write_tool in ("edit_file", "write_file", "write_scratch_test", "run_command"):
        assert registry.get_tool_schema(write_tool) is None
        with pytest.raises(PermissionError):
            getattr(registry, write_tool)({})
        with pytest.raises(KeyError):
            registry.execute(write_tool, {})


def test_search_code_and_read_file(registry: RcaToolRegistry) -> None:
    search = registry.execute("search_code", {"query": "multiply"})
    assert search.ok
    assert "pkg/calc.py" in search.content
    assert search.command.startswith("rg ")

    # A query that looks like a flag must be treated as a literal pattern.
    literal = registry.execute("search_code", {"query": "--files"})
    assert literal.ok

    read = registry.execute("read_file", {"path": "pkg/calc.py", "raw": True})
    assert read.ok
    assert "def multiply" in read.content


def test_file_tools_reject_bad_paths(registry: RcaToolRegistry) -> None:
    for tool in ("read_file", "inspect_symbols"):
        for bad_path in ("../secret.py", "/etc/passwd", ".git/config"):
            result = registry.execute(tool, {"path": bad_path})
            assert not result.ok, f"{tool} accepted {bad_path}"
    secret = registry.execute("read_file", {"path": "pkg/../.env"})
    assert not secret.ok


def test_inspect_symbols_lists_python_symbols(registry: RcaToolRegistry) -> None:
    result = registry.execute(
        "inspect_symbols",
        {"path": "pkg/calc.py"},
    )
    assert result.ok
    assert "def add" in result.content
    assert "def multiply" in result.content
    assert "1-2" in result.content.splitlines()[1]

    filtered = registry.execute(
        "inspect_symbols",
        {"path": "pkg/calc.py", "name": "multiply"},
    )
    assert filtered.ok
    assert "multiply" in filtered.content
    assert "def add" not in filtered.content


def test_inspect_symbols_handles_invalid_files(registry: RcaToolRegistry, git_repo) -> None:
    (git_repo.repo / "pkg" / "broken.py").write_text(
        "def broken(:\n    pass\n",
        encoding="utf-8",
    )
    syntax_error = registry.execute("inspect_symbols", {"path": "pkg/broken.py"})
    assert not syntax_error.ok
    assert "Syntax error" in syntax_error.content

    (git_repo.repo / "README.md").write_text("# docs\n", encoding="utf-8")
    not_python = registry.execute("inspect_symbols", {"path": "README.md"})
    assert not not_python.ok

    missing = registry.execute("inspect_symbols", {"path": "pkg/missing.py"})
    assert not missing.ok


def test_git_history_bounded_and_scoped(registry: RcaToolRegistry) -> None:
    result = registry.execute("git_history", {"max_count": 20})
    assert result.ok
    lines = [line for line in result.content.splitlines() if line.strip()]
    assert len(lines) == 2
    assert "fix multiply" in lines[0]
    assert "initial" in lines[1]

    capped = registry.execute("git_history", {"max_count": 1})
    assert len([line for line in capped.content.splitlines() if line.strip()]) == 1

    scoped = registry.execute(
        "git_history",
        {"path": "pkg/calc.py", "grep": "fix multiply"},
    )
    assert scoped.ok
    assert "fix multiply" in scoped.content

    pickaxe = registry.execute(
        "git_history",
        {"query": "return a * b"},
    )
    assert pickaxe.ok
    assert "fix multiply" in pickaxe.content
    assert "initial" not in pickaxe.content


def test_git_history_rejects_unsafe_input(registry: RcaToolRegistry) -> None:
    for arguments in (
        {"path": "../outside.py"},
        {"path": "/etc/passwd"},
        {"path": ".git/config"},
        {"max_count": 0},
        {"max_count": 10_000},
        {"grep": "a\nb"},
    ):
        result = registry.execute("git_history", arguments)
        assert not result.ok, f"git_history accepted {arguments}"


def test_git_blame_and_show(registry: RcaToolRegistry, git_repo) -> None:
    blame = registry.execute("git_blame", {"path": "pkg/calc.py"})
    assert blame.ok
    assert "multiply" in blame.content

    ranged = registry.execute(
        "git_blame",
        {"path": "pkg/calc.py", "start_line": 1, "end_line": 2},
    )
    assert ranged.ok

    show = registry.execute("git_show", {"revision": git_repo.head_sha})
    assert show.ok
    assert "fix multiply" in show.content

    stat = registry.execute(
        "git_show",
        {"revision": git_repo.head_sha[:8], "stat": True},
    )
    assert stat.ok
    assert "calc.py" in stat.content

    at_revision = registry.execute(
        "git_show",
        {"revision": git_repo.head_sha, "path": "pkg/calc.py"},
    )
    assert at_revision.ok
    assert "return a * b" in at_revision.content


def test_git_tools_reject_unsafe_input(registry: RcaToolRegistry) -> None:
    for arguments in (
        {"path": "../x.py"},
        {"path": "/abs/x.py"},
        {"path": ".git/config"},
    ):
        result = registry.execute("git_blame", arguments)
        assert not result.ok, f"git_blame accepted {arguments}"

    for arguments in (
        {"revision": "--reset"},
        {"revision": "HEAD~1"},
        {"revision": "a" * 3},
        {"revision": "good sha; rm -rf /"},
    ):
        result = registry.execute("git_show", arguments)
        assert not result.ok, f"git_show accepted {arguments}"

    bad_range = registry.execute(
        "git_blame",
        {"path": "pkg/calc.py", "start_line": 5, "end_line": 2},
    )
    assert not bad_range.ok


def test_read_external_log(registry: RcaToolRegistry) -> None:
    result = registry.execute("read_external_log", {"path": "ci.log"})
    assert result.ok
    assert "ValueError: boom" in result.content

    for bad_path in ("../outside.log", "/etc/hosts", "sub/../../x.log", ".env"):
        result = registry.execute("read_external_log", {"path": bad_path})
        assert not result.ok, f"read_external_log accepted {bad_path}"


def test_read_external_log_bounded(registry: RcaToolRegistry, tmp_path: Path) -> None:
    huge = tmp_path / "external_logs" / "huge.log"
    huge.write_text("x" * 300_000, encoding="utf-8")
    result = registry.execute("read_external_log", {"path": "huge.log"})
    assert result.ok
    assert result.truncated
    assert len(result.content) < 250_000
    assert "truncated" in result.content


def test_read_external_log_rejects_binary_and_requires_root(
    registry: RcaToolRegistry,
    git_repo,
) -> None:
    binary = registry.external_root / "binary.log"
    binary.write_bytes(b"\x00\x01\x02")
    result = registry.execute("read_external_log", {"path": "binary.log"})
    assert not result.ok
    assert "text file" in result.content

    no_root = RcaToolRegistry(Workspace(git_repo.repo))
    result = no_root.execute("read_external_log", {"path": "ci.log"})
    assert not result.ok
    assert "external log root" in result.content


def test_symlink_escape_is_rejected(registry: RcaToolRegistry, git_repo, tmp_path: Path) -> None:
    outside = tmp_path / "outside_secret.py"
    outside.write_text("SECRET = 'outside'\n", encoding="utf-8")
    link = git_repo.repo / "pkg" / "link.py"
    link.symlink_to(outside)
    linked_dir = git_repo.repo / "linked"
    linked_dir.symlink_to(tmp_path, target_is_directory=True)

    for tool, arguments in (
        ("read_file", {"path": "pkg/link.py"}),
        ("inspect_symbols", {"path": "pkg/link.py"}),
        ("read_file", {"path": "linked/outside_secret.py"}),
    ):
        result = registry.execute(tool, arguments)
        assert not result.ok, f"{tool} accepted symlink escape {arguments}"
