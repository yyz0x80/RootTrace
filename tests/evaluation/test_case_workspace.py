"""Disposable per-case workspace and local repo cache resolution tests."""

from __future__ import annotations

import subprocess

import pytest
from fixtures import build_source_repo, make_bare_mirror, make_worktree_mirror

from evaluation.manifest import ManifestCase
from evaluation.workspace import (
    create_case_workspace,
    destroy_case_workspace,
    resolve_repo_cache,
    verify_base_boundary,
)

_REPO = "acme/demo"


def _three_commit_repo(tmp_path):
    shas = build_source_repo(
        tmp_path / "build",
        [
            ("base one", {"pkg/mod.py": "def a():\n    return 1\n"}),
            ("base two", {"pkg/mod.py": "def a():\n    return 2\n"}),
            ("future fix", {"pkg/mod.py": "def a():\n    return 3\n"}),
        ],
    )
    return shas


def _case(base_commit: str) -> ManifestCase:
    return ManifestCase(
        instance_id="acme__demo-1",
        repo=_REPO,
        base_commit=base_commit,
    )


def test_resolve_repo_cache_prefers_bare_mirror(tmp_path) -> None:
    build_source_repo(tmp_path / "build", [("one", {"a.py": "x"})])
    repo = tmp_path / "build" / "source"
    cache = tmp_path / "cache"
    bare = make_bare_mirror(repo, cache, _REPO)
    worktree = make_worktree_mirror(repo, cache, _REPO)
    assert bare.exists()
    assert worktree.exists()
    resolved = resolve_repo_cache(cache, _REPO)
    assert resolved == bare.resolve()


def test_resolve_repo_cache_falls_back_to_worktree(tmp_path) -> None:
    repo = tmp_path / "build" / "source"
    build_source_repo(tmp_path / "build", [("one", {"a.py": "x"})])
    cache = tmp_path / "cache"
    worktree = make_worktree_mirror(repo, cache, _REPO)
    resolved = resolve_repo_cache(cache, _REPO)
    assert resolved == worktree.resolve()


def test_resolve_repo_cache_missing_repo_raises(tmp_path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    with pytest.raises(FileNotFoundError, match="no local git mirror"):
        resolve_repo_cache(cache, _REPO)


def test_create_workspace_is_bounded_to_base_commit(tmp_path) -> None:
    shas = _three_commit_repo(tmp_path)
    cache = tmp_path / "cache"
    make_bare_mirror(tmp_path / "build" / "source", cache, _REPO)

    workspace = create_case_workspace(cache, _case(shas[0]), work_root=tmp_path)
    try:
        assert workspace.base_commit == shas[0]
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workspace.repo,
            capture_output=True,
            text=True,
            check=True,
        )
        assert head.stdout.strip() == shas[0]
        count = subprocess.run(
            ["git", "rev-list", "--all", "--count"],
            cwd=workspace.repo,
            capture_output=True,
            text=True,
            check=True,
        )
        assert count.stdout.strip() == "1"
        refs = subprocess.run(
            ["git", "for-each-ref", "--format=%(refname)"],
            cwd=workspace.repo,
            capture_output=True,
            text=True,
            check=True,
        )
        assert refs.stdout.strip() == ""
        # The future fixing commit must not exist in the workspace at all.
        future = subprocess.run(
            ["git", "cat-file", "-e", shas[2]],
            cwd=workspace.repo,
            capture_output=True,
            text=True,
            check=False,
        )
        assert future.returncode != 0
    finally:
        destroy_case_workspace(workspace)
    assert not workspace.root.exists()


def test_verify_base_boundary_rejects_future_history(tmp_path) -> None:
    shas = _three_commit_repo(tmp_path)
    full = tmp_path / "full"
    subprocess.run(
        ["git", "clone", "--quiet", str(tmp_path / "build" / "source"), str(full)],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    subprocess.run(
        ["git", "checkout", "--quiet", shas[0]],
        cwd=full,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    with pytest.raises(Exception, match="expected exactly 1"):
        verify_base_boundary(full, shas[0])


def test_workspace_creation_failure_cleans_up(tmp_path) -> None:
    build_source_repo(tmp_path / "build", [("one", {"a.py": "x"})])
    cache = tmp_path / "cache"
    make_bare_mirror(tmp_path / "build" / "source", cache, _REPO)
    with pytest.raises(Exception, match="failed"):
        create_case_workspace(cache, _case("b" * 40), work_root=tmp_path)
    leftovers = list(tmp_path.glob("roottrace-case-*"))
    assert leftovers == []
