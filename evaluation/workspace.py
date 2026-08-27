"""Disposable per-case repositories bounded to the SWE-bench base commit.

Each case gets a fresh shallow workspace whose history contains exactly the
``base_commit``. The workspace is created with ``git fetch --depth=1`` of the
pinned SHA, so future commits, future refs, and the fixing commit are
physically absent: ``git log --all`` cannot see beyond the base and no remote
refs exist to resolve. RootTrace's Git investigation is therefore bounded to
the incident's base commit.
"""

from __future__ import annotations

import dataclasses
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from evaluation.manifest import ManifestCase

_SHA_PATTERN = re.compile(r"^[0-9a-fA-F]{7,64}$")


class WorkspaceError(RuntimeError):
    """Raised when a case workspace cannot be created or verified."""


@dataclasses.dataclass(frozen=True)
class CaseWorkspace:
    """A disposable base-commit-bounded repository checkout."""

    root: Path
    repo: Path
    base_commit: str


def _run_git(
    *args: str,
    cwd: Path | None = None,
    timeout: int = 120,
) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise WorkspaceError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout.strip()


def _is_git_repo(path: Path) -> bool:
    try:
        _run_git("rev-parse", "--git-dir", cwd=path, timeout=30)
    except WorkspaceError:
        return False
    return True


def resolve_repo_cache(repo_cache: str | Path, repo: str) -> Path:
    """Resolve ``owner/name`` to a local git mirror under ``repo_cache``.

    Prefers a bare mirror (``owner__name.git``), then a plain checkout
    (``owner__name``), then the prepared worktree backup
    (``owner__name_worktree_backup``).
    """
    cache = Path(repo_cache)
    if not cache.is_dir():
        raise FileNotFoundError(f"repo cache is not a directory: {cache}")
    owner, _, name = repo.partition("/")
    candidates = (
        f"{owner}__{name}.git",
        f"{owner}__{name}",
        f"{owner}__{name}_worktree_backup",
    )
    for relative in candidates:
        candidate = cache / relative
        if candidate.is_dir() and _is_git_repo(candidate):
            return candidate.resolve()
    raise FileNotFoundError(f"no local git mirror for {repo} under {cache}")


def _verify_commit(mirror: Path, base_commit: str) -> None:
    _run_git("cat-file", "-t", base_commit, cwd=mirror, timeout=30)


def create_case_workspace(
    repo_cache: str | Path,
    case: ManifestCase,
    *,
    work_root: str | Path | None = None,
) -> CaseWorkspace:
    """Create a disposable workspace checked out at ``case.base_commit``."""
    if not _SHA_PATTERN.fullmatch(case.base_commit):
        raise WorkspaceError(f"invalid base commit: {case.base_commit}")
    mirror = resolve_repo_cache(repo_cache, case.repo)
    _verify_commit(mirror, case.base_commit)
    parent = Path(work_root) if work_root is not None else Path(tempfile.gettempdir())
    root = Path(tempfile.mkdtemp(prefix="roottrace-case-", dir=parent))
    repo = root / "repo"
    try:
        repo.mkdir()
        _run_git("init", "--quiet", cwd=repo, timeout=60)
        _run_git("remote", "add", "origin", str(mirror), cwd=repo, timeout=60)
        _run_git(
            "fetch",
            "--depth=1",
            "origin",
            case.base_commit,
            cwd=repo,
            timeout=600,
        )
        _run_git("checkout", "--quiet", "FETCH_HEAD", cwd=repo, timeout=120)
        verify_base_boundary(repo, case.base_commit)
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise
    return CaseWorkspace(root=root, repo=repo, base_commit=case.base_commit)


def verify_base_boundary(repo: Path, base_commit: str) -> None:
    """Prove the workspace exposes exactly the base commit and no future refs."""
    head = _run_git("rev-parse", "HEAD", cwd=repo, timeout=30)
    if head != base_commit:
        raise WorkspaceError(
            f"workspace HEAD mismatch: {head[:12]} != {base_commit[:12]}"
        )
    count = _run_git("rev-list", "--all", "--count", cwd=repo, timeout=30)
    if count != "1":
        raise WorkspaceError(
            f"workspace exposes {count} commits; expected exactly 1 (base only)"
        )
    refs = _run_git("for-each-ref", "--format=%(refname)", cwd=repo, timeout=30)
    if refs:
        raise WorkspaceError("workspace contains refs beyond the base commit")


def destroy_case_workspace(workspace: CaseWorkspace) -> None:
    """Remove a disposable case workspace."""
    shutil.rmtree(workspace.root, ignore_errors=True)
