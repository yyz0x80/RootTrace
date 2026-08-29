"""Prepare disposable, revision-pinned checkouts for GitHub ingestion."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from .models import GitHubRepositoryRef

_SHA_PATTERN = re.compile(r"^[0-9a-fA-F]{7,64}$")


class GitHubRepositoryError(RuntimeError):
    """Raised when a GitHub repository cannot be prepared safely."""


@dataclass
class PreparedGitHubRepository:
    """Disposable checkout and its cache source."""

    root: Path
    repo: Path
    revision: str
    cache_path: Path

    def close(self) -> None:
        """Remove the disposable checkout while retaining the cache mirror."""
        shutil.rmtree(self.root, ignore_errors=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _run_git(
    args: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int = 600,
) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GitHubRepositoryError(f"git {' '.join(args)} failed: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise GitHubRepositoryError(f"git {' '.join(args)} failed: {detail[:500]}")
    return result.stdout.strip()


def _git_env(token: str | None) -> dict[str, str] | None:
    if not token:
        return None
    env = os.environ.copy()
    env.update(
        {
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "http.extraheader",
            "GIT_CONFIG_VALUE_0": f"Authorization: bearer {token}",
        }
    )
    return env


def _validate_revision(revision: str) -> str:
    if not isinstance(revision, str) or not _SHA_PATTERN.fullmatch(revision):
        raise ValueError("repository revision must be a 7-64 character hexadecimal SHA")
    return revision


def _mirror_path(cache_dir: Path, reference: GitHubRepositoryRef) -> Path:
    return cache_dir / f"{reference.owner}__{reference.repo}.git"


def _ensure_mirror(
    reference: GitHubRepositoryRef,
    cache_dir: Path,
    *,
    clone_url: str | None,
    token: str | None,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    mirror = _mirror_path(cache_dir, reference)
    env = _git_env(token)
    source = clone_url or f"https://github.com/{reference.full_name}.git"
    if mirror.exists():
        if not mirror.is_dir():
            raise GitHubRepositoryError(f"repository cache path is not a directory: {mirror}")
        _run_git(["rev-parse", "--git-dir"], cwd=mirror, env=env, timeout=30)
        _run_git(["fetch", "--prune", "origin"], cwd=mirror, env=env)
    else:
        _run_git(["clone", "--bare", source, str(mirror)], env=env)
    return mirror


def resolve_default_branch_revision(
    reference: GitHubRepositoryRef,
    *,
    cache_dir: str | Path,
    clone_url: str | None = None,
    token: str | None = None,
) -> str:
    """Resolve the current default branch commit from a cached mirror."""
    mirror = _ensure_mirror(
        reference,
        Path(cache_dir).expanduser().resolve(),
        clone_url=clone_url,
        token=token,
    )
    revision = _run_git(["rev-parse", "HEAD"], cwd=mirror, env=_git_env(token), timeout=30)
    return _validate_revision(revision)


def _mirror_has_revision(mirror: Path, revision: str) -> bool:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
        cwd=mirror,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    return result.returncode == 0


def _verify_checkout(repo: Path, revision: str) -> None:
    head = _run_git(["rev-parse", "HEAD"], cwd=repo, timeout=30)
    if head != revision:
        raise GitHubRepositoryError(
            f"prepared checkout HEAD mismatch: {head[:12]} != {revision[:12]}"
        )
    count = _run_git(["rev-list", "--all", "--count"], cwd=repo, timeout=30)
    if count != "1":
        raise GitHubRepositoryError(
            f"prepared checkout exposes {count} commits; expected exactly 1"
        )
    refs = _run_git(
        ["for-each-ref", "--format=%(refname)"], cwd=repo, timeout=30
    )
    if refs:
        raise GitHubRepositoryError("prepared checkout contains unexpected refs")


def prepare_github_repository(
    reference: GitHubRepositoryRef,
    revision: str,
    *,
    cache_dir: str | Path,
    clone_url: str | None = None,
    token: str | None = None,
    work_dir: str | Path | None = None,
) -> PreparedGitHubRepository:
    """Create a base-only checkout from a cached GitHub mirror.

    The cache is a local mirror and may contain history. The returned checkout
    is a separate shallow repository exposing only the requested revision.
    """
    revision = _validate_revision(revision)
    cache = Path(cache_dir).expanduser().resolve()
    mirror = _ensure_mirror(
        reference,
        cache,
        clone_url=clone_url,
        token=token,
    )
    if not _mirror_has_revision(mirror, revision):
        _run_git(["fetch", "origin", revision], cwd=mirror, env=_git_env(token))
    if not _mirror_has_revision(mirror, revision):
        raise GitHubRepositoryError(f"repository mirror does not contain revision {revision}")

    parent = Path(work_dir).expanduser().resolve() if work_dir else Path(tempfile.gettempdir())
    parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="roottrace-github-", dir=parent))
    repo = root / "repo"
    try:
        repo.mkdir()
        _run_git(["init", "--quiet"], cwd=repo, timeout=60)
        _run_git(["remote", "add", "origin", str(mirror)], cwd=repo, timeout=60)
        _run_git(
            ["fetch", "--depth=1", "origin", revision],
            cwd=repo,
            env=_git_env(token),
        )
        _run_git(["checkout", "--quiet", "--detach", "FETCH_HEAD"], cwd=repo, timeout=120)
        _verify_checkout(repo, revision)
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise
    return PreparedGitHubRepository(
        root=root,
        repo=repo,
        revision=revision,
        cache_path=mirror,
    )


__all__ = [
    "GitHubRepositoryError",
    "PreparedGitHubRepository",
    "prepare_github_repository",
    "resolve_default_branch_revision",
]
