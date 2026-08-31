"""Tests for disposable GitHub repository preparation."""

import base64
import subprocess
from pathlib import Path

from roottrace.github import GitHubRepositoryRef, prepare_github_repository
from roottrace.github import repository as repository_module


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _build_source_repo(root: Path) -> tuple[Path, list[str]]:
    repo = root / "source"
    repo.parent.mkdir(parents=True)
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Test Author")
    _git(repo, "config", "user.email", "test@example.com")
    shas: list[str] = []
    for message, contents in (
        ("bug", "broken = True\n"),
        ("fix", "broken = False\n"),
    ):
        path = repo / "src/app.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", message)
        shas.append(_git(repo, "rev-parse", "HEAD"))
    return repo, shas


def _make_bare_mirror(source: Path, cache: Path) -> Path:
    destination = cache / "acme__widget.git"
    cache.mkdir()
    subprocess.run(
        ["git", "clone", "--quiet", "--bare", str(source), str(destination)],
        check=True,
        capture_output=True,
    )
    return destination


def test_git_token_uses_basic_auth_without_exposing_plaintext() -> None:
    env = repository_module._git_env("test-token")

    assert env is not None
    expected = base64.b64encode(b"x-access-token:test-token").decode("ascii")
    assert env["GIT_CONFIG_KEY_0"] == "http.extraheader"
    assert env["GIT_CONFIG_VALUE_0"] == f"Authorization: Basic {expected}"
    assert "test-token" not in env["GIT_CONFIG_VALUE_0"]


def test_git_env_is_omitted_without_token() -> None:
    assert repository_module._git_env(None) is None


def test_prepare_repository_exposes_only_requested_revision(tmp_path) -> None:
    source, shas = _build_source_repo(tmp_path / "source-root")
    cache = tmp_path / "cache"
    _make_bare_mirror(source, cache)
    reference = GitHubRepositoryRef(owner="acme", repo="widget")

    prepared = prepare_github_repository(
        reference,
        shas[0],
        cache_dir=cache,
        clone_url=str(source),
        work_dir=tmp_path,
    )
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=prepared.repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        commits = subprocess.run(
            ["git", "rev-list", "--all", "--count"],
            cwd=prepared.repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        assert head == shas[0]
        assert commits == "1"
        assert (prepared.repo / "src/app.py").read_text() == "broken = True\n"
    finally:
        prepared.close()

    assert not prepared.root.exists()


def test_prepare_repository_exposes_only_bounded_base_ancestors(tmp_path) -> None:
    source, shas = _build_source_repo(tmp_path / "source-root")
    cache = tmp_path / "cache"
    _make_bare_mirror(source, cache)
    reference = GitHubRepositoryRef(owner="acme", repo="widget")

    prepared = prepare_github_repository(
        reference,
        shas[1],
        cache_dir=cache,
        clone_url=str(source),
        work_dir=tmp_path,
        history_depth=2,
    )
    try:
        commits = subprocess.run(
            ["git", "rev-list", "--all"],
            cwd=prepared.repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
        assert prepared.history_depth == 2
        assert set(commits) == set(shas)
        assert all(len(commit) == 40 for commit in commits)
        assert (prepared.repo / "src/app.py").read_text() == "broken = False\n"
    finally:
        prepared.close()


def test_local_cache_fetch_honors_history_depth(tmp_path) -> None:
    source, shas = _build_source_repo(tmp_path / "source-root")
    for index in range(3):
        path = source / "src/app.py"
        path.write_text(f"value = {index}\n")
        _git(source, "add", "-A")
        _git(source, "commit", "-q", "-m", f"extra {index}")
        shas.append(_git(source, "rev-parse", "HEAD"))
    cache = tmp_path / "cache"
    _make_bare_mirror(source, cache)
    reference = GitHubRepositoryRef(owner="acme", repo="widget")

    prepared = prepare_github_repository(
        reference,
        shas[-1],
        cache_dir=cache,
        clone_url=str(source),
        work_dir=tmp_path,
        history_depth=2,
    )
    try:
        commits = _git(prepared.repo, "rev-list", "--all")
        assert commits.splitlines() == [shas[-1], shas[-2]]
    finally:
        prepared.close()
