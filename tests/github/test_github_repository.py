"""Tests for disposable GitHub repository preparation."""

import subprocess
from pathlib import Path

from roottrace.github import GitHubRepositoryRef, prepare_github_repository


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
