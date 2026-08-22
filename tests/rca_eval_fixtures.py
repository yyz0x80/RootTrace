"""Shared helpers for RCA evaluation pipeline tests (no pytest dependency)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return result.stdout.strip()


def build_source_repo(
    root: Path,
    commits: list[tuple[str, dict[str, str]]],
) -> list[str]:
    """Create a git repo with one commit per ``(message, files)`` entry."""
    repo = root / "source"
    repo.mkdir(parents=True)
    git(repo, "init", "-q")
    git(repo, "config", "user.name", "Test Author")
    git(repo, "config", "user.email", "test@example.com")
    shas: list[str] = []
    for message, files in commits:
        for relative, content in files.items():
            path = repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", message)
        shas.append(git(repo, "rev-parse", "HEAD"))
    return shas


def make_bare_mirror(
    source: Path,
    cache_root: Path,
    repo_id: str,
) -> Path:
    """Create a bare mirror ``owner__name.git`` for ``owner/name``."""
    owner, name = repo_id.split("/")
    destination = cache_root / f"{owner}__{name}.git"
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--quiet", "--bare", str(source), str(destination)],
        check=True,
        capture_output=True,
        timeout=120,
    )
    return destination


def make_worktree_mirror(
    source: Path,
    cache_root: Path,
    repo_id: str,
    *,
    backup: bool = False,
) -> Path:
    """Copy a repo as a plain-checkout mirror fallback."""
    owner, name = repo_id.split("/")
    suffix = "_worktree_backup" if backup else ""
    destination = cache_root / f"{owner}__{name}{suffix}"
    shutil.copytree(source, destination)
    return destination


def public_record(
    instance_id: str,
    repo: str,
    base_commit: str,
    problem: str,
    **extra: object,
) -> dict:
    record: dict = {
        "instance_id": instance_id,
        "repo": repo,
        "base_commit": base_commit,
        "problem_statement": problem,
    }
    record.update(extra)
    return record


def gold_record(
    instance_id: str,
    patch: str,
    test_patch: str = "",
    fail_to_pass: list[str] | None = None,
    pass_to_pass: list[str] | None = None,
) -> dict:
    return {
        "instance_id": instance_id,
        "patch": patch,
        "test_patch": test_patch,
        "FAIL_TO_PASS": fail_to_pass or [],
        "PASS_TO_PASS": pass_to_pass or [],
    }


def gold_patch(source_files: list[str], test_files: list[str] | None = None) -> str:
    """Build a simple diff touching the given repository-relative files."""
    hunks: list[str] = []
    for path in [*source_files, *(test_files or [])]:
        hunks.append(
            f"diff --git a/{path} b/{path}\n"
            f"--- a/{path}\n"
            f"+++ b/{path}\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
        )
    return "\n".join(hunks)


def build_data_root(
    root: Path,
    *,
    cases: list[dict],
    gold_patches: dict[str, str],
    source_repo: Path,
    repo_id: str,
    manifest_name: str = "smoke3",
    seed: int = 42,
) -> Path:
    """Build a SWE-bench-style data root for runner tests."""
    data_root = root / "swebench"
    public = [
        public_record(
            case["instance_id"],
            case["repo"],
            case["base_commit"],
            f"Problem for {case['instance_id']}",
        )
        for case in cases
    ]
    write_jsonl(data_root / "public" / "verified_public.jsonl", public)
    gold = [
        gold_record(case_id, patch)
        for case_id, patch in sorted(gold_patches.items())
    ]
    write_jsonl(data_root / "gold" / "verified_gold.jsonl", gold)
    write_json(
        data_root / "manifests" / f"{manifest_name}.json",
        {
            "name": manifest_name,
            "seed": seed,
            "instances": [
                {
                    "instance_id": case["instance_id"],
                    "repo": case["repo"],
                    "base_commit": case["base_commit"],
                }
                for case in cases
            ],
        },
    )
    make_bare_mirror(source_repo, data_root / "repos", repo_id)
    return data_root
