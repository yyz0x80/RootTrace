"""Create a clean, reproducible Git checkout from an evaluation fixture."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

COMMIT_DATE = "2026-01-01T00:00:00Z"
COMMIT_MESSAGE = "fixture: establish evaluation baseline"


def run_git(repo: Path, *args: str, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return result.stdout.strip()


def materialize(source: Path, destination: Path, expected_commit: str) -> str:
    """Copy a fixture, initialize Git deterministically, and verify its commit."""
    source = source.resolve()
    destination = destination.resolve()
    if not source.is_dir():
        raise ValueError(f"fixture does not exist: {source}")
    if destination.exists():
        raise ValueError(f"destination already exists: {destination}")

    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(
            ".git",
            "__pycache__",
            ".pytest_cache",
            ".ruff_cache",
            "*.pyc",
        ),
    )
    run_git(destination, "init", "-q")
    run_git(destination, "config", "user.name", "PatchPilot Evaluation")
    run_git(destination, "config", "user.email", "evaluation@patchpilot.local")
    run_git(destination, "add", ".")

    commit_env = os.environ.copy()
    commit_env.update(
        {
            "GIT_AUTHOR_DATE": COMMIT_DATE,
            "GIT_COMMITTER_DATE": COMMIT_DATE,
        }
    )
    run_git(
        destination,
        "commit",
        "-q",
        "-m",
        COMMIT_MESSAGE,
        "--date",
        COMMIT_DATE,
        env=commit_env,
    )
    actual_commit = run_git(destination, "rev-parse", "HEAD")
    if actual_commit != expected_commit:
        raise RuntimeError(
            "fixture commit mismatch: "
            f"expected {expected_commit}, produced {actual_commit}"
        )
    return actual_commit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    print(materialize(args.source, args.destination, args.expected_commit))


if __name__ == "__main__":
    main()
