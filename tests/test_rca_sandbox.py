"""Tests for the M3 ephemeral runtime verification sandbox."""

from __future__ import annotations

from pathlib import Path

import pytest

from patchpilot.rca.context_builder import capture_repository_fingerprint
from patchpilot.rca.sandbox import RuntimeVerificationSandbox


def _write(repo: Path, relative_path: str, content: str) -> None:
    path = repo / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_sandbox_runs_allowed_tests_at_head(git_repo, tmp_path: Path) -> None:
    with RuntimeVerificationSandbox(git_repo.repo, work_dir=tmp_path) as sandbox:
        result = sandbox.run(
            ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_calc.py"]
        )
        assert result.exit_code == 0
        assert "2 passed" in result.stdout
        assert sandbox.head_sha == git_repo.head_sha
    assert not sandbox.work_root.exists()


def test_sandbox_reproduces_base_commit_bug(git_repo, tmp_path: Path) -> None:
    with RuntimeVerificationSandbox(
        git_repo.repo,
        base_commit=git_repo.base_sha,
        work_dir=tmp_path,
    ) as sandbox:
        result = sandbox.run(
            ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_calc.py"]
        )
        assert result.exit_code != 0
        assert "1 failed" in result.stdout
        calc = (sandbox.work_root / "pkg" / "calc.py").read_text(encoding="utf-8")
        assert "return a + b  # bug" in calc
        assert sandbox.head_sha == git_repo.base_sha


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["pytest", "tests"],
        ["python"],
        ["python", "-m"],
        ["python", "-m", "pytest", "--reset", "tests"],
        ["python", "-m", "pytest", "tests", "-p", "random"],
        ["python", "-m", "pytest", "/etc/passwd"],
        ["python", "-m", "pytest", "../outside.py"],
        ["python", "-m", "pytest", "tests/test_calc.py; rm -rf /"],
        ["python", "-m", "pytest", "tests/test_calc.py", "$(id)"],
        ["git", "commit", "-m", "hack"],
        ["rm", "-rf", "."],
    ],
)
def test_sandbox_rejects_disallowed_commands(git_repo, tmp_path: Path, argv) -> None:
    with (
        RuntimeVerificationSandbox(git_repo.repo, work_dir=tmp_path) as sandbox,
        pytest.raises(ValueError),
    ):
        sandbox.run(argv)


def test_sandbox_target_validation(git_repo, tmp_path: Path) -> None:
    with RuntimeVerificationSandbox(git_repo.repo, work_dir=tmp_path) as sandbox:
        result = sandbox.run(
            ["python", "-m", "pytest", "-q", "tests/test_calc.py::test_add"]
        )
        assert result.exit_code == 0
        directory = sandbox.run(["python", "-m", "pytest", "-q", "tests"])
        assert directory.exit_code == 0
        with pytest.raises(ValueError):
            sandbox.run(["python", "-m", "pytest", "tests/missing.py"])
        with pytest.raises(ValueError):
            sandbox.run(["python", "-m", "pytest", "README.md"])


def test_sandbox_writes_do_not_propagate(git_repo, tmp_path: Path) -> None:
    before = capture_repository_fingerprint(git_repo.repo)
    with RuntimeVerificationSandbox(git_repo.repo, work_dir=tmp_path) as sandbox:
        _write(
            sandbox.work_root,
            "tests/test_write_marker.py",
            "def test_writes_marker():\n"
            "    from pathlib import Path\n"
            "    Path('marker-from-sandbox.txt').write_text('sandbox')\n"
            "    assert Path('marker-from-sandbox.txt').exists()\n",
        )
        result = sandbox.run(
            ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_write_marker.py"]
        )
        assert result.exit_code == 0
        assert (sandbox.work_root / "marker-from-sandbox.txt").exists()
        assert not (git_repo.repo / "marker-from-sandbox.txt").exists()
    after = capture_repository_fingerprint(git_repo.repo)
    assert before.model_dump(mode="json") == after.model_dump(mode="json")


def test_sandbox_timeout(git_repo, tmp_path: Path) -> None:
    with RuntimeVerificationSandbox(git_repo.repo, work_dir=tmp_path) as sandbox:
        _write(
            sandbox.work_root,
            "tests/test_slow.py",
            "import time\n"
            "def test_slow():\n"
            "    time.sleep(30)\n",
        )
        result = sandbox.run(
            ["python", "-m", "pytest", "-q", "tests/test_slow.py"],
            timeout_seconds=1,
        )
        assert result.timed_out
        assert result.exit_code == 124


def test_sandbox_validation_errors(git_repo, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        RuntimeVerificationSandbox(git_repo.repo, base_commit="--reset", work_dir=tmp_path)
    with pytest.raises(ValueError):
        RuntimeVerificationSandbox(tmp_path / "not-a-git-repo", work_dir=tmp_path)


def test_sandbox_rejects_run_after_close(git_repo, tmp_path: Path) -> None:
    sandbox = RuntimeVerificationSandbox(git_repo.repo, work_dir=tmp_path)
    sandbox.close()
    assert not sandbox.work_root.exists()
    with pytest.raises(RuntimeError):
        sandbox.run(["python", "-m", "pytest", "-q", "tests/test_calc.py"])
