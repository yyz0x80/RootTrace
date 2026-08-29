"""Focused tests for the RootTrace incident loader."""

import json
import subprocess

import pytest

from roottrace.incident.loader import load_incident, resolve_base_commit
from roottrace.incident.schema import MAX_DIFF_CHARS, MAX_LOG_CHARS, MAX_PROBLEM_CHARS


def make_git_repo(tmp_path, files: dict[str, str]) -> str:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    for rel, content in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "init"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    return str(repo)


def head_sha(repo: str) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def test_load_markdown_incident(tmp_path) -> None:
    repo = make_git_repo(tmp_path, {"src/app.py": "def load():\n    pass\n"})
    issue = tmp_path / "issue.md"
    issue.write_text(
        "# Crash on load\n\nApp crashes when config has no path.\n",
        encoding="utf-8",
    )

    loaded = load_incident(issue, repo)
    incident = loaded.incident

    assert incident.title == "Crash on load"
    assert "config has no path" in incident.problem
    assert incident.base_commit == head_sha(repo)
    assert incident.id == f"inc-{head_sha(repo)[:12]}"
    assert incident.repo == "repo"
    assert incident.provenance.tool == "incident_loader"
    assert incident.provenance.source == "issue.md"
    assert incident.resource_kind == "issue"
    assert incident.git_verification_policy.enabled is False
    assert incident.git_verification_policy.history_depth == 1
    assert loaded.issue_chars_omitted == 0
    assert not any("truncated" in note for note in loaded.notes)


def test_load_json_incident(tmp_path) -> None:
    repo = make_git_repo(tmp_path, {"src/app.py": "def load():\n    pass\n"})
    issue = tmp_path / "issue.json"
    issue.write_text(
        json.dumps(
            {
                "id": "inc-custom",
                "title": "JSON crash",
                "problem": "Loading fails.",
                "logs": ["Traceback ..."],
                "diff": "diff --git a/src/app.py b/src/app.py",
            }
        ),
        encoding="utf-8",
    )

    loaded = load_incident(issue, repo, repo_identifier="owner/repo")
    incident = loaded.incident

    assert incident.id == "inc-custom"
    assert incident.title == "JSON crash"
    assert incident.problem == "Loading fails."
    assert incident.logs == ["Traceback ..."]
    assert incident.diff == "diff --git a/src/app.py b/src/app.py"
    assert incident.repo == "owner/repo"
    assert incident.base_commit == head_sha(repo)
    assert incident.resource_kind == "pull_request"
    assert incident.changed_files == ["src/app.py"]
    assert incident.git_verification_policy.enabled is True
    assert "pull_request" in incident.git_verification_policy.reasons


def test_explicit_base_commit_is_used(tmp_path) -> None:
    repo = make_git_repo(tmp_path, {"a.py": "x = 1\n"})
    issue = tmp_path / "issue.md"
    issue.write_text("# Problem\n\nbody\n", encoding="utf-8")
    sha = head_sha(repo)

    loaded = load_incident(issue, repo, base_commit=sha)
    assert loaded.incident.base_commit == sha
    assert loaded.notes == []

    resolved, note = resolve_base_commit(repo, sha[:12])
    assert resolved == sha
    assert note is None


def test_base_commit_falls_back_to_head(tmp_path) -> None:
    repo = make_git_repo(tmp_path, {"a.py": "x = 1\n"})
    issue = tmp_path / "issue.md"
    issue.write_text("# Problem\n\nbody\n", encoding="utf-8")

    loaded = load_incident(issue, repo)
    assert loaded.incident.base_commit == head_sha(repo)
    assert any("fell back" in note for note in loaded.notes)


def test_invalid_base_commit_rejected(tmp_path) -> None:
    repo = make_git_repo(tmp_path, {"a.py": "x = 1\n"})
    issue = tmp_path / "issue.md"
    issue.write_text("# Problem\n\nbody\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_incident(issue, repo, base_commit="bogus-sha")
    with pytest.raises(ValueError):
        resolve_base_commit(repo, "bogus-sha")


def test_missing_issue_file_rejected(tmp_path) -> None:
    repo = make_git_repo(tmp_path, {"a.py": "x = 1\n"})
    with pytest.raises(FileNotFoundError):
        load_incident(tmp_path / "missing.md", repo)


def test_stack_trace_and_ci_log_are_appended(tmp_path) -> None:
    repo = make_git_repo(tmp_path, {"src/app.py": "def load():\n    pass\n"})
    issue = tmp_path / "issue.md"
    issue.write_text("# Crash\n\nbody\n", encoding="utf-8")
    stack = tmp_path / "stack.txt"
    stack.write_text(
        'File "src/app.py", line 2, in load\nKeyError: "path"\n',
        encoding="utf-8",
    )
    ci_log = tmp_path / "ci.log"
    ci_log.write_text("FAILED tests/test_app.py::test_load\n", encoding="utf-8")

    loaded = load_incident(
        issue,
        repo,
        stack_trace_path=stack,
        ci_log_path=ci_log,
    )
    assert loaded.incident.logs == [
        'File "src/app.py", line 2, in load\nKeyError: "path"\n',
        "FAILED tests/test_app.py::test_load\n",
    ]
    assert loaded.stack_trace_chars_omitted == 0
    assert loaded.ci_log_chars_omitted == 0


def test_pr_diff_file_wins_over_json_diff(tmp_path) -> None:
    repo = make_git_repo(tmp_path, {"a.py": "x = 1\n"})
    issue = tmp_path / "issue.json"
    issue.write_text(
        json.dumps({"title": "t", "problem": "p", "diff": "old diff"}),
        encoding="utf-8",
    )
    diff_file = tmp_path / "change.diff"
    diff_file.write_text("diff --git a/a.py b/a.py\n", encoding="utf-8")

    loaded = load_incident(issue, repo, pr_diff_path=diff_file)
    assert loaded.incident.diff == "diff --git a/a.py b/a.py\n"
    assert loaded.incident.resource_kind == "pull_request"
    assert loaded.incident.git_verification_policy.enabled is True


def test_json_issue_regression_metadata_enables_history_without_diff(tmp_path) -> None:
    repo = make_git_repo(tmp_path, {"a.py": "x = 1\n"})
    issue = tmp_path / "issue.json"
    issue.write_text(
        json.dumps(
            {
                "title": "Crash after regression",
                "problem": "The behavior changed after commit " + "b" * 40,
                "labels": ["bug"],
            }
        ),
        encoding="utf-8",
    )

    loaded = load_incident(issue, repo)

    assert loaded.incident.resource_kind == "issue"
    assert loaded.incident.git_verification_policy.enabled is True
    assert loaded.incident.git_verification_policy.candidate_commits == [
        "b" * 40
    ]


def test_issue_body_truncation_is_recorded(tmp_path) -> None:
    repo = make_git_repo(tmp_path, {"a.py": "x = 1\n"})
    issue = tmp_path / "issue.md"
    body = "# Big issue\n\n" + "x" * (MAX_PROBLEM_CHARS + 100)
    issue.write_text(body, encoding="utf-8")

    loaded = load_incident(issue, repo)
    expected = len(body) - MAX_PROBLEM_CHARS
    assert loaded.issue_chars_omitted == expected
    assert "[truncated" in loaded.incident.problem
    assert any("issue body truncated" in note for note in loaded.notes)


def test_stack_trace_truncation_is_recorded(tmp_path) -> None:
    repo = make_git_repo(tmp_path, {"a.py": "x = 1\n"})
    issue = tmp_path / "issue.md"
    issue.write_text("# Crash\n\nbody\n", encoding="utf-8")
    stack = tmp_path / "stack.txt"
    stack.write_text("x" * (MAX_LOG_CHARS + 5), encoding="utf-8")

    loaded = load_incident(issue, repo, stack_trace_path=stack)
    assert loaded.stack_trace_chars_omitted == 5
    assert "[truncated" in loaded.incident.logs[0]


def test_diff_truncation_is_recorded(tmp_path) -> None:
    repo = make_git_repo(tmp_path, {"a.py": "x = 1\n"})
    issue = tmp_path / "issue.json"
    issue.write_text(
        json.dumps({"title": "t", "problem": "p", "diff": "d" * (MAX_DIFF_CHARS + 3)}),
        encoding="utf-8",
    )

    loaded = load_incident(issue, repo)
    assert loaded.diff_chars_omitted == 3
    assert "[truncated" in loaded.incident.diff


def test_json_issue_without_problem_rejected(tmp_path) -> None:
    repo = make_git_repo(tmp_path, {"a.py": "x = 1\n"})
    issue = tmp_path / "issue.json"
    issue.write_text(json.dumps({"title": "only title"}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_incident(issue, repo)


def test_invalid_json_rejected(tmp_path) -> None:
    repo = make_git_repo(tmp_path, {"a.py": "x = 1\n"})
    issue = tmp_path / "issue.json"
    issue.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        load_incident(issue, repo)


def test_issue_inside_repo_uses_relative_provenance(tmp_path) -> None:
    repo = make_git_repo(tmp_path, {"a.py": "x = 1\n"})
    issue = tmp_path / "repo" / "issues" / "crash.md"
    issue.parent.mkdir(parents=True, exist_ok=True)
    issue.write_text("# Crash\n\nbody\n", encoding="utf-8")

    loaded = load_incident(issue, repo)
    assert loaded.incident.provenance.source == "issues/crash.md"


def test_load_incident_does_not_modify_target(tmp_path) -> None:
    repo = make_git_repo(tmp_path, {"a.py": "x = 1\n"})
    issue = tmp_path / "issue.md"
    issue.write_text("# Crash\n\nbody\n", encoding="utf-8")
    before = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    load_incident(issue, repo)

    after = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert before == after == ""
