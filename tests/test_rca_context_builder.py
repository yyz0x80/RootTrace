"""Focused tests for the RootTrace deterministic context builder (M2)."""

import json
import subprocess

import pytest
from pydantic import ValidationError

from patchpilot.rca.context import MAX_SNIPPET_CHARS, SourceSnippet
from patchpilot.rca.context_builder import (
    assert_fingerprint_unchanged,
    build_incident_context,
    build_ranked_snippets,
    build_repository_inventory,
    capture_repository_fingerprint,
    extract_signals,
)
from patchpilot.rca.incident_loader import load_incident
from patchpilot.rca.schema import MAX_PROBLEM_CHARS

APP_CODE = """\
import os

def load(config):
    return parse(config["path"])

def parse(path):
    if not os.path.exists(path):
        raise KeyError("missing path key")
    return open(path, encoding="utf-8").read()
"""

UTIL_CODE = """\
def fetch(url):
    return url
"""

TEST_CODE = """\
def test_load_missing_key():
    assert True
"""


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


def make_default_repo(tmp_path) -> str:
    return make_git_repo(
        tmp_path,
        {
            "src/app.py": APP_CODE,
            "src/util.py": UTIL_CODE,
            "tests/test_app.py": TEST_CODE,
            "pyproject.toml": "[project]\nname = 'demo'\n",
            "README.md": "demo repo\n",
            "data.txt": "payload\n",
        },
    )


def make_issue(tmp_path, *, stack: bool = True, diff: bool = True, **overrides):
    problem = (
        "Loading fails with KeyError when config has no path.\n"
        "The load function crashes during startup.\n"
    )
    if stack:
        problem += (
            "Traceback (most recent call last):\n"
            '  File "/usr/src/src/app.py", line 12, in load\n'
            '  File "/usr/src/src/app.py", line 8, in parse\n'
            "KeyError: 'path'\n"
        )
    data = {
        "title": "Crash on load",
        "problem": problem,
        "logs": [],
        "diff": (
            "diff --git a/src/app.py b/src/app.py\n"
            "index 0000000..1111111\n"
            "--- a/src/app.py\n"
            "+++ b/src/app.py\n"
            "@@ -1,3 +1,4 @@\n"
            "+import os\n"
        )
        if diff
        else None,
    }
    data.update(overrides)
    issue = tmp_path / "issue.json"
    issue.write_text(json.dumps(data), encoding="utf-8")
    return issue


def load_default_incident(tmp_path):
    repo = make_default_repo(tmp_path)
    issue = make_issue(tmp_path)
    return repo, load_incident(issue, repo)


def test_context_is_deterministic(tmp_path) -> None:
    repo, loaded = load_default_incident(tmp_path)
    first = build_incident_context(loaded, repo)
    second = build_incident_context(loaded, repo)
    assert first == second
    assert first.model_dump_json() == second.model_dump_json()


def test_inventory_counts_and_sorted_lists(tmp_path) -> None:
    repo, loaded = load_default_incident(tmp_path)
    context = build_incident_context(loaded, repo)
    inventory = context.repository

    assert inventory.tracked_files == 6
    assert inventory.python_files == 3
    assert inventory.test_files == 1
    assert inventory.config_files == 1
    assert inventory.python_file_list == [
        "src/app.py",
        "src/util.py",
        "tests/test_app.py",
    ]
    assert inventory.test_file_list == ["tests/test_app.py"]
    assert inventory.config_file_list == ["pyproject.toml"]
    assert inventory.python_files_omitted == 0


def test_signals_extraction(tmp_path) -> None:
    repo, loaded = load_default_incident(tmp_path)
    context = build_incident_context(loaded, repo)
    signals = context.signals

    assert "KeyError" in signals.exception_names
    assert "load" in signals.stack_symbols
    assert "parse" in signals.stack_symbols
    assert signals.diff_paths == ["src/app.py"]
    assert "keyerror" in signals.terms
    assert "crash" in signals.terms


def test_snippets_are_ranked_bounded_and_repo_relative(tmp_path) -> None:
    repo, loaded = load_default_incident(tmp_path)
    context = build_incident_context(loaded, repo)

    assert context.snippets
    assert all(snippet.path.endswith(".py") for snippet in context.snippets)
    assert [snippet.rank for snippet in context.snippets] == list(
        range(1, len(context.snippets) + 1)
    )
    top = context.snippets[0]
    assert top.path == "src/app.py"
    assert "KeyError" in top.excerpt or "def load" in top.excerpt
    assert len(top.excerpt) <= MAX_SNIPPET_CHARS + 64
    serialized = context.model_dump_json()
    assert str(repo) not in serialized


def test_snippet_cap_truncation_is_recorded(tmp_path) -> None:
    repo, loaded = load_default_incident(tmp_path)
    context = build_incident_context(loaded, repo, max_snippets=1)
    assert len(context.snippets) == 1
    assert context.truncation.snippets_omitted > 0


def test_candidate_cap_truncation_is_recorded(tmp_path) -> None:
    repo, loaded = load_default_incident(tmp_path)
    signals, _counts = extract_signals(loaded.incident)
    inventory = build_repository_inventory(repo, loaded.incident.base_commit)
    snippets, candidates_omitted, _s, _e = build_ranked_snippets(
        repo,
        signals,
        inventory,
        max_candidates=1,
    )
    assert len(snippets) == 1
    assert candidates_omitted > 0


def test_deterministic_tie_break_by_path(tmp_path) -> None:
    repo = make_git_repo(
        tmp_path,
        {
            "src/aaa.py": "widget = 1\n",
            "src/bbb.py": "widget = 2\n",
        },
    )
    issue = tmp_path / "issue.json"
    issue.write_text(
        json.dumps({"title": "widget issue", "problem": "widget broken"}),
        encoding="utf-8",
    )
    loaded = load_incident(issue, repo)
    context = build_incident_context(loaded, repo)

    assert context.snippets[0].path == "src/aaa.py"
    assert context.snippets[1].path == "src/bbb.py"


def test_no_snippets_when_nothing_matches(tmp_path) -> None:
    repo, loaded = load_default_incident(tmp_path)
    issue = tmp_path / "issue.json"
    issue.write_text(
        json.dumps({"title": "zebra", "problem": "zebra unrelated"}),
        encoding="utf-8",
    )
    loaded = load_incident(issue, repo)
    context = build_incident_context(loaded, repo)

    assert context.snippets == []
    assert any(
        "no source snippets matched" in note for note in context.truncation.notes
    )


def test_fingerprint_unchanged_and_detects_modification(tmp_path) -> None:
    repo, loaded = load_default_incident(tmp_path)
    before = capture_repository_fingerprint(repo)
    context = build_incident_context(loaded, repo)
    after = capture_repository_fingerprint(repo)
    assert context.fingerprint == before
    assert_fingerprint_unchanged(before, after)

    with open(repo + "/src/app.py", "a", encoding="utf-8") as stream:
        stream.write("# modified\n")
    modified = capture_repository_fingerprint(repo)
    with pytest.raises(RuntimeError):
        assert_fingerprint_unchanged(before, modified)


def test_inventory_cap_records_omitted(tmp_path) -> None:
    repo = make_git_repo(
        tmp_path,
        {f"mod{i}.py": "x = 1\n" for i in range(3)},
    )
    inventory = build_repository_inventory(
        repo,
        "a" * 40,
        max_python_files=1,
    )
    assert len(inventory.python_file_list) == 1
    assert inventory.python_files_omitted == 2
    assert inventory.python_files == 3


def test_issue_body_truncation_propagates_to_context(tmp_path) -> None:
    repo, _loaded = load_default_incident(tmp_path)
    issue = tmp_path / "issue.json"
    issue.write_text(
        json.dumps({"title": "big", "problem": "x" * (MAX_PROBLEM_CHARS + 50)}),
        encoding="utf-8",
    )
    loaded = load_incident(issue, repo)
    context = build_incident_context(loaded, repo)

    assert context.truncation.issue_body_chars_omitted == 50
    assert "[truncated" in context.incident.problem


def test_terms_cap_records_omitted(tmp_path) -> None:
    repo, _loaded = load_default_incident(tmp_path)
    words = " ".join(f"word{i}" for i in range(15))
    issue = tmp_path / "issue.json"
    issue.write_text(
        json.dumps({"title": "T", "problem": words}),
        encoding="utf-8",
    )
    loaded = load_incident(issue, repo)
    context = build_incident_context(loaded, repo)

    assert len(context.signals.terms) == 10
    assert context.truncation.terms_omitted == 5


def test_diff_excerpt_truncation_is_recorded(tmp_path) -> None:
    repo, _loaded = load_default_incident(tmp_path)
    issue = tmp_path / "issue.json"
    issue.write_text(
        json.dumps(
            {
                "title": "diff",
                "problem": "problem",
                "diff": "d" * 25_000,
            }
        ),
        encoding="utf-8",
    )
    loaded = load_incident(issue, repo)
    context = build_incident_context(loaded, repo)

    assert context.diff_excerpt is not None
    assert "[truncated" in context.diff_excerpt
    assert context.truncation.diff_chars_omitted > 0


def test_snippet_schema_validates_paths_and_lines() -> None:
    with pytest.raises(ValidationError):
        SourceSnippet(
            path="../escape.py",
            start_line=1,
            end_line=2,
            excerpt="x",
            score=1,
            rank=1,
        )
    with pytest.raises(ValidationError):
        SourceSnippet(
            path="src/app.py",
            start_line=5,
            end_line=3,
            excerpt="x",
            score=1,
            rank=1,
        )
