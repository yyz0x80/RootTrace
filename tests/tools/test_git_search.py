"""Tests for deterministic layered Git history search."""

from __future__ import annotations

import re

from roottrace.tools.git_search import (
    GitSearchCommandResult,
    GitSearchExecutor,
    GitSearchPlan,
    GitSearchQuery,
)


def _sha(index: int) -> str:
    return f"{index:040x}"


class FakeGitRunner:
    """Return deterministic Git-shaped output for a synthetic history."""

    def __init__(
        self,
        commits: list[str],
        *,
        paths: dict[str, set[str]] | None = None,
        contents: dict[str, str] | None = None,
        subjects: dict[str, str] | None = None,
    ) -> None:
        self.commits = commits
        self.paths = paths or {}
        self.contents = contents or {}
        self.subjects = subjects or {}
        self.commands: list[list[str]] = []

    def __call__(self, argv: list[str]) -> GitSearchCommandResult:
        self.commands.append(argv)
        output = ""
        if argv[1] == "rev-list":
            max_count = next(
                int(value.split("=", maxsplit=1)[1])
                for value in argv
                if value.startswith("--max-count=")
            )
            output = "\n".join(self.commits[:max_count])
        elif "--name-only" in argv:
            selected = [value for value in argv if value in self.commits]
            lines: list[str] = []
            for commit in selected:
                lines.append(
                    f"ROOTTRACE_COMMIT:{commit}\t"
                    f"{self.subjects.get(commit, f'commit {commit[-4:]}')}"
                )
                lines.extend(sorted(self.paths.get(commit, set())))
            output = "\n".join(lines)
        else:
            pattern = next(value[2:] for value in argv if value.startswith("-G"))
            selected = [value for value in argv if value in self.commits]
            output = "\n".join(
                commit
                for commit in selected
                if re.search(pattern, self.contents.get(commit, ""))
            )
        return GitSearchCommandResult(
            ok=True,
            output=output,
            command=" ".join(argv),
            duration_seconds=0.001,
        )


def _query(
    *,
    kind: str,
    signal: str,
    value: str,
    weight: int,
) -> GitSearchQuery:
    return GitSearchQuery(
        kind=kind,
        signal=signal,
        source="test",
        value=value,
        weight=weight,
    )


def test_explicit_commit_stops_at_the_first_layer_that_contains_it() -> None:
    commits = [_sha(index) for index in range(1, 51)]
    runner = FakeGitRunner(commits)
    opened_depths: list[int] = []
    summary = GitSearchExecutor(
        base_commit=commits[0],
        run_command=runner,
        set_visible_depth=opened_depths.append,
    ).run(
        GitSearchPlan(
            enabled=True,
            max_depth=50,
            search_depths=[8, 16, 32, 50],
            candidate_commits=[commits[11]],
        )
    )

    assert summary.attempted_depths == [8, 16]
    assert summary.stop_reason == "explicit_commit_verified"
    assert summary.candidate_commits == [commits[11]]
    assert summary.candidates[0].strong_match is True
    assert opened_depths == [8, 16]


def test_path_only_match_is_retained_but_does_not_stop_search() -> None:
    commits = [_sha(index) for index in range(1, 13)]
    runner = FakeGitRunner(commits, paths={commits[2]: {"src/service.py"}})
    summary = GitSearchExecutor(
        base_commit=commits[0],
        run_command=runner,
    ).run(
        GitSearchPlan(
            enabled=True,
            max_depth=50,
            search_depths=[8, 16, 32, 50],
            queries=[
                _query(
                    kind="path",
                    signal="path",
                    value="src/service.py",
                    weight=4,
                )
            ],
        )
    )

    assert summary.attempted_depths == [8, 16]
    assert summary.stop_reason == "history_exhausted"
    assert summary.reached_depth == 12
    assert summary.candidate_commits == [commits[2]]
    assert summary.candidates[0].strong_match is False


def test_path_and_content_match_stops_at_the_first_layer() -> None:
    commits = [_sha(index) for index in range(1, 51)]
    target = commits[3]
    runner = FakeGitRunner(
        commits,
        paths={target: {"src/service.py"}},
        contents={target: "def load_config():"},
    )
    summary = GitSearchExecutor(
        base_commit=commits[0],
        run_command=runner,
    ).run(
        GitSearchPlan(
            enabled=True,
            max_depth=50,
            search_depths=[8, 16, 32, 50],
            queries=[
                _query(
                    kind="path",
                    signal="path",
                    value="src/service.py",
                    weight=4,
                ),
                _query(
                    kind="content",
                    signal="symbol",
                    value="load_config",
                    weight=3,
                ),
            ],
        )
    )

    assert summary.attempted_depths == [8]
    assert summary.stop_reason == "path_and_content_match"
    assert summary.candidates[0].strong_match is True
    assert summary.candidates[0].score == 7
    assert summary.candidates[0].matched_signals.count("pickaxe:symbol") == 1


def test_combined_pickaxe_query_does_not_claim_every_signal_value() -> None:
    commits = [_sha(index) for index in range(1, 9)]
    target = commits[1]
    runner = FakeGitRunner(commits, contents={target: "def selected_symbol():"})
    summary = GitSearchExecutor(
        base_commit=commits[0],
        run_command=runner,
    ).run(
        GitSearchPlan(
            enabled=True,
            max_depth=8,
            search_depths=[8],
            queries=[
                _query(
                    kind="content",
                    signal="symbol",
                    value="selected_symbol",
                    weight=3,
                ),
                _query(
                    kind="content",
                    signal="symbol",
                    value="unrelated_symbol",
                    weight=3,
                ),
            ],
        )
    )

    candidate = summary.candidates[0]
    assert candidate.score == 3
    assert candidate.matched_signals == ["pickaxe:symbol"]
    assert candidate.strong_match is False


def test_search_reaches_the_hard_limit_without_inventing_candidates() -> None:
    commits = [_sha(index) for index in range(1, 51)]
    runner = FakeGitRunner(commits)
    summary = GitSearchExecutor(
        base_commit=commits[0],
        run_command=runner,
    ).run(
        GitSearchPlan(
            enabled=True,
            max_depth=50,
            search_depths=[8, 16, 32, 50],
        )
    )

    assert summary.attempted_depths == [8, 16, 32, 50]
    assert summary.reached_depth == 50
    assert summary.stop_reason == "max_depth_reached"
    assert summary.candidates == []
    assert summary.commands_executed == 8
