"""Tests for the ``roottrace rca`` CLI and final report rendering."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from roottrace.agents.schema import PlanBudgets
from roottrace.cli import (
    add_analyze_subparser,
    add_rca_subparser,
    run_analyze_command,
    run_rca_command,
    run_rca_pipeline,
)
from roottrace.github import parse_github_resource_url
from roottrace.incident.loader import load_incident
from roottrace.llm.schema import AssistantTurn, ToolCall
from roottrace.reporting.renderer import render_rca_markdown
from roottrace.runtime.workspace import capture_repository_fingerprint


class FakeProvider:
    model = "fake-model"

    def __init__(self, *responses: AssistantTurn) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def complete(self, messages, tools, tool_choice=None) -> AssistantTurn:
        self.calls.append({"messages": messages})
        if not self._responses:
            raise AssertionError("unexpected provider call")
        return self._responses.pop(0)


def turn(content: str) -> AssistantTurn:
    return AssistantTurn(
        content=content,
        tool_calls=[],
        prompt_tokens=10,
        completion_tokens=5,
    )


def tool_turn(name: str, arguments: dict[str, Any]) -> AssistantTurn:
    return AssistantTurn(
        content=None,
        tool_calls=[ToolCall(id=f"call-{name}", name=name, arguments=arguments)],
        prompt_tokens=10,
        completion_tokens=5,
    )


PLAN_JSON = json.dumps(
    {
        "questions": [
            {
                "id": "q-issue_ci-001",
                "text": "what is the failure signature?",
                "assigned_agents": ["issue_ci"],
            },
            {
                "id": "q-code-001",
                "text": "which code path implements multiply?",
                "assigned_agents": ["code"],
            },
            {
                "id": "q-git-001",
                "text": "which change is suspected?",
                "assigned_agents": ["git_history"],
            },
        ]
    }
)

ISSUE_CI_FINAL = json.dumps(
    {
        "status": "completed",
        "ranked_locations": [],
        "evidence_ids": ["ev-issue_ci-001", "ev-issue_ci-002"],
        "uncertainty": "medium",
    }
)

CODE_FINAL = json.dumps(
    {
        "status": "completed",
        "ranked_locations": [{"path": "pkg/calc.py", "symbol": "multiply"}],
        "evidence_ids": ["ev-code-001"],
        "uncertainty": "medium",
    }
)

GIT_FINAL = json.dumps(
    {
        "status": "completed",
        "ranked_locations": [{"path": "pkg/calc.py"}],
        "evidence_ids": ["ev-git_history-001"],
        "uncertainty": "high",
    }
)

HYPOTHESES_JSON = json.dumps(
    {
        "hypotheses": [
            {
                "statement": "multiply is implemented as addition",
                "locations": [{"path": "pkg/calc.py", "symbol": "multiply"}],
                "supporting_evidence_ids": ["ev-issue_ci-001"],
                "contradicting_evidence_ids": [],
                "verification_plan": [
                    {
                        "command": "python -m pytest -q tests/test_calc.py",
                        "description": "run the calc regression tests",
                        "timeout_seconds": 60,
                    }
                ],
                "confidence": "medium",
            }
        ]
    }
)


def _report_json() -> str:
    return json.dumps(
        {
            "conclusion": "root_cause_identified",
            "conclusion_summary": "multiply is implemented as addition",
            "ranked_causes": [
                {
                    "rank": 1,
                    "hypothesis_id": "h-001",
                    "confidence": "medium",
                    "rationale": "code evidence and passing regression tests",
                    "evidence_ids": ["ev-issue_ci-001", "ev-runtime_test-001"],
                }
            ],
            "top_k_locations": [
                {"path": "pkg/calc.py", "symbol": "multiply"}
            ],
            "causal_chain": [
                {
                    "statement": "multiply returns a+b",
                    "hypothesis_id": "h-001",
                    "evidence_ids": ["ev-issue_ci-001"],
                }
            ],
            "suspected_regression": None,
            "fix_recommendation": {
                "scope": "review the multiply function in pkg/calc.py",
                "suggestions": [
                    "verify the arithmetic operation against expected behavior"
                ],
                "locations": [
                    {"path": "pkg/calc.py", "symbol": "multiply"}
                ],
                "evidence_ids": ["ev-issue_ci-001"],
            },
            "uncertainty": {
                "level": "medium",
                "insufficient_evidence": False,
                "notes": [],
            },
        }
    )


def _scripted_factory(
    final_responses: list[str] | None = None,
) -> Callable[[], FakeProvider]:
    providers = [
        FakeProvider(turn(PLAN_JSON), turn(HYPOTHESES_JSON)),
        FakeProvider(turn(ISSUE_CI_FINAL)),
        FakeProvider(
            tool_turn("read_file", {"path": "pkg/calc.py", "raw": True}),
            turn(CODE_FINAL),
        ),
        FakeProvider(
            tool_turn("git_history", {"max_count": 10}),
            turn(GIT_FINAL),
        ),
        FakeProvider(
            *(turn(response) for response in (final_responses or [_report_json()]))
        ),
    ]

    def factory() -> FakeProvider:
        return providers.pop(0)

    return factory


def _issue_files(git_repo, tmp_path: Path) -> tuple[Path, Path]:
    issue = tmp_path / "issue.md"
    issue.write_text(
        "# multiply returns the sum\n\n"
        "When multiplying 3 and 4, the result is 7 instead of 12.\n",
        encoding="utf-8",
    )
    ci_log = tmp_path / "ci.log"
    ci_log.write_text("CI FAILURE: test_multiply failed\n", encoding="utf-8")
    return issue, ci_log


def test_rca_subparser_parses_all_flags() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    add_rca_subparser(subparsers)
    args = parser.parse_args(
        [
            "rca",
            "--repo",
            "/tmp/repo",
            "--issue",
            "/tmp/issue.md",
            "--model",
            "test-model",
            "--output-dir",
            "/tmp/out",
            "--stack-trace",
            "/tmp/stack.txt",
            "--ci-log",
            "/tmp/ci.log",
            "--pr-diff",
            "/tmp/diff.patch",
        ]
    )
    assert args.command == "rca"
    assert args.repo == "/tmp/repo"
    assert args.model == "test-model"
    assert args.stack_trace == "/tmp/stack.txt"
    assert args.ci_log == "/tmp/ci.log"
    assert args.pr_diff == "/tmp/diff.patch"


def test_analyze_subparser_accepts_github_options() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    add_analyze_subparser(subparsers)

    args = parser.parse_args(
        [
            "analyze",
            "https://github.com/acme/widget/issues/7",
            "--model",
            "test-model",
            "--output-dir",
            "/tmp/out",
            "--repo-cache",
            "/tmp/cache",
            "--github-token",
            "token",
            "--github-timeout",
            "12.5",
            "--include-review-comments",
        ]
    )

    assert args.command == "analyze"
    assert args.url.endswith("/issues/7")
    assert args.model == "test-model"
    assert args.output_dir == "/tmp/out"
    assert args.repo_cache == "/tmp/cache"
    assert args.github_token == "token"
    assert args.github_timeout == 12.5
    assert args.include_review_comments is True


def test_analyze_command_routes_prepared_github_input_to_existing_pipeline(
    monkeypatch,
    tmp_path: Path,
) -> None:
    reference = parse_github_resource_url(
        "https://github.com/acme/widget/pull/8"
    )
    fetched = SimpleNamespace(reference=reference)
    normalized = SimpleNamespace(
        base_commit="a" * 40,
        incident=SimpleNamespace(
            id="github-acme-widget-pull_request-8",
            git_verification_policy=SimpleNamespace(history_depth=50),
        ),
        loaded_incident=object(),
        repository_url="https://github.com/acme/widget.git",
    )

    class FakeClient:
        token = "token"

        def __init__(self, **kwargs) -> None:
            assert kwargs["timeout"] == 30.0

    class FakeIngestor:
        def __init__(self, client, *, include_review_comments=False) -> None:
            assert isinstance(client, FakeClient)
            assert include_review_comments is True

        def fetch(self, url):
            assert url == "https://github.com/acme/widget/pull/8"
            return fetched

        def normalize(self, received):
            assert received is fetched
            return normalized

    class FakePrepared:
        repo = tmp_path / "prepared"

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return None

    captured: dict[str, Any] = {}

    def fake_pipeline(loaded, repo, output_dir, *, provider_factory):
        captured.update(
            loaded=loaded,
            repo=repo,
            output_dir=output_dir,
            provider_factory=provider_factory,
        )
        return SimpleNamespace(
            incident_id="github-acme-widget-pull_request-8",
            run=SimpleNamespace(status=SimpleNamespace(value="completed")),
            output_dir=str(output_dir),
        )

    monkeypatch.setattr("roottrace.cli.GitHubClient", FakeClient)
    monkeypatch.setattr("roottrace.cli.GitHubIngestor", FakeIngestor)
    prepared_arguments: dict[str, Any] = {}

    def fake_prepare(*args, **kwargs):
        prepared_arguments.update(kwargs)
        return FakePrepared()

    monkeypatch.setattr("roottrace.cli.prepare_github_repository", fake_prepare)
    monkeypatch.setattr("roottrace.cli.run_rca_pipeline", fake_pipeline)

    result = run_analyze_command(
        argparse.Namespace(
            url="https://github.com/acme/widget/pull/8",
            github_token="token",
            github_timeout=30.0,
            repo_cache=str(tmp_path / "cache"),
            output_dir=None,
            model=None,
            include_review_comments=True,
        )
    )

    assert result == 0
    assert captured["loaded"] is normalized.loaded_incident
    assert captured["repo"] == FakePrepared.repo
    assert captured["output_dir"] == Path("output") / normalized.incident.id
    assert prepared_arguments["history_depth"] == 50


def test_run_rca_pipeline_end_to_end(git_repo, tmp_path: Path) -> None:
    before = capture_repository_fingerprint(git_repo.repo)
    issue, ci_log = _issue_files(git_repo, tmp_path)
    loaded = load_incident(issue, git_repo.repo, ci_log_path=ci_log)
    output_dir = tmp_path / "out"
    factory = _scripted_factory()
    result = run_rca_pipeline(
        loaded,
        git_repo.repo,
        output_dir,
        provider_factory=factory,
        budgets=PlanBudgets(),
        log_sources={"ci.log": ci_log},
    )

    expected_artifacts = {
        "incident.json",
        "investigation_plan.json",
        "agents/issue_ci.json",
        "agents/code.json",
        "agents/git_history.json",
        "evidence_graph.json",
        "hypotheses.json",
        "verification.json",
        "rca_report.json",
        "rca_report.md",
        "execution_trace.jsonl",
        "run_summary.json",
    }
    assert expected_artifacts <= set(result.artifacts)
    for name in expected_artifacts:
        assert (output_dir / name).is_file()

    graph = json.loads(
        (output_dir / "evidence_graph.json").read_text(encoding="utf-8")
    )
    assert graph["hypotheses"][0]["disposition"] == "supported"
    assert any(
        item["kind"] == "test_result" for item in graph["evidence"]
    )
    verification = json.loads(
        (output_dir / "verification.json").read_text(encoding="utf-8")
    )
    assert verification["results"][0]["outcome"] == "supported"
    assert verification["results"][0]["status"] == "passed"

    report = json.loads(
        (output_dir / "rca_report.json").read_text(encoding="utf-8")
    )
    assert report["conclusion"] == "root_cause_identified"
    markdown = (output_dir / "rca_report.md").read_text(encoding="utf-8")
    assert markdown.startswith("# RootTrace RCA Report")
    assert "## Verification" in markdown
    assert "## Fix recommendation (advisory only)" in markdown

    summary = json.loads(
        (output_dir / "run_summary.json").read_text(encoding="utf-8")
    )
    assert summary["incident_id"] == loaded.incident.id
    assert summary["conclusion"] == "root_cause_identified"
    assert summary["resource_kind"] == "issue"
    assert summary["git_verification_policy"]["enabled"] is False
    assert summary["git_verification_policy"]["history_depth"] == 1

    after = capture_repository_fingerprint(git_repo.repo)
    assert before.model_dump(mode="json") == after.model_dump(mode="json")


def test_run_rca_pipeline_persists_synthesis_degradation_diagnostic(
    git_repo,
    tmp_path: Path,
) -> None:
    issue, ci_log = _issue_files(git_repo, tmp_path)
    loaded = load_incident(issue, git_repo.repo, ci_log_path=ci_log)
    payload = json.loads(_report_json())
    payload["suspected_regression"] = {
        "commit": git_repo.base_sha,
        "summary": "the base commit may contain the regression",
        "locations": [{"path": "pkg/calc.py", "symbol": "multiply"}],
    }
    invalid_response = json.dumps(payload)
    output_dir = tmp_path / "degraded-out"

    result = run_rca_pipeline(
        loaded,
        git_repo.repo,
        output_dir,
        provider_factory=_scripted_factory(
            [invalid_response, invalid_response]
        ),
        budgets=PlanBudgets(),
        log_sources={"ci.log": ci_log},
    )

    assert result.report.suspected_regression is None
    assert [diagnostic.code for diagnostic in result.run.diagnostics] == [
        "synthesis.suspected_regression_removed"
    ]
    summary = json.loads(
        (output_dir / "run_summary.json").read_text(encoding="utf-8")
    )
    assert summary["diagnostics"][0]["severity"] == "recoverable"
    assert summary["errors"] == [
        "Invalid suspected_regression was removed after synthesis repair failed"
    ]


def test_run_rca_command_with_scripted_providers(
    git_repo,
    tmp_path: Path,
    monkeypatch,
) -> None:
    issue, ci_log = _issue_files(git_repo, tmp_path)
    output_dir = tmp_path / "cli-out"
    factory = _scripted_factory()
    monkeypatch.setattr(
        "roottrace.cli.create_provider_from_config",
        lambda model_name=None: factory(),
    )
    parser = argparse.ArgumentParser()
    add_rca_subparser(parser.add_subparsers(dest="command"))
    args = parser.parse_args(
        [
            "rca",
            "--repo",
            str(git_repo.repo),
            "--issue",
            str(issue),
            "--model",
            "fake-model",
            "--output-dir",
            str(output_dir),
            "--ci-log",
            str(ci_log),
        ]
    )
    assert run_rca_command(args) == 0
    assert (output_dir / "rca_report.md").is_file()


def test_run_rca_command_reports_input_error(git_repo, tmp_path: Path) -> None:
    parser = argparse.ArgumentParser()
    add_rca_subparser(parser.add_subparsers(dest="command"))
    args = parser.parse_args(
        [
            "rca",
            "--repo",
            str(git_repo.repo),
            "--issue",
            str(tmp_path / "missing.md"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert run_rca_command(args) == 2


def test_renderer_is_deterministic(git_repo, tmp_path: Path) -> None:
    issue, ci_log = _issue_files(git_repo, tmp_path)
    loaded = load_incident(issue, git_repo.repo, ci_log_path=ci_log)
    result = run_rca_pipeline(
        loaded,
        git_repo.repo,
        tmp_path / "render-out",
        provider_factory=_scripted_factory(),
    )
    first = render_rca_markdown(result.report)
    second = render_rca_markdown(result.report)
    assert first == second
    assert "## Ranked causes" in first
    assert "## Evidence" in first
