"""Focused, network-free tests for the online GitHub smoke evaluator."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from evaluation.github import runner as smoke_runner
from evaluation.github.manifest import GitHubSmokeCase, load_manifest, select_cases
from evaluation.runner import RootTraceOutcome
from roottrace.github import (
    GitHubFetchedResource,
    GitHubIngestor,
    GitHubIssueDetail,
    GitHubPullRequestDetail,
    GitHubPullRequestFile,
    parse_github_resource_url,
)


class FakeIngestor:
    def __init__(self, resources: dict[str, GitHubFetchedResource]) -> None:
        self.resources = resources
        self._normalizer = GitHubIngestor(None)

    def fetch(self, url: str) -> GitHubFetchedResource:
        return self.resources[url]

    def normalize(self, fetched, *, base_commit: str | None = None):
        return self._normalizer.normalize(fetched, base_commit=base_commit)


class FakePrepared:
    def __init__(self, repo: Path, revision: str) -> None:
        self.repo = repo
        self.revision = revision
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeRootTrace:
    def __init__(self, prediction: str = "src/app.py") -> None:
        self.prediction = prediction
        self.received = []

    def run(self, *, case_id, repo, incident, output_dir, model):
        del case_id, repo, model
        self.received.append(incident.model_copy(deep=True))
        output_dir.mkdir(parents=True, exist_ok=True)
        report = {"top_k_locations": [{"path": self.prediction}]}
        (output_dir / "rca_report.json").write_text(json.dumps(report), encoding="utf-8")
        return RootTraceOutcome(
            status="completed",
            report=report,
            latency_seconds=0.25,
            llm_calls=2,
            prompt_tokens=100,
            completion_tokens=25,
        )


def _issue_resource(url: str, commit: str, body: str = "A reproducible failure"):
    reference = parse_github_resource_url(url)
    return GitHubFetchedResource(
        reference=reference,
        detail=GitHubIssueDetail(
            number=reference.number,
            title="Regression",
            body=body,
        ),
    )


def _pr_resource(url: str, base: str, head: str, merge: str):
    reference = parse_github_resource_url(url)
    return GitHubFetchedResource(
        reference=reference,
        detail=GitHubPullRequestDetail(
            number=reference.number,
            title="Bad change",
            body="The change introduced the regression.",
            base={"sha": base},
            head={"sha": head},
            merge_commit_sha=merge,
        ),
        files=[GitHubPullRequestFile(filename="src/app.py", patch="@@ -1 +1 @@\n-old\n+new")],
    )


def _prepare_factory(calls: list[dict], repo: Path, revision: str):
    def prepare(reference, target, *, cache_dir, clone_url, token, history_depth):
        calls.append(
            {
                "reference": reference,
                "revision": target,
                "cache_dir": Path(cache_dir),
                "clone_url": clone_url,
                "token": token,
                "history_depth": history_depth,
            }
        )
        return FakePrepared(repo, revision)

    return prepare


def test_checked_in_manifest_has_fixed_composition_and_selection() -> None:
    manifest = load_manifest()
    assert manifest.name == "github_smoke10"
    assert len(manifest.instances) == 10
    assert sum(case.source_type == "github_issue" for case in manifest.instances) == 8
    assert sum(case.source_type == "github_pr" for case in manifest.instances) == 2
    selected = select_cases(manifest, "pytest-dev__pytest-5262")
    assert len(selected) == 1
    assert selected[0].source_type == "github_issue"


def test_issue_run_uses_canonical_online_clone_and_existing_metrics(tmp_path) -> None:
    commit = "a" * 40
    url = "https://github.com/acme/demo/issues/7"
    case = GitHubSmokeCase(
        instance_id="acme__demo-7",
        source_type="github_issue",
        source_url=url,
        repo="acme/demo",
        base_commit=commit,
        gold_files=["src/app.py"],
    )
    fetched = _issue_resource(url, commit)
    ingestor = FakeIngestor({url: fetched})
    calls: list[dict] = []
    fake_client = FakeRootTrace()
    result = smoke_runner.run_case(
        case,
        ingestor=ingestor,
        roottrace_client=fake_client,
        cache_dir=tmp_path / "github-smoke-cache",
        case_dir=tmp_path / "case",
        model="fake-model",
        prepare_fn=_prepare_factory(calls, tmp_path, commit),
    )
    assert result.status == "completed"
    assert result.predicted_top5 == ["src/app.py"]
    assert result.metrics is not None and result.metrics.top_1_file_accuracy is True
    assert result.repo_acquisition_success is True
    assert result.checkout_success is True
    assert result.final_checkout_commit == commit
    assert calls[0]["clone_url"] == "https://github.com/acme/demo.git"
    assert calls[0]["revision"] == commit
    assert calls[0]["token"] is None
    assert fake_client.received[0].problem == "A reproducible failure"


def test_pr_context_combines_regression_issue_and_bad_pr_without_fix_leakage(tmp_path) -> None:
    merge = "c" * 40
    base = "a" * 40
    head = "b" * 40
    pr_url = "https://github.com/acme/demo/pull/8"
    issue_url = "https://github.com/acme/demo/issues/9"
    fix_url = "https://github.com/acme/demo/pull/10"
    case = GitHubSmokeCase(
        instance_id="acme__demo-pr8-regression9",
        source_type="github_pr",
        source_url=pr_url,
        regression_issue_url=issue_url,
        repo="acme/demo",
        base_commit=merge,
        expected_files=["src/app.py"],
        fix_evidence_url=fix_url,
        manual_review_required=True,
    )
    resources = {
        pr_url: _pr_resource(pr_url, base, head, merge),
        issue_url: _issue_resource(
            issue_url,
            merge,
            f"Regression details; fix: {fix_url}, #10, or acme/demo#10",
        ),
    }
    fake_client = FakeRootTrace()
    calls: list[dict] = []
    result = smoke_runner.run_case(
        case,
        ingestor=FakeIngestor(resources),
        roottrace_client=fake_client,
        cache_dir=tmp_path / "github-smoke-cache",
        case_dir=tmp_path / "case",
        model=None,
        prepare_fn=_prepare_factory(calls, tmp_path, merge),
    )
    assert result.status == "manual_review_required"
    assert result.metrics is None
    incident = fake_client.received[0]
    assert incident.base_commit == merge
    assert incident.diff and "src/app.py" in incident.diff
    assert fix_url not in incident.problem
    assert "#10" not in incident.problem
    assert "acme/demo#10" not in incident.problem
    assert fix_url not in "\n".join(incident.logs)
    assert "src/app.py" not in incident.problem
    assert result.expected_root_cause_files == ["src/app.py"]
    assert result.gold_files == []
    assert calls[0]["revision"] == merge


def test_preflight_runs_without_roottrace_client_or_provider(tmp_path, monkeypatch) -> None:
    manifest = load_manifest()
    case = GitHubSmokeCase(
        instance_id="acme__demo-7",
        source_type="github_issue",
        source_url="https://github.com/acme/demo/issues/7",
        repo="acme/demo",
        base_commit="a" * 40,
        gold_files=["src/app.py"],
    )
    fetched = _issue_resource(case.source_url, case.base_commit)
    fake_ingestor = FakeIngestor({case.source_url: fetched})
    monkeypatch.setattr(smoke_runner, "GitHubIngestor", lambda _client: fake_ingestor)
    calls: list[dict] = []
    report = smoke_runner.run_preflight(
        manifest,
        selected=[case],
        github_client=object(),
        cache_dir=tmp_path / "cache",
        token=None,
        prepare_fn=_prepare_factory(calls, tmp_path, case.base_commit),
    )
    assert report.ok is True
    assert report.token_configured is False
    assert any(check.warning for check in report.checks)
    assert calls and calls[0]["clone_url"] == "https://github.com/acme/demo.git"
    assert calls[0]["token"] is None


def test_checkout_failure_preserves_successful_acquisition_status(
    tmp_path, monkeypatch
) -> None:
    commit = "a" * 40
    url = "https://github.com/acme/demo/issues/7"
    case = GitHubSmokeCase(
        instance_id="acme__demo-7",
        source_type="github_issue",
        source_url=url,
        repo="acme/demo",
        base_commit=commit,
        gold_files=["src/app.py"],
    )
    context = smoke_runner.build_case_context(
        case,
        FakeIngestor({url: _issue_resource(url, commit)}),
    )
    mirror = tmp_path / "cache" / "acme__demo.git"
    mirror.mkdir(parents=True)
    monkeypatch.setattr(
        smoke_runner,
        "_validate_existing_cache_origin",
        lambda _mirror, _canonical: None,
    )

    def fail_checkout(*_args, **_kwargs):
        raise RuntimeError("repository mirror does not contain revision")

    prepared = smoke_runner.prepare_case_repository(
        case,
        context,
        cache_dir=tmp_path / "cache",
        prepare_fn=fail_checkout,
    )
    assert prepared.acquisition.success is True
    assert prepared.checkout.success is False


def test_external_dev50_and_gold_labels_are_validated(tmp_path) -> None:
    manifest = load_manifest()
    issues = [case for case in manifest.instances if case.source_type == "github_issue"]
    dev50 = {
        "name": "dev50",
        "seed": 42,
        "instances": [
            {
                "instance_id": case.instance_id,
                "repo": case.repo,
                "base_commit": case.base_commit,
            }
            for case in issues
        ],
    }
    dev50_path = tmp_path / "manifests" / "dev50.json"
    dev50_path.parent.mkdir(parents=True)
    dev50_path.write_text(json.dumps(dev50), encoding="utf-8")
    gold_path = tmp_path / "gold" / "verified_gold.jsonl"
    gold_path.parent.mkdir(parents=True)
    gold_path.write_text(
        "".join(
            json.dumps(
                {
                    "instance_id": case.instance_id,
                    "patch": "\n".join(
                        [
                            f"diff --git a/{path} b/{path}\n+++ b/{path}"
                            for path in case.gold_files
                        ]
                    ),
                }
            )
            + "\n"
            for case in issues
        ),
        encoding="utf-8",
    )
    checks = smoke_runner.validate_external_data(manifest, data_root=tmp_path)
    assert all(check.ok for check in checks)
    assert {check.name for check in checks} >= {
        "dev50 issue coverage",
        "verified gold labels",
    }


def test_run_from_args_writes_timestamped_case_stream_and_summary(tmp_path, monkeypatch) -> None:
    manifest = load_manifest()
    case = manifest.instances[6]
    fetched = _issue_resource(case.source_url, case.base_commit)
    fake_ingestor = FakeIngestor({case.source_url: fetched})
    monkeypatch.setattr(smoke_runner, "GitHubIngestor", lambda _client: fake_ingestor)
    args = Namespace(
        suite="github_smoke10",
        manifest=smoke_runner.DEFAULT_MANIFEST_PATH,
        case=case.instance_id,
        preflight=False,
        preflight_json=None,
        data_root=None,
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "results",
        github_token="token",
        github_timeout=1.0,
        model="fake-model",
    )
    now = smoke_runner.datetime(2026, 8, 30, tzinfo=smoke_runner.UTC)
    prepare_calls: list[dict] = []
    code = smoke_runner.run_from_args(
        args,
        roottrace_client=FakeRootTrace("src/_pytest/capture.py"),
        prepare_fn=_prepare_factory(prepare_calls, tmp_path, case.base_commit),
        now=now,
    )
    assert code == 0
    assert prepare_calls[0]["token"] is None
    streams = sorted((tmp_path / "results").glob("github_smoke10_*.jsonl"))
    summaries = sorted((tmp_path / "results").glob("github_smoke10_*.summary.json"))
    assert len(streams) == len(summaries) == 1
    record = json.loads(streams[0].read_text(encoding="utf-8"))
    summary = json.loads(summaries[0].read_text(encoding="utf-8"))
    assert record["case_id"] == case.instance_id
    assert record["checkout_success"] is True
    assert summary["total_cases"] == 1
