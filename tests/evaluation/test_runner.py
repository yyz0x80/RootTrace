"""Runner integration tests using a fake RootTrace client (no real model)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest
from fixtures import (
    build_data_root,
    build_source_repo,
    gold_patch,
    write_json,
)

from evaluation.manifest import load_manifest
from evaluation.metrics import CaseResult
from evaluation.runner import (
    RootTraceOutcome,
    _build_ablation_config,
    _make_run_client,
    extract_predicted_files,
    load_public_cases,
    run_from_args,
)
from evaluation.variants import AblationConfig, AblationVariant, variant_settings
from roottrace.diagnostics import DiagnosticSeverity
from roottrace.incident.schema import IncidentInput
from roottrace.runtime.docker import DockerEnvironment, EnvironmentPreparationError

_REPO = "acme/demo"


def test_real_runner_uses_official_images_unless_map_overrides(tmp_path: Path) -> None:
    variant = AblationVariant.THREE_SPECIALISTS_RETRIEVAL_OFF
    config = AblationConfig(variant=variant, verification_pull_missing=False)
    settings = variant_settings(variant)
    automatic = _make_run_client(settings, config)
    assert automatic._verification_swebench_auto is True
    assert automatic._verification_pull_missing is False
    assert automatic._verification_bootstrap_pytest is True
    image_map = tmp_path / "images.json"
    image_map.write_text("{}", encoding="utf-8")
    mapped = _make_run_client(settings, config, image_map)
    assert mapped._verification_swebench_auto is False
    assert mapped._verification_image_map == image_map


def test_verification_preflight_reports_missing_dependency_without_running_rca(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    data_root, _ = _setup_data(tmp_path)
    output = tmp_path / "preflight-out"
    calls: list[str] = []

    def fake_prepare(self, base_commit):
        assert self.bootstrap_pytest is True
        assert self.prefer_testbed_python is True
        calls.append(self.instance_id)
        if len(calls) == 1:
            return DockerEnvironment("sha256:" + "a" * 64, "sha256:" + "b" * 64,
                                     "linux/amd64", "Python 3.8", "key",
                                     "swebench/example:latest", False)
        raise EnvironmentPreparationError("verification image lacks pytest")

    monkeypatch.setattr("roottrace.runtime.docker.DockerEnvironmentPreparer.prepare", fake_prepare)
    client = FakeRootTraceClient()
    result = run_from_args(_args(
        data_root, output, max_cases=2, verification_preflight=True,
        no_verification_image_pull=True,
    ), client=client)
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert result == 1
    assert [record["status"] for record in records] == ["ready", "unverified"]
    assert "lacks pytest" in records[1]["reason"]
    assert len(calls) == 2
    assert client.received == []
    assert not output.exists()


def test_image_map_contents_are_part_of_resume_config(tmp_path: Path) -> None:
    data_root, _ = _setup_data(tmp_path)
    manifest_path = data_root / "manifests" / "smoke3.json"
    manifest = load_manifest(manifest_path)
    image_map = tmp_path / "images.json"
    args = _args(data_root, tmp_path / "out", verification_image_map=image_map)
    image_map.write_text('{"case":"first"}', encoding="utf-8")
    first = _build_ablation_config(args, manifest, manifest_path, list(manifest.instances))
    image_map.write_text('{"case":"second"}', encoding="utf-8")
    second = _build_ablation_config(args, manifest, manifest_path, list(manifest.instances))
    assert first.config_hash() != second.config_hash()


def test_pytest_bootstrap_policy_changes_resume_config(tmp_path: Path) -> None:
    data_root, _ = _setup_data(tmp_path)
    manifest_path = data_root / "manifests" / "smoke3.json"
    manifest = load_manifest(manifest_path)
    selected = list(manifest.instances)
    enabled = _build_ablation_config(
        _args(data_root, tmp_path / "out"), manifest, manifest_path, selected,
    )
    disabled = _build_ablation_config(
        _args(data_root, tmp_path / "out", no_verification_pytest_bootstrap=True),
        manifest, manifest_path, selected,
    )
    assert enabled.verification_bootstrap_pytest is True
    assert disabled.verification_bootstrap_pytest is False
    assert enabled.config_hash() != disabled.config_hash()


def test_verification_requirements_are_forwarded_and_hashed(tmp_path: Path) -> None:
    data_root, _ = _setup_data(tmp_path)
    manifest_path = data_root / "manifests" / "smoke3.json"
    manifest = load_manifest(manifest_path)
    lock = tmp_path / "verification.lock"
    lock.write_text("example==1.0 --hash=sha256:" + "a" * 64 + "\n")
    args = _args(
        data_root, tmp_path / "out", verification_requirements=lock,
        verification_index_url="https://example.invalid/simple",
    )
    first = _build_ablation_config(args, manifest, manifest_path, list(manifest.instances))
    client = _make_run_client(variant_settings(first.variant), first, None, lock,
                              args.verification_index_url)
    assert client._verification_requirements == lock
    assert client._verification_index_url == args.verification_index_url
    lock.write_text("example==2.0 --hash=sha256:" + "b" * 64 + "\n")
    second = _build_ablation_config(args, manifest, manifest_path, list(manifest.instances))
    assert first.config_hash() != second.config_hash()


class FakeRootTraceClient:
    """Records received incidents and writes deterministic fake reports."""

    def __init__(
        self,
        predictions: dict[str, list[str]] | None = None,
        *,
        fail_ids: set[str] | None = None,
        errors: list[str] | None = None,
        latency: float = 0.25,
        calls: int = 2,
        prompt: int = 100,
        completion: int = 50,
        reasoning: int | None = None,
    ) -> None:
        self.predictions = predictions or {}
        self.fail_ids = fail_ids or set()
        self.errors = errors or []
        self.received: list[IncidentInput] = []
        self.latency = latency
        self.calls = calls
        self.prompt = prompt
        self.completion = completion
        self.reasoning = reasoning

    def run(
        self,
        *,
        case_id: str,
        repo: Path,
        incident: IncidentInput,
        output_dir: Path,
        model: str | None,
    ) -> RootTraceOutcome:
        del repo, model
        self.received.append(incident.model_copy(deep=True))
        if case_id in self.fail_ids:
            raise RuntimeError(f"simulated failure for {case_id}")
        predicted = self.predictions.get(case_id, [])
        report = {"top_k_locations": [{"path": path} for path in predicted]}
        (output_dir / "rca_report.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        return RootTraceOutcome(
            status="completed",
            errors=list(self.errors),
            report=report,
            latency_seconds=self.latency,
            llm_calls=self.calls,
            prompt_tokens=self.prompt,
            completion_tokens=self.completion,
            reasoning_tokens=self.reasoning,
        )


def _args(data_root: Path, output_dir: Path, **overrides) -> argparse.Namespace:
    values = {
        "data_root": data_root,
        "manifest": None,
        "repo_cache": None,
        "gold_path": None,
        "output_dir": output_dir,
        "model": "fake-model",
        "max_cases": None,
        "dry_run": False,
        "resume": False,
        "variant": "three_specialists_retrieval_off",
        "ablation_config": None,
        "history_corpus": None,
        "history_index": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _setup_data(tmp_path: Path):
    shas = build_source_repo(
        tmp_path / "build",
        [
            ("base one", {"pkg/mod.py": "def a():\n    return 1\n"}),
            ("base two", {"pkg/mod.py": "def a():\n    return 2\n"}),
            ("future fix", {"pkg/mod.py": "def a():\n    return 3\n"}),
        ],
    )
    cases = [
        {
            "instance_id": "acme__demo-1",
            "repo": _REPO,
            "base_commit": shas[0],
        },
        {
            "instance_id": "acme__demo-2",
            "repo": _REPO,
            "base_commit": shas[1],
        },
    ]
    data_root = build_data_root(
        tmp_path,
        cases=cases,
        gold_patches={
            "acme__demo-1": gold_patch(["pkg/mod.py"]),
            "acme__demo-2": gold_patch(["pkg/mod.py"]),
        },
        source_repo=tmp_path / "build" / "source",
        repo_id=_REPO,
    )
    return data_root, shas


def test_runner_completes_cases_and_writes_artifacts(tmp_path) -> None:
    data_root, _ = _setup_data(tmp_path)
    output_dir = tmp_path / "out"
    client = FakeRootTraceClient(
        predictions={
            "acme__demo-1": ["pkg/mod.py"],
            "acme__demo-2": ["pkg/mod.py"],
        }
    )
    exit_code = run_from_args(_args(data_root, output_dir), client=client)
    assert exit_code == 0

    for case_id in ("acme__demo-1", "acme__demo-2"):
        case_dir = output_dir / "cases" / case_id
        assert (case_dir / "result.json").is_file()
        assert (case_dir / "root_trace_input.json").is_file()
        assert (case_dir / "roottrace" / "rca_report.json").is_file()
        result = CaseResult.model_validate(
            json.loads((case_dir / "result.json").read_text(encoding="utf-8"))
        )
        assert result.status == "completed"
        assert result.predicted_files == ["pkg/mod.py"]
        assert result.gold_files == ["pkg/mod.py"]
        assert result.metrics.top_1_file_accuracy is True
        assert result.metrics.reasoning_tokens is None

    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["aggregate"]["coverage"] == 1.0
    assert metrics["aggregate"]["top_1_file_accuracy"] == 1.0
    assert metrics["aggregate"]["mean_llm_calls_per_case"] == 2.0
    assert metrics["aggregate"]["reasoning_token_cases"] == 0
    assert metrics["aggregate"]["null_reasoning_cases"] == 2
    assert (output_dir / "report.md").is_file()


def test_reasoning_tokens_flow_into_case_and_aggregate_metrics(tmp_path) -> None:
    data_root, _ = _setup_data(tmp_path)
    output_dir = tmp_path / "out"
    client = FakeRootTraceClient(
        predictions={"acme__demo-1": ["pkg/mod.py"]},
        reasoning=42,
    )
    exit_code = run_from_args(_args(data_root, output_dir), client=client)
    assert exit_code == 0

    result = CaseResult.model_validate(
        json.loads(
            (
                output_dir / "cases" / "acme__demo-1" / "result.json"
            ).read_text(encoding="utf-8")
        )
    )
    assert result.metrics.reasoning_tokens == 42
    assert (result.usage or {})["reasoning_tokens"] == 42

    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["aggregate"]["reasoning_token_cases"] == 2
    assert metrics["aggregate"]["null_reasoning_cases"] == 0
    assert metrics["aggregate"]["mean_reasoning_tokens_per_case"] == 42.0


def test_gold_data_never_reaches_root_trace(tmp_path) -> None:
    data_root, _ = _setup_data(tmp_path)
    output_dir = tmp_path / "out"
    client = FakeRootTraceClient(predictions={"acme__demo-1": ["pkg/mod.py"]})
    exit_code = run_from_args(
        _args(data_root, output_dir, max_cases=1),
        client=client,
    )
    assert exit_code == 0
    assert len(client.received) == 1
    incident = client.received[0]
    assert incident.logs == []
    assert incident.diff is None
    assert "patch" not in incident.model_dump()
    input_payload = json.loads(
        (output_dir / "cases" / "acme__demo-1" / "root_trace_input.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(input_payload) == {
        "instance_id",
        "repo",
        "base_commit",
        "problem_statement",
    }


def test_case_failure_does_not_abort_benchmark(tmp_path) -> None:
    data_root, _ = _setup_data(tmp_path)
    output_dir = tmp_path / "out"
    client = FakeRootTraceClient(
        predictions={"acme__demo-1": ["pkg/mod.py"]},
        fail_ids={"acme__demo-2"},
    )
    exit_code = run_from_args(_args(data_root, output_dir), client=client)
    assert exit_code == 0

    failed = CaseResult.model_validate(
        json.loads(
            (output_dir / "cases" / "acme__demo-2" / "result.json").read_text(
                encoding="utf-8"
            )
        )
    )
    assert failed.status == "error"
    assert "simulated failure" in (failed.error or "")
    assert failed.diagnostics[0].severity is DiagnosticSeverity.FATAL
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["aggregate"]["coverage"] == 0.5
    assert metrics["aggregate"]["failed_cases"] == 1
    assert metrics["aggregate"]["top_1_file_accuracy"] == 1.0


def test_resume_skips_completed_cases(tmp_path) -> None:
    data_root, _ = _setup_data(tmp_path)
    output_dir = tmp_path / "out"
    client = FakeRootTraceClient(predictions={"acme__demo-1": ["pkg/mod.py"]})
    assert run_from_args(_args(data_root, output_dir), client=client) == 0
    calls_after_first_run = len(client.received)
    assert calls_after_first_run == 2

    second_client = FakeRootTraceClient(predictions={})
    assert (
        run_from_args(
            _args(data_root, output_dir, resume=True),
            client=second_client,
        )
        == 0
    )
    assert second_client.received == []
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["aggregate"]["covered_cases"] == 2


def test_dry_run_writes_nothing(tmp_path) -> None:
    data_root, _ = _setup_data(tmp_path)
    output_dir = tmp_path / "out"
    exit_code = run_from_args(
        _args(data_root, output_dir, dry_run=True),
        client=FakeRootTraceClient(),
    )
    assert exit_code == 0
    assert not output_dir.exists()


def test_max_cases_limits_manifest_scope(tmp_path) -> None:
    data_root, _ = _setup_data(tmp_path)
    output_dir = tmp_path / "out"
    client = FakeRootTraceClient(predictions={"acme__demo-1": ["pkg/mod.py"]})
    exit_code = run_from_args(
        _args(data_root, output_dir, max_cases=1),
        client=client,
    )
    assert exit_code == 0
    assert [received.id for received in client.received] == ["acme__demo-1"]
    assert not (output_dir / "cases" / "acme__demo-2").exists()


def test_manifest_public_mismatch_rejected(tmp_path) -> None:
    data_root, shas = _setup_data(tmp_path)
    # Rewrite the manifest so its base_commit disagrees with public metadata.
    write_json(
        data_root / "manifests" / "smoke3.json",
        {
            "name": "smoke",
            "seed": 42,
            "instances": [
                {
                    "instance_id": "acme__demo-1",
                    "repo": _REPO,
                    "base_commit": shas[1],
                }
            ],
        },
    )
    exit_code = run_from_args(
        _args(data_root, tmp_path / "out"),
        client=FakeRootTraceClient(),
    )
    assert exit_code == 2


def test_load_public_cases_rejects_gold_fields(tmp_path) -> None:
    path = tmp_path / "public.jsonl"
    path.write_text(
        json.dumps(
            {
                "instance_id": "acme__demo-1",
                "repo": _REPO,
                "base_commit": "a" * 40,
                "problem_statement": "x",
                "patch": "diff --git ...",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="disallowed fields"):
        load_public_cases(path)


def test_extract_predicted_files_from_report() -> None:
    report = {
        "top_k_locations": [
            {"path": "src/a.py"},
            {"path": "/etc/passwd"},
            {"path": "src/a.py"},
            {"path": "src/b.py"},
        ]
    }
    assert extract_predicted_files(report) == ["src/a.py", "src/b.py"]
    assert extract_predicted_files(
        {
            "conclusion": "insufficient_evidence",
            "ranked_causes": [],
            "top_k_locations": [{"path": "src/uncertain.py"}],
        }
    ) == ["src/uncertain.py"]
    assert extract_predicted_files(None) == []
    assert extract_predicted_files({"top_k_locations": "bad"}) == []


def test_rca_errors_surface_in_result(tmp_path) -> None:
    data_root, _ = _setup_data(tmp_path)
    output_dir = tmp_path / "out"
    client = FakeRootTraceClient(
        predictions={"acme__demo-1": ["pkg/mod.py"]},
        errors=["code worker failed: excerpt too long"],
    )
    assert run_from_args(_args(data_root, output_dir, max_cases=1), client=client) == 0
    result = CaseResult.model_validate(
        json.loads(
            (output_dir / "cases" / "acme__demo-1" / "result.json").read_text(
                encoding="utf-8"
            )
        )
    )
    assert result.status == "completed"
    assert result.rca_errors == ["code worker failed: excerpt too long"]
    assert len(result.diagnostics) == 1
    assert result.diagnostics[0].stage == "legacy"
    assert result.diagnostics[0].message == result.rca_errors[0]


def test_case_result_reads_legacy_payload_without_diagnostics(tmp_path) -> None:
    data_root, _ = _setup_data(tmp_path)
    output_dir = tmp_path / "out"
    client = FakeRootTraceClient(predictions={"acme__demo-1": ["pkg/mod.py"]})
    assert run_from_args(_args(data_root, output_dir, max_cases=1), client=client) == 0
    result_path = output_dir / "cases" / "acme__demo-1" / "result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload.pop("diagnostics")
    payload["schema_version"] = "1.0"
    payload["rca_errors"] = ["legacy partial result"]

    result = CaseResult.model_validate(payload)

    assert result.schema_version == "1.0"
    assert result.rca_errors == ["legacy partial result"]
    assert result.diagnostics[0].code == "legacy.error"


def test_sum_or_none_usage_merge() -> None:
    from evaluation.runner import _sum_or_none

    assert _sum_or_none(1, 2) == 3
    assert _sum_or_none(None, 2) is None
    assert _sum_or_none(1, None) is None
    assert _sum_or_none(None, None) is None


def test_variant_is_recorded_in_results_and_config(tmp_path) -> None:
    data_root, _ = _setup_data(tmp_path)
    output_dir = tmp_path / "out"
    client = FakeRootTraceClient(predictions={"acme__demo-1": ["pkg/mod.py"]})
    exit_code = run_from_args(
        _args(data_root, output_dir, variant="lead_code"),
        client=client,
    )
    assert exit_code == 0
    result = CaseResult.model_validate(
        json.loads(
            (output_dir / "cases" / "acme__demo-1" / "result.json").read_text(
                encoding="utf-8"
            )
        )
    )
    assert result.variant == "lead_code"
    assert result.config_hash
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["config"]["variant"] == "lead_code"
    assert metrics["config"]["config_hash"] == result.config_hash
    assert (output_dir / "variant.json").is_file()


def test_resume_skips_only_matching_variant_and_config(tmp_path) -> None:
    data_root, _ = _setup_data(tmp_path)
    lead_dir = tmp_path / "out-lead"
    first_client = FakeRootTraceClient(predictions={"acme__demo-1": ["pkg/mod.py"]})
    assert (
        run_from_args(
            _args(data_root, lead_dir, variant="lead_code"),
            client=first_client,
        )
        == 0
    )
    assert len(first_client.received) == 2

    # A different variant uses its own output directory.
    off_dir = tmp_path / "out-off"
    different_client = FakeRootTraceClient(predictions={"acme__demo-1": ["pkg/mod.py"]})
    assert (
        run_from_args(
            _args(
                data_root,
                off_dir,
                variant="three_specialists_retrieval_off",
                resume=True,
            ),
            client=different_client,
        )
        == 0
    )
    assert len(different_client.received) == 2

    # The same variant and config are skipped.
    resume_client = FakeRootTraceClient(predictions={})
    assert (
        run_from_args(
            _args(data_root, lead_dir, variant="lead_code", resume=True),
            client=resume_client,
        )
        == 0
    )
    assert resume_client.received == []


def test_cli_help_and_dry_run_subprocess(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    help_result = subprocess.run(
        [sys.executable, "-m", "evaluation.runner", "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert help_result.returncode == 0
    assert "--data-root" in help_result.stdout
    assert "--dry-run" in help_result.stdout
    assert "--resume" in help_result.stdout

    data_root, _ = _setup_data(tmp_path)
    dry_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "evaluation.runner",
            "--data-root",
            str(data_root),
            "--dry-run",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert dry_result.returncode == 0
    assert "DRY RUN" in dry_result.stdout
    assert "acme__demo-1" in dry_result.stdout
