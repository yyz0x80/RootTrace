"""Evaluation preflight tests (M9-C, no LLM calls)."""

from __future__ import annotations

import argparse
import json

from rca_eval_fixtures import (
    build_data_root,
    build_source_repo,
    gold_patch,
    write_jsonl,
)

from evaluation.rca.preflight import run_preflight


def _args(data_root, **overrides) -> argparse.Namespace:
    values = {
        "data_root": data_root,
        "manifest": None,
        "smoke_manifest": None,
        "repo_cache": None,
        "public": None,
        "history_corpus": None,
        "variant": "three_specialists_retrieval_off",
        "max_cases": None,
        "json": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _data_root(tmp_path):
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
            "repo": "acme/demo",
            "base_commit": shas[0],
        },
        {
            "instance_id": "acme__demo-2",
            "repo": "acme/demo",
            "base_commit": shas[1],
        },
    ]
    return build_data_root(
        tmp_path,
        cases=cases,
        gold_patches={
            "acme__demo-1": gold_patch(["pkg/mod.py"]),
            "acme__demo-2": gold_patch(["pkg/mod.py"]),
        },
        source_repo=tmp_path / "build" / "source",
        repo_id="acme/demo",
    )


def test_preflight_passes_on_valid_fixture(tmp_path) -> None:
    data_root = _data_root(tmp_path)
    manifest = data_root / "manifests" / "smoke3.json"
    exit_code = run_preflight(
        _args(data_root, manifest=manifest, smoke_manifest=manifest)
    )
    assert exit_code == 0


def test_preflight_rejects_gold_fields_in_manifest(tmp_path) -> None:
    data_root = _data_root(tmp_path)
    manifest = data_root / "manifests" / "smoke3.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["instances"][0]["patch"] = "diff --git ..."
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    exit_code = run_preflight(
        _args(data_root, manifest=manifest, smoke_manifest=manifest)
    )
    assert exit_code == 2


def test_preflight_rejects_unknown_public_case(tmp_path) -> None:
    data_root = _data_root(tmp_path)
    manifest = data_root / "manifests" / "smoke3.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["instances"][0]["instance_id"] = "nope__missing-99"
    payload["instances"][0]["base_commit"] = "a" * 40
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    exit_code = run_preflight(
        _args(data_root, manifest=manifest, smoke_manifest=manifest)
    )
    assert exit_code == 2


def test_preflight_rejects_missing_base_commit(tmp_path) -> None:
    data_root = _data_root(tmp_path)
    manifest = data_root / "manifests" / "smoke3.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["instances"][0]["base_commit"] = "b" * 40
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    exit_code = run_preflight(
        _args(data_root, manifest=manifest, smoke_manifest=manifest)
    )
    assert exit_code == 2


def test_preflight_rejects_history_overlap(tmp_path) -> None:
    data_root = _data_root(tmp_path)
    manifest = data_root / "manifests" / "smoke3.json"
    history_path = tmp_path / "history.jsonl"
    write_jsonl(
        history_path,
        [{"id": "acme__demo-1", "repo": "acme/demo"}],
    )
    exit_code = run_preflight(
        _args(
            data_root,
            manifest=manifest,
            smoke_manifest=manifest,
            history_corpus=history_path,
        )
    )
    assert exit_code == 2


def test_preflight_writes_json_report(tmp_path) -> None:
    data_root = _data_root(tmp_path)
    manifest = data_root / "manifests" / "smoke3.json"
    report_path = tmp_path / "preflight.json"
    exit_code = run_preflight(
        _args(
            data_root,
            manifest=manifest,
            smoke_manifest=manifest,
            json=report_path,
        )
    )
    assert exit_code == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert len(report["checks"]) >= 8
    assert all(check["ok"] for check in report["checks"])
