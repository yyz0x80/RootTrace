"""dev50 manifest selector tests (M9-B)."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter

import pytest
from rca_eval_fixtures import public_record, write_json, write_jsonl

from evaluation.rca.adapter import case_from_public_record, load_public_cases
from evaluation.rca.manifest import RcaManifest, load_manifest
from evaluation.rca.select_manifest import (
    allocate_quotas,
    build_dev50_manifest,
    manifest_bytes,
    run_selection,
    select_cases,
)


def _cases(
    repo: str,
    start: int,
    count: int,
    difficulty: str = "15 min - 1 hour",
) -> list[dict]:
    records = []
    for offset in range(count):
        number = start + offset
        records.append(
            public_record(
                f"{repo.replace('/', '__')}-{number}",
                repo,
                f"{number:040x}",
                f"problem {number}",
                difficulty=difficulty,
                version="1.0",
            )
        )
    return records


def _public_cases() -> list:
    records = [
        *_cases("alpha/proj", 1, 10, difficulty="15 min - 1 hour"),
        *_cases("beta/proj", 1, 10, difficulty="1 hour - 2 hours"),
        *_cases("gamma/proj", 1, 20, difficulty="2 hours - 4 hours"),
        *_cases("delta/proj", 1, 20, difficulty="4 hours - 8 hours"),
    ]
    return [case_from_public_record(record) for record in records]


def _smoke_ids() -> set[str]:
    return {"alpha__proj-1", "beta__proj-1"}


def test_selection_size_unique_and_sorted() -> None:
    cases = _public_cases()
    selected = select_cases(cases, _smoke_ids(), seed=42, size=50)
    assert len(selected) == 50
    ids = [case.instance_id for case in selected]
    assert len(set(ids)) == len(ids)
    assert set(ids) & _smoke_ids() == set()
    assert ids == sorted(ids)
    public_ids = {case.instance_id for case in cases}
    assert set(ids) <= public_ids


def test_selection_is_byte_stable() -> None:
    cases = _public_cases()
    first = build_dev50_manifest(select_cases(cases, _smoke_ids()), seed=42)
    second = build_dev50_manifest(select_cases(cases, _smoke_ids()), seed=42)
    assert manifest_bytes(first) == manifest_bytes(second)


def test_hamilton_quotas_largest_remainder() -> None:
    assert allocate_quotas({"a": 3, "b": 3, "c": 4}, 5) == {
        "a": 2,
        "b": 1,
        "c": 2,
    }
    assert allocate_quotas({"a": 1, "b": 1}, 1) == {"a": 1, "b": 0}
    quotas = allocate_quotas({"a": 10, "b": 10, "c": 20, "d": 20}, 50)
    assert sum(quotas.values()) == 50
    assert quotas == {"a": 8, "b": 8, "c": 17, "d": 17}


def test_selection_matches_per_repo_quota_and_rank() -> None:
    cases = _public_cases()
    selected = select_cases(cases, _smoke_ids(), seed=42, size=50)
    counts = Counter(case.repo for case in selected)
    assert counts == {"alpha/proj": 8, "beta/proj": 8, "gamma/proj": 17, "delta/proj": 17}

    for repo, quota in counts.items():
        pool = sorted(
            (
                case
                for case in cases
                if case.repo == repo and case.instance_id not in _smoke_ids()
            ),
            key=lambda case: hashlib.sha256(
                f"42:{case.instance_id}".encode()
            ).hexdigest(),
        )
        expected = {case.instance_id for case in pool[:quota]}
        assert {case.instance_id for case in selected if case.repo == repo} == expected


def test_seed_changes_selection() -> None:
    cases = _public_cases()
    selected_42 = select_cases(cases, _smoke_ids(), seed=42, size=30)
    selected_43 = select_cases(cases, _smoke_ids(), seed=43, size=30)
    ids_42 = {case.instance_id for case in selected_42}
    ids_43 = {case.instance_id for case in selected_43}
    assert ids_42 != ids_43


def test_size_exceeds_eligible_rejected() -> None:
    cases = _public_cases()
    with pytest.raises(ValueError, match="only .* eligible cases"):
        select_cases(cases, _smoke_ids(), seed=42, size=500)


def test_duplicate_public_cases_rejected() -> None:
    cases = _public_cases()
    duplicate = case_from_public_record(
        public_record(
            "alpha__proj-1",
            "alpha/proj",
            "a" * 40,
            "dup",
        )
    )
    with pytest.raises(ValueError, match="duplicate instance_id"):
        select_cases([*cases, duplicate], _smoke_ids(), seed=42, size=50)


def _cli_data_root(tmp_path):
    records = [
        *_cases("alpha/proj", 1, 10),
        *_cases("beta/proj", 1, 10),
        *_cases("gamma/proj", 1, 20),
        *_cases("delta/proj", 1, 20),
    ]
    data_root = tmp_path / "swebench"
    write_jsonl(data_root / "public" / "verified_public.jsonl", records)
    write_json(
        data_root / "manifests" / "smoke3.json",
        {
            "name": "roottrace-smoke3",
            "seed": 42,
            "instances": [
                {"instance_id": "alpha__proj-1", "repo": "alpha/proj", "base_commit": "a" * 40},
                {"instance_id": "beta__proj-1", "repo": "beta/proj", "base_commit": "a" * 40},
            ],
        },
    )
    return data_root


def test_cli_writes_dev50_and_report(tmp_path) -> None:
    data_root = _cli_data_root(tmp_path)
    output_dir = tmp_path / "out"
    exit_code = run_selection(
        _selector_args(data_root=data_root, output_dir=output_dir)
    )
    assert exit_code == 0

    manifest_path = output_dir / "dev50.json"
    report_path = output_dir / "dev50_selection_report.json"
    assert manifest_path.is_file()
    assert report_path.is_file()

    manifest = load_manifest(manifest_path)
    assert manifest.name == "dev50"
    assert len(manifest.instances) == 50
    for case in manifest.instances:
        assert set(case.model_dump()) == {"instance_id", "repo", "base_commit"}

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["counts"]["source_dataset_count"] == 60
    assert report["counts"]["excluded_smoke_count"] == 2
    assert report["counts"]["eligible_count"] == 58
    assert report["counts"]["selected_count"] == 50
    assert report["config"]["seed"] == 42
    assert report["config"]["requested_size"] == 50
    assert report["selector"]["name"] == "evaluation.rca.select_manifest"
    assert len(report["per_repo"]) == 4
    assert sum(entry["allocated_quota"] for entry in report["per_repo"]) == 50
    assert sum(entry["selected_count"] for entry in report["per_repo"]) == 50
    assert report["difficulty_distribution"]["dev50"]
    assert report["hashes"]["dev50_manifest_sha256"] == hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    assert report["leakage_validation"] is None


def test_cli_manifest_contains_no_gold_fields(tmp_path) -> None:
    data_root = _cli_data_root(tmp_path)
    output_dir = tmp_path / "out"
    assert run_selection(_selector_args(data_root=data_root, output_dir=output_dir)) == 0
    raw = json.loads((output_dir / "dev50.json").read_text(encoding="utf-8"))
    for instance in raw["instances"]:
        assert set(instance) == {"instance_id", "repo", "base_commit"}
    assert "patch" not in json.dumps(raw)
    assert "FAIL_TO_PASS" not in json.dumps(raw)
    assert "test_patch" not in json.dumps(raw)


def test_cli_dry_run_writes_nothing(tmp_path) -> None:
    data_root = _cli_data_root(tmp_path)
    output_dir = tmp_path / "out"
    exit_code = run_selection(
        _selector_args(data_root=data_root, output_dir=output_dir, dry_run=True)
    )
    assert exit_code == 0
    assert not output_dir.exists()


def test_cli_seed_deterministic_output(tmp_path) -> None:
    data_root = _cli_data_root(tmp_path)
    first_out = tmp_path / "first"
    second_out = tmp_path / "second"
    assert run_selection(_selector_args(data_root=data_root, output_dir=first_out)) == 0
    assert run_selection(_selector_args(data_root=data_root, output_dir=second_out)) == 0
    assert (first_out / "dev50.json").read_bytes() == (
        second_out / "dev50.json"
    ).read_bytes()


def test_cli_size_too_large_fails(tmp_path) -> None:
    data_root = _cli_data_root(tmp_path)
    exit_code = run_selection(
        _selector_args(data_root=data_root, output_dir=tmp_path / "out", size=500)
    )
    assert exit_code == 2


def _selector_args(
    *,
    data_root,
    output_dir,
    seed: int = 42,
    size: int = 50,
    dry_run: bool = False,
    history_corpus=None,
):
    return argparse.Namespace(
        data_root=data_root,
        source=None,
        smoke_manifest=None,
        history_corpus=history_corpus,
        output_dir=output_dir,
        manifest_name="dev50",
        seed=seed,
        size=size,
        dry_run=dry_run,
    )


def test_load_public_cases_rejects_gold_fields(tmp_path) -> None:
    path = tmp_path / "public.jsonl"
    path.write_text(
        json.dumps(
            {
                "instance_id": "alpha__proj-1",
                "repo": "alpha/proj",
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


def test_manifest_round_trips_as_rca_manifest(tmp_path) -> None:
    cases = _public_cases()
    manifest = build_dev50_manifest(select_cases(cases, _smoke_ids()))
    path = tmp_path / "dev50.json"
    path.write_bytes(manifest_bytes(manifest))
    loaded = RcaManifest.model_validate_json(path.read_text(encoding="utf-8"))
    assert loaded.name == "dev50"
    assert loaded.seed == 42
    assert len(loaded.instances) == 50
