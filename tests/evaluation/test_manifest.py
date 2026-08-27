"""Manifest loader tests for the RCA evaluation pipeline."""

from __future__ import annotations

import json

import pytest
from fixtures import write_json

from evaluation.manifest import load_manifest


def _manifest(instances: list[dict], name: str = "smoke") -> dict:
    return {"name": name, "seed": 42, "instances": instances}


def test_load_valid_manifest(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    write_json(
        path,
        _manifest(
            [
                {
                    "instance_id": "acme__demo-1",
                    "repo": "acme/demo",
                    "base_commit": "a" * 40,
                }
            ]
        ),
    )
    manifest = load_manifest(path)
    assert manifest.name == "smoke"
    assert manifest.instances[0].instance_id == "acme__demo-1"


def test_duplicate_instance_id_rejected(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    write_json(
        path,
        _manifest(
            [
                {
                    "instance_id": "acme__demo-1",
                    "repo": "acme/demo",
                    "base_commit": "a" * 40,
                },
                {
                    "instance_id": "acme__demo-1",
                    "repo": "acme/demo",
                    "base_commit": "b" * 40,
                },
            ]
        ),
    )
    with pytest.raises(ValueError, match="duplicate instance_id"):
        load_manifest(path)


def test_duplicate_repo_base_commit_rejected(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    write_json(
        path,
        _manifest(
            [
                {
                    "instance_id": "acme__demo-1",
                    "repo": "acme/demo",
                    "base_commit": "a" * 40,
                },
                {
                    "instance_id": "acme__demo-2",
                    "repo": "acme/demo",
                    "base_commit": "a" * 40,
                },
            ]
        ),
    )
    with pytest.raises(ValueError, match="duplicate case checkout"):
        load_manifest(path)


def test_invalid_base_commit_rejected(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    write_json(
        path,
        _manifest(
            [
                {
                    "instance_id": "acme__demo-1",
                    "repo": "acme/demo",
                    "base_commit": "not-a-sha",
                }
            ]
        ),
    )
    with pytest.raises(ValueError, match="base_commit"):
        load_manifest(path)


def test_malformed_manifest_rejected(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid manifest JSON"):
        load_manifest(path)


def test_non_object_manifest_rejected(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
    with pytest.raises(TypeError, match="must be a JSON object"):
        load_manifest(path)
