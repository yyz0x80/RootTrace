"""Historical-corpus leakage validation tests (M9-B)."""

from __future__ import annotations

import json

from test_rca_eval_select_manifest import _cli_data_root, _selector_args, run_selection

from evaluation.rca.leakage import load_history_instance_ids, validate_leakage


def test_load_history_ids_supports_id_and_instance_id(tmp_path) -> None:
    path = tmp_path / "history.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"id": "django__django-1", "repo": "django/django"}),
                json.dumps(
                    {
                        "instance_id": "sphinx-doc__sphinx-2",
                        "repo": "sphinx-doc/sphinx",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert load_history_instance_ids(path) == {
        "django__django-1",
        "sphinx-doc__sphinx-2",
    }


def test_validate_leakage_no_overlap() -> None:
    result = validate_leakage(
        target_ids={"a", "b"},
        history_ids={"c", "d"},
    )
    assert result["ok"] is True
    assert result["overlap_count"] == 0


def test_validate_leakage_detects_overlap() -> None:
    result = validate_leakage(
        target_ids={"a", "b"},
        history_ids={"b", "c"},
    )
    assert result["ok"] is False
    assert result["overlap_ids"] == ["b"]


def test_selector_rejects_history_overlap(tmp_path) -> None:
    data_root = _cli_data_root(tmp_path)
    output_dir = tmp_path / "out"
    assert run_selection(_selector_args(data_root=data_root, output_dir=output_dir)) == 0
    manifest = json.loads((output_dir / "dev50.json").read_text(encoding="utf-8"))
    target_id = manifest["instances"][0]["instance_id"]

    history_path = tmp_path / "history.jsonl"
    history_path.write_text(
        json.dumps({"id": target_id, "repo": "alpha/proj"}) + "\n",
        encoding="utf-8",
    )
    exit_code = run_selection(
        _selector_args(
            data_root=data_root,
            output_dir=tmp_path / "blocked",
            history_corpus=history_path,
        )
    )
    assert exit_code == 2
    assert not (tmp_path / "blocked" / "dev50.json").exists()


def test_selector_accepts_disjoint_history(tmp_path) -> None:
    data_root = _cli_data_root(tmp_path)
    history_path = tmp_path / "history.jsonl"
    history_path.write_text(
        json.dumps({"id": "unrelated__case-99", "repo": "other/repo"}) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    exit_code = run_selection(
        _selector_args(
            data_root=data_root,
            output_dir=output_dir,
            history_corpus=history_path,
        )
    )
    assert exit_code == 0
    report = json.loads(
        (output_dir / "dev50_selection_report.json").read_text(encoding="utf-8")
    )
    assert report["leakage_validation"]["ok"] is True
    assert report["leakage_validation"]["history_cases"] == 1
