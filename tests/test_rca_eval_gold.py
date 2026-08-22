"""Gold parser tests: non-test source file extraction."""

from __future__ import annotations

import pytest
from rca_eval_fixtures import gold_patch, gold_record, write_jsonl

from evaluation.rca.gold import GoldStore, is_test_file, parse_gold_patch


def test_parse_gold_patch_keeps_source_files_only() -> None:
    patch = gold_patch(
        ["src/package/mod.py", "src/package/other.py"],
        test_files=["tests/test_mod.py", "src/package/tests/test_other.py"],
    )
    assert parse_gold_patch(patch) == [
        "src/package/mod.py",
        "src/package/other.py",
    ]


def test_parse_gold_patch_dedupes_and_sorts() -> None:
    patch = gold_patch(["b.py", "a.py"]) + gold_patch(["b.py"])
    assert parse_gold_patch(patch) == ["a.py", "b.py"]


def test_parse_gold_patch_skips_deleted_files() -> None:
    patch = (
        "diff --git a/src/kept.py b/src/kept.py\n"
        "--- a/src/kept.py\n"
        "+++ b/src/kept.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
        "diff --git a/src/removed.py b/dev/null\n"
        "--- a/src/removed.py\n"
        "+++ b/dev/null\n"
        "@@ -1 +0,0 @@\n"
        "-gone\n"
    )
    assert parse_gold_patch(patch) == ["src/kept.py"]


def test_parse_gold_patch_empty() -> None:
    assert parse_gold_patch("") == []


@pytest.mark.parametrize(
    "path",
    [
        "tests/test_a.py",
        "src/pkg/tests/test_a.py",
        "test_a.py",
        "src/test_helpers.py",
        "conftest.py",
        "src/pkg/test/data.csv",
    ],
)
def test_is_test_file_positive(path: str) -> None:
    assert is_test_file(path)


@pytest.mark.parametrize(
    "path",
    [
        "src/a.py",
        "package/mod.py",
        "docs/guide.rst",
    ],
)
def test_is_test_file_negative(path: str) -> None:
    assert not is_test_file(path)


def test_gold_store_loads_and_parses_after_run(tmp_path) -> None:
    gold_path = tmp_path / "gold.jsonl"
    write_jsonl(
        gold_path,
        [
            gold_record(
                "acme__demo-1",
                patch=gold_patch(["src/mod.py"]),
                test_patch=gold_patch([], ["tests/test_mod.py"]),
            )
        ],
    )
    store = GoldStore(gold_path)
    assert store.gold_files("acme__demo-1") == ["src/mod.py"]


def test_gold_store_missing_case_raises(tmp_path) -> None:
    gold_path = tmp_path / "gold.jsonl"
    write_jsonl(gold_path, [gold_record("acme__demo-1", patch="")])
    store = GoldStore(gold_path)
    with pytest.raises(ValueError, match="missing"):
        store.gold_files("acme__demo-2")


def test_gold_store_rejects_duplicates(tmp_path) -> None:
    gold_path = tmp_path / "gold.jsonl"
    write_jsonl(
        gold_path,
        [
            gold_record("acme__demo-1", patch=""),
            gold_record("acme__demo-1", patch=""),
        ],
    )
    with pytest.raises(ValueError, match="duplicate gold instance_id"):
        GoldStore(gold_path).load()
