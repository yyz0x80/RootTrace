"""Tests for the shared historical RCA memory: import, index, retrieval."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from roottrace.history.clustering import build_history_index
from roottrace.history.importer import import_corpus
from roottrace.history.retrieval import HistoricalRetriever, case_text
from roottrace.history.schema import HistoricalCorpus, HistoryIndex
from roottrace.history.tfidf import build_tfidf, tokenize


def _write_corpus(tmp_path: Path, name: str = "corpus.jsonl") -> Path:
    lines = [
        {
            "id": "c-001",
            "repo": "calc",
            "title": "multiply returns the sum",
            "problem": "multiply(3, 4) returns 7 instead of 12",
            "resolved_at": "2026-01-15T00:00:00+00:00",
            "source": "history.jsonl",
            "locations": ["pkg/calc.py"],
            "summary": "multiply added instead of multiplying",
            "linked_ids": ["c-001b"],
        },
        {
            "id": "c-001",
            "repo": "calc",
            "title": "duplicate line",
            "problem": "multiply returns the sum again",
            "resolved_at": "2026-01-16T00:00:00+00:00",
            "source": "history.jsonl",
            "locations": ["pkg/calc.py"],
        },
        {
            "id": "c-002",
            "repo": "calc",
            "title": "product becomes sum",
            "problem": "the multiply operation adds its operands",
            "resolved_at": "2026-02-01T00:00:00+00:00",
            "source": "history.jsonl",
            "locations": ["pkg/calc.py", "pkg/ops.py"],
            "summary": "operator confusion in multiply",
        },
        {
            "id": "c-003",
            "repo": "calc",
            "title": "formatting issue",
            "problem": "whitespace and docstring formatting",
            "resolved_at": "2026-03-01T00:00:00+00:00",
            "source": "history.jsonl",
            "locations": ["pkg/format.py"],
        },
        {
            "id": "c-004",
            "repo": "calc",
            "title": "future multiply bug",
            "problem": "multiply returns sum in a future case",
            "resolved_at": "2026-12-31T00:00:00+00:00",
            "source": "history.jsonl",
            "locations": ["pkg/calc.py"],
        },
        {
            "id": "eval-001",
            "repo": "benchmark",
            "title": "evaluation target",
            "problem": "gold patch problem",
            "resolved_at": "2026-04-01T00:00:00+00:00",
            "source": "history.jsonl",
            "locations": ["pkg/bench.py"],
            "gold_patch": "diff --git a/pkg/bench.py b/pkg/bench.py",
        },
        {
            "id": "c-001b",
            "repo": "calc",
            "title": "linked duplicate",
            "problem": "multiply returns sum (linked case)",
            "resolved_at": "2026-01-17T00:00:00+00:00",
            "source": "history.jsonl",
            "locations": ["pkg/calc.py"],
        },
        "not json at all",
        {"id": "c-malformed", "title": "missing problem"},
        {
            "id": "c-unknown-field",
            "repo": "calc",
            "problem": "unknown field should be rejected",
            "source": "history.jsonl",
            "surprise": 1,
        },
    ]
    path = tmp_path / name
    path.write_text(
        "\n".join(
            line if isinstance(line, str) else json.dumps(line)
            for line in lines
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _build_index(corpus: HistoricalCorpus, *, n_clusters: int = 2) -> HistoryIndex:
    texts = [case_text(case) for case in corpus.cases]
    tfidf = build_tfidf(texts)
    return build_history_index(
        corpus,
        tfidf,
        seed=42,
        n_clusters=n_clusters,
        clustering=True,
    )


def test_importer_excludes_duplicates_and_strips_gold_patches(
    tmp_path: Path,
) -> None:
    corpus = import_corpus(
        _write_corpus(tmp_path),
        excluded_ids={"eval-001", "c-001b"},
    )
    ids = {case.id for case in corpus.cases}
    assert ids == {"c-001", "c-002", "c-003", "c-004"}
    assert corpus.duplicate_count == 1
    assert corpus.excluded_count == 2
    assert corpus.gold_patch_dropped_count == 0
    assert corpus.malformed_count == 3
    assert all("gold_patch" not in case.model_dump() for case in corpus.cases)
    kept = import_corpus(
        _write_corpus(tmp_path),
        excluded_ids={"c-001b"},
    )
    assert kept.gold_patch_dropped_count == 1
    assert any("gold/test patch" in note for note in kept.notes)


def test_importer_checksum_is_deterministic(tmp_path: Path) -> None:
    source = _write_corpus(tmp_path)
    first = import_corpus(source, excluded_ids={"eval-001"})
    second = import_corpus(source, excluded_ids={"eval-001"})
    assert first.checksum == second.checksum
    assert len(first.checksum) == 64
    truncated = import_corpus(source, excluded_ids={"eval-001"}, max_cases=2)
    assert len(truncated.cases) == 2
    assert any("truncated" in note for note in truncated.notes)


def test_importer_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        import_corpus(tmp_path / "missing.jsonl")


def test_tfidf_is_deterministic() -> None:
    texts = [
        "multiply returns the sum instead of the product",
        "formatting and docstring whitespace",
    ]
    first = build_tfidf(texts)
    second = build_tfidf(texts)
    assert first.vocabulary == second.vocabulary
    assert first.idf == second.idf
    assert first.vectors == second.vectors
    assert tokenize("Multiply(3, 4) returns 7") == ["multiply", "returns"]
    query = first.transform("multiply returns sum")
    assert set(query) <= set(first.vocabulary)
    assert all(value > 0 for value in query.values())


def test_seeded_clustering_is_stable(tmp_path: Path) -> None:
    corpus = import_corpus(_write_corpus(tmp_path), excluded_ids={"eval-001"})
    first = _build_index(corpus)
    second = _build_index(corpus)
    assert first.bucket_assignments == second.bucket_assignments
    assert first.cluster_centers == second.cluster_centers
    assert set(first.bucket_assignments) == {case.id for case in corpus.cases}
    assert len({value for value in first.bucket_assignments.values()}) <= 2
    assert len(first.cluster_centers) == 2


def test_retrieval_off_returns_no_hints(tmp_path: Path) -> None:
    corpus = import_corpus(_write_corpus(tmp_path), excluded_ids={"eval-001"})
    retriever = HistoricalRetriever(corpus, _build_index(corpus))
    hints = retriever.retrieve(
        "multiply returns the sum instead of the product",
        mode="off",
    )
    assert hints.mode == "off"
    assert hints.results == []
    assert hints.candidate_count == 0
    assert any("disabled" in note for note in hints.notes)


def test_retrieval_clustered_narrows_and_ranks(tmp_path: Path) -> None:
    corpus = import_corpus(_write_corpus(tmp_path), excluded_ids={"eval-001"})
    index = _build_index(corpus)
    retriever = HistoricalRetriever(corpus, index, top_k=2)
    hints = retriever.retrieve(
        "multiply returns the sum instead of the product",
        mode="clustered",
    )
    assert hints.mode == "clustered"
    assert hints.candidate_count < len(corpus.cases)
    assert len(hints.results) <= 2
    similarities = [result.similarity for result in hints.results]
    assert similarities == sorted(similarities, reverse=True)
    assert hints.results
    assert any("coarse search scope" in note for note in hints.notes)


def test_retrieval_flat_scores_all_candidates(tmp_path: Path) -> None:
    corpus = import_corpus(_write_corpus(tmp_path), excluded_ids={"eval-001"})
    retriever = HistoricalRetriever(corpus, _build_index(corpus), top_k=10)
    hints = retriever.retrieve(
        "multiply returns the sum instead of the product",
        mode="flat",
    )
    assert hints.mode == "flat"
    assert 0 < hints.candidate_count <= len(corpus.cases)
    assert len(hints.results) <= 10


def test_retrieval_leakage_guard(tmp_path: Path) -> None:
    corpus = import_corpus(_write_corpus(tmp_path), excluded_ids={"eval-001"})
    retriever = HistoricalRetriever(corpus, _build_index(corpus), top_k=10)
    hints = retriever.retrieve(
        "multiply returns the sum",
        target_id="c-001",
        excluded_ids={"c-002", "c-001b"},
        target_timestamp="2026-06-01T00:00:00+00:00",
        mode="flat",
    )
    result_ids = {result.id for result in hints.results}
    assert "c-001" not in result_ids
    assert "c-002" not in result_ids
    assert "c-004" not in result_ids  # resolved after the target incident
    assert "eval-001" not in result_ids
    assert "c-001b" not in result_ids
    assert any("resolved after the target incident" in note for note in hints.notes)


def test_retrieval_is_deterministic(tmp_path: Path) -> None:
    corpus = import_corpus(_write_corpus(tmp_path), excluded_ids={"eval-001"})
    retriever = HistoricalRetriever(corpus, _build_index(corpus), top_k=3)
    query = "multiply returns the sum instead of the product"
    first = retriever.retrieve(query, mode="clustered")
    second = retriever.retrieve(query, mode="clustered")
    assert first.model_dump() == second.model_dump()


def test_retriever_rejects_checksum_mismatch(tmp_path: Path) -> None:
    corpus = import_corpus(_write_corpus(tmp_path), excluded_ids={"eval-001"})
    index = _build_index(corpus)
    altered = corpus.model_copy(
        update={
            "cases": corpus.cases[:-1],
            "checksum": "0" * 64,
        }
    )
    with pytest.raises(ValueError):
        HistoricalRetriever(altered, index)


def test_retrieval_returns_bounded_zero_overlap(tmp_path: Path) -> None:
    corpus = import_corpus(_write_corpus(tmp_path), excluded_ids={"eval-001"})
    retriever = HistoricalRetriever(corpus, _build_index(corpus), top_k=5)
    hints = retriever.retrieve(
        "quantum entanglement teleportation protocol",
        mode="flat",
    )
    assert len(hints.results) <= 5
    for result in hints.results:
        assert 0 <= result.similarity <= 1
