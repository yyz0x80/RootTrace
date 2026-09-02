"""Lexical/cosine Top-K retrieval over the shared historical corpus.

Retrieval is the only mechanism that ranks cases; cluster buckets only narrow
the candidate scope. Leakage guards run before scoring:

- the target incident id and any explicitly excluded ids (evaluation targets,
  duplicates, linked cases) are never returned;
- when the target timestamp is available, cases resolved after the target
  incident are never returned.

The retriever is pure shared memory: it never receives target write or runtime
permissions, and retrieved cases are hints that never override current
repository evidence.
"""

from __future__ import annotations

import numpy as np

from roottrace.history.clustering import nearest_bucket
from roottrace.history.schema import (
    MAX_RETRIEVED,
    HistoricalCase,
    HistoricalCorpus,
    HistoryIndex,
    RetrievalHints,
    RetrievedCase,
    parse_timestamp,
)
from roottrace.history.tfidf import TfidfModel, build_tfidf


def case_text(case: HistoricalCase) -> str:
    """Deterministic text view of a case used for TF-IDF features."""
    parts = [case.problem]
    if case.title:
        parts.append(case.title)
    parts.extend(case.locations)
    if case.summary:
        parts.append(case.summary)
    return "\n".join(parts)


class HistoricalRetriever:
    """Retrieve bounded Top-K historical hints with leakage guards."""

    def __init__(
        self,
        corpus: HistoricalCorpus,
        index: HistoryIndex,
        *,
        top_k: int = 5,
    ) -> None:
        if index.corpus_checksum != corpus.checksum:
            raise ValueError("history index does not match the corpus checksum")
        corpus_ids = [case.id for case in corpus.cases]
        if index.case_ids != corpus_ids:
            raise ValueError("history index case order does not match the corpus")
        self._corpus = corpus
        self._index = index
        self._top_k = max(0, min(top_k, MAX_RETRIEVED))
        self._by_id = {case.id: case for case in corpus.cases}
        texts = [case_text(case) for case in corpus.cases]
        self._tfidf: TfidfModel = build_tfidf(texts)
        if index.vocabulary_size != len(self._tfidf.vocabulary):
            raise ValueError("history index vocabulary does not match the corpus")
        self._position_by_id = {
            case_id: position
            for position, case_id in enumerate(corpus_ids)
        }
        self._matrix = np.asarray(
            [_dense(self._tfidf, vector) for vector in self._tfidf.vectors],
            dtype=float,
        )
        self._norms = np.linalg.norm(self._matrix, axis=1)

    def retrieve(
        self,
        query_text: str,
        *,
        target_id: str | None = None,
        excluded_ids: set[str] | frozenset[str] = frozenset(),
        target_timestamp: str | None = None,
        mode: str = "clustered",
    ) -> RetrievalHints:
        """Return bounded, leakage-filtered Top-K historical hints."""
        if mode not in {"off", "clustered", "flat"}:
            raise ValueError("mode must be off, clustered, or flat")
        notes: list[str] = []
        if mode == "off":
            return RetrievalHints(
                mode="off",
                top_k=self._top_k,
                results=[],
                candidate_count=0,
                index_checksum=self._index.corpus_checksum,
                notes=["retrieval disabled; no historical hints attached"],
            )

        excluded = set(excluded_ids)
        if target_id is not None:
            excluded.add(target_id)
        target_time = (
            parse_timestamp(target_timestamp) if target_timestamp else None
        )
        query_vector = _dense(
            self._tfidf,
            self._tfidf.transform(query_text),
        )
        query_norm = np.linalg.norm(query_vector)

        assignments = self._index.bucket_assignments
        centers = self._index.cluster_centers
        if mode == "clustered" and assignments is not None and centers:
            bucket = nearest_bucket(centers, query_vector)
            candidate_ids = [
                case_id
                for case_id in self._index.case_ids
                if assignments.get(case_id) == bucket
            ]
            actual_mode = "clustered"
            notes.append(
                "cluster buckets are coarse search scope only, not "
                "root-cause categories"
            )
            notes.append(f"searched cluster bucket {bucket}")
        else:
            candidate_ids = list(self._index.case_ids)
            actual_mode = "flat"
            if mode == "clustered":
                notes.append(
                    "clustered retrieval requested but the index has no "
                    "buckets; fell back to flat retrieval"
                )

        scored: list[tuple[float, str]] = []
        for case_id in candidate_ids:
            if case_id in excluded:
                continue
            case = self._by_id[case_id]
            if target_time is not None and case.resolved_at is not None:
                case_time = parse_timestamp(case.resolved_at)
                if case_time is not None and case_time > target_time:
                    continue
            position = self._position_by_id[case_id]
            row = self._matrix[position]
            denominator = self._norms[position] * query_norm
            if denominator <= 0:
                continue
            similarity = float(row @ query_vector / denominator)
            similarity = max(0.0, min(1.0, similarity))
            if similarity <= 0:
                continue
            scored.append((similarity, case_id))

        scored.sort(key=lambda item: (-item[0], item[1]))
        results: list[RetrievedCase] = []
        for similarity, case_id in scored[: self._top_k]:
            case = self._by_id[case_id]
            results.append(
                RetrievedCase(
                    id=case.id,
                    similarity=round(similarity, 6),
                    locations=list(case.locations),
                    summary=case.summary,
                    resolved_at=case.resolved_at,
                    source=case.source,
                )
            )
        if target_time is not None:
            notes.append(
                "cases resolved after the target incident were excluded"
            )
        notes.append(f"{len(scored)} candidates scored after leakage filters")
        return RetrievalHints(
            mode=actual_mode,
            top_k=self._top_k,
            results=results,
            candidate_count=len(scored),
            index_checksum=self._index.corpus_checksum,
            notes=notes,
        )


def _dense(model: TfidfModel, vector: dict[str, float]) -> np.ndarray:
    values = np.zeros(len(model.vocabulary), dtype=float)
    for index, term in enumerate(model.vocabulary):
        values[index] = vector.get(term, 0.0)
    return values
