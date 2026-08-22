"""Seeded MiniBatchKMeans coarse buckets for historical RCA cases.

Buckets only narrow the retrieval scope. They are never described as verified
root-cause categories; the actual ranking is performed by cosine similarity on
TF-IDF features. Query bucketing uses the nearest stored centroid, which keeps
the index deterministic and serializable without pickling a model object.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from patchpilot.rca.history.schema import HistoricalCorpus, HistoryIndex
from patchpilot.rca.history.tfidf import TfidfModel


def fit_buckets(
    matrix: np.ndarray,
    case_ids: list[str],
    *,
    n_clusters: int,
    seed: int,
) -> tuple[dict[str, int], list[list[float]]]:
    """Fit seeded MiniBatchKMeans and return assignments plus centroids."""
    count = len(case_ids)
    if count == 0:
        return {}, []
    if n_clusters < 1:
        raise ValueError("n_clusters must be at least 1")
    if n_clusters > count:
        raise ValueError("n_clusters must not exceed the number of cases")
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        n_init=3,
        batch_size=min(256, count),
        max_iter=100,
        reassignment_ratio=0.01,
    )
    labels = kmeans.fit_predict(matrix).tolist()
    assignments = {
        case_id: int(label)
        for case_id, label in zip(case_ids, labels, strict=True)
    }
    centers = kmeans.cluster_centers_.tolist()
    return assignments, centers


def nearest_bucket(
    cluster_centers: list[list[float]],
    query_vector: np.ndarray,
) -> int:
    """Assign a query to the nearest centroid (deterministic, no refit)."""
    centers = np.asarray(cluster_centers, dtype=float)
    deltas = centers - query_vector.reshape(1, -1)
    distances = np.einsum("ij,ij->i", deltas, deltas)
    return int(np.argmin(distances))


def build_history_index(
    corpus: HistoricalCorpus,
    tfidf: TfidfModel,
    *,
    seed: int = 42,
    n_clusters: int = 8,
    clustering: bool = True,
) -> HistoryIndex:
    """Build the persisted index; clustering is optional and coarse only."""
    case_ids = [case.id for case in corpus.cases]
    matrix = np.asarray(
        [_dense(tfidf, vector) for vector in tfidf.vectors],
        dtype=float,
    )
    if clustering and len(case_ids) > 1:
        assignments, centers = fit_buckets(
            matrix,
            case_ids,
            n_clusters=min(n_clusters, len(case_ids)),
            seed=seed,
        )
    else:
        assignments, centers = None, None
    return HistoryIndex(
        corpus_checksum=corpus.checksum,
        seed=seed,
        n_clusters=min(n_clusters, len(case_ids)) if clustering else None,
        vocabulary_size=len(tfidf.vocabulary),
        case_ids=case_ids,
        bucket_assignments=assignments,
        cluster_centers=centers,
        built_at=datetime.now(UTC).isoformat(),
    )


def _dense(model: TfidfModel, vector: dict[str, float]) -> np.ndarray:
    values = np.zeros(len(model.vocabulary), dtype=float)
    for index, term in enumerate(model.vocabulary):
        values[index] = vector.get(term, 0.0)
    return values
