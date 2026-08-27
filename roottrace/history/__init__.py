"""Shared historical RCA memory: corpus import, indexing, and retrieval.

Historical RCA memory is shared infrastructure, not an evidence Agent. It
provides bounded, auditable prior-case hints: TF-IDF features, seeded
MiniBatchKMeans coarse buckets, and lexical/cosine Top-K retrieval. Retrieved
cases are hints only and never override current-repository evidence.
"""

from roottrace.history.clustering import build_history_index
from roottrace.history.importer import import_corpus
from roottrace.history.retrieval import HistoricalRetriever
from roottrace.history.schema import (
    HistoricalCase,
    HistoricalCorpus,
    HistoryIndex,
    RetrievalHints,
    RetrievedCase,
)
from roottrace.history.tfidf import TfidfModel, build_tfidf, tokenize

__all__ = [
    "HistoricalCase",
    "HistoricalCorpus",
    "HistoricalRetriever",
    "HistoryIndex",
    "RetrievalHints",
    "RetrievedCase",
    "TfidfModel",
    "build_history_index",
    "build_tfidf",
    "import_corpus",
    "tokenize",
]
