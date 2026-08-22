"""Deterministic TF-IDF features for historical RCA cases."""

from __future__ import annotations

import math
import re
from collections import Counter

from pydantic import BaseModel

_TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")

_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "from",
        "into",
        "that",
        "this",
        "should",
        "when",
        "after",
        "before",
        "using",
        "does",
        "have",
        "has",
        "are",
        "was",
        "were",
        "not",
    }
)


def tokenize(text: str) -> list[str]:
    """Return lowercased, stopword-filtered word tokens."""
    return [
        match.group(0).lower()
        for match in _TOKEN_PATTERN.finditer(text.lower())
        if match.group(0).lower() not in _STOPWORDS
    ]


class TfidfModel(BaseModel):
    """Bounded TF-IDF model built from a corpus in deterministic order."""

    vocabulary: list[str]
    idf: dict[str, float]
    vectors: list[dict[str, float]]

    def transform(self, text: str) -> dict[str, float]:
        """Transform one query text into a TF-IDF vector over the vocabulary."""
        tokens = tokenize(text)
        counts = Counter(tokens)
        total = sum(counts.values()) or 1
        return {
            term: (counts[term] / total) * self.idf[term]
            for term in set(tokens)
            if term in self.idf
        }


def build_tfidf(texts: list[str]) -> TfidfModel:
    """Build a deterministic TF-IDF model for the given case texts."""
    token_lists = [tokenize(text) for text in texts]
    vocabulary = sorted({token for tokens in token_lists for token in tokens})
    document_frequency: Counter[str] = Counter()
    for tokens in token_lists:
        document_frequency.update(set(tokens))
    count = len(texts)
    idf = {
        term: math.log((1 + count) / (1 + document_frequency[term])) + 1.0
        for term in vocabulary
    }
    vectors: list[dict[str, float]] = []
    for tokens in token_lists:
        counts = Counter(tokens)
        total = sum(counts.values()) or 1
        vectors.append(
            {
                term: (counts[term] / total) * idf[term]
                for term in set(tokens)
            }
        )
    return TfidfModel(vocabulary=vocabulary, idf=idf, vectors=vectors)
