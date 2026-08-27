"""Typed contracts for the shared historical RCA memory."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
)

from roottrace.runtime.paths import validate_relative_path

MAX_CASE_TITLE_CHARS = 200
MAX_CASE_PROBLEM_CHARS = 20_000
MAX_CASE_SUMMARY_CHARS = 2_000
MAX_CASE_SOURCE_CHARS = 1_000
MAX_CORPUS_CASES = 5_000
MAX_LOCATIONS_PER_CASE = 20
MAX_LINKED_IDS = 20
MAX_RETRIEVED = 10
MAX_HINT_NOTES = 10

CaseId = Annotated[
    str,
    StringConstraints(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$",
        max_length=128,
    ),
]
_ISO_TS_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?$"
)


def parse_timestamp(value: str | None) -> datetime | None:
    """Parse an ISO timestamp; naive values are treated as UTC."""
    if value is None:
        return None
    if not _ISO_TS_PATTERN.fullmatch(value.strip()):
        raise ValueError(f"invalid ISO timestamp: {value}")
    parsed = datetime.fromisoformat(value.strip().replace(" ", "T"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


class HistoricalCase(BaseModel):
    """One historical RCA case; gold patches are never persisted."""

    model_config = ConfigDict(extra="forbid")

    id: CaseId
    repo: str = Field(max_length=MAX_CASE_SOURCE_CHARS)
    title: str | None = Field(default=None, max_length=MAX_CASE_TITLE_CHARS)
    problem: str = Field(max_length=MAX_CASE_PROBLEM_CHARS)
    resolved_at: str | None = Field(default=None, max_length=64)
    source: str = Field(max_length=MAX_CASE_SOURCE_CHARS)
    locations: list[str] = Field(
        default_factory=list,
        max_length=MAX_LOCATIONS_PER_CASE,
    )
    summary: str | None = Field(default=None, max_length=MAX_CASE_SUMMARY_CHARS)
    linked_ids: list[CaseId] = Field(default_factory=list, max_length=MAX_LINKED_IDS)

    @field_validator("resolved_at")
    @classmethod
    def _validate_resolved_at(cls, value: str | None) -> str | None:
        if value is not None:
            parse_timestamp(value)
        return value

    @field_validator("locations")
    @classmethod
    def _validate_locations(cls, values: list[str]) -> list[str]:
        return [validate_relative_path(value) for value in values]


class HistoricalCorpus(BaseModel):
    """Validated, deduplicated historical corpus with import metadata."""

    split: str = Field(max_length=100)
    source_path: str = Field(max_length=MAX_CASE_SOURCE_CHARS)
    source_timestamp: str | None = Field(default=None, max_length=64)
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    cases: list[HistoricalCase] = Field(
        default_factory=list,
        max_length=MAX_CORPUS_CASES,
    )
    imported_count: int = Field(default=0, ge=0)
    excluded_count: int = Field(default=0, ge=0)
    duplicate_count: int = Field(default=0, ge=0)
    malformed_count: int = Field(default=0, ge=0)
    gold_patch_dropped_count: int = Field(default=0, ge=0)
    notes: list[str] = Field(default_factory=list, max_length=20)


class HistoryIndex(BaseModel):
    """Persisted index metadata: checksum, split ids, buckets, centroids."""

    corpus_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed: int = Field(ge=0)
    n_clusters: int | None = Field(default=None, ge=1)
    vocabulary_size: int = Field(ge=0)
    case_ids: list[CaseId] = Field(default_factory=list)
    bucket_assignments: dict[str, int] | None = None
    cluster_centers: list[list[float]] | None = None
    built_at: str | None = Field(default=None, max_length=64)


class RetrievedCase(BaseModel):
    """One bounded retrieval result with similarity and hint locations."""

    id: CaseId
    similarity: float = Field(ge=0, le=1)
    locations: list[str] = Field(
        default_factory=list,
        max_length=MAX_LOCATIONS_PER_CASE,
    )
    summary: str | None = Field(default=None, max_length=MAX_CASE_SUMMARY_CHARS)
    resolved_at: str | None = Field(default=None, max_length=64)
    source: str = Field(max_length=MAX_CASE_SOURCE_CHARS)


class RetrievalHints(BaseModel):
    """Bounded, auditable retrieval output attached to an RCA run."""

    mode: Literal["off", "clustered", "flat"]
    top_k: int = Field(ge=0, le=MAX_RETRIEVED)
    results: list[RetrievedCase] = Field(
        default_factory=list,
        max_length=MAX_RETRIEVED,
    )
    candidate_count: int = Field(default=0, ge=0)
    index_checksum: str = Field(default="", max_length=64)
    notes: list[str] = Field(default_factory=list, max_length=MAX_HINT_NOTES)
