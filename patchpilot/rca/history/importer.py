"""JSONL importer for the shared historical RCA corpus.

Leakage protection happens at import:
- evaluation-target, duplicate, and linked case ids are excluded via
  ``excluded_ids``;
- duplicate case ids keep only the first occurrence;
- ``gold_patch`` / ``test_patch`` fields are dropped before validation and
  are never persisted anywhere in the corpus;
- malformed lines are counted and skipped explicitly, never hidden.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from patchpilot.rca.history.schema import HistoricalCase, HistoricalCorpus

_SENSITIVE_FIELDS = ("gold_patch", "test_patch")


def import_corpus(
    path: str | Path,
    *,
    split: str = "development",
    excluded_ids: set[str] | frozenset[str] = frozenset(),
    max_cases: int = 5_000,
) -> HistoricalCorpus:
    """Import a JSONL corpus with leakage guards and import metadata."""
    if max_cases <= 0:
        raise ValueError("max_cases must be positive")
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"corpus file not found: {source}")
    excluded = set(excluded_ids)
    seen: set[str] = set()
    cases: list[HistoricalCase] = []
    excluded_count = 0
    duplicate_count = 0
    malformed_count = 0
    gold_patch_dropped_count = 0
    truncated_count = 0

    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if len(cases) >= max_cases:
                truncated_count += 1
                continue
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError:
                malformed_count += 1
                continue
            if not isinstance(raw, dict):
                malformed_count += 1
                continue
            case_id = raw.get("id")
            if not isinstance(case_id, str) or not case_id:
                malformed_count += 1
                continue
            if case_id in excluded:
                excluded_count += 1
                continue
            if case_id in seen:
                duplicate_count += 1
                continue
            if any(field in raw for field in _SENSITIVE_FIELDS):
                gold_patch_dropped_count += 1
                for field in _SENSITIVE_FIELDS:
                    raw.pop(field, None)
            try:
                case = HistoricalCase.model_validate(raw)
            except ValidationError:
                malformed_count += 1
                continue
            seen.add(case_id)
            cases.append(case)

    cases.sort(key=lambda case: case.id)
    checksum = _corpus_checksum(cases)
    notes: list[str] = []
    if truncated_count:
        notes.append(f"{truncated_count} lines truncated by max_cases")
    if gold_patch_dropped_count:
        notes.append("gold/test patch fields were dropped and never persisted")
    mtime = datetime.fromtimestamp(source.stat().st_mtime, tz=UTC).isoformat()
    return HistoricalCorpus(
        split=split,
        source_path=source.name,
        source_timestamp=mtime,
        checksum=checksum,
        cases=cases,
        imported_count=len(cases),
        excluded_count=excluded_count,
        duplicate_count=duplicate_count,
        malformed_count=malformed_count,
        gold_patch_dropped_count=gold_patch_dropped_count,
        notes=notes,
    )


def _corpus_checksum(cases: list[HistoricalCase]) -> str:
    payload = json.dumps(
        [case.model_dump(mode="json", exclude={"source"}) for case in cases],
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
