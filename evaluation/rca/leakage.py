"""Overlap/leakage validation for evaluation targets vs. historical memory.

dev50 and smoke3 targets must never be used as their own evaluation
retrieval history. The validator compares target instance ids against a
prepared historical RCA corpus (M8 ``HistoricalCase`` JSONL, keyed by ``id``
or ``instance_id``) and reports any overlap.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_history_instance_ids(path: str | Path) -> set[str]:
    """Load instance ids from a historical-corpus JSONL (``id``/``instance_id``)."""
    history_path = Path(path)
    ids: set[str] = set()
    with history_path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid history record at line {line_no}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise TypeError(f"history record at line {line_no} is not an object")
            case_id = record.get("instance_id", record.get("id"))
            if not isinstance(case_id, str) or not case_id.strip():
                raise ValueError(
                    f"history record at line {line_no} has no instance_id/id"
                )
            ids.add(case_id.strip())
    return ids


def validate_leakage(
    *,
    target_ids: set[str],
    history_ids: set[str],
) -> dict:
    """Validate that evaluation targets do not appear in retrieval history."""
    overlap = sorted(target_ids & history_ids)
    return {
        "ok": not overlap,
        "target_cases": len(target_ids),
        "history_cases": len(history_ids),
        "overlap_count": len(overlap),
        "overlap_ids": overlap,
    }
