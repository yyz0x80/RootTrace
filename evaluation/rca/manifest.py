"""Manifest loading and validation for SWE-bench-derived RCA evaluation.

A manifest pins the cases to evaluate. It must contain unique cases: a case
is identified by its SWE-bench ``instance_id`` and by the ``(repo,
base_commit)`` checkout it targets, so duplicate entries of either kind are
rejected.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

_SHA_PATTERN = re.compile(r"^[0-9a-fA-F]{7,64}$")
_CONTROL_PATTERN = re.compile(r"[\x00-\x1f\x7f]")


class ManifestCase(BaseModel):
    """One pinned evaluation case (public metadata only, no gold fields)."""

    instance_id: str = Field(min_length=1, max_length=200)
    repo: str = Field(min_length=1, max_length=200)
    base_commit: str = Field(min_length=7, max_length=64)

    @field_validator("instance_id", "repo")
    @classmethod
    def _reject_control_chars(cls, value: str) -> str:
        if _CONTROL_PATTERN.search(value):
            raise ValueError("must not contain control characters")
        return value

    @field_validator("repo")
    @classmethod
    def _validate_repo(cls, value: str) -> str:
        if value.strip() != value:
            raise ValueError("repo must not have leading/trailing whitespace")
        parts = value.split("/")
        if len(parts) != 2 or any(not part for part in parts):
            raise ValueError("repo must be an owner/name identifier")
        if any(part in {".", ".."} for part in parts):
            raise ValueError("repo must not contain '.' or '..' segments")
        return value

    @field_validator("base_commit")
    @classmethod
    def _validate_base_commit(cls, value: str) -> str:
        if not _SHA_PATTERN.fullmatch(value):
            raise ValueError("base_commit must be a 7-64 character hexadecimal SHA")
        return value


class RcaManifest(BaseModel):
    """A validated evaluation manifest with unique cases."""

    name: str = Field(min_length=1, max_length=200)
    seed: int | None = None
    instances: list[ManifestCase] = Field(min_length=1)

    @model_validator(mode="after")
    def _reject_duplicate_cases(self) -> RcaManifest:
        seen_ids: dict[str, int] = {}
        seen_checkouts: dict[tuple[str, str], int] = {}
        for index, case in enumerate(self.instances):
            if case.instance_id in seen_ids:
                raise ValueError(
                    f"duplicate instance_id at index {index}: {case.instance_id}"
                )
            seen_ids[case.instance_id] = index
            checkout = (case.repo, case.base_commit)
            if checkout in seen_checkouts:
                raise ValueError(
                    f"duplicate case checkout at index {index}: "
                    f"{case.repo}@{case.base_commit[:12]}"
                )
            seen_checkouts[checkout] = index
        return self


def load_manifest(path: str | Path) -> RcaManifest:
    """Load and validate a manifest file, rejecting duplicate cases."""
    manifest_path = Path(path)
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid manifest JSON {manifest_path.name}: {exc}") from exc
    if not isinstance(raw, dict):
        raise TypeError("manifest must be a JSON object")
    try:
        return RcaManifest(**raw)
    except ValidationError as exc:
        raise ValueError(f"invalid manifest {manifest_path.name}: {exc}") from exc


def manifest_sha256(path: str | Path) -> str:
    """Content hash of the manifest file for deterministic report provenance."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
