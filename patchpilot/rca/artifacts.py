"""Deterministic artifact persistence for RootTrace RCA runs.

All artifacts are written as JSON inside one configured output directory.
Artifact names are validated so writes can never escape that directory.
"""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import BaseModel

ARTIFACT_INCIDENT = "incident.json"
ARTIFACT_INVESTIGATION_PLAN = "investigation_plan.json"
ARTIFACT_EVIDENCE_GRAPH = "evidence_graph.json"
ARTIFACT_HYPOTHESES = "hypotheses.json"
ARTIFACT_VERIFICATION = "verification.json"
ARTIFACT_RCA_REPORT = "rca_report.json"

_SAFE_NAME_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-/"
)


class ArtifactError(ValueError):
    """Raised when an artifact name or target path is not safe."""


def model_to_json(model: BaseModel) -> str:
    """Serialize a model deterministically (sorted keys, stable indentation)."""
    payload = json.dumps(
        model.model_dump(mode="json"),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    )
    return payload + "\n"


def validate_artifact_name(name: str) -> str:
    """Validate an artifact name as a safe relative path."""
    if not name or name != name.strip():
        raise ArtifactError("artifact name must be a non-empty, trimmed string")
    if name.startswith("/") or name.endswith("/"):
        raise ArtifactError(
            "artifact name must be relative with no leading or trailing slash"
        )
    if "\\" in name:
        raise ArtifactError("artifact name must use forward slashes")
    if any(char not in _SAFE_NAME_CHARS for char in name):
        raise ArtifactError("artifact name contains unsafe characters")
    try:
        path = PurePosixPath(name)
    except ValueError as exc:
        raise ArtifactError("artifact name is not a valid path") from exc
    if (
        path.is_absolute()
        or not path.parts
        or any(part in (".", "..") for part in path.parts)
    ):
        raise ArtifactError("artifact name must stay inside the output directory")
    return name


class ArtifactWriter:
    """Write RCA models as deterministic JSON inside one output directory."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir.expanduser().resolve()

    def write_model(self, name: str, model: BaseModel) -> Path:
        """Write a Pydantic model artifact and return its path."""
        return self.write_dict(name, model.model_dump(mode="json"))

    def write_dict(self, name: str, data: dict[str, Any]) -> Path:
        """Write a plain dict as a deterministic JSON artifact."""
        safe_name = validate_artifact_name(name)
        target = (self.output_dir / safe_name).resolve()
        try:
            target.relative_to(self.output_dir)
        except ValueError as exc:
            raise ArtifactError(
                f"artifact path escapes output directory: {name}"
            ) from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False)
        target.write_text(payload + "\n", encoding="utf-8")
        return target


def write_artifact(output_dir: Path, name: str, model: BaseModel) -> Path:
    """Write one model artifact into ``output_dir`` and return its path."""
    return ArtifactWriter(output_dir).write_model(name, model)
