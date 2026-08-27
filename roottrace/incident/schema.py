"""Public input contracts for normalized RootTrace incidents."""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints, field_validator

MAX_TITLE_CHARS = 200
MAX_PROBLEM_CHARS = 20_000
MAX_LOG_CHARS = 20_000
MAX_DIFF_CHARS = 100_000
MAX_SOURCE_CHARS = 1_000
MAX_TOOL_CHARS = 200
MAX_COMMAND_CHARS = 500
MAX_LOGS = 10

StableId = Annotated[
    str,
    StringConstraints(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$",
        max_length=128,
    ),
]
BoundedLog = Annotated[str, StringConstraints(max_length=MAX_LOG_CHARS)]

_SHA_PATTERN = re.compile(r"^[0-9a-fA-F]{7,64}$")


def validate_commit_sha(value: str) -> str:
    if not _SHA_PATTERN.fullmatch(value):
        raise ValueError("commit must be a 7-64 character hexadecimal SHA")
    return value


class Provenance(BaseModel):
    """Reproducible origin of an evidence item or incident input."""

    source: str = Field(max_length=MAX_SOURCE_CHARS)
    tool: str | None = Field(default=None, max_length=MAX_TOOL_CHARS)
    command: str | None = Field(default=None, max_length=MAX_COMMAND_CHARS)
    commit: str | None = None

    @field_validator("commit")
    @classmethod
    def _validate_commit(cls, value: str | None) -> str | None:
        if value is not None:
            validate_commit_sha(value)
        return value


class IncidentInput(BaseModel):
    """Normalized incident input for one RootTrace run."""

    id: StableId
    repo: str = Field(max_length=MAX_SOURCE_CHARS)
    base_commit: str
    title: str | None = Field(default=None, max_length=MAX_TITLE_CHARS)
    problem: str = Field(max_length=MAX_PROBLEM_CHARS)
    logs: list[BoundedLog] = Field(default_factory=list, max_length=MAX_LOGS)
    diff: str | None = Field(default=None, max_length=MAX_DIFF_CHARS)
    provenance: Provenance

    @field_validator("repo")
    @classmethod
    def _validate_repo(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("repo must be a non-empty repository identifier")
        if "\\" in value or re.match(r"^[A-Za-z]:", value):
            raise ValueError("repo must be a repository identifier, not a host path")
        try:
            path = PurePosixPath(value)
        except ValueError as exc:
            raise ValueError("invalid repository identifier") from exc
        if (
            path.is_absolute()
            or not path.parts
            or any(part in (".", "..") for part in path.parts)
        ):
            raise ValueError("repo must not contain '.' or '..' segments")
        return value

    @field_validator("base_commit")
    @classmethod
    def _validate_base_commit(cls, value: str) -> str:
        return validate_commit_sha(value)
