"""Deterministic, bounded context schemas for RootTrace RCA runs.

The context prepared before any model call is a typed snapshot: repository
fingerprint, bounded file inventory, extracted search signals, ranked source
snippets, and explicit truncation metadata. Identical inputs produce stable
output; nothing here carries whole-repo or unbounded history content.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import (
    BaseModel,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from patchpilot.rca.schema import IncidentInput, validate_relative_path

MAX_FINGERPRINT_CHARS = 8_000
MAX_DIFF_EXCERPT_CHARS = 20_000
MAX_SNIPPET_CHARS = 3_000
MAX_NOTE_CHARS = 300
MAX_SNIPPETS = 50


class RepositoryFingerprint(BaseModel):
    """Read-only snapshot proving the target repository is unchanged."""

    head_sha: str = Field(max_length=64)
    status_porcelain: str = Field(max_length=MAX_FINGERPRINT_CHARS)
    diff_stat: str = Field(max_length=MAX_FINGERPRINT_CHARS)


class RepositoryInventory(BaseModel):
    """Bounded, deterministic inventory of tracked repository files."""

    base_commit: str = Field(max_length=64)
    tracked_files: int = Field(ge=0)
    python_files: int = Field(ge=0)
    test_files: int = Field(ge=0)
    config_files: int = Field(ge=0)
    python_file_list: list[str] = Field(default_factory=list, max_length=500)
    test_file_list: list[str] = Field(default_factory=list, max_length=200)
    config_file_list: list[str] = Field(default_factory=list, max_length=100)
    python_files_omitted: int = Field(default=0, ge=0)
    test_files_omitted: int = Field(default=0, ge=0)
    config_files_omitted: int = Field(default=0, ge=0)

    @field_validator("python_file_list", "test_file_list", "config_file_list")
    @classmethod
    def _validate_paths(cls, values: list[str]) -> list[str]:
        for value in values:
            validate_relative_path(value)
        return values


class IncidentSignals(BaseModel):
    """Deterministic search signals extracted from the incident."""

    terms: list[str] = Field(default_factory=list, max_length=20)
    exception_names: list[str] = Field(default_factory=list, max_length=10)
    stack_symbols: list[str] = Field(default_factory=list, max_length=10)
    diff_paths: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("terms", "exception_names", "stack_symbols")
    @classmethod
    def _validate_signal_text(cls, values: list[str]) -> list[str]:
        for value in values:
            if not value or len(value) > 200:
                raise ValueError("signal text must be non-empty and bounded")
        return values

    @field_validator("diff_paths")
    @classmethod
    def _validate_diff_paths(cls, values: list[str]) -> list[str]:
        for value in values:
            validate_relative_path(value)
        return values


class SourceSnippet(BaseModel):
    """One bounded, ranked source excerpt prepared for the model."""

    path: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    excerpt: str = Field(max_length=MAX_SNIPPET_CHARS + 64)
    score: int = Field(ge=0)
    matched_terms: list[str] = Field(default_factory=list, max_length=20)
    rank: int = Field(ge=1)

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        return validate_relative_path(value)

    @model_validator(mode="after")
    def _validate_lines(self) -> SourceSnippet:
        if self.end_line < self.start_line:
            raise ValueError("end_line must be >= start_line")
        return self


class ContextTruncation(BaseModel):
    """Visible truncation metadata for one deterministic context build."""

    issue_body_chars_omitted: int = Field(default=0, ge=0)
    title_chars_omitted: int = Field(default=0, ge=0)
    stack_trace_chars_omitted: int = Field(default=0, ge=0)
    ci_log_chars_omitted: int = Field(default=0, ge=0)
    diff_chars_omitted: int = Field(default=0, ge=0)
    terms_omitted: int = Field(default=0, ge=0)
    exception_names_omitted: int = Field(default=0, ge=0)
    stack_symbols_omitted: int = Field(default=0, ge=0)
    diff_paths_omitted: int = Field(default=0, ge=0)
    python_files_omitted: int = Field(default=0, ge=0)
    test_files_omitted: int = Field(default=0, ge=0)
    config_files_omitted: int = Field(default=0, ge=0)
    snippet_candidates_omitted: int = Field(default=0, ge=0)
    snippets_omitted: int = Field(default=0, ge=0)
    snippet_excerpt_chars_omitted: int = Field(default=0, ge=0)
    notes: list[Annotated[str, StringConstraints(max_length=MAX_NOTE_CHARS)]] = Field(
        default_factory=list,
        max_length=20,
    )


class IncidentContext(BaseModel):
    """Deterministic, bounded context prepared before any model call."""

    incident: IncidentInput
    repository: RepositoryInventory
    signals: IncidentSignals
    snippets: list[SourceSnippet] = Field(default_factory=list, max_length=MAX_SNIPPETS)
    diff_excerpt: str | None = Field(
        default=None,
        max_length=MAX_DIFF_EXCERPT_CHARS + 64,
    )
    truncation: ContextTruncation = Field(default_factory=ContextTruncation)
    fingerprint: RepositoryFingerprint

    @model_validator(mode="after")
    def _validate_ranks(self) -> IncidentContext:
        ranks = [snippet.rank for snippet in self.snippets]
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError("snippet ranks must be contiguous starting at 1")
        return self
