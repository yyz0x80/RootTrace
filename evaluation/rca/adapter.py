"""SWE-bench Verified public metadata -> RootTrace incident adapter.

The adapter is the only bridge between SWE-bench case metadata and the RCA
runtime. It forwards exactly four pieces of information to RootTrace:

- ``instance_id``
- ``repo``
- ``base_commit``
- ``problem_statement``

Gold patch, test patch, FAIL_TO_PASS/PASS_TO_PASS, PR URL, and fixing-commit
metadata never reach RootTrace: the public-case model forbids unknown fields,
so any gold record fails validation instead of leaking into the RCA run.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from patchpilot.rca.schema import MAX_PROBLEM_CHARS, IncidentInput, Provenance

# Fields that must never be forwarded to RootTrace. The model's
# ``extra="forbid"`` rejects any unknown field; this list documents the known
# gold/PR/fix metadata explicitly.
FORBIDDEN_INPUT_FIELDS = (
    "patch",
    "test_patch",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
    "pr_url",
    "fix_commit",
    "fix_patch",
)

_OMITTED_MARKER = "\n...[truncated: {n} chars omitted]"
_SHA_PATTERN = re.compile(r"^[0-9a-fA-F]{7,64}$")
_CONTROL_PATTERN = re.compile(r"[\x00-\x1f\x7f]")


class PublicCase(BaseModel):
    """SWE-bench Verified public metadata; gold fields are rejected.

    Only ``instance_id``, ``repo``, ``base_commit``, and ``problem_statement``
    are forwarded to RootTrace. The remaining public metadata fields
    (``created_at``, ``difficulty``, ``version``) are accepted for provenance
    but never forwarded.
    """

    model_config = ConfigDict(extra="forbid")

    instance_id: str = Field(min_length=1, max_length=200)
    repo: str = Field(min_length=1, max_length=200)
    base_commit: str = Field(min_length=7, max_length=64)
    problem_statement: str = Field(min_length=1)
    created_at: str | None = Field(default=None, max_length=64)
    difficulty: str | None = Field(default=None, max_length=64)
    version: str | None = Field(default=None, max_length=64)

    @field_validator("instance_id", "repo", "created_at", "difficulty", "version")
    @classmethod
    def _reject_control_chars(cls, value: str | None) -> str | None:
        if value is not None and _CONTROL_PATTERN.search(value):
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


class AdapterResult(BaseModel):
    """Validated RootTrace incident plus truncation bookkeeping."""

    incident: IncidentInput
    problem_chars_omitted: int = Field(default=0, ge=0)
    notes: list[str] = Field(default_factory=list, max_length=10)


def _truncate_problem(problem: str) -> tuple[str, int]:
    if len(problem) <= MAX_PROBLEM_CHARS:
        return problem, 0
    omitted = len(problem) - MAX_PROBLEM_CHARS
    marker = _OMITTED_MARKER.format(n=omitted)
    keep = max(0, MAX_PROBLEM_CHARS - len(marker))
    return problem[:keep] + marker, omitted


def build_incident_input(case: PublicCase) -> AdapterResult:
    """Convert public SWE-bench metadata into a validated RootTrace incident."""
    problem, omitted = _truncate_problem(case.problem_statement)
    notes = (
        [f"problem statement truncated ({omitted} chars omitted)"]
        if omitted
        else []
    )
    incident = IncidentInput(
        id=case.instance_id,
        repo=case.repo,
        base_commit=case.base_commit,
        problem=problem,
        provenance=Provenance(
            source=f"swebench:{case.instance_id}",
            tool="swebench_adapter",
            commit=case.base_commit,
        ),
    )
    return AdapterResult(
        incident=incident,
        problem_chars_omitted=omitted,
        notes=notes,
    )


def case_from_public_record(record: dict) -> PublicCase:
    """Build a public case from one JSONL record.

    Raises ``ValueError`` for any unknown field, so a gold record containing
    ``patch``/``test_patch``/FAIL_TO_PASS/PASS_TO_PASS is rejected instead of
    leaking into the RCA runtime.
    """
    try:
        return PublicCase(**record)
    except ValidationError as exc:
        raise ValueError(f"public metadata contains disallowed fields: {exc}") from exc


def load_public_cases(path: str | Path) -> dict[str, PublicCase]:
    """Load public SWE-bench metadata, rejecting gold fields and duplicates."""
    public_path = Path(path)
    cases: dict[str, PublicCase] = {}
    with public_path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                case = case_from_public_record(record)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    f"invalid public record at line {line_no}: {exc}"
                ) from exc
            if case.instance_id in cases:
                raise ValueError(f"duplicate public instance_id: {case.instance_id}")
            cases[case.instance_id] = case
    return cases


def write_root_trace_input(incident: IncidentInput, path: str | Path) -> None:
    """Persist exactly the RootTrace input as a four-field JSON snapshot."""
    payload = {
        "instance_id": incident.id,
        "repo": incident.repo,
        "base_commit": incident.base_commit,
        "problem_statement": incident.problem,
    }
    Path(path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
