"""Runtime verification result contracts."""

from enum import Enum

from pydantic import BaseModel, Field

from roottrace.evidence.schema import (
    MAX_COMMAND_CHARS,
    MAX_EVIDENCE_IDS,
    MAX_EXCERPT_CHARS,
)
from roottrace.incident.schema import StableId


class VerificationOutcome(str, Enum):
    SUPPORTED = "supported"
    REJECTED = "rejected"
    UNVERIFIED = "unverified"


class VerificationStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"


class VerificationResult(BaseModel):
    """Outcome of verifying one hypothesis in the disposable sandbox."""

    id: StableId
    hypothesis_id: StableId
    command: str = Field(max_length=MAX_COMMAND_CHARS)
    status: VerificationStatus
    outcome: VerificationOutcome
    evidence_ids: list[StableId] = Field(default_factory=list, max_length=MAX_EVIDENCE_IDS)
    output_excerpt: str | None = Field(default=None, max_length=MAX_EXCERPT_CHARS)
    exit_code: int | None = None
    duration_seconds: float | None = Field(default=None, ge=0)
