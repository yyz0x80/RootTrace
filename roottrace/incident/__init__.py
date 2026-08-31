"""Incident normalization and deterministic bounded context construction."""

from roottrace.incident.builder import build_incident_context
from roottrace.incident.context import IncidentContext
from roottrace.incident.loader import LoadedIncident, load_incident
from roottrace.incident.schema import (
    DEFAULT_GIT_SEARCH_DEPTHS,
    GitVerificationPolicy,
    IncidentInput,
    ResourceKind,
    ReviewCommentEvidence,
    ReviewCommentLocationMapping,
    ReviewCommentThread,
    ReviewCommentTruncation,
    SourceLocation,
    build_git_verification_policy,
    extract_diff_paths,
)

__all__ = [
    "DEFAULT_GIT_SEARCH_DEPTHS",
    "GitVerificationPolicy",
    "IncidentContext",
    "IncidentInput",
    "LoadedIncident",
    "ResourceKind",
    "ReviewCommentEvidence",
    "ReviewCommentLocationMapping",
    "ReviewCommentThread",
    "ReviewCommentTruncation",
    "SourceLocation",
    "build_git_verification_policy",
    "build_incident_context",
    "extract_diff_paths",
    "load_incident",
]
