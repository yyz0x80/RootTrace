"""Repository operations module for PatchPilot.

This module provides functionality for validating and inspecting
target repositories before PatchPilot operations.
"""

from patchpilot.repository.analyzer import analyze_repository
from patchpilot.repository.preflight import (
    RepositoryPreflightError,
    validate_repository,
)
from patchpilot.repository.schema import (
    RepositoryContext,
    RepositoryPreflightResult,
)

__all__ = [
    "RepositoryContext",
    "RepositoryPreflightError",
    "RepositoryPreflightResult",
    "analyze_repository",
    "validate_repository",
]
