"""Workflow module for PatchPilot.

This module provides workflow orchestration components including:
- Failure classification for verification results
- Agent workflow coordination
- Error categorization and routing
"""

from patchpilot.workflow.failure_classifier import (
    FailureType,
    classify_failure,
)

__all__ = [
    "FailureType",
    "classify_failure",
]
