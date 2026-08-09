"""Workflow module for PatchPilot.

This module provides workflow orchestration components including:
- Failure classification for verification results
- Agent workflow coordination
- Error categorization and routing
- Repair loop with early stopping logic
"""

from patchpilot.workflow.failure_classifier import (
    FailureType,
    classify_failure,
)
from patchpilot.workflow.repair_loop import (
    RepairLoop,
    RepairLoopError,
    run_repair_loop,
)

__all__ = [
    "FailureType",
    "RepairLoop",
    "RepairLoopError",
    "classify_failure",
    "run_repair_loop",
]
