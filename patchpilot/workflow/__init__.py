"""Workflow module for PatchPilot.

This module provides workflow orchestration components including:
- Failure classification for verification results
- Agent workflow coordination
- Error categorization and routing
- Repair loop with early stopping logic
- Complete workflow runner for end-to-end execution
"""

from patchpilot.workflow.failure_classifier import (
    FailureType,
    classify_failure,
)
from patchpilot.workflow.repair_loop import (
    RepairLoop,
    RepairLoopError,
    RepairLoopLimitError,
    RepairLoopStalledError,
    run_repair_loop,
)
from patchpilot.workflow.runner import (
    MAX_REPAIR_ATTEMPTS,
    WorkflowRunner,
    WorkflowRunnerError,
    WorkflowRunnerExecutionError,
    WorkflowRunnerSetupError,
    run_workflow,
)

__all__ = [
    "MAX_REPAIR_ATTEMPTS",
    "FailureType",
    "RepairLoop",
    "RepairLoopError",
    "RepairLoopLimitError",
    "RepairLoopStalledError",
    "WorkflowRunner",
    "WorkflowRunnerError",
    "WorkflowRunnerExecutionError",
    "WorkflowRunnerSetupError",
    "classify_failure",
    "run_repair_loop",
    "run_workflow",
]
