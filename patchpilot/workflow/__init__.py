"""Workflow module for PatchPilot.

This module provides workflow orchestration components including:
- Failure classification for verification results
- Agent workflow coordination
- Error categorization and routing
- Repair loop with early stopping logic
- Complete workflow runner for end-to-end execution
- Structured logging for execute workflow
- Completion state determination
"""

from patchpilot.workflow.completion import determine_completion_state
from patchpilot.workflow.execute_logger import ExecuteLogger
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
    WorkflowExecuteLogCallback,
    WorkflowRunner,
    WorkflowRunnerError,
    WorkflowRunnerExecutionError,
    WorkflowRunnerSetupError,
    run_workflow,
)

__all__ = [
    "MAX_REPAIR_ATTEMPTS",
    "ExecuteLogger",
    "FailureType",
    "RepairLoop",
    "RepairLoopError",
    "RepairLoopLimitError",
    "RepairLoopStalledError",
    "WorkflowExecuteLogCallback",
    "WorkflowRunner",
    "WorkflowRunnerError",
    "WorkflowRunnerExecutionError",
    "WorkflowRunnerSetupError",
    "classify_failure",
    "determine_completion_state",
    "run_repair_loop",
    "run_workflow",
]
