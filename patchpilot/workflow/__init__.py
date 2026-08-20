"""Workflow module for PatchPilot.

This module provides workflow orchestration components including:
- Failure classification for verification results
- Agent workflow coordination
- Error categorization and routing
- Repair loop with early stopping logic
- Repair candidate selector for relevance-aware repair
- Complete workflow runner for end-to-end execution
- Structured logging for execute workflow
- Completion state determination
- Execution tracing for workflow analysis
- Permission audit system for security enforcement
"""

from patchpilot.workflow.completion import (
    CompletionDecision,
    determine_completion_state,
)
from patchpilot.workflow.execute_logger import ExecuteLogger
from patchpilot.workflow.failure_classifier import (
    FailureType,
    classify_failure,
)
from patchpilot.workflow.permission_audit import (
    PermissionAuditor,
    PermissionDecision,
    PermissionResult,
    RuleID,
    audit_tool_permission,
)
from patchpilot.workflow.repair_loop import (
    RepairLoop,
    RepairLoopError,
    RepairLoopLimitError,
    RepairLoopStalledError,
    build_failure_repair_prompt,
    run_repair_loop,
)
from patchpilot.workflow.repair_selector import (
    ExcludedFailure,
    RepairCandidate,
    RepairSelection,
    RepairSelector,
)
from patchpilot.workflow.result import WorkflowResult
from patchpilot.workflow.runner import (
    MAX_REPAIR_ATTEMPTS,
    WorkflowExecuteLogCallback,
    WorkflowRunner,
    WorkflowRunnerError,
    WorkflowRunnerExecutionError,
    WorkflowRunnerSetupError,
    run_workflow,
)
from patchpilot.workflow.trace import TraceEvent, TraceWriter

__all__ = [
    "MAX_REPAIR_ATTEMPTS",
    "CompletionDecision",
    "ExcludedFailure",
    "ExecuteLogger",
    "FailureType",
    "PermissionAuditor",
    "PermissionDecision",
    "PermissionResult",
    "RepairCandidate",
    "RepairLoop",
    "RepairLoopError",
    "RepairLoopLimitError",
    "RepairLoopStalledError",
    "RepairSelection",
    "RepairSelector",
    "RuleID",
    "TraceEvent",
    "TraceWriter",
    "WorkflowExecuteLogCallback",
    "WorkflowResult",
    "WorkflowRunner",
    "WorkflowRunnerError",
    "WorkflowRunnerExecutionError",
    "WorkflowRunnerSetupError",
    "audit_tool_permission",
    "build_failure_repair_prompt",
    "classify_failure",
    "determine_completion_state",
    "run_repair_loop",
    "run_workflow",
]
