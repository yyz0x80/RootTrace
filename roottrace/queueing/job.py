"""Reusable RCA job entry point for RQ workers.

One complete RCA workflow is one job: the Lead agent, the three evidence
Specialists, runtime verification, and the repair loop all execute inside
``run_rca_job`` through the existing synchronous pipeline. The function takes
only plain, JSON-serializable arguments and returns a JSON-serializable
summary, so a native RQ worker can call it without a custom worker framework.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from roottrace.agents import ProviderProtocol
from roottrace.cli import run_rca_pipeline
from roottrace.incident.loader import load_incident
from roottrace.llm.provider import create_provider_from_config


class RcaJobResult(TypedDict):
    """JSON-serializable summary returned by one RCA job."""

    incident_id: str
    status: str
    conclusion: str
    output_dir: str
    artifacts: list[str]


def run_rca_job(
    repo: str,
    issue: str,
    output_dir: str,
    *,
    model: str | None = None,
    stack_trace: str | None = None,
    ci_log: str | None = None,
    pr_diff: str | None = None,
) -> RcaJobResult:
    """Run one complete RCA workflow and return its artifact summary.

    Args:
        repo: Path to the target repository (always treated as read-only).
        issue: Path to the local Issue Markdown/JSON file.
        output_dir: Directory that receives this job's RCA artifacts.
        model: Optional model name or model identifier from project config.
        stack_trace: Optional stack trace file.
        ci_log: Optional CI log file.
        pr_diff: Optional PR diff/context file.

    Returns:
        A JSON-serializable summary of the completed RCA run.
    """
    loaded = load_incident(
        issue_path=issue,
        repo_path=repo,
        stack_trace_path=stack_trace,
        ci_log_path=ci_log,
        pr_diff_path=pr_diff,
    )
    log_sources = _log_sources(ci_log=ci_log, stack_trace=stack_trace)

    def provider_factory() -> ProviderProtocol:
        return create_provider_from_config(model_name=model)

    result = run_rca_pipeline(
        loaded,
        repo,
        output_dir,
        provider_factory=provider_factory,
        log_sources=log_sources,
    )
    return {
        "incident_id": result.incident_id,
        "status": result.run.status.value,
        "conclusion": result.report.conclusion.value,
        "output_dir": result.output_dir,
        "artifacts": list(result.artifacts),
    }


def _log_sources(
    *,
    ci_log: str | None,
    stack_trace: str | None,
) -> dict[str, str]:
    """Stage optional external logs under the same names as the CLI."""
    sources: dict[str, str] = {}
    if ci_log:
        sources["ci.log"] = str(Path(ci_log))
    if stack_trace:
        sources["stack_trace.log"] = str(Path(stack_trace))
    return sources
