"""Reusable RCA job entry point for RQ workers.

One complete RCA workflow is one job: the Lead agent, the three evidence
Specialists, runtime verification, and the repair loop all execute inside
``run_rca_job`` through the existing synchronous pipeline. The function takes
only plain, JSON-serializable arguments and returns a JSON-serializable
summary, so a native RQ worker can call it without a custom worker framework.

Job-level retries stay opt-in at enqueue time because an RQ retry re-runs the
whole workflow. Transient LLM failures propagate unchanged so RQ can spend that
retry budget, while deterministic failures withdraw the budget first.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from roottrace.agents import ProviderProtocol
from roottrace.cli import run_rca_pipeline
from roottrace.incident.loader import load_incident
from roottrace.llm.errors import is_non_retryable_llm_error
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

    Transient LLM failures (rate limits, timeouts, connection errors, 5xx) and
    unknown infrastructure failures propagate unchanged, so an RQ retry policy
    stays usable. Deterministic LLM failures (bad credentials, malformed
    request, exhausted quota) clear the job retry budget before propagating:
    retrying them can never succeed.

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
    try:
        return _execute_rca_job(
            repo,
            issue,
            output_dir,
            model=model,
            stack_trace=stack_trace,
            ci_log=ci_log,
            pr_diff=pr_diff,
        )
    except Exception as error:
        _withdraw_rq_retries_for(error)
        raise


def _execute_rca_job(
    repo: str,
    issue: str,
    output_dir: str,
    *,
    model: str | None = None,
    stack_trace: str | None = None,
    ci_log: str | None = None,
    pr_diff: str | None = None,
) -> RcaJobResult:
    """Run the synchronous RCA pipeline for one job."""
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


def _withdraw_rq_retries_for(error: BaseException) -> None:
    """Clear the RQ retry budget when the failure cannot succeed on a retry.

    RQ decides retries from ``Job.retries_left`` alone and offers no
    exception-type filter, so the job itself must withdraw the budget for
    deterministic LLM failures. Transient and unknown failures keep theirs,
    which is what keeps worker crashes and infrastructure faults retryable.
    Outside an RQ worker there is no current job, so this is a no-op.
    """
    if not is_non_retryable_llm_error(error):
        return

    from rq import get_current_job

    job = get_current_job()
    if job is not None:
        job.retries_left = 0


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
