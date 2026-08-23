"""``patchpilot rca`` command: local RootTrace RCA runs.

The command accepts a local repository, a GitHub Issue-style Markdown/JSON
file, a model name, an output directory, and optional stack trace / CI log /
PR diff files. It runs the full local pipeline:

Issue -> Specialists -> Evidence -> Hypotheses -> Verification -> RCA Report.

The analyzed repository is always read-only; runtime tests execute only in a
disposable sandbox copy, and every artifact is written inside the configured
output directory.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from patchpilot.provider import create_provider_from_config
from patchpilot.rca.agents import PlanError, ProviderProtocol
from patchpilot.rca.artifacts import (
    ARTIFACT_EVIDENCE_GRAPH,
    ARTIFACT_HYPOTHESES,
    ARTIFACT_INCIDENT,
    ARTIFACT_INVESTIGATION_PLAN,
    ARTIFACT_RCA_REPORT,
    ARTIFACT_VERIFICATION,
    ArtifactWriter,
)
from patchpilot.rca.incident_loader import LoadedIncident, load_incident
from patchpilot.rca.orchestrator import RcaOrchestrator, RcaRunResult
from patchpilot.rca.renderer import render_rca_markdown
from patchpilot.rca.sandbox import RuntimeVerificationSandbox
from patchpilot.rca.schema import AgentRole, EvidenceGraph, PlanBudgets, RCAReport
from patchpilot.rca.synthesis import LeadSynthesizer, SynthesisError
from patchpilot.rca.tools import RcaToolRegistry
from patchpilot.rca.usage import UsageTracker
from patchpilot.rca.verification import RuntimeTestVerifier, VerificationRun
from patchpilot.workflow.trace import TraceEvent, TraceWriter
from patchpilot.workspace import Workspace

_TRACE_STAGE = "RCA"
_RUN_SUMMARY = "run_summary.json"
_EXECUTION_TRACE = "execution_trace.jsonl"


class RcaCliResult(BaseModel):
    """Typed outcome of one ``patchpilot rca`` pipeline run."""

    incident_id: str
    run: RcaRunResult
    verification: VerificationRun
    report: RCAReport
    output_dir: str
    artifacts: list[str] = Field(default_factory=list)


def add_rca_subparser(subparsers: Any) -> None:
    """Register the ``rca`` subcommand on the PatchPilot CLI parser."""
    parser = subparsers.add_parser(
        "rca",
        help="Run RootTrace RCA on a local repository",
    )
    parser.add_argument("--repo", required=True, help="Path to the target repository")
    parser.add_argument(
        "--issue",
        required=True,
        help="Path to the Issue Markdown/JSON file",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name from config file or direct model identifier",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for RCA artifacts",
    )
    parser.add_argument(
        "--stack-trace",
        default=None,
        help="Optional stack trace file",
    )
    parser.add_argument(
        "--ci-log",
        default=None,
        help="Optional CI log file",
    )
    parser.add_argument(
        "--pr-diff",
        default=None,
        help="Optional PR diff/context file",
    )


def run_rca_command(args: argparse.Namespace) -> int:
    """Run the RCA pipeline from parsed CLI arguments."""
    try:
        loaded = load_incident(
            issue_path=args.issue,
            repo_path=args.repo,
            stack_trace_path=args.stack_trace,
            ci_log_path=args.ci_log,
            pr_diff_path=args.pr_diff,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"RCA input error: {exc}", file=sys.stderr)
        return 2

    def provider_factory() -> ProviderProtocol:
        return create_provider_from_config(model_name=args.model)

    log_sources: dict[str, Path] = {}
    if args.ci_log:
        log_sources["ci.log"] = Path(args.ci_log)
    if args.stack_trace:
        log_sources["stack_trace.log"] = Path(args.stack_trace)
    try:
        result = run_rca_pipeline(
            loaded,
            args.repo,
            args.output_dir,
            provider_factory=provider_factory,
            log_sources=log_sources,
        )
    except (PlanError, SynthesisError, ValueError, RuntimeError) as exc:
        print(f"RCA run failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"RCA complete: incident={result.incident_id} "
        f"status={result.run.status.value} output={result.output_dir}"
    )
    return 0


def run_rca_pipeline(
    loaded: LoadedIncident,
    repo: str | Path,
    output_dir: str | Path,
    *,
    provider_factory: Callable[[], ProviderProtocol],
    budgets: PlanBudgets | None = None,
    log_sources: dict[str, str | Path] | None = None,
    worker_concurrency: int = 3,
    enabled_roles: frozenset[AgentRole] | None = None,
    retriever: Any | None = None,
    retrieval_mode: str = "off",
    history_excluded_ids: frozenset[str] = frozenset(),
) -> RcaCliResult:
    """Run the full local RCA pipeline and persist all artifacts."""
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    budgets = budgets or PlanBudgets()
    incident = loaded.incident
    repo_path = Path(repo).resolve()
    external_root = _stage_external_logs(output, log_sources or {})

    registry = RcaToolRegistry(
        Workspace(repo_path),
        external_root=external_root,
    )
    orchestrator = RcaOrchestrator(
        lead_provider=provider_factory(),
        issue_ci_provider=provider_factory(),
        code_provider=provider_factory(),
        git_history_provider=provider_factory(),
        registry=registry,
        budgets=budgets,
        worker_concurrency=worker_concurrency,
        enabled_roles=enabled_roles,
        retriever=retriever,
        retrieval_mode=retrieval_mode,
        history_excluded_ids=history_excluded_ids,
    )
    run_result = orchestrator.run(loaded, repo, output_dir=output)

    trace = TraceWriter(output / _EXECUTION_TRACE)
    _trace(trace, incident.id, "verification_start", run_result)
    verification = _run_verification(
        repo_path,
        loaded,
        run_result.graph,
    )
    _trace(trace, incident.id, "verification_end", run_result)

    _trace(trace, incident.id, "synthesis_start", run_result)
    synthesizer = LeadSynthesizer(
        provider=provider_factory(),
        usage=UsageTracker(),
    )
    report = synthesizer.synthesize(verification.graph, verification)
    _trace(trace, incident.id, "synthesis_end", run_result)

    writer = ArtifactWriter(output)
    writer.write_model(ARTIFACT_INCIDENT, incident)
    writer.write_model(ARTIFACT_INVESTIGATION_PLAN, run_result.plan)
    writer.write_model(ARTIFACT_EVIDENCE_GRAPH, verification.graph)
    _persist_hypotheses(writer, verification.graph)
    _persist_verification(writer, verification)
    writer.write_model(ARTIFACT_RCA_REPORT, report)
    (output / "rca_report.md").write_text(
        render_rca_markdown(report),
        encoding="utf-8",
    )
    _persist_agent_artifacts(writer, verification.graph)
    artifacts = _persist_run_summary(
        writer,
        output,
        run_result,
        verification,
        report,
        synthesizer_usage=report.usage,
    )
    return RcaCliResult(
        incident_id=incident.id,
        run=run_result,
        verification=verification,
        report=report,
        output_dir=str(output),
        artifacts=artifacts,
    )


def _stage_external_logs(
    output: Path,
    log_sources: dict[str, str | Path],
) -> Path:
    """Copy external logs into the output directory with stable names."""
    external_root = output / "inputs"
    external_root.mkdir(parents=True, exist_ok=True)
    for name, source in log_sources.items():
        source_path = Path(source)
        if not source_path.is_file():
            raise ValueError(f"external log file not found: {source_path}")
        shutil.copyfile(source_path, external_root / name)
    return external_root


def _run_verification(
    repo: Path,
    loaded: LoadedIncident,
    graph: EvidenceGraph,
) -> VerificationRun:
    sandbox = RuntimeVerificationSandbox(
        repo,
        base_commit=loaded.incident.base_commit,
        work_dir=tempfile.gettempdir(),
    )
    try:
        return RuntimeTestVerifier(sandbox).verify(graph)
    finally:
        sandbox.close()
def _persist_hypotheses(writer: ArtifactWriter, graph: EvidenceGraph) -> None:
    writer.write_dict(
        ARTIFACT_HYPOTHESES,
        {
            "hypotheses": [
                hypothesis.model_dump(mode="json")
                for hypothesis in graph.hypotheses
            ]
        },
    )


def _persist_verification(
    writer: ArtifactWriter,
    verification: VerificationRun,
) -> None:
    writer.write_dict(
        ARTIFACT_VERIFICATION,
        {
            "results": [
                result.model_dump(mode="json") for result in verification.results
            ],
            "evidence": [
                item.model_dump(mode="json") for item in verification.evidence
            ],
            "timing_seconds": verification.timing_seconds,
        },
    )


def _persist_agent_artifacts(
    writer: ArtifactWriter,
    graph: EvidenceGraph,
) -> None:
    for finding in graph.findings:
        evidence = [
            item.model_dump(mode="json")
            for item in graph.evidence
            if item.agent == finding.agent
        ]
        writer.write_dict(
            f"agents/{finding.agent.value}.json",
            {
                "finding": finding.model_dump(mode="json"),
                "evidence": evidence,
            },
        )


def _persist_run_summary(
    writer: ArtifactWriter,
    output: Path,
    run_result: RcaRunResult,
    verification: VerificationRun,
    report: RCAReport,
    *,
    synthesizer_usage: Any,
) -> list[str]:
    artifact_names = sorted(
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    )
    artifact_names.append(_RUN_SUMMARY)
    writer.write_dict(
        _RUN_SUMMARY,
        {
            "incident_id": run_result.incident_id,
            "status": run_result.status.value,
            "base_commit": report.evidence_graph.incident.base_commit,
            "conclusion": report.conclusion.value,
            "timing_seconds": run_result.timing_seconds,
            "verification_seconds": verification.timing_seconds,
            "usage": run_result.usage.model_dump(mode="json")
            if run_result.usage is not None
            else None,
            "synthesis_usage": synthesizer_usage.model_dump(mode="json"),
            "errors": run_result.errors,
            "isolation_violations": run_result.isolation_violations,
            "enabled_roles": run_result.enabled_roles,
            "retrieval": (
                run_result.retrieval.model_dump(mode="json")
                if run_result.retrieval is not None
                else None
            ),
            "artifacts": artifact_names,
        },
    )
    return sorted(artifact_names)


def _trace(
    writer: TraceWriter,
    run_id: str,
    event_type: str,
    run_result: RcaRunResult,
) -> None:
    writer.write(
        TraceEvent(
            run_id=run_id,
            event_type=event_type,
            workflow_stage=_TRACE_STAGE,
            final_status=run_result.status.value.upper(),
        )
    )
