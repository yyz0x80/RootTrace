"""CLI module for PatchPilot.

This module provides command-line interface for running PatchPilot
on local repositories with issue descriptions.
"""

import argparse
import json
import logging
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAIError

from patchpilot.agent_loop import AgentLoop, AgentLoopError, AgentLoopLimitError
from patchpilot.evidence import render_acceptance_coverage
from patchpilot.evidence.schema import AcceptanceCoverageReport
from patchpilot.issue.loader import load_issue
from patchpilot.issue.normalizer import normalize_issue
from patchpilot.issue.schema import NormalizedIssue
from patchpilot.planning.planner import PlanGenerationError, create_plan
from patchpilot.planning.schema import ChangePlan
from patchpilot.planning.scope_gate import check_scope
from patchpilot.planning.validator import validate_plan
from patchpilot.policy.builtins import get_builtin_policies
from patchpilot.provider import (
    LLMProvider,
    ToolCallParseError,
    create_provider_from_config,
)
from patchpilot.repository import RepositoryPreflightError, validate_repository
from patchpilot.repository.analyzer import analyze_repository
from patchpilot.tools import ToolRegistry
from patchpilot.utils import save_json
from patchpilot.workflow.execute_logger import ExecuteLogger
from patchpilot.workflow.result import PrepareSummary, RunSummary
from patchpilot.workflow.runner import (
    WorkflowRunner,
    WorkflowRunnerError,
    WorkflowRunnerExecutionError,
)
from patchpilot.workspace import Workspace


def _configure_logging() -> None:
    """Configure logging for CLI output.

    Sets up basic logging configuration to output to stdout
    with INFO level to ensure ExecuteLogger messages are visible.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )


def _create_provider(
    model_override: str | None = None,
    config_path: str | None = None,
) -> LLMProvider:
    """Create provider instance with optional model override.

    Args:
        model_override: Optional model name from config file or direct model identifier
        config_path: Optional path to model configuration file

    Returns:
        Configured provider instance

    Raises:
        ValueError: If provider initialization fails
    """
    return create_provider_from_config(model_name=model_override, config_path=config_path)


@dataclass(frozen=True)
class ProviderUsageSnapshot:
    """Exact usage captured by one provider process."""

    model: str
    llm_call_count: int
    prompt_tokens: int | None
    completion_tokens: int | None


def _provider_usage(provider: LLMProvider) -> ProviderUsageSnapshot:
    """Return exact provider usage while tolerating legacy test doubles."""
    model = getattr(provider, "model", "")
    if not isinstance(model, str):
        model = getattr(provider, "_model", "")
    llm_call_count = getattr(provider, "llm_call_count", 0)
    prompt_tokens = getattr(provider, "prompt_tokens", None)
    completion_tokens = getattr(provider, "completion_tokens", None)

    return ProviderUsageSnapshot(
        model=model if isinstance(model, str) else "",
        llm_call_count=(
            llm_call_count if isinstance(llm_call_count, int) else 0
        ),
        prompt_tokens=(
            prompt_tokens if isinstance(prompt_tokens, int) else None
        ),
        completion_tokens=(
            completion_tokens
            if isinstance(completion_tokens, int)
            else None
        ),
    )


def _save_prepare_summary(
    output_dir: Path,
    provider: object,
    *,
    outcome_code: str = "READY_FOR_APPROVAL",
    final_status: str | None = None,
    exit_code: int = 0,
    reasons: list[str] | None = None,
) -> None:
    """Persist the structured prepare outcome and exact model usage."""
    usage = _provider_usage(provider)
    summary = PrepareSummary(
        outcome_code=outcome_code,
        final_status=final_status,
        exit_code=exit_code,
        reasons=reasons or [],
        model=usage.model,
        llm_call_count=usage.llm_call_count,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        str(output_dir / "prepare_summary.json"),
        summary.model_dump_json(indent=2),
    )


def _existing_execute_artifacts(output_dir: Path) -> dict[str, str]:
    """Return only execute artifacts that currently exist on disk."""
    candidates = {
        "patch": output_dir / "patch.diff",
        "verification_report": output_dir / "verification_report.json",
        "acceptance_coverage": output_dir / "acceptance_coverage.md",
        "acceptance_evidence": output_dir / "acceptance_evidence.json",
        "execution_trace": output_dir / "execution_trace.jsonl",
    }
    return {
        name: str(path)
        for name, path in candidates.items()
        if path.exists()
    }


def _save_failed_run_summary(
    *,
    args: argparse.Namespace,
    started: float,
    provider: object | None,
    base_commit: str,
    final_status: str,
    failure_type: str,
    error_message: str,
    verification_report: dict[str, object] | None = None,
) -> None:
    """Persist a terminal execute result when the workflow raises an error."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if verification_report is not None:
        save_json(
            str(output_dir / "verification_report.json"),
            json.dumps(verification_report, indent=2),
        )
    reported_retry_count = (
        verification_report.get("retry_count", 0)
        if verification_report is not None
        else 0
    )
    retry_count = (
        reported_retry_count
        if isinstance(reported_retry_count, int)
        and reported_retry_count >= 0
        else 0
    )
    usage = _provider_usage(provider) if provider is not None else ProviderUsageSnapshot(
        model=args.model or "",
        llm_call_count=0,
        prompt_tokens=None,
        completion_tokens=None,
    )
    summary = RunSummary(
        run_id=str(uuid.uuid4()),
        task_id=args.task_id,
        phase="execute",
        base_commit=base_commit,
        model=usage.model,
        max_rounds=args.max_rounds,
        max_repairs=args.max_repairs,
        retry_count=retry_count,
        final_status=final_status,
        exit_code=1,
        duration_seconds=time.monotonic() - started,
        llm_call_count=usage.llm_call_count,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        failure_type=failure_type,
        error_message=error_message,
        artifacts=_existing_execute_artifacts(output_dir),
    )
    save_json(
        str(output_dir / "run_summary.json"),
        summary.model_dump_json(indent=2),
    )


def main() -> None:
    """Main entry point for the PatchPilot CLI."""
    _configure_logging()
    
    parser = argparse.ArgumentParser(
        description="PatchPilot: Issue-to-Patch Code Agent for Python repositories"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # prepare subcommand
    prepare_parser = subparsers.add_parser("prepare", help="Prepare a change plan for an issue")
    prepare_parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="Path to the target repository"
    )
    prepare_parser.add_argument(
        "--issue",
        type=str,
        required=True,
        help="Path to the issue description file (Markdown format) or GitHub issue URL"
    )
    prepare_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name from config file or direct model identifier"
    )
    prepare_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to model configuration file (default: ~/.patchpilot/models.json or .patchpilot/models.json)"
    )
    prepare_parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for artifacts from this run",
    )
    
    # run subcommand
    run_parser = subparsers.add_parser("run", help="Run PatchPilot on a repository")
    run_parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="Path to the target repository"
    )
    run_parser.add_argument(
        "--issue",
        type=str,
        required=True,
        help="Path to the issue description file (Markdown format)"
    )
    run_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name from config file or direct model identifier"
    )
    run_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to model configuration file (default: ~/.patchpilot/models.json or .patchpilot/models.json)"
    )
    run_parser.add_argument(
        "--max-rounds",
        type=int,
        default=16,
        help="Maximum number of agent rounds (default: 16)"
    )
    run_parser.add_argument(
        "--max-repairs",
        type=int,
        default=3,
        help="Maximum number of repair attempts (default: 3)"
    )
    run_parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for artifacts from this run",
    )
    run_parser.add_argument(
        "--task-id",
        type=str,
        default=None,
        help="Stable evaluation task identifier (optional for run command)",
    )
    
    # execute subcommand
    execute_parser = subparsers.add_parser("execute", help="Execute an approved change plan")
    execute_parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="Path to the target repository"
    )
    execute_parser.add_argument(
        "--issue",
        type=str,
        required=True,
        help="Path to the normalized issue JSON file (from prepare command)"
    )
    execute_parser.add_argument(
        "--plan",
        type=str,
        required=True,
        help="Path to the approved plan JSON file (from prepare command)"
    )
    execute_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name from config file or direct model identifier"
    )
    execute_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to model configuration file (default: ~/.patchpilot/models.json or .patchpilot/models.json)"
    )
    execute_parser.add_argument(
        "--max-rounds",
        type=int,
        default=16,
        help="Maximum number of agent rounds (default: 12)"
    )
    execute_parser.add_argument(
        "--max-repairs",
        type=int,
        default=3,
        help="Maximum number of repair attempts (default: 3)"
    )
    execute_parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for artifacts from this run",
    )
    execute_parser.add_argument(
        "--task-id",
        type=str,
        required=True,
        help="Stable evaluation task identifier",
    )

    baseline_parser = subparsers.add_parser(
        "baseline",
        help="Run the raw-issue evaluation baseline",
    )
    baseline_parser.add_argument("--repo", required=True)
    baseline_parser.add_argument("--issue", required=True)
    baseline_parser.add_argument("--model", default=None)
    baseline_parser.add_argument("--config", default=None)
    baseline_parser.add_argument("--max-rounds", type=int, default=16)
    baseline_parser.add_argument("--max-repairs", type=int, default=0)
    baseline_parser.add_argument("--output-dir", required=True)
    baseline_parser.add_argument("--task-id", required=True)
    
    args = parser.parse_args()
    
    if args.command == "prepare":
        handle_prepare(args)
    elif args.command == "run":
        handle_run(args)
    elif args.command == "execute":
        handle_execute(args)
    elif args.command == "baseline":
        handle_baseline(args)
    else:
        parser.print_help()
        sys.exit(1)


def handle_prepare(args) -> None:
    """Handle the prepare subcommand workflow.

    This function implements the prepare workflow:
    1. Load issue from file or GitHub
    2. Normalize the issue to extract structured information
    3. Repository preflight validation
    4. Repository analysis
    5. Create a change plan
    6. Validate the plan against repository context
    7. Scope gate validation
    8. Output artifacts (normalized_issue.json, repository_context.json, plan.json)
    """
    provider: LLMProvider | None = None
    outcome_code = "PREPARE_FAILED"
    final_status: str | None = "FAILED"
    exit_code = 1
    reasons: list[str] = []
    try:
        # Step 1: Load raw issue
        ExecuteLogger.log_issue_loading(args.issue)
        raw_issue = load_issue(args.issue)

        # Create provider for normalization and planning
        try:
            provider = _create_provider(args.model, args.config)
        except ValueError as e:
            outcome_code = "PROVIDER_CONFIGURATION_ERROR"
            final_status = "BLOCKED"
            reasons = [str(e)]
            print(f"Provider initialization failed: {e}", file=sys.stderr)
            sys.exit(1)

        # Step 2: Normalize the issue
        normalized_issue = normalize_issue(
            issue=raw_issue,
            generate=provider.generate_text,
        )

        # Check for ambiguous points
        if normalized_issue.ambiguous_points:
            outcome_code = "AMBIGUOUS_REQUIREMENT"
            final_status = "NEEDS_CLARIFICATION"
            reasons = list(normalized_issue.ambiguous_points)
            ExecuteLogger.log_issue_normalization(
                success=False,
                ambiguous_points=normalized_issue.ambiguous_points,
            )
            print("\nPatchPilot will not guess product behavior.")
            sys.exit(1)

        ExecuteLogger.log_issue_normalization(success=True)

        # Step 3: Repository preflight
        repo_path = Path(args.repo)
        try:
            preflight_result = validate_repository(repo_path)
            ExecuteLogger.log_repository_validation(
                is_valid=True,
                head_sha=preflight_result.head_sha,
            )
        except RepositoryPreflightError as e:
            outcome_code = "REPOSITORY_INVALID"
            final_status = "BLOCKED"
            reasons = [str(e)]
            ExecuteLogger.log_repository_validation(
                is_valid=False,
                error=str(e),
            )
            sys.exit(1)

        # Step 4: Repository analysis
        repository_context = analyze_repository(
            repo=repo_path,
            issue=normalized_issue,
            base_commit=preflight_result.head_sha,
        )
        ExecuteLogger.log_repository_analysis(
            python_files_count=len(repository_context.python_files),
            test_files_count=len(repository_context.test_files),
            keyword_matches_count=len(repository_context.keyword_matches),
        )

        # Step 5: Create change plan
        plan = create_plan(
            issue=normalized_issue,
            repository_context=repository_context,
            generate=provider.generate_text,
        )

        # Format planned changes for logging
        planned_changes = [
            f"{change.action.upper()} {change.path}"
            for change in plan.planned_changes
        ]
        planned_tests = [
            f"TEST {test.command}"
            for test in plan.planned_tests
        ]
        ExecuteLogger.log_plan_creation(planned_changes, planned_tests)

        # Step 6: Validate plan against repository context
        try:
            validation_result = validate_plan(
                plan,
                repository_context,
                issue=normalized_issue,
            )
        except ValueError as e:
            outcome_code = "PLAN_INVALID"
            final_status = "BLOCKED"
            reasons = [str(e)]
            print(f"Plan validation failed: {e}", file=sys.stderr)
            sys.exit(1)

        # Step 7: Scope gate validation
        scope_result = validation_result

        if not scope_result.allowed:
            outcome_code = "SCOPE_VIOLATION"
            final_status = "BLOCKED"
            reasons = list(scope_result.violations)
            ExecuteLogger.log_plan_validation(
                allowed=False,
                violations=scope_result.violations,
                warnings=scope_result.warnings,
            )
            sys.exit(1)

        ExecuteLogger.log_plan_validation(allowed=True)

        # Step 8: Output artifacts
        artifact_paths = []
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        normalized_issue_path = output_dir / "normalized_issue.json"
        save_json(
            str(normalized_issue_path),
            normalized_issue.model_dump_json(indent=2),
        )
        artifact_paths.append(str(normalized_issue_path))

        repository_context_path = output_dir / "repository_context.json"
        save_json(
            str(repository_context_path),
            repository_context.model_dump_json(indent=2),
        )
        artifact_paths.append(str(repository_context_path))

        plan_path = output_dir / "plan.json"
        save_json(
            str(plan_path),
            plan.model_dump_json(indent=2),
        )
        artifact_paths.append(str(plan_path))

        ExecuteLogger.log_artifacts(artifact_paths)

        outcome_code = "READY_FOR_APPROVAL"
        final_status = None
        exit_code = 0
        reasons = []

        print(f"\nReview the artifacts in {output_dir}/ directory.")
        print(f"To execute this plan, run: patchpilot execute --repo <repo> --issue {output_dir}/normalized_issue.json --plan {output_dir}/plan.json --output-dir {output_dir} --task-id <task-id>")

    except PlanGenerationError as e:
        outcome_code = "PLAN_INVALID"
        final_status = "BLOCKED"
        reasons = [str(e)]
        print(f"Plan validation failed: {e}", file=sys.stderr)
        sys.exit(1)
    except (OpenAIError, ToolCallParseError) as e:
        outcome_code = "PROVIDER_ERROR"
        final_status = "BLOCKED"
        reasons = [str(e)]
        print(f"Provider error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        if outcome_code == "PREPARE_FAILED":
            outcome_code = "INVALID_INPUT"
            reasons = [str(e)]
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        outcome_code = "FILE_SYSTEM_ERROR"
        final_status = "BLOCKED"
        reasons = [str(e)]
        print(f"File system error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        try:
            _save_prepare_summary(
                Path(args.output_dir),
                provider or object(),
                outcome_code=outcome_code,
                final_status=final_status,
                exit_code=exit_code,
                reasons=reasons,
            )
        except OSError as error:
            print(
                f"Failed to save prepare summary: {error}",
                file=sys.stderr,
            )


def handle_run(args) -> None:
    """Handle the run subcommand workflow using canonical WorkflowRunner.
    
    This function implements the full run workflow that executes the agent loop
    using the canonical WorkflowRunner and Verifier with Docker sandbox.
    """
    started = time.monotonic()
    provider: LLMProvider | None = None
    base_commit = ""

    try:
        # Setup output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load raw issue
        raw_issue = load_issue(args.issue)
        
        # Create provider for normalization
        try:
            provider = _create_provider(args.model, args.config)
        except ValueError as e:
            print(f"Provider initialization failed: {e}", file=sys.stderr)
            sys.exit(1)
        
        # Normalize the issue
        normalized_issue = normalize_issue(
            issue=raw_issue,
            generate=provider.generate_text,
        )
        
        # Save normalized issue to artifacts
        save_json(
            str(output_dir / "normalized_issue.json"),
            normalized_issue.model_dump_json(indent=2),
        )
        
        # Check for ambiguous points
        if normalized_issue.ambiguous_points:
            print("NEEDS_CLARIFICATION\n")
            print("The following requirements are ambiguous:\n")
            for i, point in enumerate(normalized_issue.ambiguous_points, start=1):
                print(f"{i}. {point}")
            print("\nPatchPilot will not guess product behavior.")
            sys.exit(1)
        
        # Create workspace
        repo_path = Path(args.repo)
        
        # Validate repository
        print("Validating repository...")
        try:
            preflight_result = validate_repository(repo_path)
            print(f"Repository validated: {preflight_result.repo_path}")
            print(f"Current HEAD: {preflight_result.head_sha[:8]}...")
            print()
        except RepositoryPreflightError as e:
            print(f"Repository validation failed: {e}", file=sys.stderr)
            sys.exit(1)
        
        base_commit = preflight_result.head_sha
        
        workspace = Workspace(root=repo_path)
        
        # Create change plan
        plan = create_plan(
            issue=normalized_issue,
            repo_path=str(repo_path),
            generate=provider.generate_text,
            base_commit=preflight_result.head_sha,
        )
        
        # Save plan to artifacts
        save_json(
            str(output_dir / "plan.json"),
            plan.model_dump_json(indent=2),
        )
        
        # Validate plan against scope restrictions
        policy_set = get_builtin_policies()
        scope_result = check_scope(plan, policy_set)
        
        if not scope_result.allowed:
            print("SCOPE_CHECK_FAILED\n")
            print("The following violations block execution:\n")
            for i, violation in enumerate(scope_result.violations, start=1):
                print(f"{i}. {violation}")
            if scope_result.warnings:
                print("\nWarnings:\n")
                for i, warning in enumerate(scope_result.warnings, start=1):
                    print(f"{i}. {warning}")
            sys.exit(1)
        
        if scope_result.warnings:
            print("SCOPE_CHECK_WARNINGS\n")
            print("The following warnings were generated:\n")
            for i, warning in enumerate(scope_result.warnings, start=1):
                print(f"{i}. {warning}")
            print()
        
        # Create tool registry
        tools = ToolRegistry(workspace=workspace)
        
        # Create agent loop
        agent_loop = AgentLoop(
            provider=provider,
            tools=tools,
            max_rounds=args.max_rounds,
        )
        
        # Create workflow runner with canonical verifier
        runner = WorkflowRunner(
            agent_loop=agent_loop,
            verifier=None,  # Use built-in sandbox verifier
            workspace=workspace,
            max_repair_attempts=args.max_repairs,
        )

        # Execute workflow
        trace_path = output_dir / "execution_trace.jsonl"
        result = runner.execute(
            issue=normalized_issue.model_dump_json(indent=2),
            plan=plan.model_dump_json(indent=2),
            change_plan=plan,
            normalized_issue=normalized_issue,
            trace_path=trace_path,
        )

        # Calculate duration and update result
        duration_seconds = time.monotonic() - started
        result.duration_seconds = duration_seconds

        # Extract retry count from verification report
        result.retry_count = int(
            result.verification_report.get("retry_count", 0)
        )

        # Preserve exact run-phase model usage without estimating tokens.
        usage = _provider_usage(provider)
        result.llm_call_count = usage.llm_call_count
        result.prompt_tokens = usage.prompt_tokens
        result.completion_tokens = usage.completion_tokens

        # Save verification report
        verification_report_path = output_dir / "verification_report.json"
        save_json(
            str(verification_report_path),
            json.dumps(result.verification_report, indent=2),
        )

        # Save patch
        if result.patch:
            patch_path = output_dir / "patch.diff"
            patch_path.write_text(result.patch, encoding="utf-8")

        # Save acceptance coverage report
        coverage_path = output_dir / "acceptance_coverage.md"
        coverage_path.write_text(
            render_acceptance_coverage(
                result.acceptance_evidence,
                result.final_status.value,
            ),
            encoding="utf-8",
        )

        # Save machine-readable acceptance evidence for deterministic metrics.
        evidence_report = AcceptanceCoverageReport(
            acceptance_evidence=result.acceptance_evidence,
            completion_state=result.final_status,
            summary="Acceptance evidence generated by the workflow.",
        )
        save_json(
            str(output_dir / "acceptance_evidence.json"),
            evidence_report.model_dump_json(indent=2),
        )

        # Save run summary
        run_summary = result.to_run_summary(
            task_id=args.task_id if args.task_id else str(uuid.uuid4()),
            base_commit=base_commit,
            model=usage.model,
            output_dir=str(output_dir),
        )
        save_json(
            str(output_dir / "run_summary.json"),
            run_summary.model_dump_json(indent=2),
        )

        # Print results
        print("\nEXECUTION_COMPLETE\n")

        # Count acceptance criteria by status
        pass_count = sum(1 for evidence in result.acceptance_evidence if evidence.status.value == "PASS")
        fail_count = sum(1 for evidence in result.acceptance_evidence if evidence.status.value == "FAIL")
        unverified_count = sum(1 for evidence in result.acceptance_evidence if evidence.status.value == "UNVERIFIED")

        print(f"Final status: {result.final_status.value}")
        print("Acceptance criteria:")
        print(f"  PASS: {pass_count}")
        print(f"  FAIL: {fail_count}")
        print(f"  UNVERIFIED: {unverified_count}")

        # Exit with code from run summary (ensures summary is saved before exit)
        sys.exit(run_summary.exit_code)
        
    except ValueError as e:
        _save_failed_run_summary(
            args=args,
            started=started,
            provider=provider,
            base_commit=base_commit,
            final_status="FAILED",
            failure_type="INPUT_ERROR",
            error_message=str(e),
        )
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except AgentLoopLimitError as e:
        _save_failed_run_summary(
            args=args,
            started=started,
            provider=provider,
            base_commit=base_commit,
            final_status="FAILED",
            failure_type="AGENT_ROUND_LIMIT",
            error_message=str(e),
        )
        print(f"Agent error: {e}", file=sys.stderr)
        sys.exit(1)
    except AgentLoopError as e:
        _save_failed_run_summary(
            args=args,
            started=started,
            provider=provider,
            base_commit=base_commit,
            final_status="FAILED",
            failure_type="AGENT_ERROR",
            error_message=str(e),
        )
        print(f"Agent error: {e}", file=sys.stderr)
        sys.exit(1)
    except WorkflowRunnerExecutionError as e:
        _save_failed_run_summary(
            args=args,
            started=started,
            provider=provider,
            base_commit=base_commit,
            final_status="FAILED",
            failure_type=e.failure_type,
            error_message=str(e),
            verification_report=e.verification_report,
        )
        print(f"Workflow execution error: {e}", file=sys.stderr)
        sys.exit(1)
    except WorkflowRunnerError as e:
        _save_failed_run_summary(
            args=args,
            started=started,
            provider=provider,
            base_commit=base_commit,
            final_status="FAILED",
            failure_type="WORKFLOW_ERROR",
            error_message=str(e),
        )
        print(f"Workflow error: {e}", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        _save_failed_run_summary(
            args=args,
            started=started,
            provider=provider,
            base_commit=base_commit,
            final_status="BLOCKED",
            failure_type="FILE_SYSTEM_ERROR",
            error_message=str(e),
        )
        print(f"File system error: {e}", file=sys.stderr)
        sys.exit(1)
    except (OpenAIError, ToolCallParseError) as e:
        _save_failed_run_summary(
            args=args,
            started=started,
            provider=provider,
            base_commit=base_commit,
            final_status="FAILED",
            failure_type="PROVIDER_ERROR",
            error_message=str(e),
        )
        print(f"Provider execution error: {e}", file=sys.stderr)
        sys.exit(1)


def handle_execute(args) -> None:
    """Handle the execute subcommand workflow.

    This function implements the execute workflow:
    1. Load normalized issue from JSON file
    2. Load approved plan from JSON file
    3. Setup workspace and sandbox
    4. Run agent implementation
    5. Execute verification
    6. Run repair loop if needed
    7. Save verification report
    """
    started = time.monotonic()
    provider: LLMProvider | None = None
    base_commit = ""

    try:
        # Step 1: Load normalized issue
        issue_path = Path(args.issue)
        if not issue_path.exists():
            message = f"Issue file not found: {args.issue}"
            _save_failed_run_summary(
                args=args,
                started=started,
                provider=provider,
                base_commit=base_commit,
                final_status="FAILED",
                failure_type="INPUT_ERROR",
                error_message=message,
            )
            print(f"Error: {message}", file=sys.stderr)
            sys.exit(1)

        with open(issue_path, encoding="utf-8") as f:
            issue_data = json.load(f)
        normalized_issue = NormalizedIssue.model_validate(issue_data)

        # Step 2: Load approved plan
        plan_path = Path(args.plan)
        if not plan_path.exists():
            message = f"Plan file not found: {args.plan}"
            _save_failed_run_summary(
                args=args,
                started=started,
                provider=provider,
                base_commit=base_commit,
                final_status="FAILED",
                failure_type="INPUT_ERROR",
                error_message=message,
            )
            print(f"Error: {message}", file=sys.stderr)
            sys.exit(1)

        with open(plan_path, encoding="utf-8") as f:
            plan_data = json.load(f)
        plan = ChangePlan.model_validate(plan_data)
        base_commit = plan.base_commit

        # Log precheck section
        ExecuteLogger.log_precheck(
            git_repo=True,
            working_tree_clean=True,
            base_commit_match=True,
        )

        # Step 3: Setup workspace
        repo_path = Path(args.repo)
        
        # Setup output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Validate repository
        try:
            preflight_result = validate_repository(repo_path)
        except RepositoryPreflightError as e:
            _save_failed_run_summary(
                args=args,
                started=started,
                provider=provider,
                base_commit=base_commit,
                final_status="BLOCKED",
                failure_type="REPOSITORY_INVALID",
                error_message=str(e),
            )
            print(f"Repository validation failed: {e}", file=sys.stderr)
            sys.exit(1)

        # Step 3: Verify repository baseline matches plan
        if preflight_result.head_sha != plan.base_commit:
            message = "Repository HEAD has changed since the plan was generated."
            _save_failed_run_summary(
                args=args,
                started=started,
                provider=provider,
                base_commit=base_commit,
                final_status="BLOCKED",
                failure_type="BASE_COMMIT_MISMATCH",
                error_message=message,
            )
            print(
                message,
                file=sys.stderr
            )
            print(f"Plan base commit: {plan.base_commit[:8]}...", file=sys.stderr)
            print(f"Current HEAD: {preflight_result.head_sha[:8]}...", file=sys.stderr)
            print("Run `patchpilot prepare` again.", file=sys.stderr)
            sys.exit(1)

        # Step 4: Create provider
        try:
            provider = _create_provider(args.model, args.config)
        except ValueError as e:
            _save_failed_run_summary(
                args=args,
                started=started,
                provider=provider,
                base_commit=base_commit,
                final_status="BLOCKED",
                failure_type="PROVIDER_CONFIGURATION_ERROR",
                error_message=str(e),
            )
            print(f"Provider initialization failed: {e}", file=sys.stderr)
            sys.exit(1)

        # Step 5: Create initial workspace with original repo path
        # This will be updated to temporary workspace path during runner execution
        workspace = Workspace(root=repo_path)

        # Step 6: Create tool registry
        tools = ToolRegistry(workspace=workspace)
        
        # Step 7: Create agent loop
        agent_loop = AgentLoop(
            provider=provider,
            tools=tools,
            max_rounds=args.max_rounds,
        )
        
        # Step 8: Create workflow runner
        runner = WorkflowRunner(
            agent_loop=agent_loop,
            verifier=None,
            workspace=workspace,
            max_repair_attempts=args.max_repairs,
        )

        # Step 9: Execute workflow
        trace_path = output_dir / "execution_trace.jsonl"
        result = runner.execute(
            issue=normalized_issue.model_dump_json(indent=2),
            plan=plan.model_dump_json(indent=2),
            change_plan=plan,
            normalized_issue=normalized_issue,
            trace_path=trace_path,
        )

        # Calculate duration and update result
        duration_seconds = time.monotonic() - started
        result.duration_seconds = duration_seconds

        # Extract retry count from verification report
        result.retry_count = int(
            result.verification_report.get("retry_count", 0)
        )

        # Preserve exact execute-phase model usage without estimating tokens.
        usage = _provider_usage(provider)
        result.llm_call_count = usage.llm_call_count
        result.prompt_tokens = usage.prompt_tokens
        result.completion_tokens = usage.completion_tokens

        # Step 10: Save verification report
        verification_report_path = output_dir / "verification_report.json"
        save_json(
            str(verification_report_path),
            json.dumps(result.verification_report, indent=2),
        )

        # Step 11: Save patch
        if result.patch:
            patch_path = output_dir / "patch.diff"
            patch_path.write_text(result.patch, encoding="utf-8")

        # Step 12: Save acceptance coverage report
        coverage_path = output_dir / "acceptance_coverage.md"
        coverage_path.write_text(
            render_acceptance_coverage(
                result.acceptance_evidence,
                result.final_status.value,
            ),
            encoding="utf-8",
        )

        # Save machine-readable acceptance evidence for deterministic metrics.
        evidence_report = AcceptanceCoverageReport(
            acceptance_evidence=result.acceptance_evidence,
            completion_state=result.final_status,
            summary="Acceptance evidence generated by the workflow.",
        )
        save_json(
            str(output_dir / "acceptance_evidence.json"),
            evidence_report.model_dump_json(indent=2),
        )

        # Step 13: Save run summary
        run_summary = result.to_run_summary(
            task_id=args.task_id,
            base_commit=plan.base_commit,
            model=usage.model,
            output_dir=str(output_dir),
        )
        save_json(
            str(output_dir / "run_summary.json"),
            run_summary.model_dump_json(indent=2),
        )

        # Step 14: Print results
        print("\nEXECUTION_COMPLETE\n")

        # Count acceptance criteria by status
        pass_count = sum(1 for evidence in result.acceptance_evidence if evidence.status.value == "PASS")
        fail_count = sum(1 for evidence in result.acceptance_evidence if evidence.status.value == "FAIL")
        unverified_count = sum(1 for evidence in result.acceptance_evidence if evidence.status.value == "UNVERIFIED")

        print(f"Final status: {result.final_status.value}")
        print("Acceptance criteria:")
        print(f"  PASS: {pass_count}")
        print(f"  FAIL: {fail_count}")
        print(f"  UNVERIFIED: {unverified_count}")

        # Exit with code from run summary (ensures summary is saved before exit)
        sys.exit(run_summary.exit_code)
        
    except json.JSONDecodeError as e:
        _save_failed_run_summary(
            args=args,
            started=started,
            provider=provider,
            base_commit=base_commit,
            final_status="FAILED",
            failure_type="JSON_ERROR",
            error_message=str(e),
        )
        print(f"JSON parsing error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        _save_failed_run_summary(
            args=args,
            started=started,
            provider=provider,
            base_commit=base_commit,
            final_status="FAILED",
            failure_type="INPUT_ERROR",
            error_message=str(e),
        )
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except AgentLoopLimitError as e:
        _save_failed_run_summary(
            args=args,
            started=started,
            provider=provider,
            base_commit=base_commit,
            final_status="FAILED",
            failure_type="AGENT_ROUND_LIMIT",
            error_message=str(e),
        )
        print(f"Agent error: {e}", file=sys.stderr)
        sys.exit(1)
    except AgentLoopError as e:
        _save_failed_run_summary(
            args=args,
            started=started,
            provider=provider,
            base_commit=base_commit,
            final_status="FAILED",
            failure_type="AGENT_ERROR",
            error_message=str(e),
        )
        print(f"Agent error: {e}", file=sys.stderr)
        sys.exit(1)
    except WorkflowRunnerExecutionError as e:
        _save_failed_run_summary(
            args=args,
            started=started,
            provider=provider,
            base_commit=base_commit,
            final_status="FAILED",
            failure_type=e.failure_type,
            error_message=str(e),
            verification_report=e.verification_report,
        )
        print(f"Workflow execution error: {e}", file=sys.stderr)
        sys.exit(1)
    except WorkflowRunnerError as e:
        _save_failed_run_summary(
            args=args,
            started=started,
            provider=provider,
            base_commit=base_commit,
            final_status="FAILED",
            failure_type="WORKFLOW_ERROR",
            error_message=str(e),
        )
        print(f"Workflow error: {e}", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        _save_failed_run_summary(
            args=args,
            started=started,
            provider=provider,
            base_commit=base_commit,
            final_status="BLOCKED",
            failure_type="FILE_SYSTEM_ERROR",
            error_message=str(e),
        )
        print(f"File system error: {e}", file=sys.stderr)
        sys.exit(1)
    except (OpenAIError, ToolCallParseError) as e:
        _save_failed_run_summary(
            args=args,
            started=started,
            provider=provider,
            base_commit=base_commit,
            final_status="FAILED",
            failure_type="PROVIDER_ERROR",
            error_message=str(e),
        )
        print(f"Provider execution error: {e}", file=sys.stderr)
        sys.exit(1)


def handle_baseline(args) -> None:
    """Run the evaluation baseline without planning or repair behavior."""
    started = time.monotonic()
    provider: LLMProvider | None = None
    base_commit = ""

    try:
        if args.max_repairs != 0:
            raise ValueError("Baseline evaluation requires --max-repairs 0")

        raw_issue = load_issue(args.issue)
        repo_path = Path(args.repo)
        preflight_result = validate_repository(repo_path)
        base_commit = preflight_result.head_sha
        provider = _create_provider(args.model, args.config)

        workspace = Workspace(root=repo_path)
        tools = ToolRegistry(workspace=workspace)
        agent_loop = AgentLoop(
            provider=provider,
            tools=tools,
            max_rounds=args.max_rounds,
        )
        runner = WorkflowRunner(
            agent_loop=agent_loop,
            verifier=None,
            workspace=workspace,
            max_repair_attempts=0,
        )

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result = runner.execute_baseline(
            issue=raw_issue.body,
            trace_path=output_dir / "execution_trace.jsonl",
        )

        usage = _provider_usage(provider)
        result.duration_seconds = time.monotonic() - started
        result.llm_call_count = usage.llm_call_count
        result.prompt_tokens = usage.prompt_tokens
        result.completion_tokens = usage.completion_tokens

        save_json(
            str(output_dir / "verification_report.json"),
            json.dumps(result.verification_report, indent=2),
        )
        if result.patch:
            (output_dir / "patch.diff").write_text(
                result.patch,
                encoding="utf-8",
            )

        coverage_path = output_dir / "acceptance_coverage.md"
        coverage_path.write_text(
            render_acceptance_coverage([], result.final_status.value),
            encoding="utf-8",
        )
        evidence_report = AcceptanceCoverageReport(
            acceptance_evidence=[],
            completion_state=result.final_status,
            summary="The raw-issue baseline does not map acceptance evidence.",
        )
        save_json(
            str(output_dir / "acceptance_evidence.json"),
            evidence_report.model_dump_json(indent=2),
        )

        run_summary = result.to_run_summary(
            task_id=args.task_id,
            base_commit=base_commit,
            model=usage.model,
            output_dir=str(output_dir),
        )
        save_json(
            str(output_dir / "run_summary.json"),
            run_summary.model_dump_json(indent=2),
        )
        print(f"Final status: {result.final_status.value}")
        sys.exit(run_summary.exit_code)

    except RepositoryPreflightError as error:
        _save_failed_run_summary(
            args=args,
            started=started,
            provider=provider,
            base_commit=base_commit,
            final_status="BLOCKED",
            failure_type="REPOSITORY_INVALID",
            error_message=str(error),
        )
        print(f"Repository validation failed: {error}", file=sys.stderr)
        sys.exit(1)
    except AgentLoopError as error:
        _save_failed_run_summary(
            args=args,
            started=started,
            provider=provider,
            base_commit=base_commit,
            final_status="FAILED",
            failure_type="AGENT_ERROR",
            error_message=str(error),
        )
        print(f"Agent error: {error}", file=sys.stderr)
        sys.exit(1)
    except WorkflowRunnerError as error:
        _save_failed_run_summary(
            args=args,
            started=started,
            provider=provider,
            base_commit=base_commit,
            final_status="FAILED",
            failure_type="WORKFLOW_ERROR",
            error_message=str(error),
        )
        print(f"Workflow error: {error}", file=sys.stderr)
        sys.exit(1)
    except (ValueError, OSError) as error:
        _save_failed_run_summary(
            args=args,
            started=started,
            provider=provider,
            base_commit=base_commit,
            final_status="FAILED",
            failure_type="BASELINE_ERROR",
            error_message=str(error),
        )
        print(f"Baseline error: {error}", file=sys.stderr)
        sys.exit(1)
    except (OpenAIError, ToolCallParseError) as error:
        _save_failed_run_summary(
            args=args,
            started=started,
            provider=provider,
            base_commit=base_commit,
            final_status="FAILED",
            failure_type="PROVIDER_ERROR",
            error_message=str(error),
        )
        print(f"Baseline provider error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
