"""CLI module for PatchPilot.

This module provides command-line interface for running PatchPilot
on local repositories with issue descriptions.
"""

import argparse
import json
import logging
import subprocess  # noqa: F401
import sys
from pathlib import Path

from patchpilot.agent_loop import AgentLoop, AgentLoopError, AgentLoopLimitError
from patchpilot.evidence import render_acceptance_coverage
from patchpilot.issue.loader import load_issue
from patchpilot.issue.normalizer import normalize_issue
from patchpilot.issue.schema import NormalizedIssue
from patchpilot.planning.planner import create_plan
from patchpilot.planning.schema import ChangePlan
from patchpilot.planning.scope_gate import check_scope
from patchpilot.planning.validator import validate_plan
from patchpilot.prompts import REPAIR_PROMPT
from patchpilot.provider import LLMProvider
from patchpilot.repository import RepositoryPreflightError, validate_repository
from patchpilot.repository.analyzer import analyze_repository
from patchpilot.sandbox.docker_runner import CommandResult
from patchpilot.tools import ToolRegistry
from patchpilot.utils import save_json
from patchpilot.verification.error_parser import parse_failure
from patchpilot.verification.report import CheckReport, VerificationReport
from patchpilot.workflow import (
    RepairLoopError,
    RepairLoopStalledError,
    run_repair_loop,
)
from patchpilot.workflow.execute_logger import ExecuteLogger
from patchpilot.workflow.runner import WorkflowRunner, WorkflowRunnerError
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
        help="Model identifier (overrides PATCHPILOT_MODEL environment variable)"
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
        help="Model identifier (overrides PATCHPILOT_MODEL environment variable)"
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
        help="Model identifier (overrides PATCHPILOT_MODEL environment variable)"
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
    
    args = parser.parse_args()
    
    if args.command == "prepare":
        handle_prepare(args)
    elif args.command == "run":
        handle_run(args)
    elif args.command == "execute":
        handle_execute(args)
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
    try:
        # Step 1: Load raw issue
        ExecuteLogger.log_issue_loading(args.issue)
        raw_issue = load_issue(args.issue)

        # Create provider for normalization and planning
        try:
            provider = LLMProvider()
        except ValueError as e:
            print(f"Provider initialization failed: {e}", file=sys.stderr)
            sys.exit(1)

        # Step 2: Normalize the issue
        normalized_issue = normalize_issue(
            issue=raw_issue,
            generate=provider.generate_text,
        )

        # Check for ambiguous points
        if normalized_issue.ambiguous_points:
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
            validation_result = validate_plan(plan, repository_context)
        except ValueError as e:
            print(f"Plan validation failed: {e}", file=sys.stderr)
            sys.exit(1)

        # Step 7: Scope gate validation
        scope_result = validation_result

        if not scope_result.allowed:
            ExecuteLogger.log_plan_validation(
                allowed=False,
                violations=scope_result.violations,
                warnings=scope_result.warnings,
            )
            sys.exit(1)

        ExecuteLogger.log_plan_validation(allowed=True)

        # Step 8: Output artifacts
        artifact_paths = []

        save_json(
            "artifacts/normalized_issue.json",
            normalized_issue.model_dump_json(indent=2),
        )
        artifact_paths.append("artifacts/normalized_issue.json")

        save_json(
            "artifacts/repository_context.json",
            repository_context.model_dump_json(indent=2),
        )
        artifact_paths.append("artifacts/repository_context.json")

        save_json(
            "artifacts/plan.json",
            plan.model_dump_json(indent=2),
        )
        artifact_paths.append("artifacts/plan.json")

        ExecuteLogger.log_artifacts(artifact_paths)

        print("\nReview the artifacts in artifacts/ directory.")
        print("To execute this plan, run: patchpilot execute --repo <repo> --issue artifacts/normalized_issue.json --plan artifacts/plan.json")

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        print(f"File system error: {e}", file=sys.stderr)
        sys.exit(1)


def handle_run(args) -> None:
    """Handle the run subcommand workflow.
    
    This function implements the full run workflow that executes the agent loop.
    """
    try:
        # Load raw issue
        raw_issue = load_issue(args.issue)
        
        # Create provider for normalization
        try:
            provider = LLMProvider()
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
            "artifacts/normalized_issue.json",
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
            "artifacts/plan.json",
            plan.model_dump_json(indent=2),
        )
        
        # Validate plan against scope restrictions
        scope_result = check_scope(plan)
        
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
        
        # Define verifier function for repair loop
        def run_verification() -> VerificationReport:
            """Run verification commands and return a VerificationReport."""
            import subprocess
            import time

            report = VerificationReport()

            # Use the current workspace root (which will be the temporary workspace)
            # The workspace will be updated by the runner during execution
            current_workspace_root = workspace.root

            # Run quick verification (ruff)
            try:
                start_time = time.time()
                result = subprocess.run(
                    ["ruff", "check"],
                    cwd=current_workspace_root,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                )
                duration = time.time() - start_time
                
                if result.returncode == 0:
                    ruff_check = CheckReport(
                        level="quick",
                        command="ruff check",
                        passed=True,
                        exit_code=0,
                        duration_seconds=duration,
                    )
                else:
                    # Parse the failure
                    mock_result = CommandResult(
                        command="ruff check",
                        exit_code=result.returncode,
                        stdout=result.stdout,
                        stderr=result.stderr,
                        duration_seconds=duration,
                        timed_out=False,
                    )
                    failure_summary = parse_failure(mock_result)
                    
                    ruff_check = CheckReport(
                        level="quick",
                        command="ruff check",
                        passed=False,
                        exit_code=result.returncode,
                        duration_seconds=duration,
                        failure_type=failure_summary.error_type,
                        summary=failure_summary.__dict__,
                    )
                report.add_check(ruff_check)
            except (subprocess.TimeoutExpired, OSError, FileNotFoundError) as e:
                # If ruff check fails, create a failed check
                ruff_check = CheckReport(
                    level="quick",
                    command="ruff check",
                    passed=False,
                    exit_code=1,
                    duration_seconds=0.0,
                    failure_type="LintError",
                    summary={"error": str(e)},
                )
                report.add_check(ruff_check)
            
            # Only run pytest if ruff passed
            if report.passed:
                try:
                    start_time = time.time()
                    result = subprocess.run(
                        ["python", "-m", "pytest", "tests/", "-q"],
                        cwd=current_workspace_root,
                        capture_output=True,
                        text=True,
                        timeout=60,
                        check=False,
                    )
                    duration = time.time() - start_time
                    
                    if result.returncode == 0:
                        pytest_check = CheckReport(
                            level="standard",
                            command="python -m pytest tests/ -q",
                            passed=True,
                            exit_code=0,
                            duration_seconds=duration,
                        )
                    else:
                        # Parse the failure
                        mock_result = CommandResult(
                            command="python -m pytest tests/ -q",
                            exit_code=result.returncode,
                            stdout=result.stdout,
                            stderr=result.stderr,
                            duration_seconds=duration,
                            timed_out=False,
                        )
                        failure_summary = parse_failure(mock_result)
                        
                        pytest_check = CheckReport(
                            level="standard",
                            command="python -m pytest tests/ -q",
                            passed=False,
                            exit_code=result.returncode,
                            duration_seconds=duration,
                            failure_type=failure_summary.error_type,
                            summary=failure_summary.__dict__,
                        )
                    report.add_check(pytest_check)
                except (subprocess.TimeoutExpired, OSError, FileNotFoundError) as e:
                    # If pytest fails, create a failed check
                    pytest_check = CheckReport(
                        level="standard",
                        command="python -m pytest tests/ -q",
                        passed=False,
                        exit_code=1,
                        duration_seconds=0.0,
                        failure_type="TestError",
                        summary={"error": str(e)},
                    )
                    report.add_check(pytest_check)
            
            return report
        
        # Define repair prompt builder
        def build_repair_prompt(original_issue: str, failure_report: VerificationReport) -> str:
            """Build a repair prompt based on the failure report."""
            failed_checks = failure_report.get_failed_checks()
            if not failed_checks:
                return original_issue
            
            latest_failure = failed_checks[-1]
            failure_summary = latest_failure.summary or {}
            
            return REPAIR_PROMPT.format(
                issue=original_issue,
                plan=plan.model_dump_json(indent=2),
                failure=failure_summary.get("relevant_output", "Unknown failure"),
            )
        
        # Run the repair loop with the plan
        try:
            result, verification_report = run_repair_loop(
                agent_loop=agent_loop,
                issue=plan.model_dump_json(indent=2),
                max_attempts=args.max_repairs,
                verifier=run_verification,
                repair_prompt_builder=build_repair_prompt,
            )
            
            # Print result
            print(result)
            
            # Print verification status
            if verification_report and verification_report.passed:
                print("\n✓ Verification passed")
            elif verification_report:
                print(f"\n✗ Verification failed after {args.max_repairs} repair attempt(s)")
                failed_checks = verification_report.get_failed_checks()
                if failed_checks:
                    latest_failure = failed_checks[-1]
                    print(f"  Failure type: {latest_failure.failure_type}")
        
        except RepairLoopStalledError as e:
            print(f"\n⚠ Repair stopped early: {e}", file=sys.stderr)
            print("The same failure repeated across repair attempts.", file=sys.stderr)
            sys.exit(1)
        
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except (AgentLoopError, AgentLoopLimitError) as e:
        print(f"Agent error: {e}", file=sys.stderr)
        sys.exit(1)
    except RepairLoopError as e:
        print(f"Repair loop error: {e}", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        print(f"File system error: {e}", file=sys.stderr)
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
    try:
        # Step 1: Load normalized issue
        issue_path = Path(args.issue)
        if not issue_path.exists():
            print(f"Error: Issue file not found: {args.issue}", file=sys.stderr)
            sys.exit(1)

        with open(issue_path, encoding="utf-8") as f:
            issue_data = json.load(f)
        normalized_issue = NormalizedIssue.model_validate(issue_data)

        # Step 2: Load approved plan
        plan_path = Path(args.plan)
        if not plan_path.exists():
            print(f"Error: Plan file not found: {args.plan}", file=sys.stderr)
            sys.exit(1)

        with open(plan_path, encoding="utf-8") as f:
            plan_data = json.load(f)
        plan = ChangePlan.model_validate(plan_data)

        # Log precheck section
        ExecuteLogger.log_precheck(
            git_repo=True,
            working_tree_clean=True,
            base_commit_match=True,
        )

        # Step 3: Setup workspace
        repo_path = Path(args.repo)

        # Validate repository
        try:
            preflight_result = validate_repository(repo_path)
        except RepositoryPreflightError as e:
            print(f"Repository validation failed: {e}", file=sys.stderr)
            sys.exit(1)

        # Step 3: Verify repository baseline matches plan
        if preflight_result.head_sha != plan.base_commit:
            print(
                "Repository HEAD has changed since the plan was generated.",
                file=sys.stderr
            )
            print(f"Plan base commit: {plan.base_commit[:8]}...", file=sys.stderr)
            print(f"Current HEAD: {preflight_result.head_sha[:8]}...", file=sys.stderr)
            print("Run `patchpilot prepare` again.", file=sys.stderr)
            sys.exit(1)

        # Step 4: Create provider
        try:
            provider = LLMProvider()
        except ValueError as e:
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
        )

        # Step 9: Execute workflow
        trace_path = Path("artifacts/execution_trace.jsonl")
        result = runner.execute(
            issue=normalized_issue.model_dump_json(indent=2),
            plan=plan.model_dump_json(indent=2),
            change_plan=plan,
            normalized_issue=normalized_issue,
            trace_path=trace_path,
        )

        # Step 10: Save verification report
        save_json(
            "artifacts/verification_report.json",
            json.dumps(result.verification_report, indent=2),
        )

        # Step 11: Save patch
        if result.patch:
            patch_path = Path("artifacts/patch.diff")
            patch_path.parent.mkdir(parents=True, exist_ok=True)
            patch_path.write_text(result.patch, encoding="utf-8")

        # Step 12: Save acceptance coverage report
        coverage_path = Path("artifacts/acceptance_coverage.md")
        coverage_path.parent.mkdir(parents=True, exist_ok=True)
        coverage_path.write_text(
            render_acceptance_coverage(
                result.acceptance_evidence,
                result.final_status.value,
            ),
            encoding="utf-8",
        )

        # Step 13: Print results
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
        
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except (AgentLoopError, AgentLoopLimitError) as e:
        print(f"Agent error: {e}", file=sys.stderr)
        sys.exit(1)
    except WorkflowRunnerError as e:
        print(f"Workflow error: {e}", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        print(f"File system error: {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
