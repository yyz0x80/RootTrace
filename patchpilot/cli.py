"""CLI module for PatchPilot.

This module provides command-line interface for running PatchPilot
on local repositories with issue descriptions.
"""

import argparse
import subprocess  # noqa: F401
import sys
from pathlib import Path

from patchpilot.agent_loop import AgentLoop, AgentLoopError, AgentLoopLimitError
from patchpilot.issue.loader import load_issue
from patchpilot.issue.normalizer import normalize_issue
from patchpilot.planning.planner import create_plan
from patchpilot.planning.scope_gate import check_scope
from patchpilot.prompts import REPAIR_PROMPT
from patchpilot.provider import LLMProvider
from patchpilot.sandbox.docker_runner import CommandResult
from patchpilot.tools import ToolRegistry
from patchpilot.utils import save_json
from patchpilot.verification.error_parser import parse_failure
from patchpilot.verification.report import CheckReport, VerificationReport
from patchpilot.workflow import (
    RepairLoopError,
    RepairLoopLimitError,
    RepairLoopStalledError,
    run_repair_loop,
)
from patchpilot.workspace import Workspace


def main() -> None:
    """Main entry point for the PatchPilot CLI."""
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
        default=12,
        help="Maximum number of agent rounds (default: 12)"
    )
    run_parser.add_argument(
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
    else:
        parser.print_help()
        sys.exit(1)


def handle_prepare(args) -> None:
    """Handle the prepare subcommand workflow.
    
    This function implements the prepare workflow:
    1. Load issue from file or GitHub
    2. Normalize the issue to extract structured information
    3. Check for ambiguous points
    4. Create a change plan
    5. Validate the plan against scope restrictions
    6. Output artifacts (normalized_issue.json, plan.json)
    7. Request user approval
    """
    try:
        # Step 1: Load raw issue
        raw_issue = load_issue(args.issue)
        print(f"Loaded issue from: {raw_issue.source}")
        print(f"Title: {raw_issue.title}\n")
        
        # Create provider for normalization and planning
        provider = LLMProvider()
        
        # Step 2: Normalize the issue
        print("Normalizing issue...")
        normalized_issue = normalize_issue(
            issue=raw_issue,
            generate=provider.generate_text,
        )
        
        # Step 3: Check for ambiguous points
        if normalized_issue.ambiguous_points:
            print("NEEDS_CLARIFICATION\n")
            print("The following requirements are ambiguous:\n")
            for i, point in enumerate(normalized_issue.ambiguous_points, start=1):
                print(f"{i}. {point}")
            print("\nPatchPilot will not guess product behavior.")
            sys.exit(1)
        
        print("Issue normalized successfully")
        print(f"Task type: {normalized_issue.task_type}")
        print(f"Acceptance criteria: {len(normalized_issue.acceptance_criteria)}")
        print()
        
        # Step 4: Create change plan
        print("Creating change plan...")
        repo_path = Path(args.repo)
        if not repo_path.exists():
            print(f"Error: Repository path not found: {args.repo}", file=sys.stderr)
            sys.exit(1)
        
        plan = create_plan(
            issue=normalized_issue,
            repo_path=str(repo_path),
            generate=provider.generate_text,
        )
        
        print(f"Plan created with {len(plan.planned_changes)} planned changes")
        print(f"Risk level: {plan.risk_level}")
        print()
        
        # Step 5: Check scope
        print("Checking scope...")
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
        
        print("Scope check passed")
        print()
        
        # Step 6: Output artifacts
        print("Saving artifacts...")
        save_json(
            "artifacts/normalized_issue.json",
            normalized_issue.model_dump_json(indent=2),
        )
        print("Saved: artifacts/normalized_issue.json")
        
        save_json(
            "artifacts/plan.json",
            plan.model_dump_json(indent=2),
        )
        print("Saved: artifacts/plan.json")
        print()
        
        # Step 7: Request user approval
        print("PREPARE_COMPLETE\n")
        print("Summary:")
        print(f"  Task type: {normalized_issue.task_type}")
        print(f"  Acceptance criteria: {len(normalized_issue.acceptance_criteria)}")
        print(f"  Planned changes: {len(plan.planned_changes)}")
        print(f"  Planned tests: {len(plan.planned_tests)}")
        print(f"  Risk level: {plan.risk_level}")
        print()
        print("Review the artifacts in artifacts/ directory.")
        print("To execute this plan, run: patchpilot run --repo <repo> --issue <issue>")
        
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
        provider = LLMProvider()
        
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
        if not repo_path.exists():
            print(f"Error: Repository path not found: {args.repo}", file=sys.stderr)
            sys.exit(1)
        
        workspace = Workspace(root=repo_path)
        
        # Create change plan
        plan = create_plan(
            issue=normalized_issue,
            repo_path=str(repo_path),
            generate=provider.generate_text,
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
            
            # Run quick verification (ruff)
            try:
                start_time = time.time()
                result = subprocess.run(
                    ["ruff", "check", "patchpilot/"],
                    cwd=workspace.root,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                )
                duration = time.time() - start_time
                
                if result.returncode == 0:
                    ruff_check = CheckReport(
                        level="quick",
                        command="ruff check patchpilot/",
                        passed=True,
                        exit_code=0,
                        duration_seconds=duration,
                    )
                else:
                    # Parse the failure
                    mock_result = CommandResult(
                        command="ruff check patchpilot/",
                        exit_code=result.returncode,
                        stdout=result.stdout,
                        stderr=result.stderr,
                        timed_out=False,
                    )
                    failure_summary = parse_failure(mock_result)
                    
                    ruff_check = CheckReport(
                        level="quick",
                        command="ruff check patchpilot/",
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
                    command="ruff check patchpilot/",
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
                        cwd=workspace.root,
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
    except (RepairLoopError, RepairLoopLimitError) as e:
        print(f"Repair loop error: {e}", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        print(f"File system error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
