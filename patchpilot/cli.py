"""CLI module for PatchPilot.

This module provides command-line interface for running PatchPilot
on local repositories with issue descriptions.
"""

import argparse
import sys
from pathlib import Path

from patchpilot.agent_loop import AgentLoop, AgentLoopError, AgentLoopLimitError
from patchpilot.issue.loader import load_issue
from patchpilot.issue.normalizer import normalize_issue
from patchpilot.planning.planner import create_plan
from patchpilot.planning.scope_gate import check_scope
from patchpilot.provider import LLMProvider
from patchpilot.tools import ToolRegistry
from patchpilot.utils import save_json
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
        
        # Run the agent with the plan
        result = agent_loop.run(issue=plan.model_dump_json(indent=2))
        
        # Print result
        print(result)
        
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except (AgentLoopError, AgentLoopLimitError) as e:
        print(f"Agent error: {e}", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        print(f"File system error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
