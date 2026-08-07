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
from patchpilot.provider import LLMProvider
from patchpilot.tools import ToolRegistry
from patchpilot.workspace import Workspace


def main() -> None:
    """Main entry point for the PatchPilot CLI."""
    parser = argparse.ArgumentParser(
        description="PatchPilot: Issue-to-Patch Code Agent for Python repositories"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
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
    
    if args.command != "run":
        parser.print_help()
        sys.exit(1)
    
    try:
        # Load raw issue
        raw_issue = load_issue(args.issue)
        
        # Create provider for normalization
        provider = LLMProvider()
        
        # Normalize the issue
        normalized_issue = normalize_issue(
            issue=raw_issue,
            generate=provider.complete_text,
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
        
        # Create tool registry
        tools = ToolRegistry(workspace=workspace)
        
        # Create agent loop
        agent_loop = AgentLoop(
            provider=provider,
            tools=tools,
            max_rounds=args.max_rounds,
        )
        
        # Run the agent with normalized issue
        result = agent_loop.run(issue=normalized_issue.model_dump_json(indent=2))
        
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
