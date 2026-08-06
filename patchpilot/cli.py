"""CLI module for PatchPilot.

This module provides command-line interface for running PatchPilot
on local repositories with issue descriptions.
"""

import argparse
import sys
from pathlib import Path

from patchpilot.agent_loop import AgentLoop, AgentLoopError, AgentLoopLimitError
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
        # Read issue file
        issue_path = Path(args.issue)
        if not issue_path.exists():
            print(f"Error: Issue file not found: {args.issue}", file=sys.stderr)
            sys.exit(1)
        
        issue_content = issue_path.read_text(encoding="utf-8")
        
        # Create workspace
        repo_path = Path(args.repo)
        if not repo_path.exists():
            print(f"Error: Repository path not found: {args.repo}", file=sys.stderr)
            sys.exit(1)
        
        workspace = Workspace(root=repo_path)
        
        # Create provider
        provider = LLMProvider()
        
        # Create tool registry
        tools = ToolRegistry(workspace=workspace)
        
        # Create agent loop
        agent_loop = AgentLoop(
            provider=provider,
            tools=tools,
            max_rounds=args.max_rounds,
        )
        
        # Run the agent
        result = agent_loop.run(issue=issue_content)
        
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
