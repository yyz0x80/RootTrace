"""Change Planner module for PatchPilot.

This module provides functionality to create implementation plans by analyzing
repository files and generating structured change plans based on normalized issues.
"""

import json
from collections.abc import Callable
from pathlib import Path

from patchpilot.issue.schema import NormalizedIssue
from patchpilot.planning.schema import ChangePlan

# Directories to ignore when scanning repository files
IGNORED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
    ".tox",
    "dist",
    "build",
}


def get_repository_files(repo_path: str) -> list[str]:
    """Get all files in the repository, excluding ignored directories.

    Args:
        repo_path: Path to the repository root directory.

    Returns:
        Sorted list of relative file paths from the repository root.
    """
    root = Path(repo_path)

    if not root.exists() or not root.is_dir():
        return []

    result = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        relative = path.relative_to(root)

        # Skip files in ignored directories
        if any(part in IGNORED_DIRS for part in relative.parts):
            continue

        result.append(str(relative))

    return sorted(result)


PLANNER_PROMPT = """
You are creating a scoped implementation plan.

Issue:
{issue}

Repository context:
{repository_context}

Rules:

1. The repository context is authoritative.
2. Never invent existing file paths.
3. For action="modify" or action="delete",
   the file must already exist in tracked_files.
4. action="create" may reference a file that does not yet exist.
5. New files are allowed only when necessary to implement the issue.
6. If the issue clearly assumes an existing component but no such
   component can be found in the repository, do not create the whole
   subsystem from scratch.
7. In that case set:
   repository_match=false
   and explain repository_mismatch_reason.
8. Stay within the requested issue scope.

Required structure:

{
  "repository_match": true,
  "repository_mismatch_reason": null,
  "relevant_files": [],
  "planned_changes": [
    {
      "path": "...",
      "action": "create|modify|delete",
      "description": "...",
      "acceptance_criteria": ["AC-1"]
    }
  ],
  "planned_tests": [
    {
      "command": "pytest ...",
      "purpose": "...",
      "acceptance_criteria": ["AC-1"]
    }
  ],
  "out_of_scope": [],
  "risk_level": "low"
}
"""


def _extract_json(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks.

    Args:
        text: Raw LLM response text.

    Returns:
        Parsed JSON dictionary.

    Raises:
        ValueError: If no valid JSON is found in the response.
    """
    text = text.strip()

    # Remove markdown code block markers if present
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines)

    # Find JSON object boundaries
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("Planner did not return JSON")

    return json.loads(text[start : end + 1])


def create_plan(
    issue: NormalizedIssue,
    repo_path: str,
    generate: Callable[[str], str],
    base_commit: str = "",
) -> ChangePlan:
    """Create a change plan for the given issue and repository.

    Args:
        issue: Normalized issue to plan changes for.
        repo_path: Path to the target repository.
        generate: Function to generate LLM responses from prompts.
        base_commit: Current Git commit SHA to use as the plan's base.

    Returns:
        Structured change plan with files, changes, tests, and risk assessment.
    """
    repository_files = get_repository_files(repo_path)
    repository_context = json.dumps(
        {"tracked_files": repository_files}, indent=2
    )

    prompt = PLANNER_PROMPT.format(
        issue=issue.model_dump_json(indent=2),
        repository_context=repository_context,
    )

    response = generate(prompt)

    data = _extract_json(response)

    plan = ChangePlan.model_validate(data)

    # Set base_commit from repository context (not from LLM)
    plan.base_commit = base_commit

    return plan
