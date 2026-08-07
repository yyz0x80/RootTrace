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
You are the Change Planner of PatchPilot.

Create a minimal file-level implementation plan.

Rules:

1. Only use files that actually exist in repository_files,
   unless a new file is clearly required.
2. Every planned change must be related to an acceptance criterion.
3. Do not expand product requirements.
4. Do not modify unrelated modules.
5. Prefer the smallest possible change.
6. Do not modify .env.
7. Do not modify CI/CD configuration.
8. Identify anything intentionally out of scope.
9. Return JSON only.

Required structure:

{
  "relevant_files": [],
  "planned_changes": [
    {
      "file": "...",
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
) -> ChangePlan:
    """Create a change plan for the given issue and repository.

    Args:
        issue: Normalized issue to plan changes for.
        repo_path: Path to the target repository.
        generate: Function to generate LLM responses from prompts.

    Returns:
        Structured change plan with files, changes, tests, and risk assessment.
    """
    repository_files = get_repository_files(repo_path)

    prompt = f"""
{PLANNER_PROMPT}

Normalized issue:

{issue.model_dump_json(indent=2)}

Repository files:

{json.dumps(repository_files, indent=2)}
"""

    response = generate(prompt)

    data = _extract_json(response)

    return ChangePlan.model_validate(data)
