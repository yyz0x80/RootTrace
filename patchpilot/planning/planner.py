"""Change Planner module for PatchPilot.

This module provides functionality to create implementation plans by analyzing
repository files and generating structured change plans based on normalized issues.
"""

import json
from collections.abc import Callable
from pathlib import Path, PurePosixPath

from patchpilot.issue.schema import NormalizedIssue
from patchpilot.planning.post_processor import post_process_plan
from patchpilot.planning.schema import ChangePlan
from patchpilot.planning.scope_gate import check_scope
from patchpilot.planning.validator import validate_acceptance_coverage
from patchpilot.repository.schema import RepositoryContext

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

MAX_PLAN_REPAIR_RESPONSE_CHARS = 5_000


def _is_test_file(path: str) -> bool:
    """Determine if a path refers to a test file.

    Args:
        path: File path to check.

    Returns:
        True if the path is a test file, False otherwise.
    """
    normalized = str(PurePosixPath(path))
    parts = normalized.split("/")

    # Check if file is in a tests/ directory
    if "tests" in parts:
        return True

    # Check if filename starts with test_
    filename = parts[-1] if parts else ""
    return filename.startswith("test_") and filename.endswith(".py")


class PlanGenerationError(ValueError):
    """Raised when a generated plan remains invalid after one repair."""


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
9. CRITICAL: Test files (files under tests/ or starting with test_) are READ-ONLY.
   NEVER include test files in planned_changes. Tests should only be in planned_tests for verification.
10. Test file modifications are FORBIDDEN. Even if the issue mentions updating tests,
    the agent should only modify source code and rely on existing tests for verification.
11. Map every acceptance criterion ID to at least one planned source change and
    one deterministic planned test. One change or test may map multiple criteria.
12. Constraints are execution boundaries, not acceptance criteria. Do not invent
    source changes merely to implement a read-only or security constraint.

Required structure:

{{
  "repository_match": true,
  "repository_mismatch_reason": null,
  "relevant_files": [],
  "planned_changes": [
    {{
      "path": "...",
      "action": "create|modify|delete",
      "description": "...",
      "acceptance_criteria": ["AC-1"]
    }}
  ],
  "planned_tests": [
    {{
      "command": "pytest ...",
      "purpose": "...",
      "acceptance_criteria": ["AC-1"]
    }}
  ],
  "out_of_scope": [],
  "risk_level": "low"
}}
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


def _parse_plan_response(response: str, base_commit: str) -> ChangePlan:
    """Parse one planner response and apply the authoritative base commit."""
    plan = ChangePlan.model_validate(_extract_json(response))
    plan.base_commit = base_commit
    return plan


def _validate_generated_plan_coverage(
    plan: ChangePlan,
    issue: NormalizedIssue,
) -> None:
    """Validate recoverable acceptance coverage errors in a model plan."""
    # Security and scope violations are terminal decisions, not omissions the
    # model should be prompted to repair into a more detailed plan.
    if plan.repository_match and check_scope(plan).allowed:
        validate_acceptance_coverage(plan, issue)


def _build_plan_repair_prompt(
    original_prompt: str,
    invalid_response: str,
    error: ValueError,
) -> str:
    """Build one bounded retry prompt for a malformed or incomplete plan."""
    bounded_response = invalid_response[-MAX_PLAN_REPAIR_RESPONSE_CHARS:]
    return f"""
{original_prompt}

Your previous plan was invalid.

Validation error:
{error}

Previous response:
{bounded_response}

Return one corrected JSON object only. Every acceptance criterion must map to
at least one planned source-code change and one deterministic planned test.
Files under tests/ and files named test_*.py are read-only and must never
appear in planned_changes. Preserve repository_match=false when the issue does
not match the repository; do not invent a missing subsystem.
"""


def _generate_plan_with_repair(
    *,
    issue: NormalizedIssue,
    prompt: str,
    generate: Callable[[str], str],
    base_commit: str,
    repository_context: RepositoryContext,
) -> ChangePlan:
    """Generate a plan and retry once with precise validation feedback."""
    response = generate(prompt)
    try:
        plan = _parse_plan_response(response, base_commit)
        plan = post_process_plan(plan, issue, repository_context)
        _validate_generated_plan_coverage(plan, issue)
        return plan
    except ValueError as error:
        repair_prompt = _build_plan_repair_prompt(
            original_prompt=prompt,
            invalid_response=response,
            error=error,
        )
        repaired_response = generate(repair_prompt)
        try:
            repaired_plan = _parse_plan_response(
                repaired_response,
                base_commit,
            )
            # Skip AC validation on repair to allow scope gate to check violations first
            repaired_plan = post_process_plan(
                repaired_plan,
                issue,
                repository_context,
                skip_ac_validation=True,
            )
            _validate_generated_plan_coverage(repaired_plan, issue)
            return repaired_plan
        except ValueError as repair_error:
            raise PlanGenerationError(
                "Planner returned an invalid plan after one repair: "
                f"{repair_error}"
            ) from repair_error


def create_plan(
    issue: NormalizedIssue,
    repository_context: RepositoryContext,
    generate: Callable[[str], str],
) -> ChangePlan:
    """Create a change plan for the given issue and repository.

    Args:
        issue: Normalized issue to plan changes for.
        repository_context: Repository context with file information.
        generate: Function to generate LLM responses from prompts.

    Returns:
        Structured change plan with files, changes, tests, and risk assessment.
    """
    repository_context_json = json.dumps(
        {
            "tracked_files": repository_context.tracked_files,
            "python_files": repository_context.python_files,
            "test_files": repository_context.test_files,
            "config_files": repository_context.config_files,
            "keyword_matches": repository_context.keyword_matches,
        },
        indent=2,
    )

    prompt = PLANNER_PROMPT.format(
        issue=issue.model_dump_json(indent=2),
        repository_context=repository_context_json,
    )

    return _generate_plan_with_repair(
        issue=issue,
        prompt=prompt,
        generate=generate,
        base_commit=repository_context.base_commit,
        repository_context=repository_context,
    )


def create_plan_with_path(
    issue: NormalizedIssue,
    repo_path: str,
    generate: Callable[[str], str],
    base_commit: str = "",
) -> ChangePlan:
    """Create a change plan for the given issue and repository path.

    This is a convenience function that creates a plan using repository path
    instead of RepositoryContext. It's kept for backward compatibility.

    Args:
        issue: Normalized issue to plan changes for.
        repo_path: Path to the target repository.
        generate: Function to generate LLM responses from prompts.
        base_commit: Current Git commit SHA to use as the plan's base.

    Returns:
        Structured change plan with files, changes, tests, and risk assessment.
    """
    repository_files = get_repository_files(repo_path)

    # Create a minimal RepositoryContext for post-processing
    repository_context = RepositoryContext(
        base_commit=base_commit,
        tracked_files=repository_files,
        python_files=[f for f in repository_files if f.endswith(".py")],
        test_files=[f for f in repository_files if _is_test_file(f)],
        config_files=[],
        keyword_matches=[],
    )

    repository_context_json = json.dumps(
        {
            "tracked_files": repository_context.tracked_files,
            "python_files": repository_context.python_files,
            "test_files": repository_context.test_files,
            "config_files": repository_context.config_files,
            "keyword_matches": repository_context.keyword_matches,
        },
        indent=2,
    )

    prompt = PLANNER_PROMPT.format(
        issue=issue.model_dump_json(indent=2),
        repository_context=repository_context_json,
    )

    return _generate_plan_with_repair(
        issue=issue,
        prompt=prompt,
        generate=generate,
        base_commit=base_commit,
        repository_context=repository_context,
    )
