import json
from collections.abc import Callable

from patchpilot.issue.loader import RawIssue
from patchpilot.issue.schema import AcceptanceCriterion, NormalizedIssue

NORMALIZER_PROMPT = """
You are the Issue Normalizer of PatchPilot.

Your job is to convert a software issue into a structured task.

Important rules:

1. Only explicit, externally observable product behavior may become
   acceptance criteria.
2. Put execution boundaries in constraints, not acceptance criteria. This
   includes read-only files, allowed file scope, forbidden commands,
   dependency-install restrictions, and security or permission rules. Apply
   this semantic distinction even when the issue lists such boundaries under
   an "Acceptance requirements" heading.
3. Do NOT invent missing product behavior.
4. If missing information could affect externally visible behavior,
   put it into ambiguous_points.
5. Reasonable implementation choices that do not change product
   behavior may go into implementation_notes.
6. Acceptance criteria must have sequential IDs:
   AC-1, AC-2, AC-3...
7. Return JSON only. Do not return Markdown.

Allowed task_type:
bug, feature, test, refactor, dependency, other.

Required JSON shape:

{
  "title": "...",
  "task_type": "feature",
  "problem_statement": "...",
  "acceptance_criteria": [
    {
      "id": "AC-1",
      "description": "..."
    }
  ],
  "constraints": [],
  "ambiguous_points": [],
  "expected_test_areas": [],
  "implementation_notes": []
}
"""

MAX_STRUCTURED_REPAIR_RESPONSE_CHARS = 4_000
_DESCRIPTION_LIST_FIELDS = (
    "constraints",
    "ambiguous_points",
    "expected_test_areas",
    "implementation_notes",
)


_EXECUTION_BOUNDARY_MARKERS = (
    "do not access",
    "do not install",
    "do not make any changes",
    "do not modify",
    "do not run",
    "keep unchanged",
    "must not access",
    "must not install",
    "must not make any changes",
    "must not modify",
    "must not run",
    "no changes outside",
    "only modify",
    "read only",
    "read-only",
)

_EXECUTION_SCOPE_MARKERS = (
    ".env",
    "ci configuration",
    "ci workflow",
    "command",
    "dependency",
    "dependencies",
    "file",
    "files",
    "git push",
    "lockfile",
    "permission",
    "secret",
    "ssh key",
    "test",
    "tests",
)


def _is_execution_constraint(description: str) -> bool:
    """Return whether an AC describes an execution boundary.

    The rule is intentionally narrow: a negative phrase must refer to a
    repository, command, dependency, or security boundary. User-visible
    negative behavior, such as rejecting invalid input, remains an acceptance
    criterion.
    """
    normalized = " ".join(description.lower().split())

    has_boundary = any(
        marker in normalized
        for marker in _EXECUTION_BOUNDARY_MARKERS
    )
    has_scope = any(
        marker in normalized
        for marker in _EXECUTION_SCOPE_MARKERS
    )
    return has_boundary and has_scope


def _separate_execution_constraints(
    issue: NormalizedIssue,
) -> NormalizedIssue:
    """Move execution-boundary ACs into constraints and renumber AC IDs."""
    product_criteria: list[AcceptanceCriterion] = []
    constraints = list(issue.constraints)

    for criterion in issue.acceptance_criteria:
        if _is_execution_constraint(criterion.description):
            if criterion.description not in constraints:
                constraints.append(criterion.description)
            continue

        product_criteria.append(criterion)

    normalized_criteria = [
        criterion.model_copy(update={"id": f"AC-{index}"})
        for index, criterion in enumerate(product_criteria, start=1)
    ]

    return issue.model_copy(
        update={
            "acceptance_criteria": normalized_criteria,
            "constraints": constraints,
        }
    )


def _extract_json(text: str) -> dict:
    """Extract JSON object from LLM response text.

    Handles cases where the model wraps JSON in markdown code blocks.

    Args:
        text: Raw response text from the LLM.

    Returns:
        Parsed JSON dictionary.

    Raises:
        ValueError: If no valid JSON object can be found in the text.
    """
    text = text.strip()

    # Remove markdown code block wrapping if present
    if text.startswith("```"):
        lines = text.splitlines()

        if lines:
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines)

    # Find JSON object boundaries
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("LLM did not return a JSON object")

    return json.loads(text[start:end + 1])


def _repair_description_lists(data: dict) -> dict:
    """Normalize safe string-list variants returned by weaker models.

    A model may wrap a string in ``{"description": "..."}`` even when the
    schema requires a plain string. Only that lossless representation is
    repaired; unknown objects remain unchanged so schema validation can reject
    them.
    """
    repaired = dict(data)
    for field_name in _DESCRIPTION_LIST_FIELDS:
        values = repaired.get(field_name)
        if not isinstance(values, list):
            continue

        normalized_values = []
        for value in values:
            if isinstance(value, dict):
                description = value.get("description")
                if isinstance(description, str) and description.strip():
                    normalized_values.append(description)
                    continue
            normalized_values.append(value)
        repaired[field_name] = normalized_values

    return repaired


def _parse_normalized_issue(response: str) -> NormalizedIssue:
    """Parse and locally repair one normalized-issue response."""
    data = _repair_description_lists(_extract_json(response))
    normalized_issue = NormalizedIssue.model_validate(data)
    return _separate_execution_constraints(normalized_issue)


def _build_normalizer_repair_prompt(
    original_prompt: str,
    invalid_response: str,
    error: ValueError,
) -> str:
    """Build one bounded retry prompt for invalid structured output."""
    bounded_response = invalid_response[-MAX_STRUCTURED_REPAIR_RESPONSE_CHARS:]
    return f"""
{original_prompt}

Your previous JSON response did not match the required schema.

Validation error:
{error}

Previous response:
{bounded_response}

Return one corrected JSON object only. Preserve the issue meaning. Every item
in constraints, ambiguous_points, expected_test_areas, and
implementation_notes must be a JSON string, not an object.
"""


def normalize_issue(
    issue: RawIssue,
    generate: Callable[[str], str],
) -> NormalizedIssue:
    """Convert a raw issue into a structured normalized issue.

    Uses an LLM to analyze the issue and extract structured information
    including acceptance criteria, constraints, and implementation notes.

    Args:
        issue: The raw issue loaded from a file or GitHub.
        generate: A callable that takes a prompt and returns LLM response text.

    Returns:
        NormalizedIssue with structured task information.

    Raises:
        ValueError: If the LLM response cannot be parsed as valid JSON.
        ValidationError: If the parsed JSON does not match NormalizedIssue schema.
    """
    prompt = f"""
{NORMALIZER_PROMPT}

Issue title:
{issue.title}

Issue body:
{issue.body}
"""

    response = generate(prompt)
    try:
        return _parse_normalized_issue(response)
    except ValueError as error:
        repair_prompt = _build_normalizer_repair_prompt(
            original_prompt=prompt,
            invalid_response=response,
            error=error,
        )
        repaired_response = generate(repair_prompt)
        return _parse_normalized_issue(repaired_response)
