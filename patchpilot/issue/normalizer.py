import json
from collections.abc import Callable

from patchpilot.issue.loader import RawIssue
from patchpilot.issue.schema import NormalizedIssue

NORMALIZER_PROMPT = """
You are the Issue Normalizer of PatchPilot.

Your job is to convert a software issue into a structured task.

Important rules:

1. Only explicit product requirements may become acceptance criteria.
2. Do NOT invent missing product behavior.
3. If missing information could affect externally visible behavior,
   put it into ambiguous_points.
4. Reasonable implementation choices that do not change product
   behavior may go into implementation_notes.
5. Acceptance criteria must have sequential IDs:
   AC-1, AC-2, AC-3...
6. Return JSON only. Do not return Markdown.

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

    data = _extract_json(response)

    return NormalizedIssue.model_validate(data)
