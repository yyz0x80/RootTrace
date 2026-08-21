import json
from collections.abc import Callable

from patchpilot.issue.loader import RawIssue
from patchpilot.issue.schema import (
    AcceptanceCriterion,
    ArtifactRequirement,
    NormalizedIssue,
    TaskConstraint,
)

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
4. Use ambiguous_points conservatively. Only report missing information when
   it would lead to at least two incompatible but reasonable externally
   observable behaviors. Do NOT report as ambiguous:
   - Multiple valid implementation approaches
   - Hypothetical scenarios not required by the issue
   - Details determinable from existing code or tests
   - Background context the model wants for understanding
   - Modifications already prohibited by security policies
5. Reasonable implementation choices that do not change product
   behavior may go into implementation_notes.
6. Acceptance criteria must have sequential IDs:
   AC-1, AC-2, AC-3...
7. Classify acceptance criteria by kind:
   - behavior: Final program behavior (e.g., "reject invalid input", "users can login")
   - preservation: Existing behavior must not regress (e.g., "keep function signature", "maintain backward compatibility")
   - structural: Required source code structure (e.g., "must call normalize_email", "use factory pattern")
8. Verification instructions (how to test) are NOT acceptance criteria. Put them in
   verification_requirements.
9. A request that a test, document, or configuration file must be included in the
   final patch is an artifact requirement, not an acceptance criterion.
10. Constraints must have sequential IDs: C-1, C-2, C-3...
11. Classify constraints by kind:
    - READ_SCOPE: Restrictions on what files can be read
    - WRITE_SCOPE: Restrictions on what files can be modified
    - COMMAND: Restrictions on commands that can be run
    - NETWORK: Restrictions on network access
    - OTHER: Other execution boundary constraints
12. Return JSON only. Do not return Markdown.

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
      "description": "...",
      "kind": "behavior|preservation|structural",
      "required": true
    }
  ],
  "constraints": [
    {
      "id": "C-1",
      "description": "...",
      "kind": "READ_SCOPE|WRITE_SCOPE|COMMAND|NETWORK|OTHER"
    }
  ],
  "verification_requirements": [],
  "artifact_requirements": [],
  "ambiguous_points": [],
  "expected_test_areas": [],
  "implementation_notes": []
}
"""

MAX_STRUCTURED_REPAIR_RESPONSE_CHARS = 4_000
_DESCRIPTION_LIST_FIELDS = (
    "ambiguous_points",
    "expected_test_areas",
    "implementation_notes",
    "verification_requirements",
)

_TEST_TERMS = ("test", "tests", "pytest", "coverage", "assertion", "assertions")
_VERIFICATION_TERMS = ("verify", "verifies", "verified", "cover", "covers", "exercise")
_PATCH_DELIVERY_TERMS = ("in the patch", "in the pr", "in the commit", "must include")


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


# High-confidence constraint kind patterns
_CONSTRAINT_KIND_PATTERNS = {
    "WRITE_SCOPE": (
        "do not modify",
        "must not modify",
        "keep unchanged",
        "read only",
        "read-only",
        "only modify",
        "no changes outside",
        "do not make any changes",
        "must not make any changes",
    ),
    "READ_SCOPE": (
        "do not access",
        "must not access",
    ),
    "COMMAND": (
        "do not run",
        "must not run",
        "do not install",
        "must not install",
        "git push",
    ),
    "NETWORK": (
        "network",
        "download",
        "fetch",
    ),
}


def _infer_constraint_kind(description: str) -> str:
    """Infer constraint kind from description with high-confidence patterns.

    Only applies high-confidence patterns. Returns OTHER if no match.
    """
    normalized = " ".join(description.lower().split())

    for kind, patterns in _CONSTRAINT_KIND_PATTERNS.items():
        if any(pattern in normalized for pattern in patterns):
            return kind

    return "OTHER"


def _migrate_string_constraints(
    constraints: list[str] | list[TaskConstraint],
) -> list[TaskConstraint]:
    """Migrate old string constraints to TaskConstraint objects.

    Args:
        constraints: List of either strings or TaskConstraint objects.

    Returns:
        List of TaskConstraint objects with inferred kinds and sequential IDs.
    """
    migrated: list[TaskConstraint] = []
    for index, constraint in enumerate(constraints, start=1):
        if isinstance(constraint, str):
            kind = _infer_constraint_kind(constraint)
            migrated.append(
                TaskConstraint(
                    id=f"C-{index}",
                    description=constraint,
                    kind=kind,  # type: ignore
                )
            )
        elif isinstance(constraint, TaskConstraint):
            # Ensure sequential ID for existing TaskConstraint objects
            migrated.append(
                constraint.model_copy(update={"id": f"C-{index}"})
            )
        else:
            # Skip invalid constraint types
            continue

    return migrated


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


def _is_test_verification_instruction(description: str) -> bool:
    """Return whether a criterion describes how behavior should be tested."""
    normalized = " ".join(description.lower().split())
    return any(term in normalized for term in _TEST_TERMS) and any(
        term in normalized for term in _VERIFICATION_TERMS
    )


def _requires_test_artifact(description: str) -> bool:
    """Return whether the text explicitly requires tests in the final patch."""
    normalized = " ".join(description.lower().split())
    return any(term in normalized for term in _TEST_TERMS) and any(
        term in normalized for term in _PATCH_DELIVERY_TERMS
    )


def _separate_execution_constraints(
    issue: NormalizedIssue,
) -> NormalizedIssue:
    """Move execution-boundary ACs into constraints and renumber AC IDs."""
    product_criteria: list[AcceptanceCriterion] = []
    constraint_descriptions = [
        c.description for c in issue.constraints
    ]
    verification_requirements = list(issue.verification_requirements)
    artifact_requirements = list(issue.artifact_requirements)

    for criterion in issue.acceptance_criteria:
        if _is_execution_constraint(criterion.description):
            if criterion.description not in constraint_descriptions:
                constraint_descriptions.append(criterion.description)
            continue

        if _is_test_verification_instruction(criterion.description):
            if criterion.description not in verification_requirements:
                verification_requirements.append(criterion.description)
            if _requires_test_artifact(criterion.description):
                artifact_requirements.append(
                    ArtifactRequirement(
                        kind="target_test_change",
                        description=criterion.description,
                        required=criterion.required,
                    )
                )
            continue

        product_criteria.append(criterion)

    normalized_criteria = [
        criterion.model_copy(update={"id": f"AC-{index}"})
        for index, criterion in enumerate(product_criteria, start=1)
    ]

    # Convert constraint descriptions to TaskConstraint objects
    migrated_constraints = _migrate_string_constraints(constraint_descriptions)

    return issue.model_copy(
        update={
            "acceptance_criteria": normalized_criteria,
            "constraints": migrated_constraints,
            "verification_requirements": verification_requirements,
            "artifact_requirements": artifact_requirements,
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

    # Handle constraints field separately for backward compatibility
    if "constraints" in repaired:
        constraints = repaired["constraints"]
        if isinstance(constraints, list):
            # If constraints are strings, migrate them to TaskConstraint objects
            if all(isinstance(c, str) for c in constraints):
                repaired["constraints"] = [
                    {
                        "id": f"C-{i+1}",
                        "description": c,
                        "kind": _infer_constraint_kind(c),
                    }
                    for i, c in enumerate(constraints)
                ]
            # If constraints are already objects, ensure they have required fields
            elif all(isinstance(c, dict) for c in constraints):
                normalized_constraints = []
                for i, c in enumerate(constraints, start=1):
                    if isinstance(c, str):
                        # Handle mixed string/object constraints
                        normalized_constraints.append({
                            "id": f"C-{i}",
                            "description": c,
                            "kind": _infer_constraint_kind(c),
                        })
                    elif isinstance(c, dict):
                        # Ensure ID and kind are present
                        if "id" not in c:
                            c["id"] = f"C-{i}"
                        if "kind" not in c:
                            c["kind"] = _infer_constraint_kind(c.get("description", ""))
                        normalized_constraints.append(c)
                repaired["constraints"] = normalized_constraints

    return repaired


def _post_process_constraints(
    constraints: list[TaskConstraint],
) -> list[TaskConstraint]:
    """Post-process constraints to fix high-confidence misclassifications.

    Only applies high-confidence patterns. Does not use broad keyword guessing.
    """
    processed: list[TaskConstraint] = []
    for constraint in constraints:
        # Re-apply high-confidence kind inference for safety
        inferred_kind = _infer_constraint_kind(constraint.description)
        if inferred_kind != "OTHER":
            # Use inferred kind if it's a high-confidence match
            processed.append(
                constraint.model_copy(update={"kind": inferred_kind})  # type: ignore
            )
        else:
            # Keep original kind if no high-confidence pattern matches
            processed.append(constraint)

    return processed


def _parse_normalized_issue(response: str) -> NormalizedIssue:
    """Parse and locally repair one normalized-issue response."""
    data = _repair_description_lists(_extract_json(response))
    normalized_issue = NormalizedIssue.model_validate(data)
    separated_issue = _separate_execution_constraints(normalized_issue)
    # Post-process constraints to fix high-confidence misclassifications
    processed_constraints = _post_process_constraints(separated_issue.constraints)
    return separated_issue.model_copy(update={"constraints": processed_constraints})


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

Return one corrected JSON object only. Preserve the issue meaning.
- acceptance_criteria must have id, description, kind (behavior|preservation|structural), and required (boolean)
- constraints must have id, description, and kind (READ_SCOPE|WRITE_SCOPE|COMMAND|NETWORK|OTHER)
- verification_requirements must contain verification instructions, not product behavior.
- artifact_requirements must describe file-level deliverables that must enter the patch.
- Items in ambiguous_points, expected_test_areas, implementation_notes, and verification_requirements must be JSON strings, not objects.
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
