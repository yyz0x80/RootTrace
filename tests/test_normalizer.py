import pytest
from pydantic import ValidationError

from patchpilot.issue.loader import RawIssue
from patchpilot.issue.normalizer import (
    _extract_json,
    _infer_constraint_kind,
    _migrate_string_constraints,
    normalize_issue,
)
from patchpilot.issue.schema import (
    NormalizedIssue,
    TaskConstraint,
)


def test_extract_json_from_plain_json():
    """Extract JSON from plain JSON response."""
    text = '{"title": "Test", "task_type": "bug"}'
    result = _extract_json(text)
    assert result == {"title": "Test", "task_type": "bug"}


def test_extract_json_from_markdown_block():
    """Extract JSON from markdown code block."""
    text = '''```json
{
  "title": "Test",
  "task_type": "bug"
}
```'''
    result = _extract_json(text)
    assert result == {"title": "Test", "task_type": "bug"}


def test_extract_json_from_markdown_without_language():
    """Extract JSON from markdown code block without language specifier."""
    text = '''```
{
  "title": "Test",
  "task_type": "bug"
}
```'''
    result = _extract_json(text)
    assert result == {"title": "Test", "task_type": "bug"}


def test_extract_json_with_extra_text():
    """Extract JSON when there's extra text around it."""
    text = 'Here is the result: {"title": "Test", "task_type": "bug"}'
    result = _extract_json(text)
    assert result == {"title": "Test", "task_type": "bug"}


def test_extract_json_invalid_no_braces():
    """Raise ValueError when no JSON object is found."""
    with pytest.raises(ValueError, match="LLM did not return a JSON object"):
        _extract_json("This is not JSON")


def test_extract_json_invalid_only_opening_brace():
    """Raise ValueError when JSON object is incomplete."""
    with pytest.raises(ValueError, match="LLM did not return a JSON object"):
        _extract_json('{"title": "Test"')


def test_normalize_issue_with_mock_generate():
    """Test normalize_issue with a mock generate function."""
    issue = RawIssue(
        title="Fix authentication bug",
        body="Users cannot login when using special characters in password",
        source="test"
    )

    def mock_generate(prompt: str) -> str:
        return """{
  "title": "Fix authentication bug",
  "task_type": "bug",
  "problem_statement": "Users cannot login when using special characters in password",
  "acceptance_criteria": [
    {
      "id": "AC-1",
      "description": "Users can login with special characters in password",
      "kind": "behavior",
      "required": true
    }
  ],
  "constraints": [],
  "ambiguous_points": [],
  "expected_test_areas": ["tests/test_auth.py"],
  "implementation_notes": []
}"""

    result = normalize_issue(issue, mock_generate)

    assert isinstance(result, NormalizedIssue)
    assert result.title == "Fix authentication bug"
    assert result.task_type == "bug"
    assert len(result.acceptance_criteria) == 1
    assert result.acceptance_criteria[0].id == "AC-1"
    assert result.acceptance_criteria[0].kind == "behavior"


def test_normalize_issue_separates_test_verification_from_product_criteria():
    """Move test instructions into verification requirements."""
    issue = RawIssue(
        title="Add description",
        body="Add a description field and update tests to verify it.",
        source="test",
    )

    def mock_generate(prompt: str) -> str:
        return """{
  "title": "Add description",
  "task_type": "feature",
  "problem_statement": "Tasks need descriptions.",
  "acceptance_criteria": [
    {
      "id": "AC-1",
      "description": "Task includes a description field.",
      "kind": "structural",
      "required": true
    },
    {
      "id": "AC-2",
      "description": "Tests must verify the description field.",
      "kind": "behavior",
      "required": true
    }
  ],
  "constraints": [],
  "verification_requirements": [],
  "artifact_requirements": [],
  "ambiguous_points": [],
  "expected_test_areas": [],
  "implementation_notes": []
}"""

    result = normalize_issue(issue, mock_generate)

    assert [criterion.id for criterion in result.acceptance_criteria] == ["AC-1"]
    assert result.verification_requirements == [
        "Tests must verify the description field."
    ]
    assert result.artifact_requirements == []


def test_normalize_issue_preserves_explicit_test_patch_requirement():
    """Record an explicit request for tests in the final patch."""
    issue = RawIssue(
        title="Add coverage",
        body="The patch must include tests that verify the behavior.",
        source="test",
    )

    def mock_generate(prompt: str) -> str:
        return """{
  "title": "Add coverage",
  "task_type": "test",
  "problem_statement": "Permanent regression coverage is required.",
  "acceptance_criteria": [
    {
      "id": "AC-1",
      "description": "The patch must include tests that verify the behavior.",
      "kind": "behavior",
      "required": true
    }
  ],
  "constraints": [],
  "ambiguous_points": [],
  "expected_test_areas": [],
  "implementation_notes": []
}"""

    result = normalize_issue(issue, mock_generate)

    assert result.acceptance_criteria == []
    assert len(result.artifact_requirements) == 1
    assert result.artifact_requirements[0].kind == "target_test_change"


def test_normalize_issue_recovers_update_tests_as_artifact_requirement():
    """Treat an explicit request to update tests as a patch deliverable."""
    issue = RawIssue(
        title="Add description",
        body="Update tests to verify the description field works correctly.",
        source="test",
    )

    def mock_generate(prompt: str) -> str:
        return """{
  "title": "Add description",
  "task_type": "feature",
  "problem_statement": "Tasks need descriptions.",
  "acceptance_criteria": [],
  "constraints": [],
  "verification_requirements": [
    "Update tests to verify the description field works correctly"
  ],
  "artifact_requirements": [],
  "ambiguous_points": [],
  "expected_test_areas": [],
  "implementation_notes": []
}"""

    result = normalize_issue(issue, mock_generate)

    assert result.verification_requirements == [
        "Update tests to verify the description field works correctly"
    ]
    assert len(result.artifact_requirements) == 1
    assert result.artifact_requirements[0].kind == "target_test_change"


def test_normalize_issue_repairs_acceptance_kind_on_execution_constraint():
    """Infer a valid constraint kind when the model uses an AC kind."""
    issue = RawIssue(
        title="Add description",
        body="Add a description without modifying tests.",
        source="test",
    )
    call_count = 0

    def mock_generate(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        return """{
  "title": "Add description",
  "task_type": "feature",
  "problem_statement": "Tasks need descriptions.",
  "acceptance_criteria": [],
  "constraints": [
    {
      "id": "C-1",
      "description": "Do not modify tests.",
      "kind": "preservation"
    }
  ],
  "ambiguous_points": [],
  "expected_test_areas": [],
  "implementation_notes": []
}"""

    result = normalize_issue(issue, mock_generate)

    assert call_count == 1
    assert result.constraints[0].kind == "WRITE_SCOPE"
    assert result.constraints[0].description == "Do not modify tests."


def test_normalize_issue_moves_misplaced_preservation_constraint_to_ac():
    """Move a product preservation requirement out of constraints."""
    issue = RawIssue(
        title="Add description",
        body="Add a description while preserving task titles.",
        source="test",
    )

    def mock_generate(prompt: str) -> str:
        return """{
  "title": "Add description",
  "task_type": "feature",
  "problem_statement": "Tasks need descriptions.",
  "acceptance_criteria": [],
  "constraints": [
    {
      "id": "C-1",
      "description": "Existing task titles remain unchanged.",
      "kind": "preservation"
    }
  ],
  "ambiguous_points": [],
  "expected_test_areas": [],
  "implementation_notes": []
}"""

    result = normalize_issue(issue, mock_generate)

    assert result.constraints == []
    assert len(result.acceptance_criteria) == 1
    assert result.acceptance_criteria[0].id == "AC-1"
    assert result.acceptance_criteria[0].kind == "preservation"


def test_normalize_issue_handles_markdown_response():
    """Test normalize_issue when LLM returns markdown-wrapped JSON."""
    issue = RawIssue(
        title="Add pagination",
        body="Need pagination for user list",
        source="test"
    )

    def mock_generate(prompt: str) -> str:
        return '''```json
{
  "title": "Add pagination",
  "task_type": "feature",
  "problem_statement": "Need pagination for user list",
  "acceptance_criteria": [
    {
      "id": "AC-1",
      "description": "User list supports page and limit parameters",
      "kind": "behavior",
      "required": true
    }
  ],
  "constraints": [],
  "ambiguous_points": [],
  "expected_test_areas": [],
  "implementation_notes": []
}
```'''

    result = normalize_issue(issue, mock_generate)

    assert isinstance(result, NormalizedIssue)
    assert result.title == "Add pagination"
    assert result.task_type == "feature"


def test_normalize_issue_validates_schema():
    """Test that normalize_issue validates against NormalizedIssue schema."""
    issue = RawIssue(
        title="Test issue",
        body="Test body",
        source="test"
    )

    def mock_generate(prompt: str) -> str:
        return '{"title": "Test", "task_type": "invalid_type"}'

    with pytest.raises(ValidationError):  # Pydantic validation error
        normalize_issue(issue, mock_generate)


def test_normalize_issue_with_ambiguous_points():
    """Test that ambiguous_points field is correctly populated."""
    issue = RawIssue(
        title="Add user preference setting",
        body="Users should be able to set preferences",
        source="test"
    )

    def mock_generate(prompt: str) -> str:
        return """{
  "title": "Add user preference setting",
  "task_type": "feature",
  "problem_statement": "Users should be able to set preferences",
  "acceptance_criteria": [
    {
      "id": "AC-1",
      "description": "Preference settings are accessible",
      "kind": "behavior",
      "required": true
    }
  ],
  "constraints": [],
  "ambiguous_points": [
    "It is unclear where preferences should be stored (database vs file)",
    "Default values for preferences are not specified"
  ],
  "expected_test_areas": [],
  "implementation_notes": []
}"""

    result = normalize_issue(issue, mock_generate)

    assert isinstance(result, NormalizedIssue)
    assert len(result.ambiguous_points) == 2
    assert "It is unclear where preferences should be stored" in result.ambiguous_points[0]
    assert "Default values for preferences are not specified" in result.ambiguous_points[1]


def test_normalize_issue_with_empty_ambiguous_points():
    """Test that ambiguous_points defaults to empty list when not provided."""
    issue = RawIssue(
        title="Simple bug fix",
        body="Fix a typo in the README",
        source="test"
    )

    def mock_generate(prompt: str) -> str:
        return """{
  "title": "Simple bug fix",
  "task_type": "bug",
  "problem_statement": "Fix a typo in the README",
  "acceptance_criteria": [
    {
      "id": "AC-1",
      "description": "Typo is fixed",
      "kind": "behavior",
      "required": true
    }
  ],
  "constraints": [],
  "ambiguous_points": [],
  "expected_test_areas": [],
  "implementation_notes": []
}"""

    result = normalize_issue(issue, mock_generate)

    assert isinstance(result, NormalizedIssue)
    assert result.ambiguous_points == []


def test_normalize_issue_with_complex_ambiguity():
    """Test normalization with multiple types of ambiguity."""
    issue = RawIssue(
        title="Implement caching",
        body="Add caching to improve performance",
        source="test"
    )

    def mock_generate(prompt: str) -> str:
        return """{
  "title": "Implement caching",
  "task_type": "feature",
  "problem_statement": "Add caching to improve performance",
  "acceptance_criteria": [
    {
      "id": "AC-1",
      "description": "Cache is implemented",
      "kind": "behavior",
      "required": true
    }
  ],
  "constraints": [],
  "ambiguous_points": [
    "Cache invalidation strategy is not specified",
    "TTL (time-to-live) values are not defined",
    "Which data should be cached is unclear",
    "Cache backend choice (Redis vs in-memory) is not specified"
  ],
  "expected_test_areas": [],
  "implementation_notes": []
}"""

    result = normalize_issue(issue, mock_generate)

    assert isinstance(result, NormalizedIssue)
    assert len(result.ambiguous_points) == 4
    assert any("invalidation" in point for point in result.ambiguous_points)
    assert any("TTL" in point for point in result.ambiguous_points)


def test_normalize_issue_separates_execution_constraints():
    """Move repository execution boundaries out of acceptance criteria."""
    issue = RawIssue(
        title="Refactor price aggregation",
        body=(
            "Use sum, keep tests read-only, and do not make any changes "
            "outside of pricing.py."
        ),
        source="test",
    )
    captured_prompt = ""

    def mock_generate(prompt: str) -> str:
        nonlocal captured_prompt
        captured_prompt = prompt
        return """{
  "title": "Refactor price aggregation",
  "task_type": "refactor",
  "problem_statement": "Replace a manual aggregation loop.",
  "acceptance_criteria": [
    {
      "id": "AC-1",
      "description": "Use sum(prices, start=0.0).",
      "kind": "behavior",
      "required": true
    },
    {
      "id": "AC-2",
      "description": "Keep the existing tests read-only.",
      "kind": "behavior",
      "required": true
    },
    {
      "id": "AC-3",
      "description": "Reject non-positive page numbers.",
      "kind": "behavior",
      "required": true
    },
    {
      "id": "AC-4",
      "description": "Do not make any changes outside of the pricing.py file.",
      "kind": "behavior",
      "required": true
    }
  ],
  "constraints": [
    {
      "id": "C-1",
      "description": "Do not run git push.",
      "kind": "COMMAND"
    }
  ],
  "ambiguous_points": [],
  "expected_test_areas": ["tests/test_pricing.py"],
  "implementation_notes": []
}"""

    result = normalize_issue(issue, mock_generate)

    assert "execution boundaries in constraints" in captured_prompt
    assert [
        criterion.id
        for criterion in result.acceptance_criteria
    ] == ["AC-1", "AC-2"]
    assert [
        criterion.description
        for criterion in result.acceptance_criteria
    ] == [
        "Use sum(prices, start=0.0).",
        "Reject non-positive page numbers.",
    ]
    # Check that constraints are TaskConstraint objects
    assert len(result.constraints) == 3
    assert all(isinstance(c, TaskConstraint) for c in result.constraints)
    assert result.constraints[0].description == "Do not run git push."
    assert result.constraints[0].kind == "COMMAND"
    assert result.constraints[1].description == "Keep the existing tests read-only."
    assert result.constraints[1].kind == "WRITE_SCOPE"
    assert result.constraints[2].description == "Do not make any changes outside of the pricing.py file."
    assert result.constraints[2].kind == "WRITE_SCOPE"


def test_normalize_issue_repairs_description_wrapped_string_lists():
    """Safely unwrap description objects used in string-list fields."""
    issue = RawIssue(
        title="Clarify ordering",
        body="The ordering rule is not defined.",
        source="test",
    )
    calls = 0

    def mock_generate(prompt: str) -> str:
        nonlocal calls
        calls += 1
        return """{
  "title": "Clarify ordering",
  "task_type": "feature",
  "problem_statement": "The ordering rule is not defined.",
  "acceptance_criteria": [],
  "constraints": [],
  "ambiguous_points": [
    {"description": "Whether oldest or newest items come first is unclear."}
  ],
  "expected_test_areas": [],
  "implementation_notes": []
}"""

    result = normalize_issue(issue, mock_generate)

    assert calls == 1
    assert result.ambiguous_points == [
        "Whether oldest or newest items come first is unclear."
    ]


def test_normalize_issue_retries_invalid_structured_output_once():
    """Retry one malformed response with the validation error."""
    issue = RawIssue(title="Fix parser", body="Parsing fails.", source="test")
    responses = iter(
        [
            '{"title": "Fix parser", "task_type": "invalid"}',
            """{
  "title": "Fix parser",
  "task_type": "bug",
  "problem_statement": "Parsing fails.",
  "acceptance_criteria": [],
  "constraints": [],
  "ambiguous_points": [],
  "expected_test_areas": [],
  "implementation_notes": []
}""",
        ]
    )
    prompts: list[str] = []

    def mock_generate(prompt: str) -> str:
        prompts.append(prompt)
        return next(responses)

    result = normalize_issue(issue, mock_generate)

    assert result.task_type == "bug"
    assert len(prompts) == 2
    assert "previous JSON response did not match" in prompts[1]


def test_behavior_criterion_classification():
    """Behavior criterion: 'reject invalid input' is classified as behavior."""
    issue = RawIssue(
        title="Inventory validation",
        body="Ensure inventory is not modified for invalid input",
        source="test"
    )

    def mock_generate(prompt: str) -> str:
        return """{
  "title": "Inventory validation",
  "task_type": "feature",
  "problem_statement": "Ensure inventory is not modified for invalid input",
  "acceptance_criteria": [
    {
      "id": "AC-1",
      "description": "Invalid input does not modify inventory",
      "kind": "behavior",
      "required": true
    }
  ],
  "constraints": [],
  "ambiguous_points": [],
  "expected_test_areas": [],
  "implementation_notes": []
}"""

    result = normalize_issue(issue, mock_generate)

    assert len(result.acceptance_criteria) == 1
    assert result.acceptance_criteria[0].kind == "behavior"
    assert result.acceptance_criteria[0].required is True


def test_write_constraint_classification():
    """Write constraint: 'do not modify tests' is classified as WRITE_SCOPE."""
    issue = RawIssue(
        title="Feature implementation",
        body="Implement feature while keeping tests read-only",
        source="test"
    )

    def mock_generate(prompt: str) -> str:
        return """{
  "title": "Feature implementation",
  "task_type": "feature",
  "problem_statement": "Implement feature while keeping tests read-only",
  "acceptance_criteria": [
    {
      "id": "AC-1",
      "description": "Feature is implemented",
      "kind": "behavior",
      "required": true
    }
  ],
  "constraints": [
    {
      "id": "C-1",
      "description": "Do not modify tests",
      "kind": "WRITE_SCOPE"
    }
  ],
  "ambiguous_points": [],
  "expected_test_areas": [],
  "implementation_notes": []
}"""

    result = normalize_issue(issue, mock_generate)

    assert len(result.constraints) == 1
    assert result.constraints[0].kind == "WRITE_SCOPE"
    assert result.constraints[0].description == "Do not modify tests"


def test_preservation_criterion_classification():
    """Preservation criterion: 'keep function signature' is classified as preservation."""
    issue = RawIssue(
        title="Refactor function",
        body="Refactor function internals while keeping signature",
        source="test"
    )

    def mock_generate(prompt: str) -> str:
        return """{
  "title": "Refactor function",
  "task_type": "refactor",
  "problem_statement": "Refactor function internals while keeping signature",
  "acceptance_criteria": [
    {
      "id": "AC-1",
      "description": "Function signature remains unchanged",
      "kind": "preservation",
      "required": true
    }
  ],
  "constraints": [],
  "ambiguous_points": [],
  "expected_test_areas": [],
  "implementation_notes": []
}"""

    result = normalize_issue(issue, mock_generate)

    assert len(result.acceptance_criteria) == 1
    assert result.acceptance_criteria[0].kind == "preservation"


def test_structural_criterion_classification():
    """Structural criterion: 'must call normalize_email' is classified as structural."""
    issue = RawIssue(
        title="Email validation",
        body="Add email validation using normalize_email function",
        source="test"
    )

    def mock_generate(prompt: str) -> str:
        return """{
  "title": "Email validation",
  "task_type": "feature",
  "problem_statement": "Add email validation using normalize_email function",
  "acceptance_criteria": [
    {
      "id": "AC-1",
      "description": "Must call normalize_email before validation",
      "kind": "structural",
      "required": true
    }
  ],
  "constraints": [],
  "ambiguous_points": [],
  "expected_test_areas": [],
  "implementation_notes": []
}"""

    result = normalize_issue(issue, mock_generate)

    assert len(result.acceptance_criteria) == 1
    assert result.acceptance_criteria[0].kind == "structural"


def test_schema_repair_old_string_constraints():
    """Old string constraints are migrated to TaskConstraint objects."""
    issue = RawIssue(
        title="Legacy constraint test",
        body="Test with old string constraint format",
        source="test"
    )

    def mock_generate(prompt: str) -> str:
        return """{
  "title": "Legacy constraint test",
  "task_type": "feature",
  "problem_statement": "Test with old string constraint format",
  "acceptance_criteria": [],
  "constraints": [
    "Keep tests read-only",
    "Do not access .env",
    "Only modify pricing.py"
  ],
  "ambiguous_points": [],
  "expected_test_areas": [],
  "implementation_notes": []
}"""

    result = normalize_issue(issue, mock_generate)

    assert len(result.constraints) == 3
    assert isinstance(result.constraints[0], TaskConstraint)
    assert result.constraints[0].id == "C-1"
    assert result.constraints[0].description == "Keep tests read-only"
    assert result.constraints[0].kind == "WRITE_SCOPE"

    assert result.constraints[1].id == "C-2"
    assert result.constraints[1].description == "Do not access .env"
    assert result.constraints[1].kind == "READ_SCOPE"

    assert result.constraints[2].id == "C-3"
    assert result.constraints[2].description == "Only modify pricing.py"
    assert result.constraints[2].kind == "WRITE_SCOPE"


def test_constraint_id_stability():
    """Constraint IDs remain stable and sequential."""
    issue = RawIssue(
        title="ID stability test",
        body="Test constraint ID assignment",
        source="test"
    )

    def mock_generate(prompt: str) -> str:
        return """{
  "title": "ID stability test",
  "task_type": "feature",
  "problem_statement": "Test constraint ID assignment",
  "acceptance_criteria": [],
  "constraints": [
    {
      "id": "C-1",
      "description": "First constraint",
      "kind": "WRITE_SCOPE"
    },
    {
      "id": "C-2",
      "description": "Second constraint",
      "kind": "READ_SCOPE"
    }
  ],
  "ambiguous_points": [],
  "expected_test_areas": [],
  "implementation_notes": []
}"""

    result = normalize_issue(issue, mock_generate)

    assert result.constraints[0].id == "C-1"
    assert result.constraints[1].id == "C-2"


def test_constraint_order_stability():
    """Constraint order is preserved during migration."""
    issue = RawIssue(
        title="Order stability test",
        body="Test constraint order preservation",
        source="test"
    )

    def mock_generate(prompt: str) -> str:
        return """{
  "title": "Order stability test",
  "task_type": "feature",
  "problem_statement": "Test constraint order preservation",
  "acceptance_criteria": [],
  "constraints": [
    "First constraint",
    "Second constraint",
    "Third constraint"
  ],
  "ambiguous_points": [],
  "expected_test_areas": [],
  "implementation_notes": []
}"""

    result = normalize_issue(issue, mock_generate)

    assert result.constraints[0].description == "First constraint"
    assert result.constraints[1].description == "Second constraint"
    assert result.constraints[2].description == "Third constraint"


def test_infer_constraint_kind():
    """Test constraint kind inference for high-confidence patterns."""
    # WRITE_SCOPE patterns
    assert _infer_constraint_kind("Do not modify tests") == "WRITE_SCOPE"
    assert _infer_constraint_kind("Keep tests read-only") == "WRITE_SCOPE"
    assert _infer_constraint_kind("Only modify pricing.py") == "WRITE_SCOPE"

    # READ_SCOPE patterns
    assert _infer_constraint_kind("Do not access .env") == "READ_SCOPE"
    assert _infer_constraint_kind("Must not access secrets") == "READ_SCOPE"

    # COMMAND patterns
    assert _infer_constraint_kind("Do not run git push") == "COMMAND"
    assert _infer_constraint_kind("Must not install dependencies") == "COMMAND"

    # NETWORK patterns
    assert _infer_constraint_kind("No network access") == "NETWORK"
    assert _infer_constraint_kind("Do not download files") == "NETWORK"

    # Default to OTHER for unknown patterns
    assert _infer_constraint_kind("Some random constraint") == "OTHER"


def test_migrate_string_constraints():
    """Test migration of string constraints to TaskConstraint objects."""
    string_constraints = [
        "Keep tests read-only",
        "Do not access .env",
        "Do not run git push",
    ]

    result = _migrate_string_constraints(string_constraints)

    assert len(result) == 3
    assert result[0].id == "C-1"
    assert result[0].description == "Keep tests read-only"
    assert result[0].kind == "WRITE_SCOPE"

    assert result[1].id == "C-2"
    assert result[1].description == "Do not access .env"
    assert result[1].kind == "READ_SCOPE"

    assert result[2].id == "C-3"
    assert result[2].description == "Do not run git push"
    assert result[2].kind == "COMMAND"


def test_migrate_mixed_constraints():
    """Test migration with mixed string and TaskConstraint objects."""
    mixed_constraints = [
        "Keep tests read-only",
        TaskConstraint(
            id="C-2",
            description="Do not access .env",
            kind="READ_SCOPE",
        ),
        "Do not run git push",
    ]

    result = _migrate_string_constraints(mixed_constraints)

    assert len(result) == 3
    assert result[0].id == "C-1"
    assert result[0].description == "Keep tests read-only"
    assert result[0].kind == "WRITE_SCOPE"

    assert result[1].id == "C-2"
    assert result[1].description == "Do not access .env"
    assert result[1].kind == "READ_SCOPE"

    assert result[2].id == "C-3"
    assert result[2].description == "Do not run git push"
    assert result[2].kind == "COMMAND"


def test_post_process_constraints_high_confidence():
    """Post-processor fixes high-confidence constraint misclassifications."""
    issue = RawIssue(
        title="Post-process test",
        body="Test constraint post-processing",
        source="test"
    )

    def mock_generate(prompt: str) -> str:
        return """{
  "title": "Post-process test",
  "task_type": "feature",
  "problem_statement": "Test constraint post-processing",
  "acceptance_criteria": [],
  "constraints": [
    {
      "id": "C-1",
      "description": "Do not modify tests",
      "kind": "OTHER"
    },
    {
      "id": "C-2",
      "description": "Keep tests read-only",
      "kind": "OTHER"
    }
  ],
  "ambiguous_points": [],
  "expected_test_areas": [],
  "implementation_notes": []
}"""

    result = normalize_issue(issue, mock_generate)

    # Post-processor should fix the high-confidence misclassifications
    assert result.constraints[0].kind == "WRITE_SCOPE"
    assert result.constraints[1].kind == "WRITE_SCOPE"


def test_verification_not_in_acceptance_criteria():
    """Verification instructions are not classified as acceptance criteria."""
    issue = RawIssue(
        title="Feature with verification",
        body="Implement feature with verification steps",
        source="test"
    )

    def mock_generate(prompt: str) -> str:
        return """{
  "title": "Feature with verification",
  "task_type": "feature",
  "problem_statement": "Implement feature with verification steps",
  "acceptance_criteria": [
    {
      "id": "AC-1",
      "description": "Feature works correctly",
      "kind": "behavior",
      "required": true
    }
  ],
  "constraints": [],
  "ambiguous_points": [],
  "expected_test_areas": [],
  "implementation_notes": [
    "Verify by running pytest tests/test_feature.py",
    "Check that feature passes integration tests"
  ]
}"""

    result = normalize_issue(issue, mock_generate)

    # Verification steps should be in implementation_notes, not acceptance_criteria
    assert len(result.acceptance_criteria) == 1
    assert result.acceptance_criteria[0].description == "Feature works correctly"
    assert len(result.implementation_notes) == 2
    assert "pytest" in result.implementation_notes[0]
