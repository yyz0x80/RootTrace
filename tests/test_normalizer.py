import pytest
from pydantic import ValidationError

from patchpilot.issue.loader import RawIssue
from patchpilot.issue.normalizer import _extract_json, normalize_issue
from patchpilot.issue.schema import NormalizedIssue


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
      "description": "Users can login with special characters in password"
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
      "description": "User list supports page and limit parameters"
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
      "description": "Preference settings are accessible"
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
      "description": "Typo is fixed"
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
      "description": "Cache is implemented"
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
