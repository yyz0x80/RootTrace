"""Tests for utility functions."""

import json

from patchpilot.utils import save_json


def test_save_json_creates_parent_directories(tmp_path):
    """Test that save_json creates parent directories."""
    nested_path = tmp_path / "artifacts" / "nested" / "test.json"
    
    save_json(str(nested_path), '{"key": "value"}')
    
    assert nested_path.exists()
    assert nested_path.parent.exists()


def test_save_json_writes_content(tmp_path):
    """Test that save_json writes the correct content."""
    output_path = tmp_path / "test.json"
    test_content = '{"key": "value", "number": 42}'
    
    save_json(str(output_path), test_content)
    
    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")
    assert content == test_content


def test_save_json_uses_utf8_encoding(tmp_path):
    """Test that save_json uses UTF-8 encoding."""
    output_path = tmp_path / "unicode.json"
    unicode_content = '{"message": "你好 世界 🌍"}'
    
    save_json(str(output_path), unicode_content)
    
    content = output_path.read_text(encoding="utf-8")
    assert content == unicode_content
    
    # Verify it's valid JSON
    parsed = json.loads(content)
    assert parsed["message"] == "你好 世界 🌍"


def test_save_json_overwrites_existing(tmp_path):
    """Test that save_json overwrites existing files."""
    output_path = tmp_path / "overwrite.json"
    
    # Write initial content
    save_json(str(output_path), '{"old": "data"}')
    assert output_path.read_text(encoding="utf-8") == '{"old": "data"}'
    
    # Overwrite with new content
    save_json(str(output_path), '{"new": "data"}')
    assert output_path.read_text(encoding="utf-8") == '{"new": "data"}'


def test_save_json_with_complex_json(tmp_path):
    """Test save_json with complex nested JSON."""
    output_path = tmp_path / "complex.json"
    complex_json = """
{
  "title": "Test Issue",
  "task_type": "bug",
  "problem_statement": "Something is broken",
  "acceptance_criteria": [
    {"id": "AC-1", "description": "Fix the bug"}
  ],
  "nested": {
    "level1": {
      "level2": {
        "value": 123
      }
    }
  }
}
""".strip()
    
    save_json(str(output_path), complex_json)
    
    content = output_path.read_text(encoding="utf-8")
    parsed = json.loads(content)
    assert parsed["title"] == "Test Issue"
    assert parsed["nested"]["level1"]["level2"]["value"] == 123
