import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from patchpilot.models import ToolFailureType, ToolResult
from patchpilot.tools import (
    ApplyPatchInput,
    EditFileInput,
    InsertTextInput,
    ReadFileInput,
    RunCommandInput,
    SearchCodeInput,
    ToolDefinition,
    ToolRegistry,
    WorkspaceChange,
    _get_workspace_changes,
    generate_json_schema,
    generate_patch,
)
from patchpilot.workspace import Workspace


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for testing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Workspace(Path(tmpdir))
        yield workspace


@pytest.fixture
def tool_registry(temp_workspace):
    """Create a ToolRegistry with temporary workspace"""
    return ToolRegistry(temp_workspace)


class TestSearchCode:
    """Tests for search_code tool"""

    def test_search_code_basic(self, tool_registry, temp_workspace):
        """Test basic code search"""
        # Create a test file
        test_file = temp_workspace.root / "test.py"
        test_file.write_text("def paginate(items, page, page_size):\n    pass\n")

        result = tool_registry.search_code({"query": "paginate", "path": "."})
        assert result.ok
        assert "paginate" in result.content
        assert str(temp_workspace.root) not in result.content

    def test_search_code_no_matches(self, tool_registry):
        """Test search with no matches"""
        result = tool_registry.search_code({"query": "nonexistent", "path": "."})
        assert result.ok
        assert result.content == ""

    def test_search_code_with_path(self, tool_registry, temp_workspace):
        """Test search with specific path"""
        subdir = temp_workspace.root / "subdir"
        subdir.mkdir()
        test_file = subdir / "test.py"
        test_file.write_text("def paginate(items):\n    pass\n")

        result = tool_registry.search_code({"query": "paginate", "path": "subdir"})
        assert result.ok
        assert "paginate" in result.content

    def test_search_code_invalid_path(self, tool_registry):
        """Test search with invalid path (outside workspace)"""
        result = tool_registry.search_code({"query": "test", "path": "/etc"})
        assert not result.ok
        assert "Path error" in result.content

    def test_search_code_absolute_path_rejected(self, tool_registry):
        """Test that absolute paths are rejected"""
        result = tool_registry.search_code({"query": "test", "path": "/absolute/path"})
        assert not result.ok
        assert "Absolute path rejected" in result.content

    def test_search_code_invalid_input(self, tool_registry):
        """Test search with invalid input"""
        result = tool_registry.search_code({"query": 123})  # Invalid type
        assert not result.ok
        assert "Invalid input" in result.content

    def test_search_code_test_files_allowed(self, tool_registry, temp_workspace):
        """Test that searching test files is allowed (read-only)"""
        tests_dir = temp_workspace.root / "tests"
        tests_dir.mkdir()
        test_file = tests_dir / "test_example.py"
        test_file.write_text("def test_something():\n    pass\n")

        result = tool_registry.search_code({"query": "test_something", "path": "tests"})
        assert result.ok
        assert "test_something" in result.content

    def test_search_code_github_workflows_allowed(self, tool_registry, temp_workspace):
        """Test that searching .github/workflows is allowed (read-only)"""
        github_dir = temp_workspace.root / ".github" / "workflows"
        github_dir.mkdir(parents=True)
        workflow_file = github_dir / "ci.yml"
        workflow_file.write_text("name: CI\n")

        result = tool_registry.search_code({"query": "CI", "path": ".github/workflows"})
        assert result.ok
        assert "CI" in result.content


class TestReadFile:
    """Tests for read_file tool"""

    def test_read_file_basic(self, tool_registry, temp_workspace):
        """Test basic file reading"""
        test_file = temp_workspace.root / "test.py"
        test_file.write_text("line1\nline2\nline3\n")

        result = tool_registry.read_file({"path": "test.py"})
        assert result.ok
        assert "1: line1" in result.content
        assert "2: line2" in result.content
        assert "3: line3" in result.content

    def test_read_file_with_line_range(self, tool_registry, temp_workspace):
        """Test reading file with line range"""
        test_file = temp_workspace.root / "test.py"
        test_file.write_text("line1\nline2\nline3\nline4\nline5\n")

        result = tool_registry.read_file({"path": "test.py", "start_line": 2, "end_line": 4})
        assert result.ok
        assert "2: line2" in result.content
        assert "3: line3" in result.content
        assert "4: line4" in result.content
        assert "line1" not in result.content
        assert "line5" not in result.content

    def test_read_file_not_found(self, tool_registry):
        """Test reading non-existent file"""
        result = tool_registry.read_file({"path": "nonexistent.py"})
        assert not result.ok
        assert "File not found" in result.content

    def test_read_file_env_rejected(self, tool_registry, temp_workspace):
        """Test that .env files are rejected"""
        env_file = temp_workspace.root / ".env"
        env_file.write_text("SECRET_KEY=abc123\n")

        result = tool_registry.read_file({"path": ".env"})
        assert not result.ok
        assert "rejected" in result.content

    def test_read_file_git_rejected(self, tool_registry, temp_workspace):
        """Test that .git directory is rejected"""
        git_dir = temp_workspace.root / ".git"
        git_dir.mkdir()
        test_file = git_dir / "config"
        test_file.write_text("[core]\n")

        result = tool_registry.read_file({"path": ".git/config"})
        assert not result.ok
        assert "rejected" in result.content

    def test_read_file_outside_workspace(self, tool_registry):
        """Test that files outside workspace are rejected"""
        result = tool_registry.read_file({"path": "/etc/passwd"})
        assert not result.ok
        assert "Path error" in result.content

    def test_read_file_exceeds_line_limit(self, tool_registry, temp_workspace):
        """Test that exceeding line limit is rejected"""
        test_file = temp_workspace.root / "test.py"
        # Create a file with more than 300 lines
        lines = ["line\n"] * 400
        test_file.write_text("".join(lines))

        result = tool_registry.read_file({"path": "test.py"})
        assert not result.ok
        assert "maximum line limit" in result.content

    def test_read_file_invalid_input(self, tool_registry):
        """Test read with invalid input"""
        result = tool_registry.read_file({"path": 123})  # Invalid type
        assert not result.ok
        assert "Invalid input" in result.content

    def test_read_file_raw_mode(self, tool_registry, temp_workspace):
        """Test reading file in raw mode (without line numbers)"""
        test_file = temp_workspace.root / "test.py"
        test_file.write_text("line1\nline2\nline3\n")

        result = tool_registry.read_file({"path": "test.py", "raw": True})
        assert result.ok
        assert "1:" not in result.content  # No line numbers
        assert "line1" in result.content
        assert "line2" in result.content
        assert "line3" in result.content


class TestEditFile:
    """Tests for edit_file tool"""

    def test_edit_file_basic(self, tool_registry, temp_workspace):
        """Test basic file editing"""
        test_file = temp_workspace.root / "test.py"
        test_file.write_text("def paginate(items, page, page_size):\n    start = (page - 1) * page_size\n")

        result = tool_registry.edit_file({
            "path": "test.py",
            "old_text": "start = (page - 1) * page_size",
            "new_text": "if page < 1:\n        page = 1\n    start = (page - 1) * page_size"
        })
        assert result.ok
        assert "if page < 1:" in result.content
        assert str(temp_workspace.root) not in result.content
        assert "--- test.py" in result.content
        assert "+++ test.py" in result.content

        # Verify file was actually modified
        updated_content = test_file.read_text()
        assert "if page < 1:" in updated_content

    def test_edit_file_validation_error_uses_relative_path(
        self,
        tool_registry,
        temp_workspace,
    ):
        """Test that validation errors do not expose the workspace root."""
        test_file = temp_workspace.root / "module.py"
        original_content = "def value():\n    return 1\n"
        test_file.write_text(original_content)

        result = tool_registry.execute(
            "edit_file",
            {
                "path": "module.py",
                "old_text": "return 1",
                "new_text": "return (",
            },
        )

        assert not result.ok
        assert result.failure_type == ToolFailureType.TOOL_FAILURE
        assert "module.py" in result.content
        assert str(temp_workspace.root) not in result.content
        assert ".patchpilot_temp" not in result.content
        assert test_file.read_text() == original_content

    def test_edit_file_repairs_multiline_block_indentation(
        self,
        tool_registry,
        temp_workspace,
    ):
        """Unindented continuation lines should inherit block indentation."""
        test_file = temp_workspace.root / "module.py"
        test_file.write_text(
            "def parse(value: str) -> str:\n"
            "    return value.strip()\n",
        )

        result = tool_registry.edit_file(
            {
                "path": "module.py",
                "old_text": "return value.strip()",
                "new_text": (
                    "normalized = value.strip()\n"
                    "return normalized.lower()"
                ),
            },
        )

        assert result.ok
        assert "automatic block-indentation repair" in result.content
        assert test_file.read_text() == (
            "def parse(value: str) -> str:\n"
            "    normalized = value.strip()\n"
            "    return normalized.lower()\n"
        )

    def test_edit_file_preserves_valid_multiline_dedent(
        self,
        tool_registry,
        temp_workspace,
    ):
        """Automatic recovery must not alter an already valid replacement."""
        test_file = temp_workspace.root / "module.py"
        test_file.write_text(
            "def first() -> int:\n"
            "    return 1\n",
        )

        result = tool_registry.edit_file(
            {
                "path": "module.py",
                "old_text": "return 1",
                "new_text": (
                    "return 2\n\n\n"
                    "def second() -> int:\n"
                    "    return 3"
                ),
            },
        )

        assert result.ok
        assert "automatic block-indentation repair" not in result.content
        assert "\ndef second() -> int:\n" in test_file.read_text()

    def test_edit_file_validation_error_returns_recovery_context(
        self,
        tool_registry,
        temp_workspace,
    ):
        """Rejected edits should return current code without changing the file."""
        test_file = temp_workspace.root / "module.py"
        original_content = (
            "def calculate(value: int) -> int:\n"
            "    result = value + 1\n"
            "    return result\n"
        )
        test_file.write_text(original_content)

        result = tool_registry.edit_file(
            {
                "path": "module.py",
                "old_text": "result = value + 1",
                "new_text": "result = (",
            },
        )

        assert not result.ok
        assert "EDIT_REJECTED" in result.content
        assert "the file was restored" in result.content
        assert "2:     result = value + 1" in result.content
        assert "Next step:" in result.content
        assert test_file.read_text() == original_content

    def test_edit_file_old_text_not_found(self, tool_registry, temp_workspace):
        """Test editing when old_text is not found"""
        test_file = temp_workspace.root / "test.py"
        test_file.write_text("def paginate(items):\n    pass\n")

        result = tool_registry.edit_file({
            "path": "test.py",
            "old_text": "nonexistent text",
            "new_text": "new text"
        })
        assert not result.ok
        assert "old_text not found" in result.content

    def test_edit_file_multiple_matches(self, tool_registry, temp_workspace):
        """Test editing when old_text appears multiple times"""
        test_file = temp_workspace.root / "test.py"
        test_file.write_text("x = 1\nx = 1\nx = 1\n")

        result = tool_registry.edit_file({
            "path": "test.py",
            "old_text": "x = 1",
            "new_text": "x = 2"
        })
        assert not result.ok
        assert "appears 3 times" in result.content

    def test_edit_file_not_found(self, tool_registry):
        """Test editing non-existent file"""
        result = tool_registry.edit_file({
            "path": "nonexistent.py",
            "old_text": "old",
            "new_text": "new"
        })
        assert not result.ok
        assert "File not found" in result.content

    def test_edit_file_env_rejected(self, tool_registry, temp_workspace):
        """Test that editing .env is rejected"""
        env_file = temp_workspace.root / ".env"
        env_file.write_text("KEY=value\n")

        result = tool_registry.edit_file({
            "path": ".env",
            "old_text": "KEY=value",
            "new_text": "KEY=newvalue"
        })
        assert not result.ok
        assert "rejected" in result.content

    def test_edit_file_tests_rejected(self, tool_registry, temp_workspace):
        """Test that editing files in tests/ is rejected"""
        tests_dir = temp_workspace.root / "tests"
        tests_dir.mkdir()
        test_file = tests_dir / "test_example.py"
        test_file.write_text("def test_something():\n    pass\n")

        result = tool_registry.edit_file({
            "path": "tests/test_example.py",
            "old_text": "def test_something():",
            "new_text": "def test_modified():"
        })
        assert not result.ok
        assert "Modifying test files is not allowed" in result.content

    def test_edit_file_test_prefix_rejected(self, tool_registry, temp_workspace):
        """Test that editing test_*.py files is rejected"""
        test_file = temp_workspace.root / "test_main.py"
        test_file.write_text("def test_something():\n    pass\n")

        result = tool_registry.edit_file({
            "path": "test_main.py",
            "old_text": "def test_something():",
            "new_text": "def test_modified():"
        })
        assert not result.ok
        assert "Modifying test files is not allowed" in result.content

    def test_edit_file_github_workflows_rejected(self, tool_registry, temp_workspace):
        """Test that editing .github/workflows files is rejected"""
        github_dir = temp_workspace.root / ".github" / "workflows"
        github_dir.mkdir(parents=True)
        workflow_file = github_dir / "ci.yml"
        workflow_file.write_text("name: CI\n")

        result = tool_registry.edit_file({
            "path": ".github/workflows/ci.yml",
            "old_text": "name: CI",
            "new_text": "name: Modified CI"
        })
        assert not result.ok
        assert "Modifying CI/CD workflows is not allowed" in result.content

    def test_edit_file_outside_workspace(self, tool_registry):
        """Test that editing files outside workspace is rejected"""
        result = tool_registry.edit_file({
            "path": "/etc/passwd",
            "old_text": "root",
            "new_text": "hacker"
        })
        assert not result.ok
        assert "Path error" in result.content

    def test_edit_file_invalid_input(self, tool_registry):
        """Test edit with invalid input"""
        result = tool_registry.edit_file({"path": 123, "old_text": "old", "new_text": "new"})
        assert not result.ok
        assert "Invalid input" in result.content

    def test_edit_file_with_context(self, tool_registry, temp_workspace):
        """Test editing with context_lines parameter"""
        test_file = temp_workspace.root / "test.py"
        test_file.write_text("line1\nline2\nline3\nline4\nline5\n")

        result = tool_registry.edit_file({
            "path": "test.py",
            "old_text": "line3",
            "new_text": "line3_modified",
            "context_lines": 1
        })
        assert result.ok
        assert "line3_modified" in result.content

        # Verify file was actually modified
        updated_content = test_file.read_text()
        assert "line3_modified" in updated_content

    def test_edit_file_preview_mode(self, tool_registry, temp_workspace):
        """Test editing in preview mode (no changes applied)"""
        test_file = temp_workspace.root / "test.py"
        original_content = "def foo():\n    pass\n"
        test_file.write_text(original_content)

        result = tool_registry.edit_file({
            "path": "test.py",
            "old_text": "pass",
            "new_text": "return 42",
            "preview": True
        })
        assert result.ok
        assert "PREVIEW MODE" in result.content
        assert "No changes applied" in result.content

        # Verify file was NOT modified
        updated_content = test_file.read_text()
        assert updated_content == original_content

    def test_edit_file_enhanced_error_message(self, tool_registry, temp_workspace):
        """Test enhanced error message when old_text not found"""
        test_file = temp_workspace.root / "test.py"
        test_file.write_text("def paginate(items):\n    return items\n")

        result = tool_registry.edit_file({
            "path": "test.py",
            "old_text": "def paginate(item):",  # Typo: 'item' instead of 'items'
            "new_text": "def paginate(items):"
        })
        assert not result.ok
        assert "old_text not found" in result.content
        # Check for enhanced error with closest match
        assert "Closest match" in result.content or "Similar content" in result.content

    def test_edit_file_empty_old_text_error(self, tool_registry, temp_workspace):
        """Test error message when old_text is empty"""
        test_file = temp_workspace.root / "test.py"
        test_file.write_text("def paginate(items):\n    return items\n")

        result = tool_registry.edit_file({
            "path": "test.py",
            "old_text": "",  # Empty old_text
            "new_text": "new text"
        })
        assert not result.ok
        assert "old_text is empty" in result.content
        assert "raw=True" in result.content
        assert "apply_patch" in result.content

    def test_edit_file_whitespace_only_old_text_error(self, tool_registry, temp_workspace):
        """Test error message when old_text contains only whitespace"""
        test_file = temp_workspace.root / "test.py"
        test_file.write_text("def paginate(items):\n    return items\n")

        result = tool_registry.edit_file({
            "path": "test.py",
            "old_text": "   ",  # Whitespace only
            "new_text": "new text"
        })
        assert not result.ok
        assert "old_text is empty" in result.content
        assert "apply_patch" in result.content

    def test_edit_file_line_prefix_error_with_example(self, tool_registry, temp_workspace):
        """Test error message with specific example when line prefixes are detected"""
        test_file = temp_workspace.root / "test.py"
        test_file.write_text("def paginate(items):\n    return items\n")

        result = tool_registry.edit_file({
            "path": "test.py",
            "old_text": "1: def paginate(items):\n2:     return items",  # With line prefixes
            "new_text": "def paginate(items):\n    return items"
        })
        assert not result.ok
        assert "line number prefixes" in result.content
        # Check for specific example in error message
        assert "Example correction" in result.content or "Wrong:" in result.content

    def test_edit_file_with_rstrip_fallback(self, tool_registry, temp_workspace):
        """Test editing with trailing whitespace fallback"""
        test_file = temp_workspace.root / "test.py"
        test_file.write_text("def paginate(items):\n    return items\n")

        # Provide old_text with extra trailing whitespace
        result = tool_registry.edit_file({
            "path": "test.py",
            "old_text": "def paginate(items):   ",  # Extra trailing spaces
            "new_text": "def paginate(items, page_size=10):"
        })
        assert result.ok
        assert "page_size" in result.content

        # Verify file was actually modified
        updated_content = test_file.read_text()
        assert "page_size" in updated_content

    def test_edit_file_with_strip_fallback(self, tool_registry, temp_workspace):
        """Test editing with leading/trailing whitespace fallback"""
        test_file = temp_workspace.root / "test.py"
        test_file.write_text("def paginate(items):\n    return items\n")

        # Provide old_text with extra leading/trailing whitespace
        result = tool_registry.edit_file({
            "path": "test.py",
            "old_text": "  def paginate(items):  ",  # Extra leading and trailing spaces
            "new_text": "def paginate(items, page_size=10):"
        })
        assert result.ok
        assert "page_size" in result.content

        # Verify file was actually modified
        updated_content = test_file.read_text()
        assert "page_size" in updated_content

    def test_edit_file_closest_match_error(self, tool_registry, temp_workspace):
        """Test enhanced error message with closest match and diff"""
        test_file = temp_workspace.root / "test.py"
        test_file.write_text("def paginate(items, page, page_size):\n    start = (page - 1) * page_size\n")

        # Provide old_text with small typo
        result = tool_registry.edit_file({
            "path": "test.py",
            "old_text": "def paginate(item, page, page_size):",  # Typo: 'item' instead of 'items'
            "new_text": "def paginate(items, page, page_size, sort=False):"
        })
        assert not result.ok
        assert "old_text not found" in result.content
        # Check for closest match feature
        assert "Closest match" in result.content or "Similar content" in result.content
        # Check for diff information
        assert "Diff between" in result.content or "similarity" in result.content

    def test_try_match_with_fallback_exact(self, tool_registry):
        """Test _try_match_with_fallback with exact match"""
        file_content = "def foo():\n    pass\n"
        old_text = "def foo():"

        result = tool_registry._try_match_with_fallback(file_content, old_text)
        assert result.matched
        assert result.matched_text == old_text
        assert result.method == "exact"

    def test_try_match_with_fallback_rstrip(self, tool_registry):
        """Test _try_match_with_fallback with rstrip fallback"""
        file_content = "def foo():\n    pass\n"
        old_text = "def foo():   "  # Extra trailing spaces

        result = tool_registry._try_match_with_fallback(file_content, old_text)
        assert result.matched
        assert result.matched_text == "def foo():"
        assert result.method == "rstrip"

    def test_try_match_with_fallback_strip(self, tool_registry):
        """Test _try_match_with_fallback with strip fallback"""
        file_content = "def foo():\n    pass\n"
        old_text = "  def foo():  "  # Extra leading and trailing whitespace

        result = tool_registry._try_match_with_fallback(file_content, old_text)
        assert result.matched
        assert result.matched_text == "def foo():"
        assert result.method == "strip"

    def test_try_match_with_fallback_no_match(self, tool_registry):
        """Test _try_match_with_fallback when no match is found"""
        file_content = "def bar():\n    pass\n"
        old_text = "def foo():"  # Not in file

        result = tool_registry._try_match_with_fallback(file_content, old_text)
        assert not result.matched
        assert result.matched_text == ""
        assert result.method == ""

    def test_find_closest_match_basic(self, tool_registry):
        """Test _find_closest_match with basic case"""
        file_content = "def paginate(items):\n    return items\n"
        old_text = "def paginate(item):"  # Small typo

        result = tool_registry._find_closest_match(file_content, old_text)
        assert result is not None
        assert "line" in result
        assert "text" in result
        assert "similarity" in result
        assert result["similarity"] > 0.5  # Should be quite similar

    def test_find_closest_match_no_good_match(self, tool_registry):
        """Test _find_closest_match when no good match exists"""
        file_content = "def completely_different():\n    pass\n"
        old_text = "def foo():"  # Very different

        result = tool_registry._find_closest_match(file_content, old_text)
        # Should return None if similarity is below threshold
        assert result is None or result["similarity"] < 0.5


class TestInsertTextTool:
    """Tests for insert_text tool"""

    def test_insert_text_input_validation(self):
        """Test InsertTextInput validation"""
        # Valid input
        input_data = InsertTextInput(
            path="test.py",
            line_number=1,
            text="new text"
        )
        assert input_data.path == "test.py"
        assert input_data.line_number == 1
        assert input_data.text == "new text"

        # Invalid line number (negative)
        with pytest.raises(ValueError, match="line_number must be >= 0"):
            InsertTextInput(path="test.py", line_number=-1, text="text")

        # Invalid path type
        with pytest.raises(TypeError, match="path must be str"):
            InsertTextInput(path=123, line_number=1, text="text")

    def test_insert_text_basic(self, tool_registry, temp_workspace):
        """Test basic text insertion"""
        test_file = temp_workspace.root / "test.py"
        test_file.write_text("line1\nline2\nline3\n")

        result = tool_registry.insert_text({
            "path": "test.py",
            "line_number": 1,
            "text": "inserted_line\n"
        })
        assert result.ok
        assert "inserted_line" in result.content

        # Verify file was actually modified
        updated_content = test_file.read_text()
        assert "line1\ninserted_line\nline2\nline3\n" == updated_content

    def test_insert_text_at_beginning(self, tool_registry, temp_workspace):
        """Test inserting at the beginning (line_number=0)"""
        test_file = temp_workspace.root / "test.py"
        test_file.write_text("line1\nline2\n")

        result = tool_registry.insert_text({
            "path": "test.py",
            "line_number": 0,
            "text": "first_line\n"
        })
        assert result.ok

        # Verify file was actually modified
        updated_content = test_file.read_text()
        assert "first_line\nline1\nline2\n" == updated_content

    def test_insert_text_at_end(self, tool_registry, temp_workspace):
        """Test inserting at the end of file"""
        test_file = temp_workspace.root / "test.py"
        test_file.write_text("line1\nline2\n")

        result = tool_registry.insert_text({
            "path": "test.py",
            "line_number": 2,
            "text": "last_line\n"
        })
        assert result.ok

        # Verify file was actually modified
        updated_content = test_file.read_text()
        assert "line1\nline2\nlast_line\n" == updated_content

    def test_insert_text_preview_mode(self, tool_registry, temp_workspace):
        """Test inserting in preview mode (no changes applied)"""
        test_file = temp_workspace.root / "test.py"
        original_content = "line1\nline2\n"
        test_file.write_text(original_content)

        result = tool_registry.insert_text({
            "path": "test.py",
            "line_number": 1,
            "text": "inserted\n",
            "preview": True
        })
        assert result.ok
        assert "PREVIEW MODE" in result.content
        assert "No changes applied" in result.content

        # Verify file was NOT modified
        updated_content = test_file.read_text()
        assert updated_content == original_content

    def test_insert_text_not_found(self, tool_registry):
        """Test inserting into non-existent file"""
        result = tool_registry.insert_text({
            "path": "nonexistent.py",
            "line_number": 1,
            "text": "new text"
        })
        assert not result.ok
        assert "File not found" in result.content

    def test_insert_text_line_number_exceeds_file(self, tool_registry, temp_workspace):
        """Test inserting with line_number beyond file length"""
        test_file = temp_workspace.root / "test.py"
        test_file.write_text("line1\nline2\n")

        result = tool_registry.insert_text({
            "path": "test.py",
            "line_number": 10,  # Beyond file length
            "text": "new text"
        })
        assert not result.ok
        assert "exceeds file length" in result.content

    def test_insert_text_auto_newline(self, tool_registry, temp_workspace):
        """Test that text without newline gets one added automatically"""
        test_file = temp_workspace.root / "test.py"
        test_file.write_text("line1\nline2\n")

        result = tool_registry.insert_text({
            "path": "test.py",
            "line_number": 1,
            "text": "inserted_line"  # No newline
        })
        assert result.ok

        # Verify file was actually modified with newline added
        updated_content = test_file.read_text()
        assert "line1\ninserted_line\nline2\n" == updated_content

    def test_insert_text_invalid_input(self, tool_registry):
        """Test insert with invalid input"""
        result = tool_registry.insert_text({"path": 123, "line_number": 1, "text": "text"})
        assert not result.ok
        assert "Invalid input" in result.content

    def test_insert_text_negative_line_number(self, tool_registry):
        """Test insert with negative line number"""
        result = tool_registry.insert_text({
            "path": "test.py",
            "line_number": -1,
            "text": "text"
        })
        assert not result.ok
        assert "Invalid input" in result.content

    def test_insert_text_empty_file(self, tool_registry, temp_workspace):
        """Test inserting into empty file"""
        test_file = temp_workspace.root / "test.py"
        test_file.write_text("")

        result = tool_registry.insert_text({
            "path": "test.py",
            "line_number": 0,
            "text": "first_line\n"
        })
        assert result.ok

        # Verify file was actually modified
        updated_content = test_file.read_text()
        assert "first_line\n" == updated_content

    def test_insert_text_test_prefix_rejected(self, tool_registry, temp_workspace):
        """Test that inserting into test_*.py files is rejected"""
        test_file = temp_workspace.root / "test_main.py"
        test_file.write_text("def test_something():\n    pass\n")

        result = tool_registry.insert_text({
            "path": "test_main.py",
            "line_number": 1,
            "text": "new line\n"
        })
        assert not result.ok
        assert "Modifying test files is not allowed" in result.content

    def test_insert_text_github_workflows_rejected(self, tool_registry, temp_workspace):
        """Test that inserting into .github/workflows files is rejected"""
        github_dir = temp_workspace.root / ".github" / "workflows"
        github_dir.mkdir(parents=True)
        workflow_file = github_dir / "ci.yml"
        workflow_file.write_text("name: CI\n")

        result = tool_registry.insert_text({
            "path": ".github/workflows/ci.yml",
            "line_number": 1,
            "text": "new step\n"
        })
        assert not result.ok
        assert "Modifying CI/CD workflows is not allowed" in result.content


class TestEditFileByLine:
    """Tests for edit_file_by_line tool"""

    def test_edit_file_by_line_basic(self, tool_registry, temp_workspace):
        """Test basic line-based editing"""
        test_file = temp_workspace.root / "test.py"
        test_file.write_text("line1\nline2\nline3\nline4\nline5\n")

        result = tool_registry.edit_file_by_line({
            "path": "test.py",
            "start_line": 2,
            "end_line": 4,
            "new_text": "modified_lines = True"
        })
        assert result.ok
        assert "modified_lines" in result.content

        # Verify file was actually modified
        updated_content = test_file.read_text()
        # The implementation adds a newline to new_text if not present
        assert "line1\nmodified_lines = True\nline5\n" == updated_content

    def test_edit_file_by_line_single_line(self, tool_registry, temp_workspace):
        """Test editing a single line"""
        test_file = temp_workspace.root / "test.py"
        test_file.write_text("line1\nline2\nline3\n")

        result = tool_registry.edit_file_by_line({
            "path": "test.py",
            "start_line": 2,
            "end_line": 2,
            "new_text": "line2_modified"
        })
        assert result.ok

        # Verify file was actually modified
        updated_content = test_file.read_text()
        # The implementation adds a newline to new_text if not present
        assert "line1\nline2_modified\nline3\n" == updated_content

    def test_edit_file_by_line_preview_mode(self, tool_registry, temp_workspace):
        """Test line-based editing in preview mode"""
        test_file = temp_workspace.root / "test.py"
        original_content = "line1\nline2\nline3\n"
        test_file.write_text(original_content)

        result = tool_registry.edit_file_by_line({
            "path": "test.py",
            "start_line": 2,
            "end_line": 2,
            "new_text": "modified",
            "preview": True
        })
        assert result.ok
        assert "PREVIEW MODE" in result.content
        assert "No changes applied" in result.content

        # Verify file was NOT modified
        updated_content = test_file.read_text()
        assert updated_content == original_content

    def test_edit_file_by_line_invalid_range(self, tool_registry, temp_workspace):
        """Test editing with invalid line range"""
        test_file = temp_workspace.root / "test.py"
        test_file.write_text("line1\nline2\nline3\n")

        result = tool_registry.edit_file_by_line({
            "path": "test.py",
            "start_line": 10,  # Beyond file length
            "end_line": 12,
            "new_text": "modified"
        })
        assert not result.ok
        assert "exceeds file length" in result.content

    def test_edit_file_by_line_end_before_start(self, tool_registry):
        """Test editing with end_line before start_line"""
        result = tool_registry.edit_file_by_line({
            "path": "test.py",
            "start_line": 5,
            "end_line": 3,  # Invalid: end before start
            "new_text": "modified"
        })
        assert not result.ok
        assert "Invalid input" in result.content

    def test_edit_file_by_line_test_prefix_rejected(self, tool_registry, temp_workspace):
        """Test that editing test_*.py files is rejected"""
        test_file = temp_workspace.root / "test_main.py"
        test_file.write_text("def test_something():\n    pass\n")

        result = tool_registry.edit_file_by_line({
            "path": "test_main.py",
            "start_line": 1,
            "end_line": 1,
            "new_text": "modified"
        })
        assert not result.ok
        assert "Modifying test files is not allowed" in result.content

    def test_edit_file_by_line_github_workflows_rejected(self, tool_registry, temp_workspace):
        """Test that editing .github/workflows files is rejected"""
        github_dir = temp_workspace.root / ".github" / "workflows"
        github_dir.mkdir(parents=True)
        workflow_file = github_dir / "ci.yml"
        workflow_file.write_text("name: CI\n")

        result = tool_registry.edit_file_by_line({
            "path": ".github/workflows/ci.yml",
            "start_line": 1,
            "end_line": 1,
            "new_text": "modified"
        })
        assert not result.ok
        assert "Modifying CI/CD workflows is not allowed" in result.content


class TestInsertText:
    """Tests for insert_text tool"""

    def test_insert_text_basic(self, tool_registry, temp_workspace):
        """Test basic text insertion"""
        test_file = temp_workspace.root / "test.py"
        test_file.write_text("line1\nline2\nline3\n")

        result = tool_registry.insert_text({
            "path": "test.py",
            "line_number": 1,
            "text": "inserted_line\n"
        })
        assert result.ok
        assert "inserted_line" in result.content

        # Verify file was actually modified
        updated_content = test_file.read_text()
        assert "line1\ninserted_line\nline2\nline3\n" == updated_content

    def test_insert_text_at_beginning(self, tool_registry, temp_workspace):
        """Test inserting at the beginning (line_number=0)"""
        test_file = temp_workspace.root / "test.py"
        test_file.write_text("line1\nline2\n")

        result = tool_registry.insert_text({
            "path": "test.py",
            "line_number": 0,
            "text": "first_line\n"
        })
        assert result.ok

        # Verify file was actually modified
        updated_content = test_file.read_text()
        assert "first_line\nline1\nline2\n" == updated_content

    def test_insert_text_at_end(self, tool_registry, temp_workspace):
        """Test inserting at the end of file"""
        test_file = temp_workspace.root / "test.py"
        test_file.write_text("line1\nline2\n")

        result = tool_registry.insert_text({
            "path": "test.py",
            "line_number": 2,
            "text": "last_line\n"
        })
        assert result.ok

        # Verify file was actually modified
        updated_content = test_file.read_text()
        assert "line1\nline2\nlast_line\n" == updated_content

    def test_insert_text_preview_mode(self, tool_registry, temp_workspace):
        """Test inserting in preview mode (no changes applied)"""
        test_file = temp_workspace.root / "test.py"
        original_content = "line1\nline2\n"
        test_file.write_text(original_content)

        result = tool_registry.insert_text({
            "path": "test.py",
            "line_number": 1,
            "text": "inserted\n",
            "preview": True
        })
        assert result.ok
        assert "PREVIEW MODE" in result.content
        assert "No changes applied" in result.content

        # Verify file was NOT modified
        updated_content = test_file.read_text()
        assert updated_content == original_content

    def test_insert_text_not_found(self, tool_registry):
        """Test inserting into non-existent file"""
        result = tool_registry.insert_text({
            "path": "nonexistent.py",
            "line_number": 1,
            "text": "new text"
        })
        assert not result.ok
        assert "File not found" in result.content

    def test_insert_text_line_number_exceeds_file(self, tool_registry, temp_workspace):
        """Test inserting with line_number beyond file length"""
        test_file = temp_workspace.root / "test.py"
        test_file.write_text("line1\nline2\n")

        result = tool_registry.insert_text({
            "path": "test.py",
            "line_number": 10,  # Beyond file length
            "text": "new text"
        })
        assert not result.ok
        assert "exceeds file length" in result.content

    def test_insert_text_auto_newline(self, tool_registry, temp_workspace):
        """Test that text without newline gets one added automatically"""
        test_file = temp_workspace.root / "test.py"
        test_file.write_text("line1\nline2\n")

        result = tool_registry.insert_text({
            "path": "test.py",
            "line_number": 1,
            "text": "inserted_line"  # No newline
        })
        assert result.ok

        # Verify file was actually modified with newline added
        updated_content = test_file.read_text()
        assert "line1\ninserted_line\nline2\n" == updated_content

    def test_insert_text_invalid_input(self, tool_registry):
        """Test insert with invalid input"""
        result = tool_registry.insert_text({"path": 123, "line_number": 1, "text": "text"})
        assert not result.ok
        assert "Invalid input" in result.content

    def test_insert_text_negative_line_number(self, tool_registry):
        """Test insert with negative line number"""
        result = tool_registry.insert_text({
            "path": "test.py",
            "line_number": -1,
            "text": "text"
        })
        assert not result.ok
        assert "Invalid input" in result.content

    def test_insert_text_empty_file(self, tool_registry, temp_workspace):
        """Test inserting into empty file"""
        test_file = temp_workspace.root / "test.py"
        test_file.write_text("")

        result = tool_registry.insert_text({
            "path": "test.py",
            "line_number": 0,
            "text": "first_line\n"
        })
        assert result.ok

        # Verify file was actually modified
        updated_content = test_file.read_text()
        assert "first_line\n" == updated_content

    def test_insert_text_test_prefix_rejected(self, tool_registry, temp_workspace):
        """Test that inserting into test_*.py files is rejected"""
        test_file = temp_workspace.root / "test_main.py"
        test_file.write_text("def test_something():\n    pass\n")

        result = tool_registry.insert_text({
            "path": "test_main.py",
            "line_number": 1,
            "text": "new line\n"
        })
        assert not result.ok
        assert "Modifying test files is not allowed" in result.content

    def test_insert_text_github_workflows_rejected(self, tool_registry, temp_workspace):
        """Test that inserting into .github/workflows files is rejected"""
        github_dir = temp_workspace.root / ".github" / "workflows"
        github_dir.mkdir(parents=True)
        workflow_file = github_dir / "ci.yml"
        workflow_file.write_text("name: CI\n")

        result = tool_registry.insert_text({
            "path": ".github/workflows/ci.yml",
            "line_number": 1,
            "text": "new step\n"
        })
        assert not result.ok
        assert "Modifying CI/CD workflows is not allowed" in result.content


class TestApplyPatch:
    """Tests for apply_patch tool"""

    def test_apply_patch_create_file(self, tool_registry, temp_workspace):
        """Test creating a new file with apply_patch"""
        result = tool_registry.apply_patch({
            "path": "new_file.py",
            "content": "def new_function():\n    pass\n"
        })
        assert result.ok
        assert "Created file" in result.content

        # Verify file was actually created
        new_file = temp_workspace.root / "new_file.py"
        assert new_file.exists()
        assert new_file.read_text() == "def new_function():\n    pass\n"

    def test_apply_patch_modify_existing_file(self, tool_registry, temp_workspace):
        """Test modifying an existing file with apply_patch"""
        test_file = temp_workspace.root / "test.py"
        test_file.write_text("old_value = 1\n")

        result = tool_registry.apply_patch({
            "path": "test.py",
            "content": "new_value = 2\n"
        })
        assert result.ok
        assert "---" in result.content or "no diff" in result.content

        # Verify file was actually modified
        updated_content = test_file.read_text()
        assert updated_content == "new_value = 2\n"

    def test_apply_patch_with_diff(self, tool_registry, temp_workspace):
        """Test that apply_patch returns proper diff for modifications"""
        test_file = temp_workspace.root / "test.py"
        test_file.write_text("line1\nline2\nline3\n")

        result = tool_registry.apply_patch({
            "path": "test.py",
            "content": "line1\nline2_modified\nline3\n"
        })
        assert result.ok
        assert "---" in result.content or "+++" in result.content

    def test_apply_patch_create_in_subdirectory(self, tool_registry, temp_workspace):
        """Test creating a file in a subdirectory"""
        result = tool_registry.apply_patch({
            "path": "subdir/new_file.py",
            "content": "def func():\n    pass\n"
        })
        assert result.ok
        assert "Created file" in result.content

        # Verify file was created in subdirectory
        new_file = temp_workspace.root / "subdir" / "new_file.py"
        assert new_file.exists()
        assert new_file.parent.exists()

    def test_apply_patch_env_rejected(self, tool_registry, temp_workspace):
        """Test that applying patch to .env is rejected"""
        result = tool_registry.apply_patch({
            "path": ".env",
            "content": "KEY=value\n"
        })
        assert not result.ok
        assert "rejected" in result.content

    def test_apply_patch_git_rejected(self, tool_registry, temp_workspace):
        """Test that applying patch to .git directory is rejected"""
        result = tool_registry.apply_patch({
            "path": ".git/config",
            "content": "[core]\n"
        })
        assert not result.ok
        assert "rejected" in result.content

    def test_apply_patch_tests_rejected(self, tool_registry, temp_workspace):
        """Test that applying patch to test files is rejected"""
        tests_dir = temp_workspace.root / "tests"
        tests_dir.mkdir()

        result = tool_registry.apply_patch({
            "path": "tests/test_example.py",
            "content": "def test_new():\n    pass\n"
        })
        assert not result.ok
        assert "Modifying test files is not allowed" in result.content

    def test_apply_patch_test_prefix_rejected(self, tool_registry, temp_workspace):
        """Test that applying patch to test_*.py files is rejected"""
        result = tool_registry.apply_patch({
            "path": "test_main.py",
            "content": "def test_new():\n    pass\n"
        })
        assert not result.ok
        assert "Modifying test files is not allowed" in result.content

    def test_apply_patch_github_workflows_rejected(self, tool_registry, temp_workspace):
        """Test that applying patch to .github/workflows is rejected"""
        result = tool_registry.apply_patch({
            "path": ".github/workflows/ci.yml",
            "content": "name: CI\n"
        })
        assert not result.ok
        assert "Modifying CI/CD workflows is not allowed" in result.content

    def test_apply_patch_outside_workspace(self, tool_registry):
        """Test that applying patch outside workspace is rejected"""
        result = tool_registry.apply_patch({
            "path": "/etc/passwd",
            "content": "malicious content"
        })
        assert not result.ok
        assert "Path error" in result.content

    def test_apply_patch_invalid_input(self, tool_registry):
        """Test apply_patch with invalid input"""
        result = tool_registry.apply_patch({"path": 123, "content": "test"})
        assert not result.ok
        assert "Invalid input" in result.content

    def test_apply_patch_empty_content(self, tool_registry, temp_workspace):
        """Test applying patch with empty content (file clearing scenario)"""
        test_file = temp_workspace.root / "test.py"
        test_file.write_text("original content\n")

        result = tool_registry.apply_patch({
            "path": "test.py",
            "content": "# Empty file with comment\n"
        })
        assert result.ok

        # Verify file was modified
        updated_content = test_file.read_text()
        assert updated_content == "# Empty file with comment\n"

    def test_apply_patch_preview_mode(self, tool_registry, temp_workspace):
        """Test apply_patch in preview mode (no changes applied)"""
        test_file = temp_workspace.root / "test.py"
        original_content = "original content\n"
        test_file.write_text(original_content)

        result = tool_registry.apply_patch({
            "path": "test.py",
            "content": "new content\n",
            "preview": True
        })
        assert result.ok
        assert "PREVIEW MODE" in result.content
        assert "No changes applied" in result.content

        # Verify file was NOT modified
        updated_content = test_file.read_text()
        assert updated_content == original_content


class TestRunCommand:
    """Tests for run_command tool"""

    def test_run_command_pytest(self, tool_registry, temp_workspace):
        """Test running pytest"""
        # Create a simple test file
        tests_dir = temp_workspace.root / "tests"
        tests_dir.mkdir()
        test_file = tests_dir / "test_example.py"
        test_file.write_text("def test_pass():\n    assert True\n")

        result = tool_registry.run_command({"command": "pytest"})
        assert result.ok or "test_pass" in result.content  # May fail if pytest not installed

    def test_run_command_python_m_pytest(self, tool_registry, temp_workspace):
        """Test running python -m pytest"""
        tests_dir = temp_workspace.root / "tests"
        tests_dir.mkdir()
        test_file = tests_dir / "test_example.py"
        test_file.write_text("def test_pass():\n    assert True\n")

        result = tool_registry.run_command({"command": "python -m pytest"})
        assert result.ok or "test_pass" in result.content

    def test_run_command_normalizes_bare_pytest(self, temp_workspace):
        """Test that bare Pytest uses the canonical module invocation."""
        commands: list[str] = []

        def run(**kwargs):
            commands.append(kwargs["command"])
            return SimpleNamespace(
                exit_code=0,
                stdout="1 passed",
                stderr="",
                timed_out=False,
            )

        registry = ToolRegistry(
            temp_workspace,
            command_runner=SimpleNamespace(run=run),
        )

        result = registry.run_command(
            {"command": "pytest tests/test_example.py -q"}
        )

        assert result.ok
        assert commands == [
            "python -m pytest tests/test_example.py -q"
        ]

    @pytest.mark.parametrize(
        "command",
        ["pytest tests/test_example.py", "python -m pytest -q"],
    )
    def test_failed_pytest_is_verification_failure(
        self,
        temp_workspace,
        command,
    ):
        """Test that a completed failing Pytest run is verification failure."""
        runner = SimpleNamespace(
            run=lambda **_kwargs: SimpleNamespace(
                exit_code=1,
                stdout="1 failed",
                stderr="",
                timed_out=False,
            )
        )
        registry = ToolRegistry(temp_workspace, command_runner=runner)

        result = registry.execute("run_command", {"command": command})

        assert not result.ok
        assert (
            result.failure_type
            == ToolFailureType.VERIFICATION_FAILURE
        )

    def test_timed_out_pytest_is_tool_failure(self, temp_workspace):
        """Test that a timed-out Pytest command remains a tool failure."""
        runner = SimpleNamespace(
            run=lambda **_kwargs: SimpleNamespace(
                exit_code=124,
                stdout="",
                stderr="timed out",
                timed_out=True,
            )
        )
        registry = ToolRegistry(temp_workspace, command_runner=runner)

        result = registry.execute(
            "run_command",
            {"command": "pytest"},
        )

        assert not result.ok
        assert result.failure_type == ToolFailureType.TOOL_FAILURE

    def test_run_command_git_status(self, tool_registry):
        """Test running git status"""
        result = tool_registry.run_command({"command": "git status"})
        # git status may fail if not a git repo, but should be allowed
        assert "exit_code" in result.content or result.ok

    def test_run_command_git_diff(self, tool_registry):
        """Test running git diff"""
        result = tool_registry.run_command({"command": "git diff"})
        # git diff may fail if not a git repo, but should be allowed
        assert "exit_code" in result.content or result.ok

    def test_run_command_ruff_check(self, tool_registry, temp_workspace):
        """Test running ruff check"""
        test_file = temp_workspace.root / "test.py"
        test_file.write_text("x = 1\n")

        result = tool_registry.run_command({"command": "ruff check test.py"})
        # ruff may not be installed, but command should be allowed
        assert "exit_code" in result.content or result.ok

    def test_run_command_not_allowed(self, tool_registry):
        """Test that disallowed commands are rejected"""
        result = tool_registry.run_command({"command": "rm -rf /"})
        assert not result.ok
        assert "not allowed" in result.content

    def test_run_command_python_m_not_pytest(self, tool_registry):
        """Test that python -m with non-pytest is rejected"""
        result = tool_registry.run_command({"command": "python -m http.server"})
        assert not result.ok
        assert "Only 'python -m pytest' is allowed" in result.content

    def test_run_command_git_push_rejected(self, tool_registry):
        """Test that git push is rejected"""
        result = tool_registry.run_command({"command": "git push"})
        assert not result.ok
        assert "Only 'git diff' and 'git status'" in result.content

    def test_run_command_pip_install_rejected(self, tool_registry):
        """Test that pip install is rejected"""
        result = tool_registry.run_command({"command": "pip install requests"})
        assert not result.ok
        assert "not allowed" in result.content
        assert "pip" in result.content

    def test_run_command_ruff_not_check(self, tool_registry):
        """Test that ruff without check is rejected"""
        result = tool_registry.run_command({"command": "ruff format"})
        assert not result.ok
        assert "Only 'ruff check' is allowed" in result.content

    def test_run_command_empty(self, tool_registry):
        """Test that empty command is rejected"""
        result = tool_registry.run_command({"command": ""})
        assert not result.ok
        assert "Empty command" in result.content

    def test_run_command_invalid_syntax(self, tool_registry):
        """Test that invalid command syntax is rejected"""
        result = tool_registry.run_command({"command": "unclosed 'quote"})
        assert not result.ok
        assert "Invalid command syntax" in result.content

    def test_run_command_invalid_input(self, tool_registry):
        """Test run with invalid input"""
        result = tool_registry.run_command({"command": 123})  # Invalid type
        assert not result.ok
        assert "Invalid input" in result.content


class TestToolDispatch:
    """Tests for tool dispatch mechanism"""

    def test_execute_sanitizes_workspace_paths(
        self,
        tool_registry,
        temp_workspace,
    ):
        """Test the central model-facing workspace path sanitizer."""
        absolute_file = temp_workspace.root / "src" / "module.py"
        tool_registry._tool_handlers["leaky_tool"] = lambda _arguments: (
            ToolResult(ok=False, content=f"Failed at {absolute_file}")
        )

        result = tool_registry.execute("leaky_tool", {})

        assert result.content == "Failed at src/module.py"
        assert result.failure_type == ToolFailureType.TOOL_FAILURE

    def test_dispatch_valid_tool(self, tool_registry):
        """Test dispatching a valid tool"""
        result = tool_registry.dispatch("search_code", {"query": "test", "path": "."})
        assert isinstance(result, ToolResult)

    def test_dispatch_unknown_tool(self, tool_registry):
        """Test dispatching an unknown tool"""
        result = tool_registry.dispatch("unknown_tool", {})
        assert not result.ok
        assert "Unknown tool" in result.content

    def test_dispatch_all_tools(self, tool_registry):
        """Test that all registered tools can be dispatched"""
        tools = ["search_code", "read_file", "edit_file", "run_command"]
        for tool_name in tools:
            result = tool_registry.dispatch(tool_name, {})
            # Should not raise "Unknown tool" error
            assert "Unknown tool" not in result.content


class TestToolSchema:
    """Tests for tool schema generation and discovery"""

    def test_tool_definition_creation(self):
        """Test creating a ToolDefinition"""
        schema = {"type": "object", "properties": {}}
        tool_def = ToolDefinition(
            name="test_tool",
            description="A test tool",
            input_schema=schema
        )
        assert tool_def.name == "test_tool"
        assert tool_def.description == "A test tool"
        assert tool_def.input_schema == schema

    def test_generate_json_schema_search_code(self):
        """Test JSON schema generation for SearchCodeInput"""
        schema = generate_json_schema(SearchCodeInput)
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "query" in schema["properties"]
        assert "path" in schema["properties"]
        assert schema["properties"]["query"]["type"] == "string"
        assert schema["properties"]["path"]["type"] == "string"
        assert "query" in schema["required"]
        assert "path" not in schema["required"]  # Has default value

    def test_generate_json_schema_read_file(self):
        """Test JSON schema generation for ReadFileInput"""
        schema = generate_json_schema(ReadFileInput)
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "path" in schema["properties"]
        assert "start_line" in schema["properties"]
        assert "end_line" in schema["properties"]
        assert schema["properties"]["path"]["type"] == "string"
        assert schema["properties"]["start_line"]["type"] == "integer"
        # end_line is int | None, so it should be nullable in JSON schema
        assert schema["properties"]["end_line"]["type"] == ["integer", "null"]
        assert "path" in schema["required"]
        assert "start_line" not in schema["required"]  # Has default
        assert "end_line" not in schema["required"]  # Optional

    def test_generate_json_schema_edit_file(self):
        """Test JSON schema generation for EditFileInput"""
        schema = generate_json_schema(EditFileInput)
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "path" in schema["properties"]
        assert "old_text" in schema["properties"]
        assert "new_text" in schema["properties"]
        assert all(schema["properties"][k]["type"] == "string" for k in ["path", "old_text", "new_text"])
        assert all(k in schema["required"] for k in ["path", "old_text", "new_text"])

    def test_generate_json_schema_apply_patch(self):
        """Test JSON schema generation for ApplyPatchInput"""
        schema = generate_json_schema(ApplyPatchInput)
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "path" in schema["properties"]
        assert "content" in schema["properties"]
        assert all(schema["properties"][k]["type"] == "string" for k in ["path", "content"])
        assert all(k in schema["required"] for k in ["path", "content"])

    def test_generate_json_schema_run_command(self):
        """Test JSON schema generation for RunCommandInput"""
        schema = generate_json_schema(RunCommandInput)
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "command" in schema["properties"]
        assert schema["properties"]["command"]["type"] == "string"
        assert "command" in schema["required"]

    def test_get_tool_schemas(self, tool_registry):
        """Test getting all tool schemas in OpenAI format"""
        schemas = tool_registry.get_tool_schemas()
        assert len(schemas) == 5
        assert all(isinstance(schema, dict) for schema in schemas)
        assert all(schema["type"] == "function" for schema in schemas)
        tool_names = {schema["function"]["name"] for schema in schemas}
        assert tool_names == {
            "search_code",
            "read_file",
            "edit_file",
            "apply_patch",
            "run_command",
        }
        assert "insert_text" not in tool_names
        assert "edit_file_by_line" not in tool_names

    def test_get_tool_schema_existing(self, tool_registry):
        """Test getting a specific tool schema that exists"""
        schema = tool_registry.get_tool_schema("search_code")
        assert schema is not None
        assert schema.name == "search_code"
        assert schema.description is not None
        assert schema.input_schema is not None
        assert isinstance(schema.input_schema, dict)

        # Compatibility methods are intentionally hidden from the model ACI.
        assert tool_registry.get_tool_schema("insert_text") is None
        assert tool_registry.get_tool_schema("edit_file_by_line") is None

    def test_get_tool_schema_nonexistent(self, tool_registry):
        """Test getting a specific tool schema that doesn't exist"""
        schema = tool_registry.get_tool_schema("nonexistent_tool")
        assert schema is None

    def test_tool_schemas_have_descriptions(self, tool_registry):
        """Test that all tool schemas have descriptions"""
        schemas = tool_registry.get_tool_schemas()
        for schema in schemas:
            function_def = schema["function"]
            assert function_def["description"], f"Tool {function_def['name']} missing description"
            assert len(function_def["description"]) > 0, f"Tool {function_def['name']} has empty description"

    def test_tool_schemas_have_valid_input_schemas(self, tool_registry):
        """Test that all tool schemas have valid input schemas"""
        schemas = tool_registry.get_tool_schemas()
        for schema in schemas:
            function_def = schema["function"]
            parameters = function_def["parameters"]
            assert parameters, f"Tool {function_def['name']} missing input schema"
            assert "type" in parameters, f"Tool {function_def['name']} schema missing type"
            assert "properties" in parameters, f"Tool {function_def['name']} schema missing properties"
            assert parameters["type"] == "object"

    def test_all_tools_registered_in_schemas(self, tool_registry):
        """Test that all tools in the handler mapping have schemas"""
        schemas = tool_registry.get_tool_schemas()
        schema_names = {schema["function"]["name"] for schema in schemas}
        handler_names = set(tool_registry._tool_handlers.keys())
        assert schema_names == handler_names, "Schema names and handler names don't match"


class TestWorkspaceChanges:
    """Tests for _get_workspace_changes function"""

    def test_get_workspace_changes_modified_files(self, temp_workspace):
        """Test detecting modified files"""
        # Create a test file and commit it
        test_file = temp_workspace.root / "test.py"
        test_file.write_text("original content\n")

        # Initialize git repo
        subprocess.run(["git", "init", "-q"], cwd=temp_workspace.root, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=temp_workspace.root, check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=temp_workspace.root, check=True)
        subprocess.run(["git", "add", "test.py"], cwd=temp_workspace.root, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "Initial commit"], cwd=temp_workspace.root, check=True)

        # Modify the file
        test_file.write_text("modified content\n")

        # Get workspace changes
        changes = _get_workspace_changes(temp_workspace.root)

        assert len(changes) == 1
        assert changes[0].path == "test.py"
        assert changes[0].action == "modify"

    def test_get_workspace_changes_new_files(self, temp_workspace):
        """Test detecting new untracked files"""
        # Initialize git repo
        subprocess.run(["git", "init", "-q"], cwd=temp_workspace.root, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=temp_workspace.root, check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=temp_workspace.root, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "Initial commit", "--allow-empty"], cwd=temp_workspace.root, check=True)

        # Create a new file
        test_file = temp_workspace.root / "new_file.py"
        test_file.write_text("new content\n")

        # Get workspace changes
        changes = _get_workspace_changes(temp_workspace.root)

        assert len(changes) == 1
        assert changes[0].path == "new_file.py"
        assert changes[0].action == "create"

    def test_get_workspace_changes_deleted_files(self, temp_workspace):
        """Test detecting deleted files"""
        # Create a test file and commit it
        test_file = temp_workspace.root / "test.py"
        test_file.write_text("original content\n")

        # Initialize git repo
        subprocess.run(["git", "init", "-q"], cwd=temp_workspace.root, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=temp_workspace.root, check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=temp_workspace.root, check=True)
        subprocess.run(["git", "add", "test.py"], cwd=temp_workspace.root, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "Initial commit"], cwd=temp_workspace.root, check=True)

        # Delete the file
        test_file.unlink()

        # Get workspace changes
        changes = _get_workspace_changes(temp_workspace.root)

        assert len(changes) == 1
        assert changes[0].path == "test.py"
        assert changes[0].action == "delete"

    def test_get_workspace_changes_multiple_changes(self, temp_workspace):
        """Test detecting multiple file changes"""
        # Create test files and commit them
        file1 = temp_workspace.root / "file1.py"
        file2 = temp_workspace.root / "file2.py"
        file1.write_text("content1\n")
        file2.write_text("content2\n")

        # Initialize git repo
        subprocess.run(["git", "init", "-q"], cwd=temp_workspace.root, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=temp_workspace.root, check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=temp_workspace.root, check=True)
        subprocess.run(["git", "add", "."], cwd=temp_workspace.root, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "Initial commit"], cwd=temp_workspace.root, check=True)

        # Modify one file, delete another, create a new one
        file1.write_text("modified content\n")
        file2.unlink()
        file3 = temp_workspace.root / "file3.py"
        file3.write_text("new content\n")

        # Get workspace changes
        changes = _get_workspace_changes(temp_workspace.root)

        assert len(changes) == 3
        paths = {change.path for change in changes}
        actions = {change.path: change.action for change in changes}

        assert "file1.py" in paths
        assert "file2.py" in paths
        assert "file3.py" in paths
        assert actions["file1.py"] == "modify"
        assert actions["file2.py"] == "delete"
        assert actions["file3.py"] == "create"

    def test_get_workspace_changes_empty_workspace(self, temp_workspace):
        """Test that empty workspace returns no changes"""
        # Initialize git repo with no changes
        subprocess.run(["git", "init", "-q"], cwd=temp_workspace.root, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=temp_workspace.root, check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=temp_workspace.root, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "Initial commit", "--allow-empty"], cwd=temp_workspace.root, check=True)

        # Get workspace changes
        changes = _get_workspace_changes(temp_workspace.root)

        assert len(changes) == 0

    def test_get_workspace_changes_rejects_rename(self, temp_workspace):
        """Test that file renames are rejected"""
        # Create a test file and commit it
        test_file = temp_workspace.root / "old_name.py"
        test_file.write_text("content\n")

        # Initialize git repo
        subprocess.run(["git", "init", "-q"], cwd=temp_workspace.root, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=temp_workspace.root, check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=temp_workspace.root, check=True)
        subprocess.run(["git", "add", "old_name.py"], cwd=temp_workspace.root, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "Initial commit"], cwd=temp_workspace.root, check=True)

        # Rename the file using git mv
        subprocess.run(["git", "mv", "old_name.py", "new_name.py"], cwd=temp_workspace.root, check=True)

        # Get workspace changes should raise RuntimeError
        with pytest.raises(RuntimeError, match="File rename is not supported"):
            _get_workspace_changes(temp_workspace.root)

    def test_generate_patch_with_new_files(self, temp_workspace):
        """Test that generate_patch handles new files correctly"""
        # Initialize git repo
        subprocess.run(["git", "init", "-q"], cwd=temp_workspace.root, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=temp_workspace.root, check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=temp_workspace.root, check=True)

        # Create and commit an initial file
        initial_file = temp_workspace.root / "initial.py"
        initial_file.write_text("initial content\n")
        subprocess.run(["git", "add", "initial.py"], cwd=temp_workspace.root, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "Initial commit"], cwd=temp_workspace.root, check=True)

        # Create a new file (not committed)
        new_file = temp_workspace.root / "new_file.py"
        new_file.write_text("new content\n")

        # Modify the initial file
        initial_file.write_text("modified content\n")

        # Get workspace changes
        changes = _get_workspace_changes(temp_workspace.root)

        # Generate patch
        patch_content = generate_patch(temp_workspace.root, changes)

        # Verify patch contains both new and modified files
        assert "new_file.py" in patch_content or "new content" in patch_content
        assert "initial.py" in patch_content or "modified content" in patch_content

    def test_generate_patch_without_new_files(self, temp_workspace):
        """Test that generate_patch works when there are no new files"""
        # Initialize git repo
        subprocess.run(["git", "init", "-q"], cwd=temp_workspace.root, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=temp_workspace.root, check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=temp_workspace.root, check=True)

        # Create and commit an initial file
        initial_file = temp_workspace.root / "initial.py"
        initial_file.write_text("initial content\n")
        subprocess.run(["git", "add", "initial.py"], cwd=temp_workspace.root, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "Initial commit"], cwd=temp_workspace.root, check=True)

        # Modify the initial file
        initial_file.write_text("modified content\n")

        # Get workspace changes
        changes = _get_workspace_changes(temp_workspace.root)

        # Generate patch
        patch_content = generate_patch(temp_workspace.root, changes)

        # Verify patch contains the modified file
        assert "initial.py" in patch_content or "modified content" in patch_content


class TestWorkspaceChange:
    """Tests for WorkspaceChange dataclass"""

    def test_workspace_change_creation(self):
        """Test creating a WorkspaceChange instance"""
        change = WorkspaceChange(path="test.py", action="modify")
        assert change.path == "test.py"
        assert change.action == "modify"

    def test_workspace_change_equality(self):
        """Test WorkspaceChange equality"""
        change1 = WorkspaceChange(path="test.py", action="modify")
        change2 = WorkspaceChange(path="test.py", action="modify")
        change3 = WorkspaceChange(path="test.py", action="create")

        assert change1 == change2
        assert change1 != change3
