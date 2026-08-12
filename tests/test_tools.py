import subprocess
import tempfile
from pathlib import Path

import pytest

from patchpilot.models import ToolResult
from patchpilot.tools import (
    ApplyPatchInput,
    EditFileInput,
    ReadFileInput,
    RunCommandInput,
    SearchCodeInput,
    ToolDefinition,
    ToolRegistry,
    WorkspaceChange,
    _get_workspace_changes,
    generate_json_schema,
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

        # Verify file was actually modified
        updated_content = test_file.read_text()
        assert "if page < 1:" in updated_content

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
        assert "tests directory rejected" in result.content

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
        test_file.write_text("old content\n")

        result = tool_registry.apply_patch({
            "path": "test.py",
            "content": "new content\n"
        })
        assert result.ok
        assert "---" in result.content or "no diff" in result.content

        # Verify file was actually modified
        updated_content = test_file.read_text()
        assert updated_content == "new content\n"

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
        assert "tests directory rejected" in result.content

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
        """Test applying patch with empty content (file deletion scenario)"""
        test_file = temp_workspace.root / "test.py"
        test_file.write_text("original content\n")

        result = tool_registry.apply_patch({
            "path": "test.py",
            "content": ""
        })
        assert result.ok

        # Verify file was emptied
        updated_content = test_file.read_text()
        assert updated_content == ""


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
        assert tool_names == {"search_code", "read_file", "edit_file", "apply_patch", "run_command"}

    def test_get_tool_schema_existing(self, tool_registry):
        """Test getting a specific tool schema that exists"""
        schema = tool_registry.get_tool_schema("search_code")
        assert schema is not None
        assert schema.name == "search_code"
        assert schema.description is not None
        assert schema.input_schema is not None
        assert isinstance(schema.input_schema, dict)

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
