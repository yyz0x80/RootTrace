"""Tool definitions and registry for PatchPilot agent operations.

This module provides the core tool system for the PatchPilot agent, including:
- Input validation dataclasses for each tool
- JSON Schema generation for tool definitions
- A tool registry for dynamic tool registration
- Tool implementations for repository operations

Available tools:
- search_code: Search for code patterns using ripgrep
- read_file: Read file content with optional line ranges
- edit_file: Edit files using exact text replacement
- insert_text: Insert text at a specific line number
- apply_patch: Apply a patch to a file (create or modify)
- run_command: Execute allowed commands in the workspace

The tool system enforces security boundaries through input validation,
output size limits, and workspace policy enforcement.
"""

import difflib
import shlex
import subprocess
import unicodedata
from dataclasses import MISSING, dataclass, fields
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, ClassVar, Protocol, Union, get_type_hints

from patchpilot.models import ToolFailureType, ToolResult
from patchpilot.validation import run_intermediate_validation
from patchpilot.workspace import Workspace

# Standard ignore patterns for temporary and compiled files
# These patterns are applied regardless of the target repository's .gitignore
DEFAULT_IGNORE_PATTERNS = [
    # Python cache files
    "__pycache__/",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "*.egg-info/",
    ".pytest_cache/",
    ".ruff_cache/",
    ".mypy_cache/",
    # macOS files
    ".DS_Store",
    ".AppleDouble",
    ".LSOverride",
    # Editor temporary files
    "*.swp",
    "*.swo",
    "*.swn",
    "*.bak",
    "*~",
    # Build artifacts
    "dist/",
    "build/",
    "*.egg",
]


def _should_ignore_file_in_changes(file_path: str) -> bool:
    """Check if a file should be ignored based on standard patterns.

    This function implements a multi-layer ignore system that does not depend
    on the target repository's .gitignore configuration, ensuring consistent
    behavior across different projects.

    Args:
        file_path: The file path to check (relative to workspace root)

    Returns:
        True if the file should be ignored, False otherwise
    """
    from pathlib import PurePosixPath

    normalized_path = str(PurePosixPath(file_path))

    for pattern in DEFAULT_IGNORE_PATTERNS:
        # Handle directory patterns (ending with /)
        if pattern.endswith("/"):
            dir_name = pattern[:-1]
            # Check if directory name appears in path
            if dir_name in normalized_path.split("/"):
                return True
        # Handle file patterns with wildcards
        elif "*" in pattern:
            # Check if filename matches the pattern
            filename = normalized_path.split("/")[-1]
            if fnmatch(filename, pattern):
                return True
        # Handle exact matches
        elif normalized_path == pattern or normalized_path.startswith(pattern + "/"):
            return True

    return False


class CommandRunnerProtocol(Protocol):
    """Protocol for command execution (DockerSandbox or subprocess fallback)."""

    def run(self, command: str, timeout_seconds: int) -> Any:
        """Execute a command and return a result with stdout, stderr, exit_code."""
        ...


@dataclass
class WorkspaceChange:
    """Represents a single file change in the workspace.

    Attributes:
        path: Relative path to the changed file
        action: Type of change - 'create', 'modify', or 'delete'
    """
    path: str
    action: str


@dataclass
class MatchResult:
    """Result of text matching attempt.

    Attributes:
        matched: Whether a match was found
        matched_text: The actual text that matched in the file
        method: The matching method used ('exact', 'rstrip', 'strip', 'unicode')
    """
    matched: bool
    matched_text: str = ""
    method: str = ""


def _get_workspace_changes(workspace: Path) -> list[WorkspaceChange]:
    """Get all workspace changes using git status --porcelain.

    Parses git status output to identify created, modified, and deleted files.
    File renames are not supported in the current MVP.
    Filters out temporary and compiled files based on standard ignore patterns.

    Args:
        workspace: Path to the workspace directory

    Returns:
        List of WorkspaceChange objects representing each file change

    Raises:
        RuntimeError: If file rename is detected (not supported in MVP)
        subprocess.CalledProcessError: If git command fails
    """
    result = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
        ],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=True,
    )

    changes = []

    for line in result.stdout.splitlines():
        if not line.strip():
            continue

        status = line[:2]
        path = line[3:].strip()

        # Skip ignored files (cache files, build artifacts, etc.)
        if _should_ignore_file_in_changes(path):
            continue

        # MVP does not support file renames
        if "R" in status:
            raise RuntimeError(
                "File rename is not supported by the current PatchPilot MVP."
            )

        if status == "??":
            action = "create"
        elif "D" in status:
            action = "delete"
        elif "A" in status:
            action = "create"
        else:
            action = "modify"

        changes.append(
            WorkspaceChange(
                path=path,
                action=action,
            )
        )

    return changes


def generate_patch(
    workspace: Path,
    changes: list[WorkspaceChange],
) -> str:
    """Generate a git diff patch including newly created files.

    This function handles the special case where newly created files
    don't appear in normal git diff output. It uses 'git add -N' to
    register new files with git without committing them, then generates
    a complete diff using 'git diff HEAD'.

    Args:
        workspace: Path to the workspace directory
        changes: List of WorkspaceChange objects representing file changes

    Returns:
        String containing the complete git diff output

    Raises:
        subprocess.CalledProcessError: If git commands fail
    """
    # Identify created files that need to be added to git index
    created_files = [
        change.path
        for change in changes
        if change.action == "create"
    ]

    # Register new files with git without committing (git add -N)
    # This tells git to include these files in the diff output
    if created_files:
        subprocess.run(
            [
                "git",
                "add",
                "-N",
                "--",
                *created_files,
            ],
            cwd=workspace,
            check=True,
        )

    # Generate the complete diff including new files
    result = subprocess.run(
        [
            "git",
            "diff",
            "--binary",
            "--no-color",
            "HEAD",
        ],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=True,
    )

    return result.stdout


@dataclass
class ToolInput:
    """Base class for tool input validation"""
    description: ClassVar[str] = ""


@dataclass
class SearchCodeInput(ToolInput):
    """Input for search_code tool"""
    description: ClassVar[str] = "Search for code patterns using ripgrep (rg). Returns matching lines with line numbers."
    query: str
    path: str = "."

    def __post_init__(self):
        if not isinstance(self.query, str):
            raise TypeError(f"query must be str, not {type(self.query).__name__}")
        if not isinstance(self.path, str):
            raise TypeError(f"path must be str, not {type(self.path).__name__}")


@dataclass
class ReadFileInput(ToolInput):
    """Input for read_file tool"""
    description: ClassVar[str] = (
        "Read one workspace file using a relative file path; absolute paths and "
        "directories are not accepted. By default returns content with line "
        "numbers prefixed (e.g., '1: content'). Set raw=True to get content "
        "without line numbers. Supports optional line ranges for partial reads."
    )
    path: str
    start_line: int = 1
    end_line: int | None = None
    raw: bool = False

    def __post_init__(self):
        if not isinstance(self.path, str):
            raise TypeError(f"path must be str, not {type(self.path).__name__}")
        if not isinstance(self.start_line, int):
            raise TypeError(f"start_line must be int, not {type(self.start_line).__name__}")
        if self.end_line is not None and not isinstance(self.end_line, int):
            raise TypeError(f"end_line must be int or None, not {type(self.end_line).__name__}")
        if not isinstance(self.raw, bool):
            raise TypeError(f"raw must be bool, not {type(self.raw).__name__}")


@dataclass
class EditFileInput(ToolInput):
    """Input for edit_file tool"""
    description: ClassVar[str] = "Edit file using exact text replacement. Replaces the first occurrence of old_text with new_text and returns a unified diff. Provide context_lines to include surrounding context for better matching. Set preview=True to see the change without applying it. CRITICAL: old_text must NOT contain line number prefixes (e.g., '1:', '2:') - use read_file with raw=True instead. NOTE: old_text cannot be empty - use insert_text tool to add new content at a specific line."
    path: str
    old_text: str
    new_text: str
    context_lines: int = 0
    preview: bool = False

    def __post_init__(self):
        if not isinstance(self.path, str):
            raise TypeError(f"path must be str, not {type(self.path).__name__}")
        if not isinstance(self.old_text, str):
            raise TypeError(f"old_text must be str, not {type(self.old_text).__name__}")
        if not isinstance(self.new_text, str):
            raise TypeError(f"new_text must be str, not {type(self.new_text).__name__}")
        if not isinstance(self.context_lines, int):
            raise TypeError(f"context_lines must be int, not {type(self.context_lines).__name__}")
        if not isinstance(self.preview, bool):
            raise TypeError(f"preview must be bool, not {type(self.preview).__name__}")
        if self.context_lines < 0:
            raise ValueError(f"context_lines must be non-negative, got {self.context_lines}")


@dataclass
class EditFileByLineInput(ToolInput):
    """Input for edit_file_by_line tool"""
    description: ClassVar[str] = "Edit file by line number range. Replaces lines from start_line to end_line (inclusive) with new_text and returns a unified diff. Set preview=True to see the change without applying it."
    path: str
    start_line: int
    end_line: int
    new_text: str
    preview: bool = False

    def __post_init__(self):
        if not isinstance(self.path, str):
            raise TypeError(f"path must be str, not {type(self.path).__name__}")
        if not isinstance(self.start_line, int):
            raise TypeError(f"start_line must be int, not {type(self.start_line).__name__}")
        if not isinstance(self.end_line, int):
            raise TypeError(f"end_line must be int, not {type(self.end_line).__name__}")
        if not isinstance(self.new_text, str):
            raise TypeError(f"new_text must be str, not {type(self.new_text).__name__}")
        if not isinstance(self.preview, bool):
            raise TypeError(f"preview must be bool, not {type(self.preview).__name__}")
        if self.start_line < 1:
            raise ValueError(f"start_line must be >= 1, got {self.start_line}")
        if self.end_line < self.start_line:
            raise ValueError(f"end_line must be >= start_line, got start_line={self.start_line}, end_line={self.end_line}")


@dataclass
class RunCommandInput(ToolInput):
    """Input for run_command tool"""
    description: ClassVar[str] = "Run allowed commands in the workspace. Only allows: pytest, python -m pytest, ruff check, git diff, git status."
    command: str

    def __post_init__(self):
        if not isinstance(self.command, str):
            raise TypeError(f"command must be str, not {type(self.command).__name__}")


@dataclass
class InsertTextInput(ToolInput):
    """Input for insert_text tool"""
    description: ClassVar[str] = "Insert text at a specific line number in a file. Inserts the text after the specified line number and returns a unified diff. Line numbers are 1-based. Set preview=True to see the change without applying it."
    path: str
    line_number: int
    text: str
    preview: bool = False

    def __post_init__(self):
        if not isinstance(self.path, str):
            raise TypeError(f"path must be str, not {type(self.path).__name__}")
        if not isinstance(self.line_number, int):
            raise TypeError(f"line_number must be int, not {type(self.line_number).__name__}")
        if not isinstance(self.text, str):
            raise TypeError(f"text must be str, not {type(self.text).__name__}")
        if not isinstance(self.preview, bool):
            raise TypeError(f"preview must be bool, not {type(self.preview).__name__}")
        if self.line_number < 0:
            raise ValueError(f"line_number must be >= 0, got {self.line_number}")


@dataclass
class ApplyPatchInput(ToolInput):
    """Input for apply_patch tool"""
    description: ClassVar[str] = "Apply a patch to a file. If the file exists, replaces its entire content and returns a unified diff. If the file doesn't exist, creates it with the given content. Set preview=True to see the change without applying it."
    path: str
    content: str
    preview: bool = False

    def __post_init__(self):
        if not isinstance(self.path, str):
            raise TypeError(f"path must be str, not {type(self.path).__name__}")
        if not isinstance(self.content, str):
            raise TypeError(f"content must be str, not {type(self.content).__name__}")
        if not isinstance(self.preview, bool):
            raise TypeError(f"preview must be bool, not {type(self.preview).__name__}")


@dataclass
class ToolDefinition:
    """Structured definition of a tool with schema"""
    name: str
    description: str
    input_schema: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert ToolDefinition to a dictionary for JSON serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


def generate_json_schema(input_class: type[ToolInput]) -> dict[str, Any]:
    """
    Generate JSON Schema from a ToolInput dataclass.

    Args:
        input_class: The ToolInput dataclass to generate schema for

    Returns:
        JSON Schema dictionary for the input class
    """
    type_hints = get_type_hints(input_class)
    properties = {}
    required = []

    for field in fields(input_class):
        field_name = field.name
        field_type = type_hints.get(field_name, field.type)

        # Skip ClassVar fields (like description)
        if field_name == "description":
            continue

        # Map Python types to JSON Schema types
        json_type, is_nullable = _python_type_to_json_type(field_type)
        property_schema = {"type": json_type}

        # Mark as nullable if it's an Optional type
        if is_nullable:
            property_schema["type"] = [json_type, "null"]

        properties[field_name] = property_schema

        # Add default value if present
        if field.default is not MISSING:
            properties[field_name]["default"] = field.default
        elif field.default_factory is not MISSING:
            properties[field_name]["default"] = field.default_factory()
        # Add to required if no default and not nullable
        elif not is_nullable:
            required.append(field_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def _python_type_to_json_type(python_type: type) -> tuple[str, bool]:
    """
    Convert Python type to JSON Schema type string and nullable flag.

    Args:
        python_type: Python type to convert

    Returns:
        Tuple of (json_type_string, is_nullable)
    """
    # Check for Union types (including Optional which is Union[T, None])
    if hasattr(python_type, "__args__"):
        args = python_type.__args__

        # Handle Union types (Python 3.10+ union syntax like int | None)
        if hasattr(python_type, "__origin__"):
            origin = python_type.__origin__
            if origin is Union:  # type: ignore
                # Check if None is in the union
                has_none = type(None) in args
                # Get the first non-None type
                for arg in args:
                    if arg is not type(None):
                        json_type, _ = _python_type_to_json_type(arg)
                        return json_type, has_none
                return "string", has_none
            # Handle list types
            elif origin is list:  # type: ignore
                return "array", False
        else:
            # Python 3.10+ union syntax (int | None) has __args__ but no __origin__
            # Check if None is in the union
            has_none = type(None) in args
            # Get the first non-None type
            for arg in args:
                if arg is not type(None):
                    json_type, _ = _python_type_to_json_type(arg)
                    return json_type, has_none
            return "string", has_none

    # Handle basic types
    if python_type == str:
        return "string", False
    elif python_type == int:
        return "integer", False
    elif python_type == float:
        return "number", False
    elif python_type == bool:
        return "boolean", False
    elif python_type == type(None):
        return "null", True

    return "string", False  # Default fallback


class ToolRegistry:
    """Registry for available tools with dynamic schema registration"""

    # Day 1 allowed commands
    ALLOWED_COMMANDS: ClassVar[set[str]] = {
        "pytest",
        "python",
        "ruff",
        "git",
    }

    # Maximum output size limits
    MAX_SEARCH_OUTPUT = 100_000  # characters
    MAX_FILE_LINES = 300
    COMMAND_TIMEOUT = 60

    def __init__(self, workspace: Workspace, command_runner: CommandRunnerProtocol | None = None):
        self.workspace = workspace
        self.command_runner = command_runner
        # Dynamic tool registration storage
        self._tool_definitions: dict[str, ToolDefinition] = {}
        self._tool_handlers: dict[str, Any] = {}

        # Register default tools
        self._register_default_tools()

    def update_command_runner(
        self,
        command_runner: CommandRunnerProtocol | None,
    ) -> None:
        """Update the isolated runner used by the command tool."""
        self.command_runner = command_runner

    def _register_default_tools(self) -> None:
        """Register the default set of tools"""
        self.register_tool(
            name="search_code",
            input_class=SearchCodeInput,
            handler=self.search_code,
        )
        self.register_tool(
            name="read_file",
            input_class=ReadFileInput,
            handler=self.read_file,
        )
        self.register_tool(
            name="edit_file",
            input_class=EditFileInput,
            handler=self.edit_file,
        )
        self.register_tool(
            name="insert_text",
            input_class=InsertTextInput,
            handler=self.insert_text,
        )
        self.register_tool(
            name="edit_file_by_line",
            input_class=EditFileByLineInput,
            handler=self.edit_file_by_line,
        )
        self.register_tool(
            name="apply_patch",
            input_class=ApplyPatchInput,
            handler=self.apply_patch,
        )
        self.register_tool(
            name="run_command",
            input_class=RunCommandInput,
            handler=self.run_command,
        )

    def update_workspace(self, workspace: Workspace) -> None:
        """Update the workspace used by all tools.

        Args:
            workspace: New Workspace instance to use for path resolution
        """
        self.workspace = workspace

    def register_tool(
        self,
        name: str,
        input_class: type[ToolInput],
        handler: Any,
    ) -> None:
        """
        Register a tool with its input schema and handler.

        Args:
            name: Tool name
            input_class: ToolInput dataclass for schema generation
            handler: Callable that implements the tool logic
        """
        # Extract description from the input class
        description = getattr(input_class, "description", "")

        # Generate JSON schema from the input class
        input_schema = generate_json_schema(input_class)

        # Create tool definition
        tool_def = ToolDefinition(
            name=name,
            description=description,
            input_schema=input_schema,
        )

        # Register the tool
        self._tool_definitions[name] = tool_def
        self._tool_handlers[name] = handler

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """
        Get all available tool definitions in OpenAI-compatible format.

        Returns:
            List of tool dictionaries in OpenAI format with type, function name, description, and parameters
        """
        openai_tools = []
        for tool_def in self._tool_definitions.values():
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool_def.name,
                    "description": tool_def.description,
                    "parameters": tool_def.input_schema,
                }
            })
        return openai_tools

    def get_tool_schema(self, tool_name: str) -> ToolDefinition | None:
        """
        Get a specific tool's definition by name.

        Args:
            tool_name: Name of the tool to look up

        Returns:
            ToolDefinition if found, None otherwise
        """
        return self._tool_definitions.get(tool_name)

    def get_available_tools(self) -> list[str]:
        """
        Get list of all available tool names.

        Returns:
            List of registered tool names
        """
        return list(self._tool_definitions.keys())

    def unregister_tool(self, name: str) -> bool:
        """
        Unregister a tool by name.

        Args:
            name: Name of the tool to unregister

        Returns:
            True if tool was removed, False if not found
        """
        if name in self._tool_definitions:
            del self._tool_definitions[name]
            del self._tool_handlers[name]
            return True
        return False

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """
        Execute a tool by name with the given arguments.

        Args:
            name: Name of the tool to execute
            arguments: Dictionary of arguments to pass to the tool

        Returns:
            ToolResult containing the execution result

        Raises:
            KeyError: If the tool name is not registered
        """
        handler = self._tool_handlers.get(name)
        if handler is None:
            raise KeyError(f"Tool not found: {name}")
        result = handler(arguments)
        failure_type = result.failure_type
        if not result.ok and failure_type is None:
            failure_type = ToolFailureType.TOOL_FAILURE

        return ToolResult(
            ok=result.ok,
            content=self.sanitize_workspace_paths(result.content),
            failure_type=failure_type,
        )

    def sanitize_workspace_paths(self, content: str) -> str:
        """Replace internal workspace paths in model-facing tool output."""
        root = str(self.workspace.root)
        return content.replace(f"{root}/", "").replace(root, ".")

    def search_code(self, arguments: dict[str, Any]) -> ToolResult:
        """
        Search for code using ripgrep (rg).
        
        Args:
            arguments: Dict with 'query' and optional 'path' (default ".")
        
        Returns:
            ToolResult with search results or empty string if no matches
        """
        try:
            input_data = SearchCodeInput(**arguments)
        except (TypeError, ValueError) as e:
            return ToolResult(ok=False, content=f"Invalid input: {e}")

        try:
            self.workspace.resolve(input_data.path)
        except ValueError as e:
            return ToolResult(ok=False, content=f"Path error: {e}")

        # Run ripgrep with subprocess
        try:
            args = ["rg", "-n", input_data.query, input_data.path]
            result = subprocess.run(
                args,
                cwd=self.workspace.root,
                capture_output=True,
                text=True,
                timeout=self.COMMAND_TIMEOUT,
                check=False,
            )

            # Limit output size
            output = result.stdout
            if len(output) > self.MAX_SEARCH_OUTPUT:
                output = output[:self.MAX_SEARCH_OUTPUT] + "\n... (output truncated)"

            # No matches is not an error
            return ToolResult(ok=True, content=output)

        except subprocess.TimeoutExpired:
            return ToolResult(ok=False, content="Search timed out")
        except FileNotFoundError:
            return ToolResult(ok=False, content="ripgrep (rg) not found in PATH")
        except OSError as e:
            return ToolResult(ok=False, content=f"Search failed: {e}")

    def read_file(self, arguments: dict[str, Any]) -> ToolResult:
        """
        Read file content with optional line numbers.
        
        Args:
            arguments: Dict with 'path', optional 'start_line' (default 1), 
                      optional 'end_line' (default None), and optional 'raw' (default False)
        
        Returns:
            ToolResult with file content. If raw=False, returns content with line numbers.
            If raw=True, returns raw content without line numbers.
        """
        try:
            input_data = ReadFileInput(**arguments)
        except (TypeError, ValueError) as e:
            return ToolResult(ok=False, content=f"Invalid input: {e}")

        try:
            resolved_path = self.workspace.assert_read_allowed(input_data.path)
        except (ValueError, PermissionError) as e:
            return ToolResult(ok=False, content=f"Path error: {e}")

        # Check file exists
        if not resolved_path.exists():
            return ToolResult(ok=False, content=f"File not found: {input_data.path}")

        if not resolved_path.is_file():
            return ToolResult(ok=False, content=f"Not a file: {input_data.path}")

        try:
            # Read file content
            with open(resolved_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Apply line range limits
            start_idx = max(0, input_data.start_line - 1)
            if input_data.end_line is None:
                end_idx = len(lines)
            else:
                end_idx = min(len(lines), input_data.end_line)

            # Enforce maximum line limit
            if end_idx - start_idx > self.MAX_FILE_LINES:
                return ToolResult(
                    ok=False,
                    content=f"Request exceeds maximum line limit of {self.MAX_FILE_LINES}"
                )

            # Get selected lines
            selected_lines = lines[start_idx:end_idx]

            # Return raw content if requested
            if input_data.raw:
                return ToolResult(ok=True, content="".join(selected_lines))

            # Format with line numbers (default behavior)
            output_lines = []
            for i, line in enumerate(selected_lines, start=start_idx + 1):
                output_lines.append(f"{i}: {line.rstrip()}")

            return ToolResult(ok=True, content="\n".join(output_lines))

        except UnicodeDecodeError:
            return ToolResult(ok=False, content="File is not valid UTF-8 text")
        except OSError as e:
            return ToolResult(ok=False, content=f"Read failed: {e}")

    def edit_file(self, arguments: dict[str, Any]) -> ToolResult:
        """
        Edit file using multi-tier text replacement with optional context and preview.
        
        Implements a fallback matching strategy:
        1. Exact match (preferred)
        2. Right-stripped match (trailing whitespace)
        3. Trimmed match (leading/trailing whitespace)
        4. Unicode-normalized match
        
        Args:
            arguments: Dict with 'path', 'old_text', 'new_text', optional 'context_lines' (default 0),
                      and optional 'preview' (default False)
        
        Returns:
            ToolResult with unified diff or error message. If preview=True, shows change without applying.
        """
        try:
            input_data = EditFileInput(**arguments)
        except (TypeError, ValueError) as e:
            return ToolResult(ok=False, content=f"Invalid input: {e}")

        # Early validation for empty old_text
        if not input_data.old_text or not input_data.old_text.strip():
            return ToolResult(
                ok=False,
                content="ERROR: old_text is empty. The edit_file tool requires existing text to replace. "
                       "If you want to insert new content at a specific location, use the insert_text tool instead. "
                       "Example: insert_text(path='file.py', line_number=10, text='new content')"
            )

        # Early validation for line number prefixes in old_text
        old_text_lines = input_data.old_text.splitlines()
        has_line_prefixes = any(
            line.strip().startswith(tuple(f"{i}:" for i in range(1, 1000)))
            for line in old_text_lines
        )
        if has_line_prefixes:
            return ToolResult(
                ok=False,
                content="ERROR: old_text contains line number prefixes (e.g., '1:', '2:'). "
                       "Line numbers from read_file output are NOT part of the actual file content. "
                       "Use read_file with raw=True to get content without line numbers, "
                       "or remove the line number prefixes before using the text as old_text. "
                       "Example correction: Wrong: old_text='1: def hello():\\n2:     return world' "
                       "Right: old_text='def hello():\\n    return world'"
            )

        try:
            resolved_path = self.workspace.assert_write_allowed(input_data.path)
        except (ValueError, PermissionError) as e:
            return ToolResult(ok=False, content=f"Path error: {e}")

        # Check file exists
        if not resolved_path.exists():
            return ToolResult(ok=False, content=f"File not found: {input_data.path}")

        if not resolved_path.is_file():
            return ToolResult(ok=False, content=f"Not a file: {input_data.path}")

        try:
            # Read current content
            with open(resolved_path, "r", encoding="utf-8") as f:
                original_content = f.read()

            # Try multi-tier matching strategy
            match_result = self._try_match_with_fallback(original_content, input_data.old_text)
            
            if not match_result.matched:
                return self._enhanced_edit_error(original_content, input_data.old_text, resolved_path)
            
            # Use the matched text for replacement
            matched_text = match_result.matched_text
            match_method = match_result.method
            
            # Verify uniqueness
            count = original_content.count(matched_text)
            if count > 1:
                return ToolResult(
                    ok=False,
                    content=f"Matched text appears {count} times in file (matched using {match_method}). "
                           f"Please provide more specific context or use context_lines parameter."
                )

            # Perform replacement
            new_content = original_content.replace(matched_text, input_data.new_text, 1)

            # Generate unified diff
            original_lines = original_content.splitlines(keepends=True)
            new_lines = new_content.splitlines(keepends=True)
            diff = difflib.unified_diff(
                original_lines,
                new_lines,
                fromfile=input_data.path,
                tofile=input_data.path,
                lineterm=""
            )
            diff_text = "".join(diff)

            # If preview mode, return diff without writing
            if input_data.preview:
                return ToolResult(ok=True, content=f"PREVIEW MODE - No changes applied:\n{diff_text or '(no diff)'}")

            # Write new content
            with open(resolved_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            # Run intermediate validation to catch syntax errors early
            validation_passed, validation_errors = run_intermediate_validation(
                resolved_path,
                display_path=input_data.path,
            )
            if not validation_passed:
                # Revert the change if validation fails
                with open(resolved_path, "w", encoding="utf-8") as f:
                    f.write(original_content)
                return ToolResult(
                    ok=False,
                    content="Edit reverted due to validation failure:\n" + "\n".join(validation_errors)
                )

            return ToolResult(ok=True, content=diff_text or "(no diff)")

        except UnicodeDecodeError:
            return ToolResult(ok=False, content="File is not valid UTF-8 text")
        except OSError as e:
            return ToolResult(ok=False, content=f"Edit failed: {e}")

    def _try_match_with_fallback(self, file_content: str, old_text: str) -> MatchResult:
        """Try to match old_text in file_content using multi-tier fallback strategy.

        Args:
            file_content: The current file content
            old_text: The text to search for

        Returns:
            MatchResult with matching status and method used
        """
        # Tier 1: Exact match (preferred)
        if old_text in file_content:
            return MatchResult(matched=True, matched_text=old_text, method="exact")
        
        # Tier 2: Right-stripped match (trailing whitespace)
        old_text_rstrip = old_text.rstrip()
        if old_text_rstrip and old_text_rstrip in file_content:
            return MatchResult(matched=True, matched_text=old_text_rstrip, method="rstrip")
        
        # Tier 3: Trimmed match (leading and trailing whitespace)
        old_text_stripped = old_text.strip()
        if old_text_stripped and old_text_stripped in file_content:
            return MatchResult(matched=True, matched_text=old_text_stripped, method="strip")
        
        # Tier 4: Unicode-normalized match
        # Normalize both strings to NFKC form for consistent comparison
        old_text_normalized = unicodedata.normalize('NFKC', old_text)
        file_content_normalized = unicodedata.normalize('NFKC', file_content)
        if old_text_normalized in file_content_normalized:
            return MatchResult(matched=True, matched_text=old_text_normalized, method="unicode")
        
        # No match found
        return MatchResult(matched=False, matched_text="", method="")

    def _enhanced_edit_error(self, file_content: str, old_text: str, file_path: Path) -> ToolResult:
        """Generate enhanced error message for edit_file failures with closest match and diff.

        Args:
            file_content: Current file content
            old_text: The text that was not found
            file_path: Path to the file for context

        Returns:
            ToolResult with detailed error information including closest match and diff
        """
        lines = file_content.splitlines()
        old_text_lines = old_text.splitlines()

        # Check if old_text is empty
        if not old_text or not old_text.strip():
            error_msg = "ERROR: old_text is empty.\n"
            error_msg += "The edit_file tool requires existing text to replace.\n"
            error_msg += "SOLUTION: If you want to insert new content, use the insert_text tool instead.\n"
            error_msg += "Example: insert_text(path='file.py', line_number=10, text='new content')\n"
            return ToolResult(ok=False, content=error_msg)

        # Check if old_text contains line number prefixes (common mistake)
        has_line_prefixes = any(
            line.strip().startswith(tuple(f"{i}:" for i in range(1, 1000)))
            for line in old_text_lines
        )

        error_msg = "old_text not found in file. The exact text you provided:\n"
        error_msg += f"  {old_text[:100]!r}{'...' if len(old_text) > 100 else ''}\n\n"

        if has_line_prefixes:
            error_msg += "ERROR: Your old_text contains line number prefixes (e.g., '1:', '2:').\n"
            error_msg += "Line numbers from read_file output are NOT part of the actual file content.\n"
            error_msg += "SOLUTION: Use read_file with raw=True to get content without line numbers,\n"
            error_msg += "or remove the line number prefixes before using the text as old_text.\n\n"
            error_msg += "Example correction:\n"
            error_msg += "  Wrong: old_text='1: def hello():\\n2:     return world'\n"
            error_msg += "  Right: old_text='def hello():\\n    return world'\n\n"

        # Find the closest match using sliding window for multi-line blocks
        closest_match = self._find_closest_match(file_content, old_text)

        if closest_match:
            error_msg += f"Closest match found at line {closest_match['line']} "
            error_msg += f"(similarity: {closest_match['similarity']:.2f}):\n"
            error_msg += f"  {closest_match['text'][:100]!r}{'...' if len(closest_match['text']) > 100 else ''}\n\n"

            # Generate diff between old_text and closest match
            error_msg += "Diff between your old_text and the closest match:\n"
            diff_lines = difflib.unified_diff(
                old_text.splitlines(keepends=True),
                closest_match['text'].splitlines(keepends=True),
                fromfile="your old_text",
                tofile="closest match",
                lineterm=""
            )
            error_msg += "".join(diff_lines)
            error_msg += "\n"

            error_msg += "SOLUTION: Re-read the file with raw=True and use the exact text from the file.\n"
            error_msg += "Copy the closest match above as your old_text, adjusting if needed.\n"
        else:
            # Fallback to line-by-line similarity check
            similar_lines = []
            for i, line in enumerate(lines):
                # Check if any line from old_text is similar to this line
                for old_line in old_text_lines:
                    if old_line and line:
                        similarity = difflib.SequenceMatcher(None, old_line, line).ratio()
                        if similarity > 0.5:  # More than 50% similar
                            similar_lines.append(f"Line {i+1}: {line[:80]}{'...' if len(line) > 80 else ''}")
                            break

            if similar_lines:
                error_msg += "Similar content found in file:\n"
                for similar_line in similar_lines[:5]:  # Show at most 5 similar lines
                    error_msg += f"  {similar_line}\n"
                error_msg += "\nPlease re-read the file and verify the exact text including whitespace and indentation.\n"
                error_msg += "Tip: Use read_file with raw=True to get clean content without line numbers.\n"
            else:
                error_msg += "No similar content found. Please re-read the file to get the exact text.\n"
                error_msg += "Tip: If you want to insert new content at a specific location, use insert_text instead.\n"

        return ToolResult(ok=False, content=error_msg)

    def _find_closest_match(self, file_content: str, old_text: str) -> dict[str, Any] | None:
        """Find the closest matching text block in the file content.

        Uses a sliding window approach to find the most similar multi-line block.
        Similarity is calculated using difflib.SequenceMatcher.ratio().

        Args:
            file_content: The current file content
            old_text: The text to search for

        Returns:
            Dictionary with 'line', 'text', and 'similarity' keys, or None if no good match found
        """
        lines = file_content.splitlines()
        old_text_lines = old_text.splitlines()
        num_old_lines = len(old_text_lines)

        if num_old_lines == 0:
            return None

        best_match = None
        best_similarity = 0.0

        # Try different window sizes around the expected size
        for window_size in [num_old_lines, num_old_lines + 1, num_old_lines - 1]:
            if window_size <= 0 or window_size > len(lines):
                continue

            # Slide through the file with the window
            for start_line in range(len(lines) - window_size + 1):
                window_lines = lines[start_line:start_line + window_size]
                window_text = "\n".join(window_lines)

                # Calculate similarity using SequenceMatcher
                similarity = difflib.SequenceMatcher(None, old_text, window_text).ratio()

                if similarity > best_similarity and similarity > 0.3:  # Minimum threshold
                    best_similarity = similarity
                    best_match = {
                        'line': start_line + 1,  # 1-based line number
                        'text': window_text,
                        'similarity': similarity
                    }

        return best_match

    def edit_file_by_line(self, arguments: dict[str, Any]) -> ToolResult:
        """
        Edit file by line number range with optional preview.
        
        Args:
            arguments: Dict with 'path', 'start_line', 'end_line', 'new_text', and optional 'preview' (default False)
        
        Returns:
            ToolResult with unified diff or error message. If preview=True, shows change without applying.
        """
        try:
            input_data = EditFileByLineInput(**arguments)
        except (TypeError, ValueError) as e:
            return ToolResult(ok=False, content=f"Invalid input: {e}")

        try:
            resolved_path = self.workspace.assert_write_allowed(input_data.path)
        except (ValueError, PermissionError) as e:
            return ToolResult(ok=False, content=f"Path error: {e}")

        # Check file exists
        if not resolved_path.exists():
            return ToolResult(ok=False, content=f"File not found: {input_data.path}")

        if not resolved_path.is_file():
            return ToolResult(ok=False, content=f"Not a file: {input_data.path}")

        try:
            # Read current content
            with open(resolved_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Save original content for potential revert
            original_content = "".join(lines)

            # Validate line range
            if input_data.start_line > len(lines):
                return ToolResult(
                    ok=False,
                    content=f"start_line {input_data.start_line} exceeds file length ({len(lines)} lines)"
                )
            if input_data.end_line > len(lines):
                return ToolResult(
                    ok=False,
                    content=f"end_line {input_data.end_line} exceeds file length ({len(lines)} lines)"
                )

            # Convert to 0-based indexing
            start_idx = input_data.start_line - 1
            end_idx = input_data.end_line  # end is inclusive for slicing

            # Build new content
            # Ensure new_text ends with newline if it doesn't already
            if input_data.new_text and not input_data.new_text.endswith('\n'):
                new_text_with_newline = input_data.new_text + '\n'
            else:
                new_text_with_newline = input_data.new_text

            new_lines = (
                lines[:start_idx] +
                [new_text_with_newline] +
                lines[end_idx:]
            )
            new_content = "".join(new_lines)

            # Generate unified diff
            original_lines = lines[:]  # Copy for diff
            diff = difflib.unified_diff(
                original_lines,
                new_lines,
                fromfile=input_data.path,
                tofile=input_data.path,
                lineterm=""
            )
            diff_text = "".join(diff)

            # If preview mode, return diff without writing
            if input_data.preview:
                return ToolResult(ok=True, content=f"PREVIEW MODE - No changes applied:\n{diff_text or '(no diff)'}")

            # Write new content
            with open(resolved_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            # Run intermediate validation to catch syntax errors early
            validation_passed, validation_errors = run_intermediate_validation(
                resolved_path,
                display_path=input_data.path,
            )
            if not validation_passed:
                # Revert the change if validation fails
                with open(resolved_path, "w", encoding="utf-8") as f:
                    f.write(original_content)
                return ToolResult(
                    ok=False,
                    content="Edit reverted due to validation failure:\n" + "\n".join(validation_errors)
                )

            return ToolResult(ok=True, content=diff_text or "(no diff)")

        except UnicodeDecodeError:
            return ToolResult(ok=False, content="File is not valid UTF-8 text")
        except OSError as e:
            return ToolResult(ok=False, content=f"Edit failed: {e}")

    def insert_text(self, arguments: dict[str, Any]) -> ToolResult:
        """
        Insert text at a specific line number in a file.

        Args:
            arguments: Dict with 'path', 'line_number', 'text', and optional 'preview' (default False)

        Returns:
            ToolResult with unified diff or error message. If preview=True, shows change without applying.
        """
        try:
            input_data = InsertTextInput(**arguments)
        except (TypeError, ValueError) as e:
            return ToolResult(ok=False, content=f"Invalid input: {e}")

        try:
            resolved_path = self.workspace.assert_write_allowed(input_data.path)
        except (ValueError, PermissionError) as e:
            return ToolResult(ok=False, content=f"Path error: {e}")

        # Check file exists
        if not resolved_path.exists():
            return ToolResult(ok=False, content=f"File not found: {input_data.path}")

        if not resolved_path.is_file():
            return ToolResult(ok=False, content=f"Not a file: {input_data.path}")

        try:
            # Read current content
            with open(resolved_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Save original content for potential revert
            original_content = "".join(lines)

            # Validate line number
            if input_data.line_number > len(lines):
                return ToolResult(
                    ok=False,
                    content=f"line_number {input_data.line_number} exceeds file length ({len(lines)} lines)"
                )

            # Convert to 0-based indexing
            # line_number=0 means insert at the beginning, line_number=N means insert after line N
            insert_idx = input_data.line_number

            # Ensure text ends with newline if it doesn't already
            if input_data.text and not input_data.text.endswith('\n'):
                text_with_newline = input_data.text + '\n'
            else:
                text_with_newline = input_data.text

            # Build new content by inserting text at the specified position
            new_lines = lines[:insert_idx] + [text_with_newline] + lines[insert_idx:]
            new_content = "".join(new_lines)

            # Generate unified diff
            original_lines = lines[:]  # Copy for diff
            diff = difflib.unified_diff(
                original_lines,
                new_lines,
                fromfile=input_data.path,
                tofile=input_data.path,
                lineterm=""
            )
            diff_text = "".join(diff)

            # If preview mode, return diff without writing
            if input_data.preview:
                return ToolResult(ok=True, content=f"PREVIEW MODE - No changes applied:\n{diff_text or '(no diff)'}")

            # Write new content
            with open(resolved_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            # Run intermediate validation to catch syntax errors early
            validation_passed, validation_errors = run_intermediate_validation(
                resolved_path,
                display_path=input_data.path,
            )
            if not validation_passed:
                # Revert the change if validation fails
                with open(resolved_path, "w", encoding="utf-8") as f:
                    f.write(original_content)
                return ToolResult(
                    ok=False,
                    content="Insert reverted due to validation failure:\n" + "\n".join(validation_errors)
                )

            return ToolResult(ok=True, content=diff_text or "(no diff)")

        except UnicodeDecodeError:
            return ToolResult(ok=False, content="File is not valid UTF-8 text")
        except OSError as e:
            return ToolResult(ok=False, content=f"Insert failed: {e}")

    def apply_patch(self, arguments: dict[str, Any]) -> ToolResult:
        """
        Apply a patch to a file. Creates the file if it doesn't exist.

        Args:
            arguments: Dict with 'path', 'content', and optional 'preview' (default False)

        Returns:
            ToolResult with unified diff (for modifications) or creation message (for new files).
            If preview=True, shows change without applying.
        """
        try:
            input_data = ApplyPatchInput(**arguments)
        except (TypeError, ValueError) as e:
            return ToolResult(ok=False, content=f"Invalid input: {e}")

        try:
            resolved_path = self.workspace.assert_write_allowed(input_data.path)
        except (ValueError, PermissionError) as e:
            return ToolResult(ok=False, content=f"Path error: {e}")

        # Check if file exists
        file_exists = resolved_path.exists()

        if file_exists:
            if not resolved_path.is_file():
                return ToolResult(ok=False, content=f"Not a file: {input_data.path}")

            try:
                # Read current content
                with open(resolved_path, "r", encoding="utf-8") as f:
                    original_content = f.read()

                # Generate unified diff
                original_lines = original_content.splitlines(keepends=True)
                new_lines = input_data.content.splitlines(keepends=True)
                diff = difflib.unified_diff(
                    original_lines,
                    new_lines,
                    fromfile=input_data.path,
                    tofile=input_data.path,
                    lineterm=""
                )
                diff_text = "".join(diff)

                # If preview mode, return diff without writing
                if input_data.preview:
                    return ToolResult(ok=True, content=f"PREVIEW MODE - No changes applied:\n{diff_text or '(no diff)'}")

                # Write new content
                with open(resolved_path, "w", encoding="utf-8") as f:
                    f.write(input_data.content)

                # Run intermediate validation to catch syntax errors early
                validation_passed, validation_errors = run_intermediate_validation(
                    resolved_path,
                    display_path=input_data.path,
                )
                if not validation_passed:
                    # Revert the change if validation fails
                    with open(resolved_path, "w", encoding="utf-8") as f:
                        f.write(original_content)
                    return ToolResult(
                        ok=False,
                        content="Patch reverted due to validation failure:\n" + "\n".join(validation_errors)
                    )

                return ToolResult(ok=True, content=diff_text or "(no diff)")

            except UnicodeDecodeError:
                return ToolResult(ok=False, content="File is not valid UTF-8 text")
            except OSError as e:
                return ToolResult(ok=False, content=f"Patch failed: {e}")
        else:
            # File doesn't exist, create it
            try:
                # If preview mode, return message without creating
                if input_data.preview:
                    return ToolResult(
                        ok=True,
                        content=f"PREVIEW MODE - Would create file: {input_data.path}\nContent preview:\n{input_data.content[:500]}{'...' if len(input_data.content) > 500 else ''}"
                    )

                # Ensure parent directory exists
                resolved_path.parent.mkdir(parents=True, exist_ok=True)

                # Write new content
                with open(resolved_path, "w", encoding="utf-8") as f:
                    f.write(input_data.content)

                # Run intermediate validation to catch syntax errors early
                validation_passed, validation_errors = run_intermediate_validation(
                    resolved_path,
                    display_path=input_data.path,
                )
                if not validation_passed:
                    # Remove the file if validation fails
                    resolved_path.unlink()
                    return ToolResult(
                        ok=False,
                        content="File creation reverted due to validation failure:\n" + "\n".join(validation_errors)
                    )

                return ToolResult(
                    ok=True,
                    content=f"Created file: {input_data.path}"
                )

            except OSError as e:
                return ToolResult(ok=False, content=f"File creation failed: {e}")

    @staticmethod
    def _is_verification_command(args: list[str]) -> bool:
        """Return whether a parsed command runs deterministic verification."""
        if not args:
            return False
        if args[0] == "pytest":
            return True
        if args[:3] == ["python", "-m", "pytest"]:
            return True
        return args[:2] == ["ruff", "check"]

    @classmethod
    def _command_failure_type(
        cls,
        args: list[str],
        exit_code: int,
        *,
        timed_out: bool = False,
    ) -> ToolFailureType:
        """Classify a non-zero command result by execution outcome."""
        if (
            cls._is_verification_command(args)
            and exit_code == 1
            and not timed_out
        ):
            return ToolFailureType.VERIFICATION_FAILURE
        return ToolFailureType.TOOL_FAILURE

    def run_command(self, arguments: dict[str, Any]) -> ToolResult:
        """
        Run an allowed command in the workspace.
        
        Args:
            arguments: Dict with 'command' string
        
        Returns:
            ToolResult with command output or error
        """
        try:
            input_data = RunCommandInput(**arguments)
        except (TypeError, ValueError) as e:
            return ToolResult(ok=False, content=f"Invalid input: {e}")

        # Parse command using shlex to handle quotes properly
        try:
            args = shlex.split(input_data.command)
        except ValueError as e:
            return ToolResult(ok=False, content=f"Invalid command syntax: {e}")

        # Validate command is allowed
        if not args:
            return ToolResult(ok=False, content="Empty command")

        base_command = args[0]
        if base_command not in self.ALLOWED_COMMANDS:
            return ToolResult(
                ok=False,
                content=f"Command '{base_command}' is not allowed. Allowed: {', '.join(sorted(self.ALLOWED_COMMANDS))}"
            )

        # Additional validation for specific commands
        if base_command == "python":
            # Only allow python -m pytest
            if len(args) >= 2 and args[1] == "-m":
                if len(args) >= 3 and args[2] != "pytest":
                    return ToolResult(
                        ok=False,
                        content="Only 'python -m pytest' is allowed for python command"
                    )
            else:
                return ToolResult(
                    ok=False,
                    content="Only 'python -m pytest' is allowed for python command"
                )

        if base_command == "git" and len(args) >= 2 and args[1] not in {"diff", "status"}:
            # Only allow git diff and git status
            return ToolResult(
                ok=False,
                content="Only 'git diff' and 'git status' are allowed for git command"
            )

        if base_command == "ruff" and len(args) >= 2 and args[1] != "check":
            # Only allow ruff check
            return ToolResult(
                ok=False,
                content="Only 'ruff check' is allowed for ruff command"
            )

        # Run command with DockerSandbox if available, otherwise subprocess
        if self.command_runner is not None:
            try:
                result = self.command_runner.run(
                    command=input_data.command,
                    timeout_seconds=self.COMMAND_TIMEOUT,
                )

                # Combine stdout and stderr
                output = result.stdout
                if result.stderr:
                    output += f"\n{result.stderr}" if output else result.stderr

                if result.exit_code != 0:
                    return ToolResult(
                        ok=False,
                        content=f"exit_code={result.exit_code}\n{output}",
                        failure_type=self._command_failure_type(
                            args,
                            result.exit_code,
                            timed_out=getattr(result, "timed_out", False),
                        ),
                    )

                return ToolResult(ok=True, content=output)

            except (OSError, subprocess.SubprocessError, AttributeError) as e:
                return ToolResult(ok=False, content=f"Docker execution failed: {e}")
        else:
            # Fallback to subprocess
            try:
                result = subprocess.run(
                    args,
                    cwd=self.workspace.root,
                    capture_output=True,
                    text=True,
                    timeout=self.COMMAND_TIMEOUT,
                    check=False,
                )

                # Combine stdout and stderr
                output = result.stdout
                if result.stderr:
                    output += f"\n{result.stderr}" if output else result.stderr

                if result.returncode != 0:
                    return ToolResult(
                        ok=False,
                        content=f"exit_code={result.returncode}\n{output}",
                        failure_type=self._command_failure_type(
                            args,
                            result.returncode,
                        ),
                    )

                return ToolResult(ok=True, content=output)

            except subprocess.TimeoutExpired:
                return ToolResult(ok=False, content="Command timed out")
            except FileNotFoundError:
                return ToolResult(ok=False, content=f"Command '{base_command}' not found in PATH")
            except OSError as e:
                return ToolResult(ok=False, content=f"Command failed: {e}")

    def dispatch(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        """
        Dispatch a tool call to the appropriate handler.

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments as a dict

        Returns:
            ToolResult from the tool execution
        """
        handler = self._tool_handlers.get(tool_name)
        if handler is None:
            return ToolResult(ok=False, content=f"Unknown tool: {tool_name}")

        return handler(arguments)
