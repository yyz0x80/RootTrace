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
- write_file: Write complete file content when exact replacement is unsuitable
- write_scratch_test: Write an isolated, non-patch test for supplemental evidence
- run_command: Execute allowed commands in the workspace

The model-facing ACI exposes two patch-editing tools. Specialized
line-based helpers remain available as Python methods for compatibility, but
are not registered as model tools.

The tool system enforces security boundaries through input validation,
output size limits, and workspace policy enforcement.
"""

import ast
import difflib
import logging
import re
import shlex
import subprocess
import unicodedata
from dataclasses import MISSING, dataclass, field, fields
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar, Protocol, Union, get_type_hints

from patchpilot.models import ToolFailureType, ToolResult
from patchpilot.policy.builtins import get_builtin_policies
from patchpilot.policy.evaluator import PolicyEvaluator
from patchpilot.policy.schema import PolicySet
from patchpilot.validation import run_intermediate_validation
from patchpilot.workspace import Workspace

logger = logging.getLogger(__name__)

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
    ".patchpilot_checks/",
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
    description: ClassVar[str] = (
        "Make a small change to an existing file using unique text replacement. "
        "The tool preserves block indentation for multiline Python replacements, "
        "validates the edited file, reverts invalid edits, and returns a unified "
        "diff. Read the file with raw=True first. old_text must be non-empty and "
        "must not include displayed line-number prefixes."
    )
    path: str = field(
        metadata={
            "description": (
                "Workspace-relative path of the existing source file to edit."
            )
        }
    )
    old_text: str = field(
        metadata={
            "description": (
                "Non-empty, unique text copied exactly from read_file(raw=True). "
                "It must already exist in the file."
            )
        }
    )
    new_text: str = field(
        metadata={
            "description": (
                "Replacement text for old_text. Include every line needed for "
                "the focused source-code change."
            )
        }
    )
    context_lines: int = field(
        default=0,
        metadata={"model_exposed": False},
    )
    preview: bool = field(
        default=False,
        metadata={"model_exposed": False},
    )

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
    description: ClassVar[str] = (
        "Run allowed commands in the workspace. Use 'python -m pytest' for "
        "tests. Bare 'pytest' input is normalized to the same invocation. "
        "Also allows: ruff check, git diff, git status."
    )
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
class WriteFileInput(ToolInput):
    """Input for write_file tool"""
    description: ClassVar[str] = (
        "Write the complete content of one file and return a unified diff. Use "
        "this for planned file creation or when a change cannot be expressed as "
        "a small unique replacement. Prefer edit_file for existing files. The "
        "write is validated and reverted if invalid."
    )
    path: str = field(
        metadata={
            "description": (
                "Workspace-relative path of the planned file to create or rewrite."
            )
        }
    )
    content: str = field(
        metadata={
            "description": (
                "Complete UTF-8 file content. This replaces the entire file, "
                "so include all content that must remain."
            )
        }
    )
    preview: bool = field(
        default=False,
        metadata={"model_exposed": False},
    )

    def __post_init__(self):
        if not isinstance(self.path, str):
            raise TypeError(f"path must be str, not {type(self.path).__name__}")
        if not isinstance(self.content, str):
            raise TypeError(f"content must be str, not {type(self.content).__name__}")
        if not isinstance(self.preview, bool):
            raise TypeError(f"preview must be bool, not {type(self.preview).__name__}")


@dataclass
class WriteScratchTestInput(ToolInput):
    """Input for the isolated scratch-test writer."""

    description: ClassVar[str] = (
        "Write a temporary pytest file under .patchpilot_checks for behavior "
        "that existing repository tests do not cover. Scratch tests are "
        "supplemental verification only and are excluded from the final patch."
    )
    name: str = field(
        metadata={
            "description": (
                "Short lowercase identifier using letters, numbers, and underscores."
            )
        }
    )
    content: str = field(
        metadata={"description": "Complete Python pytest source for the scratch test."}
    )

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError(f"name must be str, not {type(self.name).__name__}")
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", self.name):
            raise ValueError(
                "name must start with a lowercase letter and contain only "
                "lowercase letters, numbers, and underscores"
            )
        if not isinstance(self.content, str):
            raise TypeError(f"content must be str, not {type(self.content).__name__}")
        if not self.content.strip():
            raise ValueError("content must not be empty")


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

    for model_field in fields(input_class):
        field_name = model_field.name
        field_type = type_hints.get(field_name, model_field.type)

        # Skip ClassVar fields (like description)
        if field_name == "description":
            continue
        if model_field.metadata.get("model_exposed") is False:
            continue

        # Map Python types to JSON Schema types
        json_type, is_nullable = _python_type_to_json_type(field_type)
        property_schema = {"type": json_type}

        field_description = model_field.metadata.get("description")
        if field_description:
            property_schema["description"] = field_description

        # Mark as nullable if it's an Optional type
        if is_nullable:
            property_schema["type"] = [json_type, "null"]

        properties[field_name] = property_schema

        # Add default value if present
        if model_field.default is not MISSING:
            properties[field_name]["default"] = model_field.default
        elif model_field.default_factory is not MISSING:
            properties[field_name]["default"] = (
                model_field.default_factory()
            )
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

    # Maximum output size limits
    MAX_SEARCH_OUTPUT = 100_000  # characters
    MAX_FILE_LINES = 300
    COMMAND_TIMEOUT = 60

    def __init__(
        self,
        workspace: Workspace,
        policy_set: PolicySet | None = None,
        command_runner: CommandRunnerProtocol | None = None,
    ):
        self.workspace = workspace
        self.policy_set = policy_set or get_builtin_policies()
        self.command_runner = command_runner
        self._allowed_new_test_files: set[str] = set()
        # Dynamic tool registration storage
        self._tool_definitions: dict[str, ToolDefinition] = {}
        self._tool_handlers: dict[str, Any] = {}

        # Initialize policy evaluator if policy_set is provided
        self.policy_evaluator = PolicyEvaluator(self.policy_set)

        # Register default tools
        self._register_default_tools()

    def update_command_runner(
        self,
        command_runner: CommandRunnerProtocol | None,
    ) -> None:
        """Update the isolated runner used by the command tool."""
        self.command_runner = command_runner

    def update_policy_set(self, policy_set: PolicySet) -> None:
        """Update the policy set used for permission checking.

        Args:
            policy_set: New PolicySet to use for permission evaluation
        """
        self.policy_set = policy_set
        self.policy_evaluator = PolicyEvaluator(policy_set) if policy_set else None

    def _register_default_tools(self) -> None:
        """Register the compact set of tools exposed to the coding model."""
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
            name="write_file",
            input_class=WriteFileInput,
            handler=self.write_file,
        )
        self.register_tool(
            name="write_scratch_test",
            input_class=WriteScratchTestInput,
            handler=self.write_scratch_test,
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

    def configure_test_writes(self, allowed_new_files: set[str]) -> None:
        """Set the plan-authorized new test files for the current run."""
        self._allowed_new_test_files = {
            str(PurePosixPath(path))
            for path in allowed_new_files
            if not self.workspace.resolve(path).exists()
        }

    @staticmethod
    def _is_test_file(path: str) -> bool:
        """Return whether a workspace-relative path is a Python test file."""
        normalized = PurePosixPath(path)
        return "tests" in normalized.parts or (
            normalized.name.startswith("test_") and normalized.suffix == ".py"
        )

    def _assert_test_write_allowed(self, path: str) -> None:
        """Protect existing tests while allowing approved new test artifacts."""
        normalized = str(PurePosixPath(path))
        if (
            self._is_test_file(normalized)
            and normalized not in self._allowed_new_test_files
        ):
            raise PermissionError(
                "Modifying test files is not allowed: existing tests are immutable. "
                "Only a new test file explicitly "
                f"authorized by the approved plan may be written: {normalized}"
            )

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
                content=(
                    "EDIT_REJECTED: old_text is empty or whitespace-only; it "
                    "must contain existing file text. "
                    "Re-read the file with raw=True and choose a unique anchor. "
                    "Use write_file only when the plan creates a file or a "
                    "small replacement cannot express the change."
                ),
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
            # Additional policy check if policy_evaluator is available
            if self.policy_evaluator:
                self.policy_evaluator.assert_write_allowed(input_data.path)
            self._assert_test_write_allowed(input_data.path)
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
                           "Re-read the relevant block with raw=True and use a larger unique old_text value."
                )

            replacement_text = input_data.new_text
            indentation_repaired = False
            new_content = original_content.replace(
                matched_text,
                replacement_text,
                1,
            )

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
                # Exact replacement preserves indentation before the first
                # matched line, but not before continuation lines supplied by
                # the model. Retry once with conservative block indentation.
                repaired_text, indentation_repaired = (
                    self._prepare_replacement_text(
                        original_content=original_content,
                        matched_text=matched_text,
                        new_text=input_data.new_text,
                        is_python=resolved_path.suffix == ".py",
                    )
                )
                if indentation_repaired:
                    repaired_content = original_content.replace(
                        matched_text,
                        repaired_text,
                        1,
                    )
                    with open(resolved_path, "w", encoding="utf-8") as f:
                        f.write(repaired_content)
                    validation_passed, validation_errors = (
                        run_intermediate_validation(
                            resolved_path,
                            display_path=input_data.path,
                        )
                    )
                    if validation_passed:
                        new_content = repaired_content
                        new_lines = new_content.splitlines(keepends=True)
                        diff = difflib.unified_diff(
                            original_lines,
                            new_lines,
                            fromfile=input_data.path,
                            tofile=input_data.path,
                            lineterm="",
                        )
                        diff_text = "".join(diff)

            if not validation_passed:
                # Revert the change if validation fails
                with open(resolved_path, "w", encoding="utf-8") as f:
                    f.write(original_content)
                return ToolResult(
                    ok=False,
                    content=self._format_validation_rejection(
                        file_content=original_content,
                        matched_text=matched_text,
                        file_path=input_data.path,
                        validation_errors=validation_errors,
                    ),
                )

            result_content = diff_text or "(no diff)"
            if indentation_repaired:
                result_content = (
                    "Applied with automatic block-indentation repair.\n"
                    + result_content
                )
            return ToolResult(ok=True, content=result_content)

        except UnicodeDecodeError:
            return ToolResult(ok=False, content="File is not valid UTF-8 text")
        except OSError as e:
            return ToolResult(ok=False, content=f"Edit failed: {e}")

    @staticmethod
    def _prepare_replacement_text(
        original_content: str,
        matched_text: str,
        new_text: str,
        is_python: bool,
    ) -> tuple[str, bool]:
        """Convert relative continuation indentation to file indentation.

        The first replacement line inherits the indentation before the match.
        Every later non-blank line needs the same base indentation in addition
        to any relative indentation supplied by the model.
        """
        if not is_python or "\n" not in new_text:
            return new_text, False

        match_start = original_content.index(matched_text)
        line_start = original_content.rfind("\n", 0, match_start) + 1
        leading_text = original_content[line_start:match_start]
        if not leading_text or not leading_text.isspace():
            return new_text, False

        lines = new_text.splitlines(keepends=True)
        repaired = False
        for index in range(1, len(lines)):
            content = lines[index].rstrip("\r\n")
            line_ending = lines[index][len(content):]
            if content.strip():
                lines[index] = leading_text + content + line_ending
                repaired = True

        return "".join(lines), repaired

    @staticmethod
    def _format_validation_rejection(
        file_content: str,
        matched_text: str,
        file_path: str,
        validation_errors: list[str],
    ) -> str:
        """Return concise recovery context after an invalid edit is reverted."""
        lines = file_content.splitlines()
        match_start = file_content.index(matched_text)
        match_line = file_content.count("\n", 0, match_start)
        context_start = max(0, match_line - 2)
        context_end = min(len(lines), match_line + 4)
        context = "\n".join(
            f"{line_number}: {lines[line_number - 1]}"
            for line_number in range(context_start + 1, context_end + 1)
        )
        errors = "\n".join(
            f"- {error}"
            for error in validation_errors
        )
        return (
            "EDIT_REJECTED: validation failed and the file was restored.\n"
            f"File: {file_path}\n"
            f"Validation errors:\n{errors}\n"
            "Current code near the rejected edit:\n"
            f"{context}\n"
            "Next step: use this current code as the exact replacement anchor "
            "or re-read a slightly larger block with raw=True."
        )

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
            error_msg += "SOLUTION: Re-read the file and choose a unique existing anchor.\n"
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
                error_msg += "Tip: Replace a larger unique block when a small anchor is unavailable.\n"

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
            # Additional policy check if policy_evaluator is available
            if self.policy_evaluator:
                self.policy_evaluator.assert_write_allowed(input_data.path)
            self._assert_test_write_allowed(input_data.path)
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
            # Additional policy check if policy_evaluator is available
            if self.policy_evaluator:
                self.policy_evaluator.assert_write_allowed(input_data.path)
            self._assert_test_write_allowed(input_data.path)
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

    def write_file(self, arguments: dict[str, Any]) -> ToolResult:
        """
        Write complete file content. Creates the file if it doesn't exist.

        Args:
            arguments: Dict with 'path', 'content', and optional 'preview' (default False)

        Returns:
            ToolResult with unified diff (for modifications) or creation message (for new files).
            If preview=True, shows change without applying.
        """
        try:
            input_data = WriteFileInput(**arguments)
        except (TypeError, ValueError) as e:
            return ToolResult(ok=False, content=f"Invalid input: {e}")

        try:
            resolved_path = self.workspace.assert_write_allowed(input_data.path)
            # Additional policy check if policy_evaluator is available
            if self.policy_evaluator:
                self.policy_evaluator.assert_write_allowed(input_data.path)
            self._assert_test_write_allowed(input_data.path)
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

    def write_scratch_test(self, arguments: dict[str, Any]) -> ToolResult:
        """Write a constrained scratch test that is excluded from the patch."""
        try:
            input_data = WriteScratchTestInput(**arguments)
        except (TypeError, ValueError) as error:
            return ToolResult(ok=False, content=f"Invalid input: {error}")

        if len(input_data.content) > 50_000:
            return ToolResult(ok=False, content="Scratch test exceeds 50000 characters")

        try:
            tree = ast.parse(input_data.content)
        except SyntaxError as error:
            return ToolResult(ok=False, content=f"Scratch test syntax error: {error}")

        forbidden_imports = {
            "ctypes",
            "http",
            "os",
            "requests",
            "shutil",
            "socket",
            "subprocess",
            "urllib",
        }
        forbidden_calls = {
            "__import__",
            "compile",
            "eval",
            "exec",
            "open",
        }
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                modules = (
                    [alias.name for alias in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                )
                if any(module.split(".", 1)[0] in forbidden_imports for module in modules):
                    return ToolResult(
                        ok=False,
                        content="Scratch tests may not import process, network, or filesystem modules",
                    )
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in forbidden_calls
            ):
                return ToolResult(
                    ok=False,
                    content=f"Scratch tests may not call {node.func.id}",
                )

        relative_path = f".patchpilot_checks/test_{input_data.name}.py"
        try:
            resolved_path = self.workspace.assert_write_allowed(relative_path)
            if self.policy_evaluator:
                self.policy_evaluator.assert_write_allowed(relative_path)
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            resolved_path.write_text(input_data.content, encoding="utf-8")
        except (OSError, PermissionError, ValueError) as error:
            return ToolResult(ok=False, content=f"Scratch test write failed: {error}")

        return ToolResult(
            ok=True,
            content=(
                f"Created isolated scratch test: {relative_path}\n"
                f"Run: python -m pytest {relative_path} -q -p no:cacheprovider"
            ),
        )

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

    @staticmethod
    def _canonicalize_command(args: list[str]) -> list[str]:
        """Normalize supported commands to their deterministic invocation."""
        if args and args[0] == "pytest":
            return ["python", "-m", "pytest", *args[1:]]
        if args[:2] == ["ruff", "check"] and ".patchpilot_checks" not in args:
            return [
                *args[:2],
                "--extend-exclude",
                ".patchpilot_checks",
                *args[2:],
            ]
        return args

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

        # Enforce the narrow command grammar before evaluating extra policies.
        base_command = args[0]
        allowed_commands = {"pytest", "python", "ruff", "git"}
        if base_command not in allowed_commands:
            return ToolResult(
                ok=False,
                content=(
                    f"Command '{base_command}' is not allowed. Allowed: "
                    f"{', '.join(sorted(allowed_commands))}"
                ),
            )

        if (
            base_command == "python"
            and args[:3] != ["python", "-m", "pytest"]
        ):
            return ToolResult(
                ok=False,
                content="Only 'python -m pytest' is allowed for python command",
            )

        if base_command == "git" and (
            len(args) < 2 or args[1] not in {"diff", "status"}
        ):
            return ToolResult(
                ok=False,
                content="Only 'git diff' and 'git status' are allowed for git command",
            )

        if base_command == "ruff" and args[:2] != ["ruff", "check"]:
            return ToolResult(
                ok=False,
                content="Only 'ruff check' is allowed for ruff command",
            )

        if self.policy_evaluator:
            try:
                self.policy_evaluator.assert_command_allowed(input_data.command)
            except PermissionError as e:
                return ToolResult(ok=False, content=str(e))

        execution_args = self._canonicalize_command(args)

        # Run command with DockerSandbox if available, otherwise subprocess
        if self.command_runner is not None:
            try:
                result = self.command_runner.run(
                    command=shlex.join(execution_args),
                    timeout_seconds=self.COMMAND_TIMEOUT,
                )

                if (
                    execution_args[:3] == ["python", "-m", "pytest"]
                    and result.exit_code == 2
                    and not getattr(result, "timed_out", False)
                ):
                    logger.info(
                        "Retrying Pytest once after exit code 2 before "
                        "returning control to the model"
                    )
                    result = self.command_runner.run(
                        command=shlex.join(execution_args),
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
                            execution_args,
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
                    execution_args,
                    cwd=self.workspace.root,
                    capture_output=True,
                    text=True,
                    timeout=self.COMMAND_TIMEOUT,
                    check=False,
                )

                if (
                    execution_args[:3] == ["python", "-m", "pytest"]
                    and result.returncode == 2
                ):
                    logger.info(
                        "Retrying Pytest once after exit code 2 before "
                        "returning control to the model"
                    )
                    result = subprocess.run(
                        execution_args,
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
                            execution_args,
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
