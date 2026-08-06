import difflib
import shlex
import subprocess
from dataclasses import MISSING, dataclass, fields
from typing import Any, ClassVar, Union, get_type_hints

from patchpilot.models import ToolResult
from patchpilot.workspace import Workspace


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
    description: ClassVar[str] = "Read file content with line numbers. Supports optional line range for partial reads."
    path: str
    start_line: int = 1
    end_line: int | None = None

    def __post_init__(self):
        if not isinstance(self.path, str):
            raise TypeError(f"path must be str, not {type(self.path).__name__}")
        if not isinstance(self.start_line, int):
            raise TypeError(f"start_line must be int, not {type(self.start_line).__name__}")
        if self.end_line is not None and not isinstance(self.end_line, int):
            raise TypeError(f"end_line must be int or None, not {type(self.end_line).__name__}")


@dataclass
class EditFileInput(ToolInput):
    """Input for edit_file tool"""
    description: ClassVar[str] = "Edit file using exact text replacement. Replaces the first occurrence of old_text with new_text and returns a unified diff."
    path: str
    old_text: str
    new_text: str

    def __post_init__(self):
        if not isinstance(self.path, str):
            raise TypeError(f"path must be str, not {type(self.path).__name__}")
        if not isinstance(self.old_text, str):
            raise TypeError(f"old_text must be str, not {type(self.old_text).__name__}")
        if not isinstance(self.new_text, str):
            raise TypeError(f"new_text must be str, not {type(self.new_text).__name__}")


@dataclass
class RunCommandInput(ToolInput):
    """Input for run_command tool"""
    description: ClassVar[str] = "Run allowed commands in the workspace. Only allows: pytest, python -m pytest, ruff check, git diff, git status."
    command: str

    def __post_init__(self):
        if not isinstance(self.command, str):
            raise TypeError(f"command must be str, not {type(self.command).__name__}")


@dataclass
class ToolDefinition:
    """Structured definition of a tool with schema"""
    name: str
    description: str
    input_schema: dict[str, Any]


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

    def __init__(self, workspace: Workspace):
        self.workspace = workspace
        # Dynamic tool registration storage
        self._tool_definitions: dict[str, ToolDefinition] = {}
        self._tool_handlers: dict[str, Any] = {}

        # Register default tools
        self._register_default_tools()

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
            name="run_command",
            input_class=RunCommandInput,
            handler=self.run_command,
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
        return handler(arguments)

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
            resolved_path = self.workspace.resolve(input_data.path)
        except ValueError as e:
            return ToolResult(ok=False, content=f"Path error: {e}")

        # Run ripgrep with subprocess
        try:
            args = ["rg", "-n", input_data.query, str(resolved_path)]
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
        Read file content with line numbers.
        
        Args:
            arguments: Dict with 'path', optional 'start_line' (default 1), 
                      and optional 'end_line' (default None)
        
        Returns:
            ToolResult with file content prefixed with line numbers
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

            # Format with line numbers
            selected_lines = lines[start_idx:end_idx]
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
        Edit file using exact text replacement.
        
        Args:
            arguments: Dict with 'path', 'old_text', and 'new_text'
        
        Returns:
            ToolResult with unified diff or error message
        """
        try:
            input_data = EditFileInput(**arguments)
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
                original_content = f.read()

            # Verify old_text appears exactly once
            count = original_content.count(input_data.old_text)
            if count == 0:
                return ToolResult(
                    ok=False,
                    content="old_text not found in file. Please re-read the file and verify the exact text."
                )
            if count > 1:
                return ToolResult(
                    ok=False,
                    content=f"old_text appears {count} times in file. Please provide more specific context."
                )

            # Perform replacement
            new_content = original_content.replace(input_data.old_text, input_data.new_text, 1)

            # Generate unified diff
            original_lines = original_content.splitlines(keepends=True)
            new_lines = new_content.splitlines(keepends=True)
            diff = difflib.unified_diff(
                original_lines,
                new_lines,
                fromfile=str(resolved_path),
                tofile=str(resolved_path),
                lineterm=""
            )
            diff_text = "".join(diff)

            # Write new content
            with open(resolved_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            return ToolResult(ok=True, content=diff_text or "(no diff)")

        except UnicodeDecodeError:
            return ToolResult(ok=False, content="File is not valid UTF-8 text")
        except OSError as e:
            return ToolResult(ok=False, content=f"Edit failed: {e}")

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

        # Run command with subprocess
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
                    content=f"exit_code={result.returncode}\n{output}"
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
