"""Typed, read-only tool primitives for RootTrace RCA agents."""

from __future__ import annotations

from dataclasses import MISSING, dataclass, fields
from types import UnionType
from typing import Any, ClassVar, Union, get_args, get_origin, get_type_hints

from roottrace.runtime.workspace import Workspace
from roottrace.tools.schema import ToolFailureType, ToolResult


@dataclass
class ToolInput:
    """Base class for model-facing tool inputs."""

    description: ClassVar[str] = ""


@dataclass
class SearchCodeInput(ToolInput):
    """Input for bounded repository text search."""

    description: ClassVar[str] = (
        "Search for code patterns using ripgrep and return matching lines "
        "with line numbers."
    )
    query: str
    path: str = "."

    def __post_init__(self) -> None:
        if not isinstance(self.query, str):
            raise TypeError(f"query must be str, not {type(self.query).__name__}")
        if not isinstance(self.path, str):
            raise TypeError(f"path must be str, not {type(self.path).__name__}")


@dataclass
class ReadFileInput(ToolInput):
    """Input for bounded reads of one repository file."""

    description: ClassVar[str] = (
        "Read one repository file using a relative path. Absolute paths and "
        "directories are rejected. By default, returned lines include line "
        "numbers; set raw=true to return the original text."
    )
    path: str
    start_line: int = 1
    end_line: int | None = None
    raw: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.path, str):
            raise TypeError(f"path must be str, not {type(self.path).__name__}")
        if not isinstance(self.start_line, int):
            raise TypeError(
                f"start_line must be int, not {type(self.start_line).__name__}"
            )
        if self.end_line is not None and not isinstance(self.end_line, int):
            raise TypeError(
                f"end_line must be int or None, not {type(self.end_line).__name__}"
            )
        if not isinstance(self.raw, bool):
            raise TypeError(f"raw must be bool, not {type(self.raw).__name__}")


@dataclass
class ToolDefinition:
    """A model-facing tool definition and its JSON input schema."""

    name: str
    description: str
    input_schema: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


def _python_type_to_json_type(python_type: Any) -> tuple[str, bool]:
    """Map a supported Python annotation to JSON Schema type metadata."""

    origin = get_origin(python_type)
    args = get_args(python_type)
    if origin in (Union, UnionType):
        nullable = type(None) in args
        for argument in args:
            if argument is not type(None):
                json_type, _ = _python_type_to_json_type(argument)
                return json_type, nullable
        return "string", nullable
    if origin is list:
        return "array", False
    mapping = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        type(None): "null",
    }
    return mapping.get(python_type, "string"), python_type is type(None)


def generate_json_schema(input_class: type[ToolInput]) -> dict[str, Any]:
    """Generate the bounded JSON Schema used for provider tool calls."""

    type_hints = get_type_hints(input_class)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for model_field in fields(input_class):
        if model_field.name == "description":
            continue
        field_type = type_hints.get(model_field.name, model_field.type)
        json_type, nullable = _python_type_to_json_type(field_type)
        property_schema: dict[str, Any] = {
            "type": [json_type, "null"] if nullable else json_type
        }
        field_description = model_field.metadata.get("description")
        if field_description:
            property_schema["description"] = field_description
        if model_field.default is not MISSING:
            property_schema["default"] = model_field.default
        elif model_field.default_factory is not MISSING:
            property_schema["default"] = model_field.default_factory()
        elif not nullable:
            required.append(model_field.name)
        properties[model_field.name] = property_schema
    return {"type": "object", "properties": properties, "required": required}


class ToolRegistry:
    """Minimal registry for RootTrace's read-only evidence tools."""

    MAX_FILE_LINES = 300
    COMMAND_TIMEOUT = 60

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self._tool_definitions: dict[str, ToolDefinition] = {}
        self._tool_handlers: dict[str, Any] = {}
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        self.register_tool("read_file", ReadFileInput, self.read_file)

    def register_tool(
        self,
        name: str,
        input_class: type[ToolInput],
        handler: Any,
    ) -> None:
        self._tool_definitions[name] = ToolDefinition(
            name=name,
            description=getattr(input_class, "description", ""),
            input_schema=generate_json_schema(input_class),
        )
        self._tool_handlers[name] = handler

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": definition.name,
                    "description": definition.description,
                    "parameters": definition.input_schema,
                },
            }
            for definition in self._tool_definitions.values()
        ]

    def get_tool_schema(self, tool_name: str) -> ToolDefinition | None:
        return self._tool_definitions.get(tool_name)

    def get_available_tools(self) -> list[str]:
        return list(self._tool_definitions)

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
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
        root = str(self.workspace.root)
        return content.replace(f"{root}/", "").replace(root, ".")

    def read_file(self, arguments: dict[str, Any]) -> ToolResult:
        """Read a UTF-8 file while enforcing workspace and line bounds."""

        try:
            input_data = ReadFileInput(**arguments)
        except (TypeError, ValueError) as exc:
            return ToolResult(ok=False, content=f"Invalid input: {exc}")
        try:
            path = self.workspace.assert_read_allowed(input_data.path)
        except (ValueError, PermissionError) as exc:
            return ToolResult(ok=False, content=f"Path error: {exc}")
        if not path.exists():
            return ToolResult(ok=False, content=f"File not found: {input_data.path}")
        if not path.is_file():
            return ToolResult(ok=False, content=f"Not a file: {input_data.path}")
        try:
            lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        except UnicodeDecodeError:
            return ToolResult(ok=False, content="File is not valid UTF-8 text")
        except OSError as exc:
            return ToolResult(ok=False, content=f"Read failed: {exc}")

        start_index = max(0, input_data.start_line - 1)
        end_index = len(lines) if input_data.end_line is None else min(
            len(lines), input_data.end_line
        )
        if end_index - start_index > self.MAX_FILE_LINES:
            return ToolResult(
                ok=False,
                content=(
                    f"Request exceeds maximum line limit of {self.MAX_FILE_LINES}"
                ),
            )
        selected = lines[start_index:end_index]
        if input_data.raw:
            return ToolResult(ok=True, content="".join(selected))
        numbered = [
            f"{line_number}: {line.rstrip()}"
            for line_number, line in enumerate(selected, start=start_index + 1)
        ]
        return ToolResult(ok=True, content="\n".join(numbered))
