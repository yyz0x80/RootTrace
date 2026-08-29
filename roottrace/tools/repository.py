"""RCA-safe read-only tool registry for RootTrace evidence gathering.

This registry is the read-only tool surface used by the three evidence
specialists. It provides typed inputs, bounded observations, AST-based symbol
inspection, and validated repository access.

The registry exposes exactly seven tools:

- ``search_code``, ``read_file``: bounded repository inspection.
- ``inspect_symbols``: Python ``ast`` only, no LSP or full Tree-sitter.
- ``git_history``, ``git_blame``, ``git_show``: allowlisted argv,
  ``subprocess.run(..., shell=False)``, bounded output, finite timeouts.
- ``read_external_log``: bounded read of an external CI/stack log.

Write-capable tools are never registered, and their handler methods raise so
RCA agents cannot mutate the target repository.
"""

from __future__ import annotations

import ast
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from roottrace.runtime.paths import validate_relative_path
from roottrace.runtime.workspace import Workspace
from roottrace.tools.registry import (
    ReadFileInput,
    SearchCodeInput,
    ToolInput,
    ToolRegistry,
)
from roottrace.tools.schema import ToolFailureType

MAX_SEARCH_QUERY_CHARS = 200
MAX_SEARCH_OUTPUT_CHARS = 100_000
MAX_SYMBOLS = 200
MAX_SYMBOL_NAME_CHARS = 120
MAX_SYMBOL_OUTPUT_CHARS = 100_000
MAX_GIT_OUTPUT_CHARS = 100_000
MAX_GIT_LOG_COUNT = 200
MAX_GIT_PATTERN_CHARS = 200
MAX_BLAME_LINES = 200
MAX_EXTERNAL_LOG_CHARS = 200_000
MAX_EXTERNAL_LOG_LINES = 5_000
GIT_TIMEOUT_SECONDS = 30

_CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x1f\x7f]")
_REVISION_PATTERN = re.compile(
    r"^(?:[0-9a-fA-F]{4,64}|HEAD|[A-Za-z0-9][A-Za-z0-9._/-]{0,127})$"
)


def _reject_control_chars(value: str, label: str) -> None:
    if _CONTROL_CHAR_PATTERN.search(value):
        raise ValueError(f"{label} must not contain control characters")


def _bound_text(value: str, limit: int, marker: str = "\n... (output truncated)") -> tuple[str, bool]:
    if len(value) <= limit:
        return value, False
    return value[:limit] + marker, True


@dataclass
class RcaToolResult:
    """Bounded, auditable result returned by an RCA tool invocation."""

    ok: bool
    content: str
    tool: str = ""
    command: str = ""
    duration_seconds: float = 0.0
    truncated: bool = False
    failure_type: ToolFailureType | None = None


@dataclass
class InspectSymbolsInput(ToolInput):
    """Input for the inspect_symbols tool."""

    description: ClassVar[str] = (
        "List Python symbols (classes, functions, async functions) defined in a "
        "repository file using AST, with line ranges and signatures. Optionally "
        "filter by symbol name substring."
    )
    path: str
    name: str | None = None
    max_symbols: int = 100

    def __post_init__(self) -> None:
        if not isinstance(self.path, str):
            raise TypeError(f"path must be str, not {type(self.path).__name__}")
        if self.name is not None and not isinstance(self.name, str):
            raise TypeError(f"name must be str or None, not {type(self.name).__name__}")
        if not isinstance(self.max_symbols, int):
            raise TypeError(f"max_symbols must be int, not {type(self.max_symbols).__name__}")
        if not 1 <= self.max_symbols <= MAX_SYMBOLS:
            raise ValueError(f"max_symbols must be between 1 and {MAX_SYMBOLS}")
        if self.name is not None:
            if not self.name.strip() or len(self.name) > MAX_SYMBOL_NAME_CHARS:
                raise ValueError("name must be non-empty and bounded")
            _reject_control_chars(self.name, "name")


@dataclass
class GitHistoryInput(ToolInput):
    """Input for the git_history tool."""

    description: ClassVar[str] = (
        "Show bounded git commit history. Optionally scope to one repository "
        "file, filter by commit message grep, or pickaxe-search a code change "
        "with -S."
    )
    path: str | None = None
    max_count: int = 20
    grep: str | None = None
    query: str | None = None

    def __post_init__(self) -> None:
        if self.path is not None and not isinstance(self.path, str):
            raise TypeError(f"path must be str or None, not {type(self.path).__name__}")
        if not isinstance(self.max_count, int):
            raise TypeError(f"max_count must be int, not {type(self.max_count).__name__}")
        if not 1 <= self.max_count <= MAX_GIT_LOG_COUNT:
            raise ValueError(f"max_count must be between 1 and {MAX_GIT_LOG_COUNT}")
        for label, value in (("grep", self.grep), ("query", self.query)):
            if value is None:
                continue
            if not isinstance(value, str):
                raise TypeError(f"{label} must be str or None")
            if not value.strip() or len(value) > MAX_GIT_PATTERN_CHARS:
                raise ValueError(f"{label} must be non-empty and bounded")
            _reject_control_chars(value, label)


@dataclass
class GitBlameInput(ToolInput):
    """Input for the git_blame tool."""

    description: ClassVar[str] = (
        "Show git blame for a repository file, with an optional 1-based line "
        "range of at most 200 lines."
    )
    path: str
    start_line: int | None = None
    end_line: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.path, str):
            raise TypeError(f"path must be str, not {type(self.path).__name__}")
        for label, value in (("start_line", self.start_line), ("end_line", self.end_line)):
            if value is not None and not isinstance(value, int):
                raise TypeError(f"{label} must be int or None")
        if self.start_line is not None and self.start_line < 1:
            raise ValueError("start_line must be at least 1")
        if self.end_line is not None and self.end_line < 1:
            raise ValueError("end_line must be at least 1")
        start = self.start_line or 1
        end = self.end_line or self.start_line or 1
        if start > end:
            raise ValueError("start_line must not exceed end_line")
        if end - start + 1 > MAX_BLAME_LINES:
            raise ValueError(f"blame line range must not exceed {MAX_BLAME_LINES} lines")


@dataclass
class GitShowInput(ToolInput):
    """Input for the git_show tool."""

    description: ClassVar[str] = (
        "Show a git commit (hex SHA, HEAD, or ref name), optionally restricted "
        "to one repository file at that revision, with optional --stat."
    )
    revision: str
    path: str | None = None
    stat: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.revision, str):
            raise TypeError(f"revision must be str, not {type(self.revision).__name__}")
        if self.path is not None and not isinstance(self.path, str):
            raise TypeError(f"path must be str or None, not {type(self.path).__name__}")
        if not isinstance(self.stat, bool):
            raise TypeError(f"stat must be bool, not {type(self.stat).__name__}")
        if not self.revision.strip() or len(self.revision) > 200:
            raise ValueError("revision must be non-empty and bounded")
        _reject_control_chars(self.revision, "revision")
        if not _REVISION_PATTERN.fullmatch(self.revision):
            raise ValueError("revision must be a hex SHA, HEAD, or plain ref name")


@dataclass
class ReadExternalLogInput(ToolInput):
    """Input for the read_external_log tool."""

    description: ClassVar[str] = (
        "Read a bounded excerpt of an external log file (stack trace or CI log) "
        "relative to the configured external log root."
    )
    path: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, str):
            raise TypeError(f"path must be str, not {type(self.path).__name__}")


def _collect_symbols(tree: ast.AST, name_filter: str | None) -> list[str]:
    """Return bounded, deterministic symbol lines from a parsed Python file."""
    parent_map: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parent_map[child] = parent

    symbols: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        qualified = node.name
        parent = parent_map.get(node)
        while parent is not None and isinstance(
            parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            qualified = f"{parent.name}.{qualified}"
            parent = parent_map.get(parent)
        if name_filter is not None and name_filter not in qualified:
            continue
        if isinstance(node, ast.ClassDef):
            kind = "class"
            bases = ", ".join(ast.unparse(base) for base in node.bases)
            detail = f"bases: ({bases})" if bases else ""
        else:
            kind = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            raw_signature = ast.unparse(node.args)
            detail = f"({raw_signature})"
        line = (
            f"{node.lineno:>6}-{node.end_lineno:<6} "
            f"{kind} {qualified}{detail}"
        )
        symbols.append((node.lineno, _bound_text(line, MAX_SYMBOL_NAME_CHARS)[0]))

    symbols.sort(key=lambda item: item[0])
    return [line for _, line in symbols]


class RcaToolRegistry(ToolRegistry):
    """Read-only RCA tool registry; the only tools RCA Agents may call.

    Uses ``Workspace`` path/secret enforcement and typed schema generation.
    Git-backed tools build allowlisted argv lists and run with
    ``subprocess.run(..., shell=False)`` against the original repository
    (read-only by construction).
    """

    def __init__(
        self,
        workspace: Workspace,
        external_root: str | Path | None = None,
    ) -> None:
        self.external_root = (
            Path(external_root).resolve() if external_root is not None else None
        )
        super().__init__(workspace=workspace)

    def _register_default_tools(self) -> None:
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
            name="inspect_symbols",
            input_class=InspectSymbolsInput,
            handler=self.inspect_symbols,
        )
        self.register_tool(
            name="git_history",
            input_class=GitHistoryInput,
            handler=self.git_history,
        )
        self.register_tool(
            name="git_blame",
            input_class=GitBlameInput,
            handler=self.git_blame,
        )
        self.register_tool(
            name="git_show",
            input_class=GitShowInput,
            handler=self.git_show,
        )
        self.register_tool(
            name="read_external_log",
            input_class=ReadExternalLogInput,
            handler=self.read_external_log,
        )

    def execute(self, name: str, arguments: dict[str, Any]) -> RcaToolResult:
        """Execute a registered RCA tool and return a bounded result."""
        handler = self._tool_handlers.get(name)
        if handler is None:
            raise KeyError(f"Tool not found: {name}")
        result = handler(arguments)
        failure_type = result.failure_type
        if failure_type is None and not result.ok:
            failure_type = ToolFailureType.TOOL_FAILURE
        return RcaToolResult(
            ok=result.ok,
            content=self.sanitize_workspace_paths(result.content),
            tool=name,
            command=getattr(result, "command", ""),
            duration_seconds=getattr(result, "duration_seconds", 0.0),
            truncated=getattr(result, "truncated", False),
            failure_type=failure_type,
        )

    # Write-capable tools are deliberately unavailable to RCA agents.

    def edit_file(self, arguments: dict[str, Any]) -> RcaToolResult:
        raise PermissionError("edit_file is not available to RCA agents")

    def write_file(self, arguments: dict[str, Any]) -> RcaToolResult:
        raise PermissionError("write_file is not available to RCA agents")

    def write_scratch_test(self, arguments: dict[str, Any]) -> RcaToolResult:
        raise PermissionError("write_scratch_test is not available to RCA agents")

    def run_command(self, arguments: dict[str, Any]) -> RcaToolResult:
        raise PermissionError("run_command is not available to RCA agents")

    # -- file tools ------------------------------------------------------------

    def search_code(self, arguments: dict[str, Any]) -> RcaToolResult:
        """Harden the inherited ripgrep search: bounded query, ``--`` separator."""
        try:
            input_data = SearchCodeInput(**arguments)
        except (TypeError, ValueError) as exc:
            return RcaToolResult(ok=False, content=f"Invalid input: {exc}", tool="search_code")
        if not input_data.query or len(input_data.query) > MAX_SEARCH_QUERY_CHARS:
            return RcaToolResult(
                ok=False,
                content=f"query must be non-empty and at most {MAX_SEARCH_QUERY_CHARS} chars",
                tool="search_code",
            )
        if "\x00" in input_data.query:
            return RcaToolResult(ok=False, content="query must not contain NUL", tool="search_code")
        try:
            path = "." if input_data.path == "." else validate_relative_path(input_data.path)
            self.workspace.resolve(path)
        except ValueError as exc:
            return RcaToolResult(ok=False, content=f"Path error: {exc}", tool="search_code")

        started = time.monotonic()
        argv = ["rg", "-n", "--", input_data.query, path]
        try:
            result = subprocess.run(
                argv,
                cwd=self.workspace.root,
                capture_output=True,
                text=True,
                timeout=self.COMMAND_TIMEOUT,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return RcaToolResult(
                ok=False,
                content="Search timed out",
                tool="search_code",
                command=" ".join(argv),
                duration_seconds=time.monotonic() - started,
            )
        except FileNotFoundError:
            return RcaToolResult(
                ok=False,
                content="ripgrep (rg) not found in PATH",
                tool="search_code",
                command=" ".join(argv),
                duration_seconds=time.monotonic() - started,
            )
        except OSError as exc:
            return RcaToolResult(
                ok=False,
                content=f"Search failed: {exc}",
                tool="search_code",
                command=" ".join(argv),
                duration_seconds=time.monotonic() - started,
            )
        if result.returncode not in (0, 1):
            detail = (result.stderr or result.stdout).strip()
            if not detail:
                detail = f"exit code {result.returncode}"
            return RcaToolResult(
                ok=False,
                content=f"Search failed: {detail[:2_000]}",
                tool="search_code",
                command=" ".join(argv),
                duration_seconds=time.monotonic() - started,
                failure_type=ToolFailureType.TOOL_FAILURE,
            )
        output, truncated = _bound_text(result.stdout, MAX_SEARCH_OUTPUT_CHARS)
        return RcaToolResult(
            ok=True,
            content=output,
            tool="search_code",
            command=" ".join(argv),
            duration_seconds=time.monotonic() - started,
            truncated=truncated,
        )

    # -- symbol tool -----------------------------------------------------------

    def inspect_symbols(self, arguments: dict[str, Any]) -> RcaToolResult:
        """List Python symbols from one repository file using ``ast``."""
        try:
            input_data = InspectSymbolsInput(**arguments)
        except (TypeError, ValueError) as exc:
            return RcaToolResult(ok=False, content=f"Invalid input: {exc}", tool="inspect_symbols")
        try:
            resolved = self.workspace.assert_read_allowed(input_data.path)
        except (ValueError, PermissionError) as exc:
            return RcaToolResult(ok=False, content=f"Path error: {exc}", tool="inspect_symbols")
        if not resolved.is_file():
            return RcaToolResult(ok=False, content=f"File not found: {input_data.path}", tool="inspect_symbols")
        if not input_data.path.endswith(".py"):
            return RcaToolResult(
                ok=False,
                content="inspect_symbols supports Python (.py) files only",
                tool="inspect_symbols",
            )
        started = time.monotonic()
        try:
            tree = ast.parse(resolved.read_text(encoding="utf-8"), filename=str(resolved))
        except SyntaxError as exc:
            return RcaToolResult(
                ok=False,
                content=f"Syntax error: {exc.msg} at line {exc.lineno}",
                tool="inspect_symbols",
                duration_seconds=time.monotonic() - started,
            )
        except (OSError, UnicodeDecodeError) as exc:
            return RcaToolResult(
                ok=False,
                content=f"Read failed: {exc}",
                tool="inspect_symbols",
                duration_seconds=time.monotonic() - started,
            )
        symbols = _collect_symbols(tree, input_data.name)
        omitted = max(0, len(symbols) - input_data.max_symbols)
        selected = symbols[: input_data.max_symbols]
        lines = [f"symbols in {input_data.path}: {len(selected)} shown"]
        lines.extend(selected)
        if omitted:
            lines.append(f"... ({omitted} symbols omitted)")
        output = "\n".join(lines)
        output, truncated = _bound_text(output, MAX_SYMBOL_OUTPUT_CHARS)
        return RcaToolResult(
            ok=True,
            content=output,
            tool="inspect_symbols",
            command=f"ast.parse({input_data.path})",
            duration_seconds=time.monotonic() - started,
            truncated=truncated or omitted > 0,
        )

    # -- git tools -------------------------------------------------------------

    def _validate_git_path(self, path: str) -> str:
        normalized = validate_relative_path(path)
        resolved = self.workspace.resolve(normalized)
        if resolved.name == ".git" or ".git" in resolved.parts:
            raise PermissionError(f"git path must not reference .git: {path}")
        return normalized

    def _run_git_capture(
        self,
        tool: str,
        argv: list[str],
        *,
        timeout: int = GIT_TIMEOUT_SECONDS,
    ) -> RcaToolResult:
        started = time.monotonic()
        try:
            result = subprocess.run(
                argv,
                cwd=self.workspace.root,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return RcaToolResult(
                ok=False,
                content="git command timed out",
                tool=tool,
                command=" ".join(argv),
                duration_seconds=time.monotonic() - started,
            )
        duration = time.monotonic() - started
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip() or f"exit code {result.returncode}"
            return RcaToolResult(
                ok=False,
                content=f"git command failed: {detail[:2_000]}",
                tool=tool,
                command=" ".join(argv),
                duration_seconds=duration,
                failure_type=ToolFailureType.TOOL_FAILURE,
            )
        output, truncated = _bound_text(result.stdout, MAX_GIT_OUTPUT_CHARS)
        return RcaToolResult(
            ok=True,
            content=output,
            tool=tool,
            command=" ".join(argv),
            duration_seconds=duration,
            truncated=truncated,
        )

    def git_history(self, arguments: dict[str, Any]) -> RcaToolResult:
        try:
            input_data = GitHistoryInput(**arguments)
        except (TypeError, ValueError) as exc:
            return RcaToolResult(ok=False, content=f"Invalid input: {exc}", tool="git_history")
        argv = [
            "git",
            "log",
            "--no-ext-diff",
            "--date=short",
            "--format=%h %ad %s",
            f"--max-count={input_data.max_count}",
        ]
        if input_data.grep:
            argv.append(f"--grep={input_data.grep}")
        if input_data.query:
            argv.append(f"-S{input_data.query}")
        if input_data.path is not None:
            try:
                path = self._validate_git_path(input_data.path)
            except (ValueError, PermissionError) as exc:
                return RcaToolResult(ok=False, content=f"Path error: {exc}", tool="git_history")
            argv.extend(["--", path])
        return self._run_git_capture("git_history", argv)

    def git_blame(self, arguments: dict[str, Any]) -> RcaToolResult:
        try:
            input_data = GitBlameInput(**arguments)
        except (TypeError, ValueError) as exc:
            return RcaToolResult(ok=False, content=f"Invalid input: {exc}", tool="git_blame")
        try:
            path = self._validate_git_path(input_data.path)
        except (ValueError, PermissionError) as exc:
            return RcaToolResult(ok=False, content=f"Path error: {exc}", tool="git_blame")
        argv = ["git", "blame", "--no-ext-diff", "-w"]
        if input_data.start_line is not None:
            end = input_data.end_line or input_data.start_line
            argv.append(f"-L{input_data.start_line},{end}")
        argv.extend(["--", path])
        return self._run_git_capture("git_blame", argv)

    def git_show(self, arguments: dict[str, Any]) -> RcaToolResult:
        try:
            input_data = GitShowInput(**arguments)
        except (TypeError, ValueError) as exc:
            return RcaToolResult(ok=False, content=f"Invalid input: {exc}", tool="git_show")
        argv = ["git", "show", "--no-ext-diff", "--format=fuller"]
        if input_data.stat:
            argv.append("--stat")
        if input_data.path is not None:
            try:
                path = self._validate_git_path(input_data.path)
            except (ValueError, PermissionError) as exc:
                return RcaToolResult(ok=False, content=f"Path error: {exc}", tool="git_show")
            argv.append(f"{input_data.revision}:{path}")
        else:
            argv.append(input_data.revision)
        return self._run_git_capture("git_show", argv)

    # -- external log tool ------------------------------------------------------

    def read_external_log(self, arguments: dict[str, Any]) -> RcaToolResult:
        try:
            input_data = ReadExternalLogInput(**arguments)
        except (TypeError, ValueError) as exc:
            return RcaToolResult(ok=False, content=f"Invalid input: {exc}", tool="read_external_log")
        if self.external_root is None:
            return RcaToolResult(
                ok=False,
                content="read_external_log requires an external log root",
                tool="read_external_log",
            )
        try:
            normalized = validate_relative_path(input_data.path)
            resolved = (self.external_root / normalized).resolve()
            resolved.relative_to(self.external_root)
        except ValueError as exc:
            return RcaToolResult(ok=False, content=f"Path error: {exc}", tool="read_external_log")
        if resolved.name == ".env" or resolved.name == ".git" or ".git" in resolved.parts:
            return RcaToolResult(
                ok=False,
                content="reading sensitive files is rejected",
                tool="read_external_log",
            )
        if not resolved.is_file():
            return RcaToolResult(ok=False, content=f"File not found: {input_data.path}", tool="read_external_log")
        started = time.monotonic()
        try:
            with resolved.open("rb") as stream:
                head = stream.read(MAX_EXTERNAL_LOG_CHARS + 1)
            total_bytes = resolved.stat().st_size
        except OSError as exc:
            return RcaToolResult(
                ok=False,
                content=f"Read failed: {exc}",
                tool="read_external_log",
                duration_seconds=time.monotonic() - started,
            )
        if b"\x00" in head:
            return RcaToolResult(
                ok=False,
                content="external log must be a text file",
                tool="read_external_log",
                duration_seconds=time.monotonic() - started,
            )
        text = head.decode("utf-8", errors="replace")
        truncated = len(text) > MAX_EXTERNAL_LOG_CHARS
        if truncated:
            text = text[:MAX_EXTERNAL_LOG_CHARS] + f"\n...[truncated: {total_bytes - MAX_EXTERNAL_LOG_CHARS} bytes omitted]"
        lines = text.splitlines()
        total_lines = len(lines)
        if total_lines > MAX_EXTERNAL_LOG_LINES:
            lines = lines[:MAX_EXTERNAL_LOG_LINES]
            lines.append(f"... ({total_lines - MAX_EXTERNAL_LOG_LINES} lines omitted)")
            truncated = True
        return RcaToolResult(
            ok=True,
            content="\n".join(lines),
            tool="read_external_log",
            command=f"read {resolved.name}",
            duration_seconds=time.monotonic() - started,
            truncated=truncated,
        )
