"""Incident loading for RootTrace RCA runs.

Loads a local GitHub Issue-style Markdown or JSON file plus optional stack
traces, CI logs, and PR diffs into a validated ``IncidentInput``. Resolves an
explicit base commit or falls back to the recorded current HEAD. All reads are
bounded and truncation is recorded explicitly.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from roottrace.incident.schema import (
    MAX_DIFF_CHARS,
    MAX_LOG_CHARS,
    MAX_PROBLEM_CHARS,
    MAX_TITLE_CHARS,
    IncidentInput,
    Provenance,
)

_OMITTED_MARKER = "\n...[truncated: {n} chars omitted]"


class LoadedIncident(BaseModel):
    """Validated incident plus explicit truncation bookkeeping."""

    incident: IncidentInput
    issue_chars_omitted: int = Field(default=0, ge=0)
    title_chars_omitted: int = Field(default=0, ge=0)
    stack_trace_chars_omitted: int = Field(default=0, ge=0)
    ci_log_chars_omitted: int = Field(default=0, ge=0)
    diff_chars_omitted: int = Field(default=0, ge=0)
    notes: list[str] = Field(default_factory=list, max_length=20)


@dataclass
class _JsonIssue:
    title: str | None
    problem: str | None
    logs: list[str]
    diff: str | None
    incident_id: str | None


def _run_git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise ValueError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout.strip()


def _truncate_text(text: str, limit: int) -> tuple[str, int]:
    if len(text) <= limit:
        return text, 0
    omitted = len(text) - limit
    return _apply_marker(text[:limit], omitted, limit), omitted


def _apply_marker(text: str, omitted: int, limit: int) -> str:
    marker = _OMITTED_MARKER.format(n=omitted)
    keep = max(0, limit - len(marker))
    return text[:keep] + marker


def _read_limited(path: Path, limit: int) -> tuple[str, int]:
    """Read at most ``limit`` chars; return ``(text, chars_omitted)``."""
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        text = stream.read(limit + 1)
        if len(text) <= limit:
            return text, 0
        total = len(text)
        while chunk := stream.read(8192):
            total += len(chunk)
        return text[:limit], total - limit


def _display_path(path: Path, repo: Path) -> str:
    try:
        return path.resolve().relative_to(repo).as_posix()
    except ValueError:
        return path.name


def resolve_base_commit(
    repo: Path,
    base_commit: str | None = None,
) -> tuple[str, str | None]:
    """Resolve the explicit base commit or fall back to recorded HEAD.

    Returns ``(canonical_sha, note)`` where ``note`` is set only on fallback.
    """
    if base_commit is not None:
        if not base_commit.strip() or len(base_commit) > 64:
            raise ValueError("base commit must be a valid git SHA")
        canonical = _run_git(
            repo,
            "rev-parse",
            "--verify",
            f"{base_commit}^{{commit}}",
        )
        return canonical, None
    head = _run_git(repo, "rev-parse", "HEAD")
    return head, f"base commit fell back to current HEAD {head[:12]}"


def _load_json_issue(path: Path) -> _JsonIssue:
    raw, omitted = _read_limited(path, MAX_DIFF_CHARS + 20_000)
    if omitted:
        raise ValueError(f"JSON issue file too large: {path.name}")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON issue file {path.name}: {exc}") from exc
    if not isinstance(data, dict):
        raise TypeError("JSON issue must be an object")
    title = data.get("title")
    problem = data.get("problem", data.get("body"))
    logs = data.get("logs", [])
    diff = data.get("diff")
    incident_id = data.get("id")
    if not isinstance(title, str) and title is not None:
        raise ValueError("JSON issue 'title' must be a string")
    if not isinstance(problem, str) and problem is not None:
        raise ValueError("JSON issue 'problem'/'body' must be a string")
    if not isinstance(logs, list) or any(not isinstance(item, str) for item in logs):
        raise ValueError("JSON issue 'logs' must be a list of strings")
    if diff is not None and not isinstance(diff, str):
        raise ValueError("JSON issue 'diff' must be a string")
    if incident_id is not None and not isinstance(incident_id, str):
        raise ValueError("JSON issue 'id' must be a string")
    return _JsonIssue(
        title=title,
        problem=problem,
        logs=logs,
        diff=diff,
        incident_id=incident_id,
    )


def _load_markdown_issue(path: Path) -> tuple[str, str, int]:
    body, omitted = _read_limited(path, MAX_PROBLEM_CHARS)
    title = path.stem
    for line in body.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
            break
    return title, body, omitted


def load_incident(
    issue_path: str | Path,
    repo_path: str | Path,
    *,
    base_commit: str | None = None,
    stack_trace_path: str | Path | None = None,
    ci_log_path: str | Path | None = None,
    pr_diff_path: str | Path | None = None,
    incident_id: str | None = None,
    repo_identifier: str | None = None,
) -> LoadedIncident:
    """Load a local issue plus optional evidence into a validated incident."""
    repo = Path(repo_path).resolve()
    if not repo.is_dir():
        raise ValueError(f"repository path is not a directory: {repo}")
    resolved_base, base_note = resolve_base_commit(repo, base_commit)

    issue = Path(issue_path)
    if not issue.is_file():
        raise FileNotFoundError(f"issue file not found: {issue}")

    notes: list[str] = []
    if base_note:
        notes.append(base_note)

    if issue.suffix.lower() == ".json":
        parsed = _load_json_issue(issue)
        title = parsed.title or issue.stem
        problem = parsed.problem or ""
        json_logs = parsed.logs
        json_diff = parsed.diff
        json_id = parsed.incident_id
        issue_omitted = 0
    else:
        title, problem, issue_omitted = _load_markdown_issue(issue)
        json_logs = []
        json_diff = None
        json_id = None
    if issue_omitted:
        notes.append(f"issue body truncated ({issue_omitted} chars omitted)")
        problem = _apply_marker(problem, issue_omitted, MAX_PROBLEM_CHARS)
    else:
        problem, problem_omitted = _truncate_text(problem, MAX_PROBLEM_CHARS)
        issue_omitted = problem_omitted
        if problem_omitted:
            notes.append(
                f"issue body truncated ({problem_omitted} chars omitted)"
            )

    if not problem.strip():
        raise ValueError("issue must contain a problem/body")

    title, title_omitted = _truncate_text(title, MAX_TITLE_CHARS)
    if title_omitted:
        notes.append(f"issue title truncated ({title_omitted} chars omitted)")

    logs: list[str] = []
    stack_omitted = 0
    ci_omitted = 0
    if stack_trace_path is not None:
        stack_path = Path(stack_trace_path)
        if not stack_path.is_file():
            raise FileNotFoundError(f"stack trace file not found: {stack_path}")
        stack_text, stack_omitted = _read_limited(stack_path, MAX_LOG_CHARS)
        if stack_omitted:
            stack_text = _apply_marker(stack_text, stack_omitted, MAX_LOG_CHARS)
            notes.append(f"stack trace truncated ({stack_omitted} chars omitted)")
        logs.append(stack_text)
    if ci_log_path is not None:
        ci_path = Path(ci_log_path)
        if not ci_path.is_file():
            raise FileNotFoundError(f"CI log file not found: {ci_path}")
        ci_text, ci_omitted = _read_limited(ci_path, MAX_LOG_CHARS)
        if ci_omitted:
            ci_text = _apply_marker(ci_text, ci_omitted, MAX_LOG_CHARS)
            notes.append(f"CI log truncated ({ci_omitted} chars omitted)")
        logs.append(ci_text)

    json_logs_omitted = 0
    for entry in json_logs:
        if len(logs) >= 10:
            json_logs_omitted += 1
            continue
        bounded_entry, entry_omitted = _truncate_text(entry, MAX_LOG_CHARS)
        if entry_omitted:
            notes.append(f"JSON issue log truncated ({entry_omitted} chars omitted)")
        logs.append(bounded_entry)
    if json_logs_omitted:
        notes.append(f"{json_logs_omitted} JSON issue logs omitted (log cap reached)")

    diff: str | None = None
    diff_omitted = 0
    if pr_diff_path is not None:
        diff_path = Path(pr_diff_path)
        if not diff_path.is_file():
            raise FileNotFoundError(f"PR diff file not found: {diff_path}")
        diff, diff_omitted = _read_limited(diff_path, MAX_DIFF_CHARS)
        if diff_omitted:
            diff = _apply_marker(diff, diff_omitted, MAX_DIFF_CHARS)
            notes.append(f"PR diff truncated ({diff_omitted} chars omitted)")
    elif json_diff:
        diff, diff_omitted = _truncate_text(json_diff, MAX_DIFF_CHARS)
        if diff_omitted:
            notes.append(f"PR diff truncated ({diff_omitted} chars omitted)")

    final_id = incident_id or json_id or f"inc-{resolved_base[:12]}"
    final_repo = repo_identifier or repo.name

    provenance = Provenance(
        source=_display_path(issue, repo),
        tool="incident_loader",
        commit=resolved_base,
    )

    try:
        incident = IncidentInput(
            id=final_id,
            repo=final_repo,
            base_commit=resolved_base,
            title=title,
            problem=problem,
            logs=logs,
            diff=diff,
            provenance=provenance,
        )
    except ValidationError as exc:
        raise ValueError(f"invalid incident input: {exc}") from exc

    return LoadedIncident(
        incident=incident,
        issue_chars_omitted=issue_omitted,
        title_chars_omitted=title_omitted,
        stack_trace_chars_omitted=stack_omitted,
        ci_log_chars_omitted=ci_omitted,
        diff_chars_omitted=diff_omitted,
        notes=notes,
    )
