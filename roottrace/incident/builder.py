"""Deterministic, bounded context construction for RootTrace RCA runs.

The builder collects a repository fingerprint, a capped file inventory, search
signals (issue terms, exception names, stack-frame symbols, diff paths), and
ranked source snippets. Output is deterministic for identical inputs and every
trim is recorded in ``ContextTruncation``. Whole repositories and unbounded Git
history are never included.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import NamedTuple

from roottrace.incident.context import (
    MAX_DIFF_EXCERPT_CHARS,
    MAX_SNIPPET_CHARS,
    ContextTruncation,
    IncidentContext,
    IncidentSignals,
    RepositoryInventory,
    SourceSnippet,
)
from roottrace.incident.loader import LoadedIncident
from roottrace.incident.schema import IncidentInput
from roottrace.runtime.paths import validate_relative_path
from roottrace.runtime.workspace import (
    capture_repository_fingerprint,
)

_CONFIG_FILES = {
    "pyproject.toml",
    "requirements.txt",
    "setup.py",
    "setup.cfg",
}

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "from",
        "has",
        "in",
        "into",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "we",
        "when",
        "with",
        "not",
    }
)

_EXCEPTION_PATTERN = re.compile(r"\b[A-Z][A-Za-z0-9_]*(?:Error|Exception)\b")
_SYMBOL_PATTERN = re.compile(r"line \d+,\s+in\s+([A-Za-z_][A-Za-z0-9_.]*)")
_TERM_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_DIFF_PLUS_PATTERN = re.compile(r"^\+\+\+ b/(.+)$")
_DIFF_HEADER_PATTERN = re.compile(r"^diff --git a/(.+) b/(.+)$")


class _SignalCounts(NamedTuple):
    terms_omitted: int
    exception_names_omitted: int
    stack_symbols_omitted: int
    diff_paths_omitted: int


@dataclass
class _Candidate:
    score: int = 0
    matched: set[str] = field(default_factory=set)
    excerpt: str = ""
    excerpt_omitted: int = 0


def _git_ls_files(repo: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError("target is not a valid git repository")
    return sorted(line for line in result.stdout.splitlines() if line.strip())


def _is_test_file(path: str) -> bool:
    normalized = PurePosixPath(path)
    return "tests" in normalized.parts or normalized.name.startswith("test_")


def _cap_list(values: list[str], limit: int) -> tuple[list[str], int]:
    if limit < 0:
        raise ValueError("cap must be non-negative")
    return values[:limit], max(0, len(values) - limit)


def build_repository_inventory(
    repo: str | Path,
    base_commit: str,
    *,
    max_python_files: int = 300,
    max_test_files: int = 100,
    max_config_files: int = 50,
) -> RepositoryInventory:
    """Build a capped, sorted inventory of tracked repository files."""
    repo_path = Path(repo).resolve()
    tracked = _git_ls_files(repo_path)
    python_files = [path for path in tracked if path.endswith(".py")]
    test_files = [path for path in python_files if _is_test_file(path)]
    config_files = [
        path for path in tracked if PurePosixPath(path).name in _CONFIG_FILES
    ]

    python_list, python_omitted = _cap_list(python_files, max_python_files)
    test_list, test_omitted = _cap_list(test_files, max_test_files)
    config_list, config_omitted = _cap_list(config_files, max_config_files)

    return RepositoryInventory(
        base_commit=base_commit,
        tracked_files=len(tracked),
        python_files=len(python_files),
        test_files=len(test_files),
        config_files=len(config_files),
        python_file_list=python_list,
        test_file_list=test_list,
        config_file_list=config_list,
        python_files_omitted=python_omitted,
        test_files_omitted=test_omitted,
        config_files_omitted=config_omitted,
    )


def extract_signals(
    incident: IncidentInput,
    *,
    max_terms: int = 10,
    max_exceptions: int = 5,
    max_symbols: int = 8,
    max_diff_paths: int = 20,
) -> tuple[IncidentSignals, _SignalCounts]:
    """Extract deterministic search signals from the incident."""
    if max_terms <= 0 or max_exceptions <= 0 or max_symbols <= 0:
        raise ValueError("signal caps must be positive")

    issue_text = f"{incident.title or ''}\n{incident.problem}"
    searchable = "\n".join([issue_text, *incident.logs])

    terms: list[str] = []
    for match in _TERM_PATTERN.finditer(issue_text):
        word = match.group(0).lower()
        if len(word) > 200:
            continue
        if word in _STOPWORDS or word in terms:
            continue
        terms.append(word)

    exceptions: list[str] = []
    for match in _EXCEPTION_PATTERN.finditer(searchable):
        name = match.group(0)
        if name not in exceptions:
            exceptions.append(name)

    symbols: list[str] = []
    for match in _SYMBOL_PATTERN.finditer(searchable):
        symbol = match.group(1)
        if symbol.startswith("<") or symbol in symbols:
            continue
        symbols.append(symbol)

    diff_paths: list[str] = []
    if incident.diff:
        for line in incident.diff.splitlines():
            plus = _DIFF_PLUS_PATTERN.match(line)
            if plus:
                candidate = plus.group(1).strip()
            else:
                header = _DIFF_HEADER_PATTERN.match(line)
                if not header:
                    continue
                candidate = header.group(2).strip()
            try:
                candidate = validate_relative_path(candidate)
            except ValueError:
                continue
            if candidate not in diff_paths:
                diff_paths.append(candidate)
        diff_paths.sort()

    terms_capped, terms_omitted = _cap_list(terms, max_terms)
    exceptions_capped, exceptions_omitted = _cap_list(exceptions, max_exceptions)
    symbols_capped, symbols_omitted = _cap_list(symbols, max_symbols)
    paths_capped, paths_omitted = _cap_list(diff_paths, max_diff_paths)

    signals = IncidentSignals(
        terms=terms_capped,
        exception_names=exceptions_capped,
        stack_symbols=symbols_capped,
        diff_paths=paths_capped,
    )
    counts = _SignalCounts(
        terms_omitted=terms_omitted,
        exception_names_omitted=exceptions_omitted,
        stack_symbols_omitted=symbols_omitted,
        diff_paths_omitted=paths_omitted,
    )
    return signals, counts


def _rg_matches(
    repo: Path,
    pattern: str,
    *,
    max_per_file: int = 5,
    max_total: int = 30,
    timeout: int = 30,
) -> list[tuple[str, int, str]]:
    try:
        result = subprocess.run(
            [
                "rg",
                "-n",
                "-i",
                "--no-heading",
                "--color",
                "never",
                "--glob",
                "*.py",
                "-m",
                str(max_per_file),
                "-F",
                "-e",
                pattern,
                ".",
            ],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return []
    if result.returncode not in (0, 1):
        return []

    matches: list[tuple[str, int, str]] = []
    for line in result.stdout.splitlines()[:max_total]:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        rel_path, lineno, content = parts
        try:
            number = int(lineno)
        except ValueError:
            continue
        try:
            rel_path = validate_relative_path(rel_path)
        except ValueError:
            continue
        matches.append((rel_path, number, content))
    return matches


def _read_window(repo: Path, rel_path: str, start: int, end: int) -> str:
    path = repo / rel_path
    if not path.is_file():
        return ""
    lines: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for lineno, line in enumerate(stream, start=1):
            if lineno > end:
                break
            if lineno >= start:
                lines.append(line.rstrip("\n"))
    return "\n".join(f"{n}: {text}" for n, text in enumerate(lines, start=start))


def _truncate_excerpt(
    text: str,
    limit: int = MAX_SNIPPET_CHARS,
) -> tuple[str, int]:
    if len(text) <= limit:
        return text, 0
    marker = "\n...[truncated]"
    keep = max(0, limit - len(marker))
    return text[:keep] + marker, len(text) - limit


def build_ranked_snippets(
    repo: str | Path,
    signals: IncidentSignals,
    inventory: RepositoryInventory,
    *,
    window_lines: int = 5,
    max_candidates: int = 100,
    max_snippets: int = 20,
) -> tuple[list[SourceSnippet], int, int, int]:
    """Search tracked Python files and return ranked, bounded snippets.

    Returns ``(snippets, candidates_omitted, snippets_omitted,
    excerpt_chars_omitted)``.
    """
    if window_lines <= 0 or max_candidates <= 0 or max_snippets <= 0:
        raise ValueError("snippet caps must be positive")

    repo_path = Path(repo).resolve()
    python_set = set(inventory.python_file_list)
    candidates: dict[tuple[str, int, int], _Candidate] = {}

    search_pool: list[tuple[str, str, int]] = []
    search_pool.extend(("term", term, 1) for term in signals.terms)
    search_pool.extend(("exception", name, 5) for name in signals.exception_names)
    search_pool.extend(("symbol", symbol, 4) for symbol in signals.stack_symbols)

    for _kind, pattern, weight in search_pool:
        for rel_path, lineno, _content in _rg_matches(repo_path, pattern):
            if rel_path not in python_set:
                continue
            start = max(1, lineno - window_lines)
            end = lineno + window_lines
            key = (rel_path, start, end)
            candidate = candidates.setdefault(key, _Candidate())
            if pattern not in candidate.matched:
                candidate.score += weight
                candidate.matched.add(pattern)
            if not candidate.excerpt:
                raw = _read_window(repo_path, rel_path, start, end)
                candidate.excerpt, candidate.excerpt_omitted = _truncate_excerpt(raw)

    for diff_path in signals.diff_paths:
        if diff_path not in python_set:
            continue
        key = (diff_path, 1, min(10, window_lines * 2))
        candidate = candidates.setdefault(key, _Candidate())
        if "diff_changed_file" not in candidate.matched:
            candidate.score += 8
            candidate.matched.add("diff_changed_file")
        if not candidate.excerpt:
            raw = _read_window(repo_path, diff_path, key[1], key[2])
            candidate.excerpt, candidate.excerpt_omitted = _truncate_excerpt(raw)

    if not candidates:
        return [], 0, 0, 0

    ordered = sorted(
        candidates.items(),
        key=lambda item: (-item[1].score, item[0][0], item[0][1], item[0][2]),
    )
    candidates_omitted = max(0, len(ordered) - max_candidates)
    ordered = ordered[:max_candidates]

    snippets: list[SourceSnippet] = []
    for rank, ((path, start, end), candidate) in enumerate(
        ordered[:max_snippets],
        start=1,
    ):
        snippets.append(
            SourceSnippet(
                path=path,
                start_line=start,
                end_line=end,
                excerpt=candidate.excerpt,
                score=candidate.score,
                matched_terms=sorted(candidate.matched),
                rank=rank,
            )
        )
    snippets_omitted = max(0, len(ordered) - max_snippets)
    excerpt_omitted = sum(
        candidate.excerpt_omitted for _, candidate in ordered[:max_snippets]
    )
    return snippets, candidates_omitted, snippets_omitted, excerpt_omitted


def build_incident_context(
    loaded: LoadedIncident,
    repo: str | Path,
    *,
    max_snippets: int = 20,
    window_lines: int = 5,
    max_candidates: int = 100,
) -> IncidentContext:
    """Build the deterministic, bounded context for one RCA run."""
    repo_path = Path(repo).resolve()
    fingerprint = capture_repository_fingerprint(repo_path)
    inventory = build_repository_inventory(
        repo_path,
        loaded.incident.base_commit,
    )
    signals, counts = extract_signals(loaded.incident)
    snippets, candidates_omitted, snippets_omitted, excerpt_omitted = (
        build_ranked_snippets(
            repo_path,
            signals,
            inventory,
            window_lines=window_lines,
            max_candidates=max_candidates,
            max_snippets=max_snippets,
        )
    )

    diff_excerpt: str | None = None
    diff_omitted = 0
    if loaded.incident.diff:
        diff_excerpt, diff_omitted = _truncate_excerpt(
            loaded.incident.diff,
            MAX_DIFF_EXCERPT_CHARS,
        )

    notes = list(loaded.notes)
    if not snippets and (
        signals.terms or signals.exception_names or signals.stack_symbols
    ):
        notes.append("no source snippets matched search signals")
    if loaded.incident.diff and diff_omitted:
        notes.append(f"PR diff excerpt truncated ({diff_omitted} chars omitted)")
    review_truncation = loaded.incident.review_comment_truncation
    if review_truncation.threads_omitted:
        notes.append(
            f"review comment threads truncated "
            f"({review_truncation.threads_omitted} omitted)"
        )
    if review_truncation.comments_omitted:
        notes.append(
            f"review comments truncated "
            f"({review_truncation.comments_omitted} omitted)"
        )

    truncation = ContextTruncation(
        issue_body_chars_omitted=loaded.issue_chars_omitted,
        title_chars_omitted=loaded.title_chars_omitted,
        stack_trace_chars_omitted=loaded.stack_trace_chars_omitted,
        ci_log_chars_omitted=loaded.ci_log_chars_omitted,
        diff_chars_omitted=diff_omitted,
        terms_omitted=counts.terms_omitted,
        exception_names_omitted=counts.exception_names_omitted,
        stack_symbols_omitted=counts.stack_symbols_omitted,
        diff_paths_omitted=counts.diff_paths_omitted,
        python_files_omitted=inventory.python_files_omitted,
        test_files_omitted=inventory.test_files_omitted,
        config_files_omitted=inventory.config_files_omitted,
        snippet_candidates_omitted=candidates_omitted,
        snippets_omitted=snippets_omitted,
        snippet_excerpt_chars_omitted=excerpt_omitted,
        review_threads_omitted=review_truncation.threads_omitted,
        review_comments_omitted=review_truncation.comments_omitted,
        review_comment_chars_omitted=review_truncation.chars_omitted,
        review_comment_locations_unmapped=review_truncation.locations_unmapped,
        review_comment_invalid_paths=review_truncation.invalid_paths,
        notes=notes,
    )

    return IncidentContext(
        incident=loaded.incident,
        repository=inventory,
        signals=signals,
        snippets=snippets,
        diff_excerpt=diff_excerpt,
        truncation=truncation,
        fingerprint=fingerprint,
    )
