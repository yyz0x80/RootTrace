"""Thread-safe provider usage accounting for RootTrace RCA agents.

``UsageTracker`` mirrors PatchPilot's exact-usage semantics: missing per-call
usage makes the aggregate ``None`` rather than an estimate. Concurrent workers
record through a lock, and ``snapshot()`` returns an immutable ``Usage`` model
for artifacts.
"""

from __future__ import annotations

import threading

from patchpilot.rca.schema import Usage


class UsageTracker:
    """Thread-safe aggregator of exact or null LLM token usage."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._llm_calls = 0
        self._prompt_tokens: int | None = 0
        self._completion_tokens: int | None = 0
        self._reasoning_tokens: int | None = 0

    def record(
        self,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        reasoning_tokens: int | None = None,
    ) -> None:
        """Record one provider call; missing usage makes the aggregate null."""
        with self._lock:
            self._llm_calls += 1
            if prompt_tokens is None:
                self._prompt_tokens = None
            elif self._prompt_tokens is not None:
                self._prompt_tokens += prompt_tokens

            if completion_tokens is None:
                self._completion_tokens = None
            elif self._completion_tokens is not None:
                self._completion_tokens += completion_tokens

            if reasoning_tokens is None:
                self._reasoning_tokens = None
            elif self._reasoning_tokens is not None:
                self._reasoning_tokens += reasoning_tokens

    def snapshot(self) -> Usage:
        """Return an immutable aggregate usage snapshot."""
        with self._lock:
            return Usage(
                llm_calls=self._llm_calls,
                prompt_tokens=self._prompt_tokens,
                completion_tokens=self._completion_tokens,
                reasoning_tokens=self._reasoning_tokens,
            )
