"""Tests for thread-safe RCA provider usage accounting."""

from __future__ import annotations

import threading

from patchpilot.rca.usage import UsageTracker


def test_usage_tracker_accumulates_exact_usage() -> None:
    tracker = UsageTracker()
    tracker.record(10, 4)
    tracker.record(20, 6)
    snapshot = tracker.snapshot()
    assert snapshot.llm_calls == 2
    assert snapshot.prompt_tokens == 30
    assert snapshot.completion_tokens == 10


def test_usage_tracker_missing_usage_becomes_null() -> None:
    tracker = UsageTracker()
    tracker.record(10, 4)
    tracker.record(None, None)
    snapshot = tracker.snapshot()
    assert snapshot.llm_calls == 2
    assert snapshot.prompt_tokens is None
    assert snapshot.completion_tokens is None


def test_usage_tracker_is_thread_safe() -> None:
    tracker = UsageTracker()
    threads = [
        threading.Thread(
            target=lambda: [tracker.record(7, 3) for _ in range(10)]
        )
        for _ in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    snapshot = tracker.snapshot()
    assert snapshot.llm_calls == 80
    assert snapshot.prompt_tokens == 560
    assert snapshot.completion_tokens == 240
