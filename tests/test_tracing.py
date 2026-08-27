"""Tests for execution trace module."""

import json
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from roottrace.tracing import (
    TraceEvent,
    TraceWriter,
    summarize_tool_arguments,
)


class TestSummarizeToolArguments:
    """Tests for tool argument summarization and redaction."""

    def test_redact_sensitive_arguments(self) -> None:
        """Test that sensitive arguments are redacted."""
        arguments = {
            "old_text": "sensitive content",
            "new_text": "new sensitive content",
            "content": "more sensitive data",
            "prompt": "secret prompt",
            "api_key": "sk-1234567890",
            "token": "secret-token",
            "password": "mypassword",
            "secret": "mysecret",
            "normal_arg": "normal value",
        }

        result = summarize_tool_arguments(arguments)

        assert result["old_text"] == "<redacted>"
        assert result["new_text"] == "<redacted>"
        assert result["content"] == "<redacted>"
        assert result["prompt"] == "<redacted>"
        assert result["api_key"] == "<redacted>"
        assert result["token"] == "<redacted>"
        assert result["password"] == "<redacted>"
        assert result["secret"] == "<redacted>"
        assert result["normal_arg"] == "normal value"

    def test_case_insensitive_redaction(self) -> None:
        """Test that redaction is case-insensitive."""
        arguments = {
            "Old_Text": "uppercase version",
            "API_KEY": "uppercase api key",
            "Password": "capitalized password",
        }

        result = summarize_tool_arguments(arguments)

        assert result["Old_Text"] == "<redacted>"
        assert result["API_KEY"] == "<redacted>"
        assert result["Password"] == "<redacted>"

    def test_truncate_long_strings(self) -> None:
        """Test that long strings are truncated."""
        arguments = {
            "short_string": "short",
            "long_string": "a" * 250,
            "normal_arg": "normal value",
        }

        result = summarize_tool_arguments(arguments)

        assert result["short_string"] == "short"
        assert result["long_string"] == "<250 chars>"
        assert result["normal_arg"] == "normal value"

    def test_mixed_types(self) -> None:
        """Test handling of mixed value types."""
        arguments = {
            "string_value": "test string",
            "number_value": 42,
            "list_value": [1, 2, 3],
            "dict_value": {"key": "value"},
            "none_value": None,
        }

        result = summarize_tool_arguments(arguments)

        assert result["string_value"] == "test string"
        assert result["number_value"] == 42
        assert result["list_value"] == [1, 2, 3]
        assert result["dict_value"] == {"key": "value"}
        assert result["none_value"] is None

    def test_empty_arguments(self) -> None:
        """Test handling of empty arguments dictionary."""
        result = summarize_tool_arguments({})
        assert result == {}


class TestTraceEvent:
    """Tests for TraceEvent data model."""

    def test_minimal_event_creation(self) -> None:
        """Test creating a minimal trace event with required fields."""
        event = TraceEvent(
            run_id="test-run-123",
            event_type="test_event",
            workflow_stage="TEST_STAGE",
        )
        assert event.run_id == "test-run-123"
        assert event.event_type == "test_event"
        assert event.workflow_stage == "TEST_STAGE"
        assert event.model is None
        assert event.tool_name is None
        assert event.modified_files == []
        assert event.retry_count == 0

    def test_full_event_creation(self) -> None:
        """Test creating a trace event with all fields populated."""
        event = TraceEvent(
            run_id="test-run-456",
            event_type="tool_call",
            workflow_stage="CODING",
            model="gpt-4",
            tool_name="read_file",
            tool_arguments={"path": "src/main.py"},
            tool_duration=0.5,
            permission_result="ALLOWED",
            modified_files=["src/main.py"],
            verification_result={"ruff": True, "pytest": True},
            degradation={"reason": "bounded_fallback"},
            retry_count=1,
            final_status="SUCCESS",
            prompt_tokens=100,
            completion_tokens=50,
            total_cost=0.01,
        )
        assert event.run_id == "test-run-456"
        assert event.event_type == "tool_call"
        assert event.workflow_stage == "CODING"
        assert event.model == "gpt-4"
        assert event.tool_name == "read_file"
        assert event.tool_arguments == {"path": "src/main.py"}
        assert event.tool_duration == 0.5
        assert event.permission_result == "ALLOWED"
        assert event.modified_files == ["src/main.py"]
        assert event.verification_result == {"ruff": True, "pytest": True}
        assert event.degradation == {"reason": "bounded_fallback"}
        assert event.retry_count == 1
        assert event.final_status == "SUCCESS"
        assert event.prompt_tokens == 100
        assert event.completion_tokens == 50
        assert event.total_cost == 0.01

    def test_timestamp_auto_generation(self) -> None:
        """Test that timestamp is automatically generated in ISO format."""
        event = TraceEvent(
            run_id="test-run-789",
            event_type="timestamp_test",
            workflow_stage="TEST",
        )
        assert event.timestamp is not None
        # Verify ISO format by attempting to parse
        datetime.fromisoformat(event.timestamp)

    def test_model_dump_json(self) -> None:
        """Test that event can be serialized to JSON."""
        event = TraceEvent(
            run_id="test-run-json",
            event_type="json_test",
            workflow_stage="TEST",
            tool_name="edit_file",
            tool_arguments={"path": "test.py", "old_text": "old", "new_text": "new"},
        )
        json_str = event.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["run_id"] == "test-run-json"
        assert parsed["event_type"] == "json_test"
        assert parsed["tool_name"] == "edit_file"


class TestTraceWriter:
    """Tests for TraceWriter file operations."""

    def test_write_creates_parent_directories(self) -> None:
        """Test that writer creates parent directories automatically."""
        with TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "nested" / "dir" / "trace.jsonl"
            writer = TraceWriter(trace_path)

            event = TraceEvent(
                run_id="dir-test",
                event_type="directory_test",
                workflow_stage="TEST",
            )
            writer.write(event)

            assert trace_path.exists()
            assert trace_path.parent.exists()

    def test_write_single_event(self) -> None:
        """Test writing a single event to trace file."""
        with TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "trace.jsonl"
            writer = TraceWriter(trace_path)

            event = TraceEvent(
                run_id="single-event",
                event_type="single_test",
                workflow_stage="TEST",
                tool_name="search_code",
                tool_arguments={"query": "test"},
            )
            writer.write(event)

            content = trace_path.read_text(encoding="utf-8")
            lines = content.strip().split("\n")
            assert len(lines) == 1

            parsed = json.loads(lines[0])
            assert parsed["run_id"] == "single-event"
            assert parsed["tool_name"] == "search_code"

    def test_write_multiple_events(self) -> None:
        """Test writing multiple events to trace file."""
        with TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "trace.jsonl"
            writer = TraceWriter(trace_path)

            events = [
                TraceEvent(
                    run_id="multi-event",
                    event_type=f"event_{i}",
                    workflow_stage="TEST",
                    tool_name=f"tool_{i}",
                )
                for i in range(3)
            ]

            for event in events:
                writer.write(event)

            content = trace_path.read_text(encoding="utf-8")
            lines = content.strip().split("\n")
            assert len(lines) == 3

            for i, line in enumerate(lines):
                parsed = json.loads(line)
                assert parsed["event_type"] == f"event_{i}"
                assert parsed["tool_name"] == f"tool_{i}"

    def test_append_to_existing_file(self) -> None:
        """Test that writer appends to existing file without overwriting."""
        with TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "trace.jsonl"
            writer = TraceWriter(trace_path)

            # Write first event
            event1 = TraceEvent(
                run_id="append-test",
                event_type="first",
                workflow_stage="TEST",
            )
            writer.write(event1)

            # Write second event
            event2 = TraceEvent(
                run_id="append-test",
                event_type="second",
                workflow_stage="TEST",
            )
            writer.write(event2)

            content = trace_path.read_text(encoding="utf-8")
            lines = content.strip().split("\n")
            assert len(lines) == 2

            first_parsed = json.loads(lines[0])
            second_parsed = json.loads(lines[1])
            assert first_parsed["event_type"] == "first"
            assert second_parsed["event_type"] == "second"

    def test_start_run_clears_events_from_previous_run(self) -> None:
        """Test that a new workflow run starts with an empty trace artifact."""
        with TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "trace.jsonl"
            trace_path.write_text("stale event\n", encoding="utf-8")
            writer = TraceWriter(trace_path)

            writer.start_run()
            writer.write(
                TraceEvent(
                    run_id="current-run",
                    event_type="workflow_started",
                    workflow_stage="WORKSPACE",
                )
            )

            lines = trace_path.read_text(encoding="utf-8").splitlines()
            assert len(lines) == 1
            assert json.loads(lines[0])["run_id"] == "current-run"

    def test_utf8_encoding(self) -> None:
        """Test that writer handles UTF-8 characters correctly."""
        with TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "trace.jsonl"
            writer = TraceWriter(trace_path)

            event = TraceEvent(
                run_id="utf8-test",
                event_type="unicode_test",
                workflow_stage="TEST",
                tool_name="测试工具",
                tool_arguments={"path": "文件路径.py", "description": "测试描述"},
            )
            writer.write(event)

            content = trace_path.read_text(encoding="utf-8")
            parsed = json.loads(content.strip())
            assert parsed["tool_name"] == "测试工具"
            assert parsed["tool_arguments"]["path"] == "文件路径.py"

    def test_complex_nested_data(self) -> None:
        """Test writing events with complex nested data structures."""
        with TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "trace.jsonl"
            writer = TraceWriter(trace_path)

            event = TraceEvent(
                run_id="complex-test",
                event_type="complex_data",
                workflow_stage="VERIFY",
                verification_result={
                    "tests": {
                        "unit_tests": {"passed": 10, "failed": 0},
                        "integration_tests": {"passed": 5, "failed": 1},
                    },
                    "linting": {"ruff": True, "mypy": False},
                },
                modified_files=["src/a.py", "src/b.py", "tests/test_a.py"],
            )
            writer.write(event)

            content = trace_path.read_text(encoding="utf-8")
            parsed = json.loads(content.strip())
            assert parsed["verification_result"]["tests"]["unit_tests"]["passed"] == 10
            assert len(parsed["modified_files"]) == 3

    def test_sensitive_argument_redaction(self) -> None:
        """Test that sensitive tool arguments are redacted when writing."""
        with TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "trace.jsonl"
            writer = TraceWriter(trace_path)

            event = TraceEvent(
                run_id="sensitive-test",
                event_type="tool_call",
                workflow_stage="CODING",
                tool_name="edit_file",
                tool_arguments={
                    "path": "test.py",
                    "old_text": "sensitive old content",
                    "new_text": "sensitive new content",
                    "api_key": "sk-secret-key",
                },
            )
            writer.write(event)

            content = trace_path.read_text(encoding="utf-8")
            parsed = json.loads(content.strip())

            assert parsed["tool_arguments"]["path"] == "test.py"
            assert parsed["tool_arguments"]["old_text"] == "<redacted>"
            assert parsed["tool_arguments"]["new_text"] == "<redacted>"
            assert parsed["tool_arguments"]["api_key"] == "<redacted>"

    def test_tool_failure_event_recording(self) -> None:
        """Test that tool failure events are still recorded to trace."""
        with TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "trace.jsonl"
            writer = TraceWriter(trace_path)

            # Simulate a failed tool call event
            failure_event = TraceEvent(
                run_id="failure-test",
                event_type="tool_call",
                workflow_stage="CODING",
                tool_name="edit_file",
                tool_arguments={"path": "nonexistent.py"},
                tool_duration=0.1,
                permission_result="ALLOWED",
                final_status="FAILURE",
                retry_count=2,
            )
            writer.write(failure_event)

            # Also write a success event to ensure both are recorded
            success_event = TraceEvent(
                run_id="failure-test",
                event_type="tool_call",
                workflow_stage="CODING",
                tool_name="read_file",
                tool_arguments={"path": "exists.py"},
                tool_duration=0.05,
                permission_result="ALLOWED",
                final_status="SUCCESS",
            )
            writer.write(success_event)

            content = trace_path.read_text(encoding="utf-8")
            lines = content.strip().split("\n")
            assert len(lines) == 2

            # Verify failure event was recorded
            failure_parsed = json.loads(lines[0])
            assert failure_parsed["final_status"] == "FAILURE"
            assert failure_parsed["retry_count"] == 2
            assert failure_parsed["tool_name"] == "edit_file"

            # Verify success event was also recorded
            success_parsed = json.loads(lines[1])
            assert success_parsed["final_status"] == "SUCCESS"
            assert success_parsed["tool_name"] == "read_file"

    def test_usage_fields_null_when_not_provided(self) -> None:
        """Test that usage fields are null when Provider does not provide them."""
        with TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "trace.jsonl"
            writer = TraceWriter(trace_path)

            # Create event without usage information (simulating Provider not providing usage)
            event = TraceEvent(
                run_id="usage-test",
                event_type="model_call",
                workflow_stage="CODING",
                model="gpt-4",
                # Do not set prompt_tokens, completion_tokens, or total_cost
                # They should default to None and be serialized as null in JSON
            )
            writer.write(event)

            content = trace_path.read_text(encoding="utf-8")
            parsed = json.loads(content.strip())

            # Verify that usage fields exist and are null in JSON
            assert "prompt_tokens" in parsed
            assert "completion_tokens" in parsed
            assert "total_cost" in parsed
            assert parsed["prompt_tokens"] is None
            assert parsed["completion_tokens"] is None
            assert parsed["total_cost"] is None
