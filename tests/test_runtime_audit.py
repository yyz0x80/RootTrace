"""Tests for deterministic runtime diff auditing."""

from patchpilot.policy.runtime_audit import _should_ignore_file


def test_runtime_audit_ignores_patchpilot_scratch_tests() -> None:
    """Scratch tests are verifier evidence, not planned patch artifacts."""
    assert _should_ignore_file(
        ".patchpilot_checks/test_task_description.py"
    )


def test_runtime_audit_keeps_repository_source_files() -> None:
    """Repository source changes must remain visible to the policy audit."""
    assert not _should_ignore_file("src/task_service.py")
