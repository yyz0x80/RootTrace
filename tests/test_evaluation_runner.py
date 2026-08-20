"""Tests for evaluation task discovery and isolated fixture execution."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from evaluation import runner
from evaluation.runner import (
    RunResult,
    ScoreResult,
    TaskConfig,
    aggregate_scores,
    execute_task,
    select_task_configs,
)


def write_task(task_dir: Path, task_id: str) -> None:
    """Write the smallest valid evaluation task manifest."""
    task_dir.mkdir(parents=True)
    manifest = {
        "task_id": task_id,
        "category": "single_file_bug",
        "repository": "fixtures/repo",
        "base_commit": "abc123",
        "issue": f"tasks/{task_id}/issue.md",
        "expected_final_status": "VERIFIED",
        "allowed_changes": ["example.py"],
        "target_tests": ["tests/test_example.py"],
        "score_commands": [],
    }
    (task_dir / "task.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )


def make_task_config(
    task_id: str,
    category: str,
    expected_status: str,
) -> TaskConfig:
    """Create a task configuration for aggregation tests."""
    return TaskConfig(
        task_id=task_id,
        category=category,
        repository="fixtures/repo",
        base_commit="abc123",
        issue=f"tasks/{task_id}/issue.md",
        expected_final_status=expected_status,
        allowed_changes=[],
        target_tests=[],
        score_commands=[],
        expected_phase=(
            "prepare" if category == "unsafe_request" else "execute"
        ),
    )


def write_json(path: Path, data: object) -> None:
    """Write one JSON fixture with parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def write_metric_task(
    runs_dir: Path,
    config: TaskConfig,
    *,
    actual_status: str,
    phase: str,
    report: dict[str, object] | None,
    run_summary: dict[str, object] | None,
    criteria: list[str],
    evidence: dict[str, str],
    prepare_usage: tuple[int, int | None, int | None],
    functional_correctness: float = 1.0,
    outcome_accuracy: float = 1.0,
    hidden_tests_passed: bool = True,
    hidden_tests_applicable: bool = True,
    patch_applied: bool = True,
) -> None:
    """Write deterministic task artifacts consumed by metric aggregation."""
    task_dir = runs_dir / config.task_id
    write_json(
        task_dir / "score.json",
        {
            "schema_version": "2.0",
            "task_id": config.task_id,
            "category": config.category,
            "expected_status": config.expected_final_status,
            "actual_status": actual_status,
            "phase_reached": phase,
            "functional_correctness": functional_correctness,
            "outcome_accuracy": outcome_accuracy,
            "hidden_tests_passed": hidden_tests_passed,
            "hidden_tests_applicable": hidden_tests_applicable,
            "verification_report_present": report is not None,
            "patch_generated": patch_applied,
            "patch_applied": patch_applied,
            "details": {},
        },
    )
    prepare_calls, prepare_prompt, prepare_completion = prepare_usage
    write_json(
        task_dir / "prepare" / "prepare_summary.json",
        {
            "phase": "prepare",
            "model": "test-model",
            "llm_call_count": prepare_calls,
            "prompt_tokens": prepare_prompt,
            "completion_tokens": prepare_completion,
        },
    )
    if phase != "execute":
        return

    write_json(
        task_dir / "prepare" / "normalized_issue.json",
        {
            "acceptance_criteria": [
                {"id": criterion_id, "description": criterion_id}
                for criterion_id in criteria
            ]
        },
    )
    if report is not None:
        write_json(task_dir / "execute" / "verification_report.json", report)
    if run_summary is not None:
        write_json(task_dir / "execute" / "run_summary.json", run_summary)
    write_json(
        task_dir / "execute" / "acceptance_evidence.json",
        {
            "acceptance_evidence": [
                {"criterion_id": criterion_id, "status": status}
                for criterion_id, status in evidence.items()
            ]
        },
    )


def test_select_task_configs_resolves_evaluation_relative_paths(
    tmp_path: Path,
) -> None:
    tasks_dir = tmp_path / "evaluation" / "tasks"
    write_task(tasks_dir / "task-a", "task-a")
    write_task(tasks_dir / "task-b", "task-b")

    selected = select_task_configs(
        tasks_dir,
        ["tasks/task-a/task.json", "tasks/task-b/task.json"],
        task_id="task-b",
    )

    assert [config.task_id for config in selected] == ["task-b"]


def test_select_task_configs_returns_all_tasks_without_filter(
    tmp_path: Path,
) -> None:
    tasks_dir = tmp_path / "evaluation" / "tasks"
    write_task(tasks_dir / "task-a", "task-a")
    write_task(tasks_dir / "task-b", "task-b")

    selected = select_task_configs(
        tasks_dir,
        ["task-a/task.json", "task-b/task.json"],
    )

    assert [config.task_id for config in selected] == ["task-a", "task-b"]


def test_select_task_configs_rejects_unknown_task_id(tmp_path: Path) -> None:
    tasks_dir = tmp_path / "evaluation" / "tasks"
    write_task(tasks_dir / "task-a", "task-a")

    with pytest.raises(ValueError, match="Unknown task_id: missing-task"):
        select_task_configs(
            tasks_dir,
            ["task-a/task.json"],
            task_id="missing-task",
        )


def test_select_task_configs_rejects_duplicate_task_ids(tmp_path: Path) -> None:
    tasks_dir = tmp_path / "evaluation" / "tasks"
    write_task(tasks_dir / "first", "duplicate")
    write_task(tasks_dir / "second", "duplicate")

    with pytest.raises(ValueError, match="Duplicate task_id in index: duplicate"):
        select_task_configs(
            tasks_dir,
            ["first/task.json", "second/task.json"],
            task_id="duplicate",
        )


def test_main_runs_only_requested_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tasks_dir = tmp_path / "evaluation" / "tasks"
    write_task(tasks_dir / "task-a", "task-a")
    write_task(tasks_dir / "task-b", "task-b")
    (tasks_dir / "index.json").write_text(
        json.dumps(
            {
                "tasks": ["task-a/task.json", "task-b/task.json"],
                "defaults": {"max_rounds": 16, "max_repairs": 2},
            }
        ),
        encoding="utf-8",
    )

    execute_mock = Mock(
        return_value=RunResult(
            task_id="task-b",
            phase="execute",
            status="completed",
            prepare_success=True,
            execute_success=True,
            score_success=True,
        )
    )
    score_mock = Mock(
        return_value=ScoreResult(
            task_id="task-b",
            category="single_file_bug",
            expected_status="VERIFIED",
            actual_status="VERIFIED",
            phase_reached="execute",
            functional_correctness=1.0,
            outcome_accuracy=1.0,
            hidden_tests_passed=True,
            hidden_tests_applicable=True,
            verification_report_present=True,
            patch_generated=True,
            patch_applied=True,
        )
    )
    monkeypatch.setattr(runner, "execute_task", execute_mock)
    monkeypatch.setattr(runner, "score_task", score_mock)
    monkeypatch.setattr(runner, "save_score_result", Mock())
    monkeypatch.setattr(runner, "save_aggregate_results", Mock())
    monkeypatch.setattr(
        runner,
        "aggregate_scores",
        Mock(
            return_value={
                "total_tasks": 1,
                "completed_tasks": 1,
                "average_functional_correctness": 1.0,
                "average_outcome_accuracy": 1.0,
                "category_scores": {},
            }
        ),
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "runner.py",
            "--tasks",
            "evaluation/tasks",
            "--task-id",
            "task-b",
            "--model",
            "test-model",
        ],
    )

    runner.main()

    execute_mock.assert_called_once()
    selected_config = execute_mock.call_args.kwargs["task_config"]
    assert selected_config.task_id == "task-b"
    assert execute_mock.call_args.kwargs["evaluation_root"] == (
        tmp_path / "evaluation"
    )
    assert execute_mock.call_args.kwargs["project_root"] == tmp_path
    assert execute_mock.call_args.kwargs["variant"] == "patchpilot"


def test_score_task_resolves_patch_and_hidden_test_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = Path(__file__).parents[1]
    source_fixture = (
        project_root / "evaluation" / "fixtures" / "day5_python_repo"
    )
    evaluation_root = tmp_path / "evaluation"
    fixture = evaluation_root / "fixtures" / "day5_python_repo"
    shutil.copytree(source_fixture, fixture)

    task_dir = evaluation_root / "tasks" / "relative-path-task"
    task_dir.mkdir(parents=True)
    hidden_check = task_dir / "hidden_check.py"
    hidden_check.write_text(
        "from benchmark.booleans import parse_bool\n"
        "\n"
        "\n"
        "def test_surrounding_whitespace() -> None:\n"
        "    assert parse_bool('  YES  ') is True\n",
        encoding="utf-8",
    )

    timestamp = "relative-path-run"
    execute_dir = (
        evaluation_root
        / "runs"
        / timestamp
        / "relative-path-task"
        / "execute"
    )
    execute_dir.mkdir(parents=True)
    (execute_dir / "verification_report.json").write_text(
        "{}",
        encoding="utf-8",
    )
    (execute_dir / "run_summary.json").write_text(
        json.dumps({"final_status": "VERIFIED"}),
        encoding="utf-8",
    )
    (execute_dir / "patch.diff").write_text(
        "diff --git a/benchmark/booleans.py b/benchmark/booleans.py\n"
        "index 996992d..de4b8ae 100644\n"
        "--- a/benchmark/booleans.py\n"
        "+++ b/benchmark/booleans.py\n"
        "@@ -3,7 +3,7 @@\n"
        " \n"
        " def parse_bool(value: str) -> bool:\n"
        "     \"\"\"Parse a supported textual boolean value.\"\"\"\n"
        "-    normalized = value.lower()\n"
        "+    normalized = value.strip().lower()\n"
        "     if normalized in {\"true\", \"yes\", \"1\"}:\n"
        "         return True\n"
        "     if normalized in {\"false\", \"no\", \"0\"}:\n",
        encoding="utf-8",
    )
    config = TaskConfig(
        task_id="relative-path-task",
        category="single_file_bug",
        repository="fixtures/day5_python_repo",
        base_commit="e32138dad45ca3652677aa9aaef4417975047d0e",  # Use actual fixture commit
        issue="tasks/relative-path-task/issue.md",
        expected_final_status="VERIFIED",
        allowed_changes=["benchmark/booleans.py"],
        target_tests=["tests/test_booleans.py"],
        score_commands=[
            "python -m pytest -q -p no:cacheprovider {task_dir}/hidden_check.py"
        ],
    )
    run_result = RunResult(
        task_id="relative-path-task",
        phase="execute",
        status="completed",
        prepare_success=True,
        execute_success=True,
        score_success=False,
    )

    monkeypatch.chdir(tmp_path)
    result = runner.score_task(
        task_config=config,
        run_result=run_result,
        evaluation_root=Path("evaluation"),
        timestamp=timestamp,
    )

    assert result.actual_status == "VERIFIED"
    assert result.hidden_tests_passed is True
    assert result.functional_correctness == 1.0
    assert result.outcome_accuracy == 1.0


def test_score_task_preserves_failed_status_without_verification_report(
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).parents[1]
    evaluation_root = tmp_path / "evaluation"
    shutil.copytree(
        project_root / "evaluation" / "fixtures" / "day5_python_repo",
        evaluation_root / "fixtures" / "day5_python_repo",
    )
    timestamp = "failed-run"
    execute_dir = (
        evaluation_root
        / "runs"
        / timestamp
        / "failed-task"
        / "execute"
    )
    write_json(
        execute_dir / "run_summary.json",
        {
            "final_status": "FAILED",
            "failure_type": "AGENT_ERROR",
        },
    )
    config = TaskConfig(
        task_id="failed-task",
        category="single_file_bug",
        repository="fixtures/day5_python_repo",
        base_commit="e32138dad45ca3652677aa9aaef4417975047d0e",  # Use actual fixture commit
        issue="tasks/failed-task/issue.md",
        expected_final_status="VERIFIED",
        allowed_changes=[],
        target_tests=[],
        score_commands=[],
    )
    result = runner.score_task(
        task_config=config,
        run_result=RunResult(
            task_id="failed-task",
            phase="execute",
            status="patch_not_generated",
            prepare_success=True,
            execute_success=True,
            score_success=False,
            actual_status="FAILED",
            failure_type="AGENT_ERROR",
        ),
        evaluation_root=evaluation_root,
        timestamp=timestamp,
    )

    assert result.actual_status == "FAILED"
    assert result.verification_report_present is False
    assert result.patch_generated is False
    assert result.functional_correctness == 0.0
    assert result.outcome_accuracy == 0.0
    assert result.details["run_status"] == "patch_not_generated"
    assert result.details["failure_type"] == "AGENT_ERROR"
    runner.save_score_result(execute_dir.parent, result)
    saved = json.loads(
        (execute_dir.parent / "score.json").read_text(encoding="utf-8")
    )
    assert saved["actual_status"] == "FAILED"
    assert saved["verification_report_present"] is False
    assert saved["patch_generated"] is False
    assert saved["functional_correctness"] == 0.0
    assert saved["outcome_accuracy"] == 0.0


def test_extract_prepare_status_maps_null_final_status() -> None:
    assert runner.extract_prepare_status(
        {
            "outcome_code": "READY_FOR_APPROVAL",
            "final_status": None,
        }
    ) == "READY_FOR_APPROVAL"


def test_aggregate_scores_calculates_deterministic_metrics(
    tmp_path: Path,
) -> None:
    evaluation_root = tmp_path / "evaluation"
    timestamp = "metrics-run"
    runs_dir = evaluation_root / "runs" / timestamp
    fix = make_task_config("fix", "single_file_bug", "VERIFIED")
    retry = make_task_config("retry", "repair_loop", "VERIFIED")
    unsafe = make_task_config("unsafe", "unsafe_request", "BLOCKED")
    environment = make_task_config(
        "environment",
        "environment_failure",
        "BLOCKED",
    )

    passing_report = {
        "passed": True,
        "checks": [
            {"level": "LEVEL_3_REGRESSION", "passed": True},
        ],
    }
    write_metric_task(
        runs_dir,
        fix,
        actual_status="VERIFIED",
        phase="execute",
        report=passing_report,
        run_summary={
            "duration_seconds": 10.0,
            "retry_count": 0,
            "llm_call_count": 3,
            "prompt_tokens": 20,
            "completion_tokens": 10,
        },
        criteria=["AC-1", "AC-2"],
        evidence={"AC-1": "PASS"},
        prepare_usage=(2, 10, 5),
        functional_correctness=1.0,
        outcome_accuracy=1.0,
        hidden_tests_passed=True,
        hidden_tests_applicable=True,
        patch_applied=True,
    )
    write_metric_task(
        runs_dir,
        retry,
        actual_status="VERIFIED",
        phase="execute",
        report=passing_report,
        run_summary={
            "duration_seconds": 20.0,
            "retry_count": 1,
            "llm_call_count": 4,
            "prompt_tokens": 30,
            "completion_tokens": 15,
        },
        criteria=["AC-1"],
        evidence={"AC-1": "PASS"},
        prepare_usage=(2, 10, 5),
        functional_correctness=1.0,
        outcome_accuracy=1.0,
        hidden_tests_passed=True,
        hidden_tests_applicable=True,
        patch_applied=True,
    )
    write_metric_task(
        runs_dir,
        unsafe,
        actual_status="BLOCKED",
        phase="prepare",
        report=None,
        run_summary=None,
        criteria=[],
        evidence={},
        prepare_usage=(1, 5, 2),
        functional_correctness=0.0,  # Not applicable for prepare-only
        outcome_accuracy=1.0,
        hidden_tests_passed=False,
        hidden_tests_applicable=False,
        patch_applied=False,
    )
    write_metric_task(
        runs_dir,
        environment,
        actual_status="BLOCKED",
        phase="execute",
        report={"passed": False, "checks": []},
        run_summary={
            "duration_seconds": 5.0,
            "retry_count": 0,
            "llm_call_count": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        },
        criteria=["AC-1"],
        evidence={"AC-1": "FAIL"},
        prepare_usage=(2, 8, 4),
        functional_correctness=0.0,
        outcome_accuracy=1.0,
        hidden_tests_passed=False,
        hidden_tests_applicable=True,
        patch_applied=False,
    )

    aggregate = aggregate_scores(
        evaluation_root,
        timestamp,
        task_configs=[fix, retry, unsafe, environment],
    )
    metrics = aggregate["metrics"]

    # New separated metrics
    # functional_correctness_rate: 2/3 = 0.67 (fix and retry have hidden tests, environment doesn't count)
    assert metrics["functional_correctness_rate"]["value"] == pytest.approx(2/3)
    assert metrics["outcome_accuracy_rate"]["value"] == 1.0
    assert metrics["false_verified_rate"]["value"] == 0.0
    # patch_applicability_rate: 2/3 = 0.67 (fix and retry applied, environment didn't)
    assert metrics["patch_applicability_rate"]["value"] == pytest.approx(2/3)
    
    # Legacy metrics (should still work)
    assert metrics["expected_outcome_match_rate"]["value"] == 1.0
    assert metrics["verified_task_rate"]["value"] == 1.0
    assert metrics["verifier_pass_rate"] == {
        "value": 2 / 3,
        "numerator": 2,
        "denominator": 3,
        "missing_count": 0,
    }
    assert metrics["acceptance_criteria_coverage"]["value"] == 0.5
    assert metrics["acceptance_criteria_coverage"][
        "missing_evidence_count"
    ] == 1
    assert metrics["regression_pass_rate"]["value"] == 1.0
    assert metrics["retry_recovery_rate"]["value"] == 1.0
    assert metrics["unsafe_action_block_rate"]["value"] == 1.0
    assert metrics["average_execute_duration_seconds"]["value"] == pytest.approx(
        35 / 3
    )
    assert metrics["average_llm_call_count"]["value"] == 3.5
    assert metrics["average_prompt_tokens"]["value"] == pytest.approx(83 / 4)
    assert metrics["average_completion_tokens"]["value"] == pytest.approx(41 / 4)
    assert metrics["average_total_tokens"]["value"] == pytest.approx(124 / 4)
    
    # Check aggregate-level averages
    assert aggregate["average_functional_correctness"] == pytest.approx(0.5)
    assert aggregate["average_outcome_accuracy"] == 1.0


def test_empty_metric_denominator_is_not_reported_as_zero(tmp_path: Path) -> None:
    runs_dir = tmp_path / "evaluation" / "runs" / "empty-run"
    runs_dir.mkdir(parents=True)

    aggregate = aggregate_scores(tmp_path / "evaluation", "empty-run")

    for metric in aggregate["metrics"].values():
        assert metric["value"] is None


def test_missing_token_usage_is_not_treated_as_zero(tmp_path: Path) -> None:
    evaluation_root = tmp_path / "evaluation"
    timestamp = "missing-usage-run"
    config = make_task_config("unsafe", "unsafe_request", "BLOCKED")
    write_metric_task(
        evaluation_root / "runs" / timestamp,
        config,
        actual_status="BLOCKED",
        phase="prepare",
        report=None,
        run_summary=None,
        criteria=[],
        evidence={},
        prepare_usage=(1, None, None),
    )

    aggregate = aggregate_scores(
        evaluation_root,
        timestamp,
        task_configs=[config],
    )

    assert aggregate["metrics"]["average_llm_call_count"]["value"] == 1.0
    assert aggregate["metrics"]["average_total_tokens"] == {
        "value": None,
        "count": 0,
        "missing_count": 1,
    }


def test_verified_with_passing_hidden_tests(tmp_path: Path) -> None:
    """Test VERIFIED status with passing hidden tests yields full functional correctness."""
    evaluation_root = tmp_path / "evaluation"
    timestamp = "verified-passing-run"
    config = make_task_config("fix", "single_file_bug", "VERIFIED")
    write_metric_task(
        evaluation_root / "runs" / timestamp,
        config,
        actual_status="VERIFIED",
        phase="execute",
        report={"passed": True, "checks": []},
        run_summary={"final_status": "VERIFIED"},
        criteria=[],
        evidence={},
        prepare_usage=(2, 10, 5),
        functional_correctness=1.0,
        outcome_accuracy=1.0,
        hidden_tests_passed=True,
        hidden_tests_applicable=True,
        patch_applied=True,
    )

    aggregate = aggregate_scores(
        evaluation_root,
        timestamp,
        task_configs=[config],
    )

    assert aggregate["metrics"]["functional_correctness_rate"]["value"] == 1.0
    assert aggregate["metrics"]["outcome_accuracy_rate"]["value"] == 1.0
    assert aggregate["metrics"]["false_verified_rate"]["value"] == 0.0


def test_verified_with_failing_hidden_tests(tmp_path: Path) -> None:
    """Test VERIFIED status with failing hidden tests yields zero functional correctness."""
    evaluation_root = tmp_path / "evaluation"
    timestamp = "verified-failing-run"
    config = make_task_config("fix", "single_file_bug", "VERIFIED")
    write_metric_task(
        evaluation_root / "runs" / timestamp,
        config,
        actual_status="VERIFIED",
        phase="execute",
        report={"passed": True, "checks": []},
        run_summary={"final_status": "VERIFIED"},
        criteria=[],
        evidence={},
        prepare_usage=(2, 10, 5),
        functional_correctness=0.0,  # Hidden tests failed
        outcome_accuracy=1.0,  # Status matches expected
        hidden_tests_passed=False,
        hidden_tests_applicable=True,
        patch_applied=True,
    )

    aggregate = aggregate_scores(
        evaluation_root,
        timestamp,
        task_configs=[config],
    )

    assert aggregate["metrics"]["functional_correctness_rate"]["value"] == 0.0
    assert aggregate["metrics"]["outcome_accuracy_rate"]["value"] == 1.0
    assert aggregate["metrics"]["false_verified_rate"]["value"] == 1.0  # False VERIFIED


def test_wrong_status_with_passing_hidden_tests(tmp_path: Path) -> None:
    """Test wrong status with passing hidden tests yields zero outcome accuracy."""
    evaluation_root = tmp_path / "evaluation"
    timestamp = "wrong-status-run"
    config = make_task_config("fix", "single_file_bug", "VERIFIED")
    write_metric_task(
        evaluation_root / "runs" / timestamp,
        config,
        actual_status="FAILED",  # Wrong status
        phase="execute",
        report={"passed": True, "checks": []},
        run_summary={"final_status": "FAILED"},
        criteria=[],
        evidence={},
        prepare_usage=(2, 10, 5),
        functional_correctness=0.0,  # Should be 0 since status is wrong
        outcome_accuracy=0.0,  # Status doesn't match
        hidden_tests_passed=True,
        hidden_tests_applicable=True,
        patch_applied=True,
    )

    aggregate = aggregate_scores(
        evaluation_root,
        timestamp,
        task_configs=[config],
    )

    assert aggregate["metrics"]["functional_correctness_rate"]["value"] == 0.0
    assert aggregate["metrics"]["outcome_accuracy_rate"]["value"] == 0.0


def test_prepare_only_blocked(tmp_path: Path) -> None:
    """Test prepare-only task with expected BLOCKED status."""
    evaluation_root = tmp_path / "evaluation"
    timestamp = "prepare-blocked-run"
    config = make_task_config("unsafe", "unsafe_request", "BLOCKED")
    config.expected_phase = "prepare"
    write_metric_task(
        evaluation_root / "runs" / timestamp,
        config,
        actual_status="BLOCKED",
        phase="prepare",
        report=None,
        run_summary=None,
        criteria=[],
        evidence={},
        prepare_usage=(1, 5, 2),
        functional_correctness=0.0,  # Not applicable for prepare-only
        outcome_accuracy=1.0,  # Status matches expected
        hidden_tests_passed=False,
        hidden_tests_applicable=False,  # No hidden tests for prepare-only
        patch_applied=False,
    )

    aggregate = aggregate_scores(
        evaluation_root,
        timestamp,
        task_configs=[config],
    )

    # Functional correctness not applicable for prepare-only
    assert aggregate["metrics"]["functional_correctness_rate"]["value"] is None
    assert aggregate["metrics"]["outcome_accuracy_rate"]["value"] == 1.0


def test_prepare_only_needs_clarification(tmp_path: Path) -> None:
    """Test prepare-only task with expected NEEDS_CLARIFICATION status."""
    evaluation_root = tmp_path / "evaluation"
    timestamp = "prepare-clarification-run"
    config = make_task_config("ambiguous", "ambiguous_requirement", "NEEDS_CLARIFICATION")
    config.expected_phase = "prepare"
    write_metric_task(
        evaluation_root / "runs" / timestamp,
        config,
        actual_status="NEEDS_CLARIFICATION",
        phase="prepare",
        report=None,
        run_summary=None,
        criteria=[],
        evidence={},
        prepare_usage=(1, 5, 2),
        functional_correctness=0.0,  # Not applicable for prepare-only
        outcome_accuracy=1.0,  # Status matches expected
        hidden_tests_passed=False,
        hidden_tests_applicable=False,
        patch_applied=False,
    )

    aggregate = aggregate_scores(
        evaluation_root,
        timestamp,
        task_configs=[config],
    )

    assert aggregate["metrics"]["functional_correctness_rate"]["value"] is None
    assert aggregate["metrics"]["outcome_accuracy_rate"]["value"] == 1.0


def test_execute_no_hidden_tests(tmp_path: Path) -> None:
    """Test execute task with no hidden tests configured."""
    evaluation_root = tmp_path / "evaluation"
    timestamp = "no-hidden-tests-run"
    config = make_task_config("fix", "single_file_bug", "VERIFIED")
    config.score_commands = []  # No hidden tests
    write_metric_task(
        evaluation_root / "runs" / timestamp,
        config,
        actual_status="VERIFIED",
        phase="execute",
        report={"passed": True, "checks": []},
        run_summary={"final_status": "VERIFIED"},
        criteria=[],
        evidence={},
        prepare_usage=(2, 10, 5),
        functional_correctness=1.0,  # Patch applied successfully
        outcome_accuracy=1.0,
        hidden_tests_passed=False,  # Not applicable
        hidden_tests_applicable=False,  # No hidden tests configured
        patch_applied=True,
    )

    aggregate = aggregate_scores(
        evaluation_root,
        timestamp,
        task_configs=[config],
    )

    # Should not count as hidden test eligible
    assert aggregate["metrics"]["functional_correctness_rate"]["value"] is None
    assert aggregate["metrics"]["outcome_accuracy_rate"]["value"] == 1.0


def test_aggregate_false_verified_metrics(tmp_path: Path) -> None:
    """Test aggregation of false VERIFIED metrics."""
    evaluation_root = tmp_path / "evaluation"
    timestamp = "false-verified-run"
    
    # Task 1: True VERIFIED (functional correctness = 1)
    task1 = make_task_config("fix1", "single_file_bug", "VERIFIED")
    write_metric_task(
        evaluation_root / "runs" / timestamp,
        task1,
        actual_status="VERIFIED",
        phase="execute",
        report={"passed": True, "checks": []},
        run_summary={"final_status": "VERIFIED"},
        criteria=[],
        evidence={},
        prepare_usage=(2, 10, 5),
        functional_correctness=1.0,
        outcome_accuracy=1.0,
        hidden_tests_passed=True,
        hidden_tests_applicable=True,
        patch_applied=True,
    )
    
    # Task 2: False VERIFIED (reported VERIFIED but hidden tests failed)
    task2 = make_task_config("fix2", "single_file_bug", "VERIFIED")
    write_metric_task(
        evaluation_root / "runs" / timestamp,
        task2,
        actual_status="VERIFIED",
        phase="execute",
        report={"passed": True, "checks": []},
        run_summary={"final_status": "VERIFIED"},
        criteria=[],
        evidence={},
        prepare_usage=(2, 10, 5),
        functional_correctness=0.0,  # Hidden tests failed
        outcome_accuracy=1.0,
        hidden_tests_passed=False,
        hidden_tests_applicable=True,
        patch_applied=True,
    )
    
    # Task 3: Correctly reported FAILED
    task3 = make_task_config("fix3", "single_file_bug", "VERIFIED")
    write_metric_task(
        evaluation_root / "runs" / timestamp,
        task3,
        actual_status="FAILED",
        phase="execute",
        report={"passed": False, "checks": []},
        run_summary={"final_status": "FAILED"},
        criteria=[],
        evidence={},
        prepare_usage=(2, 10, 5),
        functional_correctness=0.0,
        outcome_accuracy=0.0,
        hidden_tests_passed=False,
        hidden_tests_applicable=True,
        patch_applied=False,
    )

    aggregate = aggregate_scores(
        evaluation_root,
        timestamp,
        task_configs=[task1, task2, task3],
    )

    # Functional correctness: 1/3 = 0.33
    assert aggregate["metrics"]["functional_correctness_rate"]["value"] == pytest.approx(1/3)
    # Outcome accuracy: 2/3 = 0.67 (task1 and task2 have correct status)
    assert aggregate["metrics"]["outcome_accuracy_rate"]["value"] == pytest.approx(2/3)
    # False VERIFIED: 1/3 = 0.33 (task2 is false VERIFIED out of 3 VERIFIED-expected tasks)
    assert aggregate["metrics"]["false_verified_rate"]["value"] == pytest.approx(1/3)


def test_missing_artifacts_distinguished_from_zero(tmp_path: Path) -> None:
    """Test that missing artifacts are distinguished from genuine zero scores."""
    evaluation_root = tmp_path / "evaluation"
    timestamp = "missing-artifacts-run"
    
    # Task with genuine zero score
    task_zero = make_task_config("zero", "single_file_bug", "VERIFIED")
    write_metric_task(
        evaluation_root / "runs" / timestamp,
        task_zero,
        actual_status="FAILED",
        phase="execute",
        report={"passed": False, "checks": []},
        run_summary={"final_status": "FAILED"},
        criteria=[],
        evidence={},
        prepare_usage=(2, 10, 5),
        functional_correctness=0.0,
        outcome_accuracy=0.0,
        hidden_tests_passed=False,
        hidden_tests_applicable=True,
        patch_applied=False,
    )
    
    # Missing score.json (no artifact)
    task_missing = make_task_config("missing", "single_file_bug", "VERIFIED")
    task_dir = evaluation_root / "runs" / timestamp / "missing"
    task_dir.mkdir(parents=True)
    # Don't write score.json - artifact is missing

    aggregate = aggregate_scores(
        evaluation_root,
        timestamp,
        task_configs=[task_zero, task_missing],
    )

    # Should count one completed task with zero score
    assert aggregate["completed_tasks"] == 1
    assert aggregate["missing_score_count"] == 1
    assert aggregate["average_functional_correctness"] == 0.0
    assert aggregate["average_outcome_accuracy"] == 0.0


def test_execute_task_materializes_expected_git_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = Path(__file__).parents[1]
    source_fixture = (
        project_root / "evaluation" / "fixtures" / "day5_python_repo"
    )
    evaluation_root = tmp_path / "evaluation"
    fixture = evaluation_root / "fixtures" / "day5_python_repo"
    shutil.copytree(source_fixture, fixture)
    issue = evaluation_root / "tasks" / "prepare-only" / "issue.md"
    issue.parent.mkdir(parents=True)
    issue.write_text("# Ambiguous task\n", encoding="utf-8")

    prepare = Mock()

    def fake_prepare(**kwargs: object) -> subprocess.CompletedProcess[str]:
        repo = kwargs["repo"]
        output_dir = kwargs["output_dir"]
        assert isinstance(repo, Path)
        assert isinstance(output_dir, Path)
        # Don't check base commit since we're using a fake materialize
        write_json(
            output_dir / "prepare_summary.json",
            {
                "phase": "prepare",
                "outcome_code": "AMBIGUOUS_REQUIREMENT",
                "final_status": "NEEDS_CLARIFICATION",
                "exit_code": 1,
                "reasons": ["Priority ordering is unspecified."],
            },
        )
        prepare(repo=repo)
        return subprocess.CompletedProcess(
            args=["patchpilot", "prepare"],
            returncode=1,
            stdout="PatchPilot will not guess product behavior.",
            stderr="",
        )

    monkeypatch.setattr(
        "evaluation.runner.run_patchpilot_prepare",
        fake_prepare,
    )
    # Mock materialize to avoid actual git operations
    def fake_materialize(source: Path, dest: Path, commit: str) -> None:
        dest.mkdir(parents=True, exist_ok=True)
    
    monkeypatch.setattr(
        "evaluation.runner.materialize",
        fake_materialize,
    )
    # Mock verify_base_commit to return True
    monkeypatch.setattr(
        "evaluation.runner.verify_base_commit",
        lambda *args, **kwargs: True,
    )
    config = TaskConfig(
        task_id="prepare-only",
        category="ambiguous_requirement",
        repository="fixtures/day5_python_repo",
        base_commit="e32138dad45ca3652677aa9aaef4417975047d0e",  # Use actual fixture commit
        issue="tasks/prepare-only/issue.md",
        expected_final_status="NEEDS_CLARIFICATION",
        allowed_changes=[],
        target_tests=[],
        score_commands=[],
        expected_phase="prepare",
    )

    result = execute_task(
        task_config=config,
        evaluation_root=evaluation_root,
        project_root=project_root,
        model="test-model",
        max_rounds=4,
        max_repairs=1,
        timestamp="test-run",
    )

    assert result.status == "stopped_at_prepare"
    assert result.outcome_matched is True
    prepare.assert_called_once()


def test_execute_task_uses_summary_for_prepare_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = Path(__file__).parents[1]
    evaluation_root = tmp_path / "evaluation"
    shutil.copytree(
        project_root / "evaluation" / "fixtures" / "day5_python_repo",
        evaluation_root / "fixtures" / "day5_python_repo",
    )
    issue = evaluation_root / "tasks" / "plan-invalid" / "issue.md"
    issue.parent.mkdir(parents=True)
    issue.write_text("# Add a feature\n", encoding="utf-8")

    def fake_prepare(**kwargs: object) -> subprocess.CompletedProcess[str]:
        output_dir = kwargs["output_dir"]
        assert isinstance(output_dir, Path)
        write_json(
            output_dir / "prepare_summary.json",
            {
                "outcome_code": "PLAN_INVALID",
                "final_status": "BLOCKED",
                "reasons": ["AC-2 has no planned source change."],
            },
        )
        return subprocess.CompletedProcess(
            args=["patchpilot", "prepare"],
            returncode=1,
            stdout="",
            stderr="Plan validation failed",
        )

    monkeypatch.setattr(
        "evaluation.runner.run_patchpilot_prepare",
        fake_prepare,
    )
    execute = Mock()
    monkeypatch.setattr(
        "evaluation.runner.run_patchpilot_execute",
        execute,
    )
    # Mock materialize to avoid actual git operations
    def fake_materialize(source: Path, dest: Path, commit: str) -> None:
        dest.mkdir(parents=True, exist_ok=True)
    
    monkeypatch.setattr(
        "evaluation.runner.materialize",
        fake_materialize,
    )
    # Mock verify_base_commit to return True
    monkeypatch.setattr(
        "evaluation.runner.verify_base_commit",
        lambda *args, **kwargs: True,
    )
    config = TaskConfig(
        task_id="plan-invalid",
        category="small_feature",
        repository="fixtures/day5_python_repo",
        base_commit="e32138dad45ca3652677aa9aaef4417975047d0e",  # Use actual fixture commit
        issue="tasks/plan-invalid/issue.md",
        expected_final_status="VERIFIED",
        allowed_changes=[],
        target_tests=[],
        score_commands=[],
    )

    result = execute_task(
        task_config=config,
        evaluation_root=evaluation_root,
        project_root=project_root,
        model="test-model",
        max_rounds=4,
        max_repairs=1,
        timestamp="prepare-failure-run",
    )

    assert result.status == "prepare_failed"
    assert result.actual_status == "BLOCKED"
    assert result.failure_type == "PLAN_INVALID"
    assert result.outcome_matched is False
    execute.assert_not_called()


def test_execute_task_baseline_skips_prepare_and_uses_run_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that baseline tasks execute raw issues without prepare."""
    project_root = Path(__file__).parents[1]
    source_fixture = (
        project_root / "evaluation" / "fixtures" / "day5_python_repo"
    )
    evaluation_root = tmp_path / "evaluation"
    fixture = evaluation_root / "fixtures" / "day5_python_repo"
    shutil.copytree(source_fixture, fixture)
    issue = evaluation_root / "tasks" / "baseline-task" / "issue.md"
    issue.parent.mkdir(parents=True)
    issue.write_text("# Raw baseline task\n", encoding="utf-8")

    def fake_baseline(**kwargs: object) -> subprocess.CompletedProcess[str]:
        output_dir = kwargs["output_dir"]
        assert isinstance(output_dir, Path)
        write_json(
            output_dir / "run_summary.json",
            {"phase": "execute", "final_status": "FAILED"},
        )
        return subprocess.CompletedProcess(
            args=["patchpilot", "baseline"],
            returncode=1,
            stdout="baseline stdout",
            stderr="baseline stderr",
        )

    # Mock materialize to avoid actual git operations
    def fake_materialize(source: Path, dest: Path, commit: str) -> None:
        dest.mkdir(parents=True, exist_ok=True)
    
    monkeypatch.setattr(
        "evaluation.runner.materialize",
        fake_materialize,
    )
    # Mock verify_base_commit to return True
    monkeypatch.setattr(
        "evaluation.runner.verify_base_commit",
        lambda *args, **kwargs: True,
    )
    
    baseline = Mock(side_effect=fake_baseline)
    monkeypatch.setattr("evaluation.runner.run_patchpilot_baseline", baseline)
    prepare = Mock()
    monkeypatch.setattr("evaluation.runner.run_patchpilot_prepare", prepare)
    
    config = TaskConfig(
        task_id="baseline-task",
        category="ambiguous_requirement",
        repository="fixtures/day5_python_repo",
        base_commit="e32138dad45ca3652677aa9aaef4417975047d0e",  # Use actual fixture commit
        issue="tasks/baseline-task/issue.md",
        expected_final_status="FAILED",  # Baseline expects actual status
        allowed_changes=[],
        target_tests=[],
        score_commands=[],
        expected_phase="execute",  # Baseline always goes to execute
    )

    result = execute_task(
        task_config=config,
        evaluation_root=evaluation_root,
        project_root=project_root,
        model="test-model",
        max_rounds=4,
        max_repairs=0,
        timestamp="baseline-run",
        variant="baseline",
    )

    assert result.phase == "execute"
    assert result.execute_success is True
    assert result.actual_status == "FAILED"
    assert result.outcome_matched is True  # Actual FAILED matches expected FAILED
    prepare.assert_not_called()
    baseline.assert_called_once()
    execute_dir = (
        evaluation_root / "runs" / "baseline-run" / "baseline-task" / "execute"
    )
    assert (execute_dir / "stdout.log").read_text() == "baseline stdout"
    assert (execute_dir / "stderr.log").read_text() == "baseline stderr"
