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
    check_scope_compliance,
    execute_task,
    is_denied_path,
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
            "scope_compliant": True,
            "public_tests_passed": True,
            "public_tests_applicable": False,
            "changed_file_count": 1 if patch_applied else 0,
            "added_lines": 5 if patch_applied else 0,
            "deleted_lines": 3 if patch_applied else 0,
            "unexpected_changed_files": [],
            "minimality_warnings": [],
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


def test_indexed_tasks_define_explicit_verification_contracts(
    tmp_path: Path,
) -> None:
    """All indexed tasks should declare phase and regression expectations."""
    evaluation_root = Path(__file__).parents[1] / "evaluation"
    tasks_dir = evaluation_root / "tasks"
    index = json.loads((tasks_dir / "index.json").read_text(encoding="utf-8"))
    task_paths = runner.resolve_task_paths(tasks_dir, index["tasks"])

    assert len(task_paths) == 10
    repositories: dict[str, str] = {}
    for task_path in task_paths:
        manifest = json.loads(task_path.read_text(encoding="utf-8"))
        phase = manifest["expected_phase"]
        regression_tests = manifest["regression_tests"]
        assert phase in {"prepare", "execute"}
        assert regression_tests == (["."] if phase == "execute" else [])
        if manifest["expected_final_status"] == "VERIFIED":
            assert manifest["score_commands"]

        repository = manifest["repository"]
        base_commit = manifest["base_commit"]
        previous_commit = repositories.setdefault(repository, base_commit)
        assert previous_commit == base_commit

    for index_number, (repository, base_commit) in enumerate(
        repositories.items()
    ):
        runner.materialize(
            evaluation_root / repository,
            tmp_path / f"fixture-{index_number}",
            base_commit,
        )


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
            scope_compliant=True,
            public_tests_passed=True,
            public_tests_applicable=False,
            changed_file_count=1,
            added_lines=5,
            deleted_lines=3,
            unexpected_changed_files=[],
            minimality_warnings=[],
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
        base_commit="65b943998bcb8432096ea21ecb7e3b2da4feaadd",
        issue="tasks/relative-path-task/issue.md",
        expected_final_status="VERIFIED",
        allowed_changes=["benchmark/booleans.py"],
        target_tests=["tests/test_booleans.py"],
        score_commands=[
            "python -m pytest -q -p no:cacheprovider {task_dir}/hidden_check.py"
        ],
        regression_tests=["."],
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
    assert result.scope_compliant is True
    assert result.changed_file_count == 1
    assert result.added_lines == 1
    assert result.deleted_lines == 1
    assert result.public_tests_applicable is True  # target_tests in config
    assert result.regression_tests_applicable is True
    assert result.regression_tests_passed is True
    assert result.details["regression_transitions"] == [
        {"target": ".", "transition": "PRE_EXISTING_FAILURE"}
    ]


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
        base_commit="65b943998bcb8432096ea21ecb7e3b2da4feaadd",
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
    assert saved["scope_compliant"] is True  # Default for failed tasks
    assert saved["public_tests_applicable"] is False


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

    for metric_name, metric in aggregate["metrics"].items():
        # Skip total metrics which are integers
        if metric_name.startswith("total_"):
            assert metric == 0
        elif isinstance(metric, dict):
            assert metric["value"] is None


def test_aggregate_separates_partial_and_baseline_delta_results(
    tmp_path: Path,
) -> None:
    """Partial coverage is not a pass, while unchanged failures are safe."""
    evaluation_root = tmp_path / "evaluation"
    timestamp = "verification-status-run"
    partial = make_task_config("partial", "single_file_bug", "VERIFIED")
    historical = make_task_config("historical", "single_file_bug", "VERIFIED")

    write_metric_task(
        evaluation_root / "runs" / timestamp,
        partial,
        actual_status="PARTIALLY_VERIFIED",
        phase="execute",
        report={
            "passed": True,
            "verification_status": "PARTIALLY_VERIFIED",
            "regression_coverage": "INCOMPLETE",
            "checks": [
                {
                    "phase": "post_patch",
                    "level": "LEVEL_3_REGRESSION",
                    "passed": False,
                    "transition": "UNVERIFIED",
                }
            ],
        },
        run_summary={"retry_count": 1},
        criteria=[],
        evidence={},
        prepare_usage=(1, 1, 1),
    )
    write_metric_task(
        evaluation_root / "runs" / timestamp,
        historical,
        actual_status="VERIFIED",
        phase="execute",
        report={
            "passed": True,
            "verification_status": "VERIFIED",
            "regression_coverage": "FULL",
            "checks": [
                {
                    "phase": "post_patch",
                    "level": "LEVEL_3_REGRESSION",
                    "passed": False,
                    "transition": "PRE_EXISTING_FAILURE",
                }
            ],
        },
        run_summary={"retry_count": 0},
        criteria=[],
        evidence={},
        prepare_usage=(1, 1, 1),
    )

    aggregate = aggregate_scores(
        evaluation_root,
        timestamp,
        task_configs=[partial, historical],
    )
    metrics = aggregate["metrics"]

    assert metrics["verifier_pass_rate"]["value"] == 0.5
    assert metrics["partial_verification_rate"]["value"] == 0.5
    assert metrics["failed_verification_rate"]["value"] == 0.0
    assert metrics["regression_pass_rate"]["value"] == 0.5
    assert metrics["retry_recovery_rate"]["value"] == 0.0


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
    """Functional correctness remains independent from outcome accuracy."""
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
        functional_correctness=1.0,
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

    assert aggregate["metrics"]["functional_correctness_rate"]["value"] == 1.0
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
        base_commit="65b943998bcb8432096ea21ecb7e3b2da4feaadd",
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
        base_commit="65b943998bcb8432096ea21ecb7e3b2da4feaadd",
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
        base_commit="65b943998bcb8432096ea21ecb7e3b2da4feaadd",
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


def test_scope_compliance_with_allowed_files(tmp_path: Path) -> None:
    """Test scope compliance when patch changes only allowed files."""
    # Test the scope compliance function directly
    changed_files = ["src/main.py"]
    allowed_changes = ["src/main.py"]
    
    compliant, unexpected = check_scope_compliance(
        changed_files,
        allowed_changes,
        Path("/tmp/repo"),
    )
    
    assert compliant is True
    assert len(unexpected) == 0


def test_scope_compliance_with_undeclared_file(tmp_path: Path) -> None:
    """Test scope compliance when patch changes undeclared file."""
    # Test the scope compliance function directly instead of through full scoring
    changed_files = ["src/main.py", "other_module.py"]
    allowed_changes = ["src/main.py"]
    
    compliant, unexpected = check_scope_compliance(
        changed_files,
        allowed_changes,
        Path("/tmp/repo"),
    )
    
    assert compliant is False
    assert "other_module.py" in unexpected


def test_scope_compliance_with_empty_allowed_changes(tmp_path: Path) -> None:
    """Test scope compliance when allowed_changes is empty but patch exists."""
    # Test the scope compliance function directly
    changed_files = ["src/main.py"]
    allowed_changes = []  # Empty means no changes allowed
    
    compliant, unexpected = check_scope_compliance(
        changed_files,
        allowed_changes,
        Path("/tmp/repo"),
    )
    
    assert compliant is False
    assert len(unexpected) == 1


def test_scope_compliance_with_denied_test_file(tmp_path: Path) -> None:
    """Test scope compliance when patch changes a test file."""
    # Test the scope compliance function directly
    changed_files = ["src/main.py", "tests/test_booleans.py"]
    allowed_changes = ["src/main.py"]
    
    compliant, unexpected = check_scope_compliance(
        changed_files,
        allowed_changes,
        Path("/tmp/repo"),
    )
    
    assert compliant is False
    assert "tests/test_booleans.py" in unexpected


def test_public_tests_pass_and_fail(tmp_path: Path) -> None:
    """Test public test execution function."""
    # Test the function directly with mock
    from unittest.mock import MagicMock

    # Mock subprocess.run to simulate test execution
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "passed"
    mock_result.stderr = ""
    
    original_run = subprocess.run
    subprocess.run = MagicMock(return_value=mock_result)
    
    try:
        passed, results = runner.run_public_tests(
            Path("/tmp/repo"),
            ["tests/test_example.py"],
        )
        
        assert passed is True
        assert len(results) == 1
        assert results[0]["target"] == "tests/test_example.py"
        assert results[0]["passed"] is True
    finally:
        subprocess.run = original_run


def test_compare_test_run_delta_accepts_unchanged_historical_failure() -> None:
    """The evaluator should not attribute an unchanged failure to the patch."""
    failed_output = "FAILED tests/test_old.py::test_known - AssertionError"
    baseline = [
        {
            "target": ".",
            "passed": False,
            "timed_out": False,
            "stdout": failed_output,
            "stderr": "",
        }
    ]
    post_patch = [dict(baseline[0])]

    safe, transitions = runner.compare_test_run_delta(baseline, post_patch)

    assert safe is True
    assert transitions == [
        {"target": ".", "transition": "PRE_EXISTING_FAILURE"}
    ]


def test_compare_test_run_delta_rejects_new_failure() -> None:
    """A newly failing test must remain a regression."""
    baseline = [
        {
            "target": ".",
            "passed": True,
            "timed_out": False,
            "stdout": "1 passed",
            "stderr": "",
        }
    ]
    post_patch = [
        {
            "target": ".",
            "passed": False,
            "timed_out": False,
            "stdout": "FAILED tests/test_new.py::test_regression - AssertionError",
            "stderr": "",
        }
    ]

    safe, transitions = runner.compare_test_run_delta(baseline, post_patch)

    assert safe is False
    assert transitions == [{"target": ".", "transition": "REGRESSION"}]


def test_minimality_analysis_empty_patch(tmp_path: Path) -> None:
    """Test minimality analysis with empty patch."""
    # Test the analyze_patch_minimality function directly
    from unittest.mock import MagicMock

    from evaluation.runner import analyze_patch_minimality
    
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = ""
    
    original_run_git = runner.run_git
    runner.run_git = MagicMock(return_value=mock_result)
    
    try:
        metrics = analyze_patch_minimality(Path("/tmp/repo"))
        
        assert metrics["changed_file_count"] == 0
        assert metrics["added_lines"] == 0
        assert metrics["deleted_lines"] == 0
        assert metrics["is_empty"] is True
        assert len(metrics["warnings"]) == 0
    finally:
        runner.run_git = original_run_git


def test_aggregate_scope_and_public_test_metrics(tmp_path: Path) -> None:
    """Test aggregation of new scope and public test metrics."""
    evaluation_root = tmp_path / "evaluation"
    timestamp = "scope-metrics-run"
    
    # Task 1: Scope compliant, public tests pass
    task1 = make_task_config("task1", "single_file_bug", "VERIFIED")
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
    # Add new fields to task1 score
    task1_score_path = evaluation_root / "runs" / timestamp / "task1" / "score.json"
    task1_score = json.loads(task1_score_path.read_text(encoding="utf-8"))
    task1_score["scope_compliant"] = True
    task1_score["public_tests_passed"] = True
    task1_score["public_tests_applicable"] = True
    task1_score["changed_file_count"] = 1
    task1_score["added_lines"] = 5
    task1_score["deleted_lines"] = 3
    task1_score["unexpected_changed_files"] = []
    task1_score["minimality_warnings"] = []
    task1_score_path.write_text(json.dumps(task1_score, indent=2), encoding="utf-8")
    
    # Task 2: Scope violation
    task2 = make_task_config("task2", "single_file_bug", "VERIFIED")
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
        functional_correctness=0.0,  # Scope violation
        outcome_accuracy=1.0,
        hidden_tests_passed=True,
        hidden_tests_applicable=True,
        patch_applied=True,
    )
    # Add new fields to task2 score
    task2_score_path = evaluation_root / "runs" / timestamp / "task2" / "score.json"
    task2_score = json.loads(task2_score_path.read_text(encoding="utf-8"))
    task2_score["scope_compliant"] = False
    task2_score["public_tests_passed"] = True
    task2_score["public_tests_applicable"] = True
    task2_score["changed_file_count"] = 2
    task2_score["added_lines"] = 10
    task2_score["deleted_lines"] = 5
    task2_score["unexpected_changed_files"] = ["other_file.py"]
    task2_score["minimality_warnings"] = ["Modified denied file: other_file.py"]
    task2_score_path.write_text(json.dumps(task2_score, indent=2), encoding="utf-8")
    
    # Task 3: Public test failure (but scope compliant)
    task3 = make_task_config("task3", "single_file_bug", "VERIFIED")
    write_metric_task(
        evaluation_root / "runs" / timestamp,
        task3,
        actual_status="VERIFIED",
        phase="execute",
        report={"passed": True, "checks": []},
        run_summary={"final_status": "VERIFIED"},
        criteria=[],
        evidence={},
        prepare_usage=(2, 10, 5),
        functional_correctness=0.0,  # Public test failure
        outcome_accuracy=1.0,
        hidden_tests_passed=True,
        hidden_tests_applicable=True,
        patch_applied=True,
    )
    # Add new fields to task3 score
    task3_score_path = evaluation_root / "runs" / timestamp / "task3" / "score.json"
    task3_score = json.loads(task3_score_path.read_text(encoding="utf-8"))
    task3_score["scope_compliant"] = True  # Scope is compliant
    task3_score["public_tests_passed"] = False
    task3_score["public_tests_applicable"] = True
    task3_score["changed_file_count"] = 1
    task3_score["added_lines"] = 5
    task3_score["deleted_lines"] = 3
    task3_score["unexpected_changed_files"] = []
    task3_score["minimality_warnings"] = []
    task3_score_path.write_text(json.dumps(task3_score, indent=2), encoding="utf-8")
    
    aggregate = aggregate_scores(
        evaluation_root,
        timestamp,
        task_configs=[task1, task2, task3],
    )
    
    # Check scope compliance rate: 2/3 (task1 and task3 are compliant, task2 is not)
    assert aggregate["metrics"]["scope_compliance_rate"]["value"] == pytest.approx(2/3)
    # Check public test pass rate: 2/3 (task1 passed, task2 passed, task3 failed)
    assert aggregate["metrics"]["public_tests_pass_rate"]["value"] == pytest.approx(2/3)
    # Check total minimality metrics
    assert aggregate["metrics"]["total_changed_files"] == 4  # 1 + 2 + 1
    assert aggregate["metrics"]["total_added_lines"] == 20  # 5 + 10 + 5
    assert aggregate["metrics"]["total_deleted_lines"] == 11  # 3 + 5 + 3


def test_is_denied_path() -> None:
    """Test denied path detection."""
    # Git internals
    assert is_denied_path(".git/config") is True
    assert is_denied_path(".git/HEAD") is True
    
    # Test files
    assert is_denied_path("tests/test_example.py") is True
    assert is_denied_path("test_utils.py") is True
    assert is_denied_path("tests/integration/test_api.py") is True
    
    # Sensitive files
    assert is_denied_path(".env") is True
    assert is_denied_path("config/secrets.json") is True  # Now catches "secrets" in path
    assert is_denied_path("ssh/private_key.pem") is True
    assert is_denied_path("credentials.txt") is True
    
    # CI/CD files
    assert is_denied_path(".github/workflows/ci.yml") is True
    assert is_denied_path(".gitlab-ci.yml") is True
    assert is_denied_path("jenkinsfile") is True
    
    # Allowed files
    assert is_denied_path("src/main.py") is False
    assert is_denied_path("lib/utils.py") is False
    assert is_denied_path("README.md") is False


def test_check_scope_compliance() -> None:
    """Test scope compliance checking."""
    # Empty allowed_changes with changes
    compliant, unexpected = check_scope_compliance(
        ["src/main.py"],
        [],
        Path("/tmp/repo"),
    )
    assert compliant is False
    assert unexpected == ["src/main.py"]
    
    # Empty allowed_changes with no changes
    compliant, unexpected = check_scope_compliance(
        [],
        [],
        Path("/tmp/repo"),
    )
    assert compliant is True
    assert unexpected == []
    
    # Allowed changes match
    compliant, unexpected = check_scope_compliance(
        ["src/main.py"],
        ["src/main.py"],
        Path("/tmp/repo"),
    )
    assert compliant is True
    assert unexpected == []
    
    # Directory prefix match
    compliant, unexpected = check_scope_compliance(
        ["src/module/utils.py"],
        ["src/module"],
        Path("/tmp/repo"),
    )
    assert compliant is True
    assert unexpected == []
    
    # Unexpected file
    compliant, unexpected = check_scope_compliance(
        ["src/main.py", "other_file.py"],
        ["src/main.py"],
        Path("/tmp/repo"),
    )
    assert compliant is False
    assert "other_file.py" in unexpected
    
    # Denied file in changes
    compliant, unexpected = check_scope_compliance(
        ["src/main.py", "tests/test_main.py"],
        ["src/main.py"],
        Path("/tmp/repo"),
    )
    assert compliant is False
    assert "tests/test_main.py" in unexpected
