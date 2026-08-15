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
    execute_task,
    select_task_configs,
    verify_base_commit,
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
            score=1.0,
            hidden_tests_passed=True,
            signal_matched=False,
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
                "average_score": 1.0,
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
        base_commit="a3df5b5f8aadf0015070e07ad21c22f744de3230",
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
    assert result.score == 1.0


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
        assert isinstance(repo, Path)
        assert verify_base_commit(
            repo,
            "a3df5b5f8aadf0015070e07ad21c22f744de3230",
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
    config = TaskConfig(
        task_id="prepare-only",
        category="ambiguous_requirement",
        repository="fixtures/day5_python_repo",
        base_commit="a3df5b5f8aadf0015070e07ad21c22f744de3230",
        issue="tasks/prepare-only/issue.md",
        expected_final_status="NEEDS_CLARIFICATION",
        allowed_changes=[],
        target_tests=[],
        score_commands=[],
        expected_phase="prepare",
        expected_signal="PatchPilot will not guess product behavior.",
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
    assert result.signal_matched is True
    prepare.assert_called_once()
