"""Evaluation runner for PatchPilot task execution and scoring.

This module implements the evaluation pipeline that:
1. Reads task definitions from task.json files
2. Validates base commits against expected values
3. Creates clean repository copies for isolation
4. Executes patchpilot prepare and execute phases
5. Applies generated patches to scoring copies
6. Runs hidden tests for evaluation
7. Aggregates and saves scoring results
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__:
    from .materialize_fixture import materialize
else:
    from materialize_fixture import materialize


@dataclass
class TaskConfig:
    """Configuration for a single evaluation task."""

    task_id: str
    category: str
    repository: str
    base_commit: str
    issue: str
    expected_final_status: str
    allowed_changes: list[str]
    target_tests: list[str]
    score_commands: list[str]
    expected_phase: str = "execute"
    expected_signal: str = ""


@dataclass
class RunResult:
    """Result of a single task execution."""

    task_id: str
    phase: str
    status: str
    prepare_success: bool
    execute_success: bool
    score_success: bool
    error_message: str = ""
    signal_matched: bool = False


@dataclass
class ScoreResult:
    """Scoring result for a task."""

    task_id: str
    category: str
    expected_status: str
    actual_status: str
    phase_reached: str
    score: float
    hidden_tests_passed: bool
    signal_matched: bool
    details: dict[str, Any] = field(default_factory=dict)


def load_task_config(task_dir: Path) -> TaskConfig:
    """Load task configuration from task.json file.

    Args:
        task_dir: Directory containing task.json

    Returns:
        TaskConfig object with task parameters

    Raises:
        FileNotFoundError: If task.json does not exist
        ValueError: If required fields are missing
    """
    task_json = task_dir / "task.json"
    if not task_json.exists():
        raise FileNotFoundError(f"task.json not found in {task_dir}")

    with open(task_json) as f:
        data = json.load(f)

    return TaskConfig(
        task_id=data["task_id"],
        category=data["category"],
        repository=data["repository"],
        base_commit=data["base_commit"],
        issue=data["issue"],
        expected_final_status=data.get("expected_final_status", "VERIFIED"),
        allowed_changes=data.get("allowed_changes", []),
        target_tests=data.get("target_tests", []),
        score_commands=data.get("score_commands", []),
        expected_phase=data.get("expected_phase", "execute"),
        expected_signal=data.get("expected_signal", ""),
    )


def resolve_task_paths(tasks_dir: Path, task_entries: list[str]) -> list[Path]:
    """Resolve task manifest paths declared in an evaluation index.

    Index entries may be relative to either the tasks directory or the
    evaluation root. Every resolved manifest must remain under the evaluation
    root.

    Args:
        tasks_dir: Directory containing index.json.
        task_entries: Manifest paths declared by the index.

    Returns:
        Resolved task.json paths in index order.

    Raises:
        ValueError: If an entry is absolute, escapes the evaluation root, or
            does not identify a task manifest.
    """
    tasks_dir = tasks_dir.resolve()
    evaluation_root = tasks_dir.parent
    resolved_paths: list[Path] = []

    for entry in task_entries:
        entry_path = Path(entry)
        if entry_path.is_absolute():
            raise ValueError(f"Task index entry must be relative: {entry}")

        candidates = (tasks_dir / entry_path, evaluation_root / entry_path)
        task_path = next(
            (candidate.resolve() for candidate in candidates if candidate.is_file()),
            None,
        )
        if task_path is None:
            raise ValueError(f"Task manifest not found: {entry}")

        try:
            task_path.relative_to(evaluation_root)
        except ValueError as error:
            raise ValueError(
                f"Task manifest is outside the evaluation root: {entry}"
            ) from error

        if task_path.name != "task.json":
            raise ValueError(f"Task index entry must reference task.json: {entry}")
        resolved_paths.append(task_path)

    return resolved_paths


def select_task_configs(
    tasks_dir: Path,
    task_entries: list[str],
    task_id: str | None = None,
) -> list[TaskConfig]:
    """Load indexed task configurations and optionally select one task.

    Args:
        tasks_dir: Directory containing index.json.
        task_entries: Manifest paths declared by the index.
        task_id: Optional exact task identifier to select.

    Returns:
        All indexed task configurations, or a single selected configuration.

    Raises:
        ValueError: If task IDs are duplicated or the requested ID is unknown.
    """
    configs = [
        load_task_config(task_path.parent)
        for task_path in resolve_task_paths(tasks_dir, task_entries)
    ]
    configs_by_id: dict[str, TaskConfig] = {}
    for config in configs:
        if config.task_id in configs_by_id:
            raise ValueError(f"Duplicate task_id in index: {config.task_id}")
        configs_by_id[config.task_id] = config

    if task_id is None:
        return configs
    if task_id not in configs_by_id:
        available_ids = ", ".join(configs_by_id)
        raise ValueError(
            f"Unknown task_id: {task_id}. Available task IDs: {available_ids}"
        )
    return [configs_by_id[task_id]]


def run_git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a git command in the specified repository.

    Args:
        repo: Path to the repository
        *args: Git command arguments
        check: Whether to raise an exception on non-zero exit

    Returns:
        CompletedProcess result with stdout and stderr
    """
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=check,
        capture_output=True,
        text=True,
    )


def verify_base_commit(repo: Path, expected_commit: str) -> bool:
    """Verify that the repository's HEAD matches the expected commit.

    Args:
        repo: Path to the repository
        expected_commit: Expected Git commit hash

    Returns:
        True if commit matches, False otherwise
    """
    result = run_git(repo, "rev-parse", "HEAD", check=False)
    if result.returncode != 0:
        return False
    actual_commit = result.stdout.strip()
    return actual_commit == expected_commit


def create_repo_copy(source: Path, destination: Path) -> None:
    """Create a clean copy of a repository for evaluation.

    Args:
        source: Source repository path
        destination: Destination path for the copy

    Raises:
        ValueError: If source does not exist or destination already exists
    """
    if not source.is_dir():
        raise ValueError(f"Source repository does not exist: {source}")
    if destination.exists():
        raise ValueError(f"Destination already exists: {destination}")

    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(
            ".git",
            "__pycache__",
            ".pytest_cache",
            ".ruff_cache",
            "*.pyc",
        ),
    )


def run_patchpilot_prepare(
    repo: Path,
    issue: Path,
    model: str,
    output_dir: Path,
    project_root: Path,
) -> subprocess.CompletedProcess[str]:
    """Execute patchpilot prepare command.

    Args:
        repo: Path to target repository
        issue: Path to issue file
        model: Model identifier
        output_dir: Directory for prepare artifacts
        project_root: PatchPilot project root directory

    Returns:
        CompletedProcess result from subprocess execution
    """
    return subprocess.run(
        [
            "patchpilot",
            "prepare",
            "--repo",
            str(repo),
            "--issue",
            str(issue),
            "--model",
            model,
            "--output-dir",
            str(output_dir),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )


def run_patchpilot_execute(
    repo: Path,
    issue: Path,
    plan: Path,
    task_id: str,
    model: str,
    max_rounds: int,
    max_repairs: int,
    output_dir: Path,
    project_root: Path,
) -> subprocess.CompletedProcess[str]:
    """Execute patchpilot execute command.

    Args:
        repo: Path to target repository
        issue: Path to normalized issue JSON
        plan: Path to plan JSON
        task_id: Task identifier
        model: Model identifier
        max_rounds: Maximum agent rounds
        max_repairs: Maximum repair attempts
        output_dir: Directory for execute artifacts
        project_root: PatchPilot project root directory

    Returns:
        CompletedProcess result from subprocess execution
    """
    return subprocess.run(
        [
            "patchpilot",
            "execute",
            "--repo",
            str(repo),
            "--issue",
            str(issue),
            "--plan",
            str(plan),
            "--task-id",
            task_id,
            "--model",
            model,
            "--max-rounds",
            str(max_rounds),
            "--max-repairs",
            str(max_repairs),
            "--output-dir",
            str(output_dir),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )


def apply_patch(
    repo: Path,
    patch_file: Path,
) -> subprocess.CompletedProcess[str]:
    """Apply a patch file to a repository using git apply.

    Args:
        repo: Path to the repository
        patch_file: Path to the patch.diff file

    Returns:
        Completed Git process with captured output.
    """
    patch_content = patch_file.resolve().read_text(encoding="utf-8")
    return subprocess.run(
        ["git", "apply", "-"],
        cwd=repo.resolve(),
        input=patch_content,
        capture_output=True,
        text=True,
        check=False,
    )


def run_score_command(
    command: str,
    repo: Path,
    task_dir: Path,
) -> subprocess.CompletedProcess[str]:
    """Execute a scoring command with {task_dir} substitution.

    Args:
        command: Command string with {task_dir} placeholder
        repo: Path to the repository for command execution
        task_dir: Absolute path to task directory for substitution

    Returns:
        CompletedProcess result from subprocess execution
    """
    resolved_command = command.replace("{task_dir}", str(task_dir.resolve()))
    parts = shlex.split(resolved_command)
    return subprocess.run(
        parts,
        cwd=repo.resolve(),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def check_plan_exists(prepare_dir: Path) -> bool:
    """Check if plan.json was generated during prepare phase.

    Args:
        prepare_dir: Directory containing prepare artifacts

    Returns:
        True if plan.json exists, False otherwise
    """
    return (prepare_dir / "plan.json").exists()


def check_patch_exists(execute_dir: Path) -> bool:
    """Check if patch.diff was generated during execute phase.

    Args:
        execute_dir: Directory containing execute artifacts

    Returns:
        True if patch.diff exists, False otherwise
    """
    return (execute_dir / "patch.diff").exists()


def check_verification_report(execute_dir: Path) -> bool:
    """Check if verification_report.json was generated.

    Args:
        execute_dir: Directory containing execute artifacts

    Returns:
        True if verification_report.json exists, False otherwise
    """
    return (execute_dir / "verification_report.json").exists()


def extract_final_status(execute_dir: Path) -> str:
    """Extract the final status from the machine-readable run summary.

    Args:
        execute_dir: Directory containing execute artifacts.

    Returns:
        Final status string (e.g., "VERIFIED", "FAILED")
    """
    summary_path = execute_dir / "run_summary.json"
    if not summary_path.exists():
        return "UNKNOWN"

    with open(summary_path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("final_status", "UNKNOWN")


def execute_task(
    task_config: TaskConfig,
    evaluation_root: Path,
    project_root: Path,
    model: str,
    max_rounds: int,
    max_repairs: int,
    timestamp: str,
) -> RunResult:
    """Execute a single evaluation task.

    Args:
        task_config: Task configuration
        evaluation_root: Root directory for evaluation files
        project_root: PatchPilot project root directory
        model: Model identifier for patchpilot
        max_rounds: Maximum agent rounds
        max_repairs: Maximum repair attempts
        timestamp: Timestamp for run directory naming

    Returns:
        RunResult with execution details
    """
    evaluation_root = evaluation_root.resolve()
    project_root = project_root.resolve()
    task_id = task_config.task_id
    fixture_path = evaluation_root / task_config.repository
    issue_path = evaluation_root / task_config.issue

    # Create task run directory
    run_dir = evaluation_root / "runs" / timestamp / task_id
    prepare_dir = run_dir / "prepare"
    execute_dir = run_dir / "execute"
    prepare_dir.mkdir(parents=True, exist_ok=True)
    execute_dir.mkdir(parents=True, exist_ok=True)

    result = RunResult(
        task_id=task_id,
        phase="prepare",
        status="pending",
        prepare_success=False,
        execute_success=False,
        score_success=False,
    )

    # Create temporary repository copy for this task
    temp_dir = Path(tempfile.mkdtemp())
    try:
        temp_repo = temp_dir / "repo"
        materialize(fixture_path, temp_repo, task_config.base_commit)

        # Verify base commit
        if not verify_base_commit(temp_repo, task_config.base_commit):
            result.status = "base_commit_mismatch"
            result.error_message = (
                f"Base commit mismatch: expected {task_config.base_commit}"
            )
            return result

        # Run prepare phase
        prepare_result = run_patchpilot_prepare(
            repo=temp_repo,
            issue=issue_path,
            model=model,
            output_dir=prepare_dir,
            project_root=project_root,
        )

        result.prepare_success = prepare_result.returncode == 0

        # Prepare-only tasks use a deterministic CLI signal as their outcome.
        if task_config.expected_phase == "prepare":
            result.phase = "prepare"
            combined_output = f"{prepare_result.stdout}\n{prepare_result.stderr}"
            result.signal_matched = (
                bool(task_config.expected_signal)
                and task_config.expected_signal in combined_output
            )
            result.prepare_success = result.signal_matched
            result.status = (
                "stopped_at_prepare"
                if result.signal_matched
                else "unexpected_prepare_result"
            )
            if not result.signal_matched:
                result.error_message = combined_output.strip()
            return result

        if not result.prepare_success:
            result.status = "prepare_failed"
            result.error_message = prepare_result.stderr or prepare_result.stdout
            return result

        # Check if plan was generated
        if not check_plan_exists(prepare_dir):
            result.status = "plan_not_generated"
            result.error_message = "plan.json not found after prepare"
            return result

        # Run execute phase
        result.phase = "execute"
        normalized_issue = prepare_dir / "normalized_issue.json"
        plan = prepare_dir / "plan.json"

        execute_result = run_patchpilot_execute(
            repo=temp_repo,
            issue=normalized_issue,
            plan=plan,
            task_id=task_id,
            model=model,
            max_rounds=max_rounds,
            max_repairs=max_repairs,
            output_dir=execute_dir,
            project_root=project_root,
        )

        run_summary_exists = (execute_dir / "run_summary.json").exists()
        result.execute_success = run_summary_exists

        if not run_summary_exists:
            result.status = "execute_failed"
            result.error_message = execute_result.stderr or execute_result.stdout
            return result

        # Tasks with hidden tests require a patch for the scoring checkout.
        if task_config.score_commands and not check_patch_exists(execute_dir):
            result.status = "patch_not_generated"
            result.error_message = "patch.diff not found after execute"
            return result

        result.status = "completed"
        return result

    except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as e:
        result.status = "error"
        result.error_message = str(e)
        return result
    finally:
        # Clean up temporary directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def score_task(
    task_config: TaskConfig,
    run_result: RunResult,
    evaluation_root: Path,
    timestamp: str,
) -> ScoreResult:
    """Score a completed task by applying patch and running hidden tests.

    Args:
        task_config: Task configuration
        run_result: Result from task execution
        evaluation_root: Root directory for evaluation files
        timestamp: Timestamp for run directory naming

    Returns:
        ScoreResult with scoring details
    """
    evaluation_root = evaluation_root.resolve()
    task_id = task_config.task_id
    run_dir = evaluation_root / "runs" / timestamp / task_id
    execute_dir = run_dir / "execute"
    patch_file = execute_dir / "patch.diff"
    task_dir = evaluation_root / Path(task_config.issue).parent

    result = ScoreResult(
        task_id=task_id,
        category=task_config.category,
        expected_status=task_config.expected_final_status,
        actual_status="UNKNOWN",
        phase_reached=run_result.phase,
        score=0.0,
        hidden_tests_passed=False,
        signal_matched=run_result.signal_matched,
    )

    # For tasks that stopped at prepare phase
    if run_result.phase == "prepare":
        result.actual_status = (
            task_config.expected_final_status
            if run_result.signal_matched
            else "STOPPED_AT_PREPARE"
        )
        result.score = 1.0 if run_result.signal_matched else 0.0
        return result

    # For tasks that failed during execution
    if not run_result.execute_success:
        result.actual_status = "EXECUTION_FAILED"
        result.score = 0.0
        return result

    # Extract actual status from verification report
    if check_verification_report(execute_dir):
        result.actual_status = extract_final_status(execute_dir)
    else:
        result.actual_status = "NO_VERIFICATION_REPORT"

    # Create scoring repository copy
    temp_dir = Path(tempfile.mkdtemp())
    try:
        fixture_path = evaluation_root / task_config.repository
        score_repo = temp_dir / "score_repo"
        materialize(fixture_path, score_repo, task_config.base_commit)

        # Apply the patch when present. A patch is mandatory only for tasks
        # that declare hidden score commands.
        if patch_file.exists():
            patch_result = apply_patch(score_repo, patch_file)
            if patch_result.returncode != 0:
                result.details["patch_error"] = (
                    patch_result.stderr.strip()
                    or patch_result.stdout.strip()
                    or "git apply failed without output"
                )
                result.score = 0.0
                return result
        if task_config.score_commands and not patch_file.exists():
            result.details["patch_error"] = "Patch file was not generated"
            result.score = 0.0
            return result

        # Run score commands (hidden tests)
        all_passed = True
        test_results = []

        for command in task_config.score_commands:
            cmd_result = run_score_command(command, score_repo, task_dir)
            passed = cmd_result.returncode == 0
            all_passed = all_passed and passed
            test_results.append(
                {
                    "command": command,
                    "passed": passed,
                    "stdout": cmd_result.stdout,
                    "stderr": cmd_result.stderr,
                }
            )

        result.hidden_tests_passed = all_passed
        result.details["test_results"] = test_results

        # Calculate score based on status match and hidden tests
        status_match = result.actual_status == result.expected_status
        if status_match and result.hidden_tests_passed:
            result.score = 1.0
        elif status_match:
            result.score = 0.5
        else:
            result.score = 0.0

    finally:
        # Clean up temporary directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

    return result


def save_score_result(run_dir: Path, score: ScoreResult) -> None:
    """Save score result to score.json file.

    Args:
        run_dir: Task run directory
        score: ScoreResult to save
    """
    score_file = run_dir / "score.json"
    with open(score_file, "w") as f:
        json.dump(
            {
                "task_id": score.task_id,
                "category": score.category,
                "expected_status": score.expected_status,
                "actual_status": score.actual_status,
                "phase_reached": score.phase_reached,
                "score": score.score,
                "hidden_tests_passed": score.hidden_tests_passed,
                "signal_matched": score.signal_matched,
                "details": score.details,
            },
            f,
            indent=2,
        )


def aggregate_scores(
    evaluation_root: Path,
    timestamp: str,
) -> dict[str, Any]:
    """Aggregate scores from all completed tasks.

    Args:
        evaluation_root: Root directory for evaluation files
        timestamp: Timestamp for run directory naming

    Returns:
        Dictionary with aggregated scoring results
    """
    runs_dir = evaluation_root / "runs" / timestamp
    aggregate = {
        "timestamp": timestamp,
        "total_tasks": 0,
        "completed_tasks": 0,
        "total_score": 0.0,
        "average_score": 0.0,
        "category_scores": {},
        "task_results": [],
    }

    task_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    aggregate["total_tasks"] = len(task_dirs)

    for task_dir in task_dirs:
        score_file = task_dir / "score.json"
        if not score_file.exists():
            continue

        with open(score_file) as f:
            task_score = json.load(f)

        aggregate["completed_tasks"] += 1
        aggregate["total_score"] += task_score["score"]
        aggregate["task_results"].append(task_score)

        category = task_score["category"]
        if category not in aggregate["category_scores"]:
            aggregate["category_scores"][category] = {
                "count": 0,
                "total_score": 0.0,
            }
        aggregate["category_scores"][category]["count"] += 1
        aggregate["category_scores"][category]["total_score"] += task_score["score"]

    # Calculate averages
    if aggregate["completed_tasks"] > 0:
        aggregate["average_score"] = (
            aggregate["total_score"] / aggregate["completed_tasks"]
        )

    for category in aggregate["category_scores"]:
        cat_data = aggregate["category_scores"][category]
        if cat_data["count"] > 0:
            cat_data["average_score"] = cat_data["total_score"] / cat_data["count"]

    return aggregate


def save_aggregate_results(
    evaluation_root: Path,
    timestamp: str,
    aggregate: dict[str, Any],
) -> None:
    """Save aggregate results to aggregate.json file.

    Args:
        evaluation_root: Root directory for evaluation files
        timestamp: Timestamp for run directory naming
        aggregate: Aggregate results dictionary
    """
    runs_dir = evaluation_root / "runs" / timestamp
    aggregate_file = runs_dir / "aggregate.json"
    with open(aggregate_file, "w") as f:
        json.dump(aggregate, f, indent=2)


def main() -> None:
    """Main entry point for evaluation runner."""
    parser = argparse.ArgumentParser(
        description="Run PatchPilot evaluation tasks"
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        required=True,
        help="Path to tasks directory containing task.json files",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="patchpilot",
        help="Variant identifier for the evaluation run",
    )
    parser.add_argument(
        "--task-id",
        type=str,
        default=None,
        help="Run only the task with this exact task_id",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model identifier for patchpilot",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=16,
        help="Maximum agent rounds per task",
    )
    parser.add_argument(
        "--max-repairs",
        type=int,
        default=2,
        help="Maximum repair attempts per task",
    )
    args = parser.parse_args()

    # Resolve paths
    project_root = Path.cwd().resolve()
    tasks_dir = args.tasks.expanduser().resolve()
    evaluation_root = tasks_dir.parent

    # Generate timestamp for this run
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")

    # Load task index
    index_file = tasks_dir / "index.json"
    if not index_file.exists():
        print(f"Error: Task index not found at {index_file}")
        return

    with open(index_file) as f:
        index = json.load(f)

    try:
        task_configs = select_task_configs(
            tasks_dir=tasks_dir,
            task_entries=index["tasks"],
            task_id=args.task_id,
        )
    except (KeyError, TypeError, ValueError) as error:
        parser.error(str(error))

    # Get defaults from index
    defaults = index.get("defaults", {})
    default_max_rounds = defaults.get("max_rounds", 16)
    default_max_repairs = defaults.get("max_repairs", 2)

    # Use command-line args if provided, otherwise use defaults
    max_rounds = args.max_rounds if args.max_rounds != 16 else default_max_rounds
    max_repairs = args.max_repairs if args.max_repairs != 2 else default_max_repairs

    print(f"Starting evaluation run: {timestamp}")
    print(f"Model: {args.model}")
    print(f"Max rounds: {max_rounds}, Max repairs: {max_repairs}")
    print(f"Tasks to process: {len(task_configs)}")

    # Process each task
    for task_config in task_configs:
        try:
            print(f"\nProcessing task: {task_config.task_id}")

            # Execute task
            run_result = execute_task(
                task_config=task_config,
                evaluation_root=evaluation_root,
                project_root=project_root,
                model=args.model,
                max_rounds=max_rounds,
                max_repairs=max_repairs,
                timestamp=timestamp,
            )

            print(f"  Phase: {run_result.phase}, Status: {run_result.status}")

            # Score task
            score_result = score_task(
                task_config=task_config,
                run_result=run_result,
                evaluation_root=evaluation_root,
                timestamp=timestamp,
            )

            print(f"  Score: {score_result.score:.1f}, Status: {score_result.actual_status}")

            # Save score result
            run_dir = evaluation_root / "runs" / timestamp / task_config.task_id
            save_score_result(run_dir, score_result)

        except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as e:
            print(f"  Error processing task {task_config.task_id}: {e}")
            continue

    # Aggregate and save results
    print("\nAggregating results...")
    aggregate = aggregate_scores(evaluation_root, timestamp)
    save_aggregate_results(evaluation_root, timestamp, aggregate)

    print("\nEvaluation complete!")
    print(f"Results saved to: evaluation/runs/{timestamp}/")
    print(f"Total tasks: {aggregate['total_tasks']}")
    print(f"Completed: {aggregate['completed_tasks']}")
    print(f"Average score: {aggregate['average_score']:.2f}")

    # Print category breakdown
    if aggregate["category_scores"]:
        print("\nCategory scores:")
        for category, data in aggregate["category_scores"].items():
            print(f"  {category}: {data['average_score']:.2f} ({data['count']} tasks)")


if __name__ == "__main__":
    main()
