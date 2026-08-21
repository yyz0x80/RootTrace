from __future__ import annotations

import ast
import shlex
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath

from patchpilot.planning.schema import ChangePlan


class SelectionReasonType(str, Enum):
    """Classification of why a test was selected for verification."""

    DIRECT = "direct"
    AFFECTED = "affected"
    UNRELATED = "unrelated"


@dataclass(frozen=True)
class TestSelectionReason:
    """Structured reason for why a test was selected."""

    classification: SelectionReasonType
    description: str
    changed_modules: list[str] = field(default_factory=list)
    import_path: str = ""


@dataclass(frozen=True)
class SelectedTest:
    """A single selected test with classification metadata."""

    test_id: str
    reason: TestSelectionReason
    acceptance_criteria: list[str] = field(default_factory=list)
    is_direct_evidence: bool = False


@dataclass(frozen=True)
class TargetTestSelection:
    """Target tests and their mapped acceptance criteria."""

    tests: list[str]
    acceptance_criteria: list[str]
    direct_acceptance_criteria: list[str]
    selected_tests: list[SelectedTest] = field(default_factory=list)


def _append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _pytest_arguments(command: str) -> list[str] | None:
    """Return pytest arguments or None for a non-pytest command."""
    args = shlex.split(command)

    if not args:
        raise ValueError("Planned test command must not be empty")

    if args[0] == "pytest":
        return args[1:]

    if args[:3] == ["python", "-m", "pytest"]:
        return args[3:]

    if args[:2] == ["ruff", "check"]:
        # Ruff is always executed by Verifier Level 1.
        return None

    raise ValueError(
        "Unsupported planned test command: "
        f"{command}. Only pytest, python -m pytest, "
        "and ruff check are supported."
    )


def _is_test_target(argument: str) -> bool:
    if argument.startswith("-"):
        return False

    file_part = argument.split("::", maxsplit=1)[0]
    path = PurePosixPath(file_part)

    if path.is_absolute() or ".." in path.parts:
        raise ValueError(
            f"Unsafe planned test target: {argument}"
        )

    return (
        file_part.endswith(".py")
        or "tests" in path.parts
        or path.name.startswith("test_")
    )


def _path_to_module(path: str, repo_root: Path) -> str | None:
    """Convert a file path to its Python module name.

    Args:
        path: Relative file path from repository root
        repo_root: Path to repository root

    Returns:
        Module name (e.g., "package.module") or None if not a Python module
    """
    if not path.endswith(".py"):
        return None

    # Remove .py extension
    module_path = path[:-3].replace("/", ".")

    # Handle src-layout repositories (src/ prefix)
    module_path = module_path.removeprefix("src.")

    return module_path


def _extract_imports_from_file(file_path: Path, repo_root: Path) -> set[str]:
    """Extract imported module names from a Python file using AST.

    Args:
        file_path: Path to the Python file
        repo_root: Path to repository root for module resolution

    Returns:
        Set of imported module names
    """
    try:
        source = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return set()

    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError:
        return set()

    imports = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)
            # Also handle "from . import x" style imports
            if node.level > 0:
                # Relative import - calculate the module path
                rel_path = file_path.relative_to(repo_root)
                module_parts = list(rel_path.parent.parts)
                # Remove the .py extension from the file itself
                if rel_path.stem != "__init__":
                    module_parts.append(rel_path.stem)
                
                # Go up the directory tree based on level
                for _ in range(node.level - 1):
                    if module_parts:
                        module_parts.pop()
                
                if module_parts:
                    relative_module = ".".join(module_parts)
                    imports.add(relative_module)

    return imports


def _find_module_dependencies(
    module_name: str,
    all_modules: dict[str, str],
    max_depth: int = 3,
) -> set[str]:
    """Find modules that depend on the given module (reverse dependencies).

    Args:
        module_name: Module to find dependents for
        all_modules: Dict mapping module names to their file paths
        max_depth: Maximum recursion depth for dependency traversal

    Returns:
        Set of module names that depend on the given module
    """
    dependents = set()
    visited = set()
    to_check = {module_name}
    current_depth = 0

    while to_check and current_depth < max_depth:
        next_check = set()
        
        for check_module in to_check:
            if check_module in visited:
                continue
            visited.add(check_module)

            # Find all modules that import this module
            for other_module, file_path in all_modules.items():
                if other_module in visited:
                    continue

                try:
                    imports = _extract_imports_from_file(Path(file_path), Path(file_path).parent)
                    if check_module in imports or any(
                        imp.startswith(check_module + ".") for imp in imports
                    ):
                        dependents.add(other_module)
                        next_check.add(other_module)
                except (OSError, SyntaxError, ValueError):
                    # If we can't analyze a file, skip it conservatively
                    continue

        to_check = next_check
        current_depth += 1

    return dependents


def _build_module_index(repo_root: Path, python_files: list[str]) -> dict[str, str]:
    """Build an index of module names to file paths.

    Args:
        repo_root: Path to repository root
        python_files: List of relative Python file paths

    Returns:
        Dict mapping module names to their file paths
    """
    module_index = {}

    for file_path in python_files:
        module_name = _path_to_module(file_path, repo_root)
        if module_name:
            module_index[module_name] = str(repo_root / file_path)

    return module_index


def _classify_test_selection(
    test_id: str,
    changed_modules: set[str],
    module_index: dict[str, str],
    repo_root: Path,
    planned_tests: set[str],
) -> TestSelectionReason:
    """Classify why a test was selected based on dependency analysis.

    Args:
        test_id: Pytest test identifier
        changed_modules: Set of module names that were changed
        module_index: Index of all modules in the repository
        repo_root: Path to repository root
        planned_tests: Set of test IDs explicitly planned in ChangePlan

    Returns:
        TestSelectionReason with classification and metadata
    """
    # Check if this is a directly planned test
    if test_id in planned_tests:
        return TestSelectionReason(
            classification=SelectionReasonType.DIRECT,
            description="Explicitly planned in ChangePlan",
        )

    # Extract file path from test_id
    test_file = test_id.split("::")[0]
    test_path = repo_root / test_file

    if not test_path.exists():
        return TestSelectionReason(
            classification=SelectionReasonType.UNRELATED,
            description="Test file not found in workspace",
        )

    # Get imports from the test file
    test_imports = _extract_imports_from_file(test_path, repo_root)

    # Check if test imports any changed module
    affected_modules = changed_modules & test_imports

    if affected_modules:
        return TestSelectionReason(
            classification=SelectionReasonType.AFFECTED,
            description=f"Test imports changed modules: {', '.join(sorted(affected_modules))}",
            changed_modules=sorted(affected_modules),
            import_path=test_file,
        )

    # Check for transitive dependencies
    test_module = _path_to_module(test_file, repo_root)
    if test_module:
        for changed_module in changed_modules:
            dependents = _find_module_dependencies(changed_module, module_index)
            if test_module in dependents:
                return TestSelectionReason(
                    classification=SelectionReasonType.AFFECTED,
                    description=f"Test transitively depends on changed module {changed_module}",
                    changed_modules=[changed_module],
                    import_path=test_file,
                )

    return TestSelectionReason(
        classification=SelectionReasonType.UNRELATED,
        description="No dependency relationship found with changed modules",
    )


def select_target_tests(
    plan: ChangePlan | None,
    changed_files: list[str] | None = None,
    repo_root: Path | None = None,
    python_files: list[str] | None = None,
) -> TargetTestSelection:
    """Extract safe pytest targets and mapped AC IDs from a plan with dependency analysis.

    Extended to support intelligent test selection based on actual changed files:
    - DIRECT: Explicitly mapped target tests or exact pytest node IDs from ChangePlan
    - AFFECTED: Tests importing or depending on changed modules (via AST analysis)
    - UNRELATED: Tests not connected by available deterministic evidence

    Direct evidence rules:
    - Exact node (contains ::) with single AC → direct
    - Entire file (no ::) → indirect by default
    - Entire directory/all tests → indirect
    - Acceptance Probe → direct
    - Dedicated AST checker → direct
    - Ruff → not AC direct evidence

    Args:
        plan: Optional ChangePlan with planned tests
        changed_files: List of actually changed source file paths (relative to repo root)
        repo_root: Path to repository root for dependency analysis
        python_files: List of all Python files in repository for module indexing

    Returns:
        TargetTestSelection with classified test selection and acceptance criteria
    """
    if plan is None:
        return TargetTestSelection(
            tests=[],
            acceptance_criteria=[],
            direct_acceptance_criteria=[],
            selected_tests=[],
        )

    targets: list[str] = []
    criterion_ids: list[str] = []
    direct_criterion_ids: list[str] = []
    selected_tests: list[SelectedTest] = []

    # Extract planned test targets
    planned_test_ids: set[str] = set()
    direct_test_ids: set[str] = set()

    for planned_test in plan.planned_tests:
        pytest_args = _pytest_arguments(planned_test.command)

        if pytest_args is None:
            continue

        planned_targets = [
            argument
            for argument in pytest_args
            if _is_test_target(argument)
        ]

        for target in planned_targets:
            _append_unique(targets, target)
            planned_test_ids.add(target)

        if planned_targets:
            for criterion_id in planned_test.acceptance_criteria:
                _append_unique(criterion_ids, criterion_id)

        # Determine directness based on target granularity
        if len(planned_test.acceptance_criteria) == 1:
            criterion_id = planned_test.acceptance_criteria[0]

            # Check if this is direct evidence
            is_direct = False

            # Acceptance Probe is always direct
            # Dedicated AST checker is direct
            # Exact node (contains ::) can be direct
            if "acceptance probe" in planned_test.purpose.lower() or "ast checker" in planned_test.purpose.lower() or any("::" in target for target in planned_targets):
                is_direct = True
            # File-level targets are indirect by default
            # (no :: indicates entire file)
            else:
                is_direct = False

            if is_direct:
                _append_unique(direct_criterion_ids, criterion_id)
                direct_test_ids.update(planned_targets)

    # Perform dependency analysis if we have the required information
    if changed_files and repo_root and python_files:
        try:
            # Build module index for the repository
            module_index = _build_module_index(repo_root, python_files)

            # Convert changed files to module names
            changed_modules = set()
            for file_path in changed_files:
                module_name = _path_to_module(file_path, repo_root)
                if module_name:
                    changed_modules.add(module_name)

            # Find test files that might be affected by changes
            for test_file in python_files:
                if not (test_file.startswith("tests/") or Path(test_file).name.startswith("test_")):
                    continue

                test_path = repo_root / test_file
                if not test_path.exists():
                    continue

                # Get the base test ID (file path, may include :: for specific tests)
                test_id = test_file

                # Skip if already in planned tests
                if test_id in planned_test_ids:
                    # Add classification for planned tests
                    reason = _classify_test_selection(
                        test_id,
                        changed_modules,
                        module_index,
                        repo_root,
                        planned_test_ids,
                    )
                    selected_tests.append(
                        SelectedTest(
                            test_id=test_id,
                            reason=reason,
                            acceptance_criteria=[],
                            is_direct_evidence=test_id in direct_test_ids,
                        )
                    )
                    continue

                # Classify this test
                reason = _classify_test_selection(
                    test_id,
                    changed_modules,
                    module_index,
                    repo_root,
                    planned_test_ids,
                )

                # Only add AFFECTED tests to the selection
                if (
                    reason.classification == SelectionReasonType.AFFECTED
                    and test_id not in targets
                ):
                    _append_unique(targets, test_id)
                    selected_tests.append(
                        SelectedTest(
                            test_id=test_id,
                            reason=reason,
                            acceptance_criteria=[],
                            is_direct_evidence=False,
                        )
                    )

        except (OSError, SyntaxError, ValueError, subprocess.SubprocessError):
            # If dependency analysis fails, fall back to planned tests only
            # This is conservative - we don't want to break verification due to analysis errors
            selected_tests = [
                SelectedTest(
                    test_id=test_id,
                    reason=TestSelectionReason(
                        classification=SelectionReasonType.DIRECT,
                        description="Explicitly planned in ChangePlan (dependency analysis failed)",
                    ),
                    acceptance_criteria=[],
                    is_direct_evidence=test_id in direct_test_ids,
                )
                for test_id in planned_test_ids
            ]
    else:
        # No dependency analysis available, classify planned tests as DIRECT
        selected_tests = [
            SelectedTest(
                test_id=test_id,
                reason=TestSelectionReason(
                    classification=SelectionReasonType.DIRECT,
                    description="Explicitly planned in ChangePlan (no dependency analysis)",
                ),
                acceptance_criteria=[],
                is_direct_evidence=test_id in direct_test_ids,
            )
            for test_id in planned_test_ids
        ]

    selected_test_ids = {selected.test_id for selected in selected_tests}
    for test_id in planned_test_ids - selected_test_ids:
        selected_tests.append(
            SelectedTest(
                test_id=test_id,
                reason=TestSelectionReason(
                    classification=SelectionReasonType.DIRECT,
                    description="Explicitly planned in ChangePlan",
                ),
                acceptance_criteria=[],
                is_direct_evidence=test_id in direct_test_ids,
            )
        )

    # Ensure deterministic ordering by sorting
    targets.sort()
    selected_tests.sort(key=lambda st: st.test_id)

    return TargetTestSelection(
        tests=targets,
        acceptance_criteria=criterion_ids,
        direct_acceptance_criteria=direct_criterion_ids,
        selected_tests=selected_tests,
    )
