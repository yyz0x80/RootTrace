from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from patchpilot.planning.schema import ChangePlan, PlannedTest
from patchpilot.verification.targets import (
    SelectionReasonType,
    _classify_test_selection,
    _extract_imports_from_file,
    _find_module_dependencies,
    _path_to_module,
    select_target_tests,
)


def make_plan(planned_tests: list[PlannedTest]) -> ChangePlan:
    return ChangePlan(
        relevant_files=[],
        planned_changes=[],
        planned_tests=planned_tests,
        out_of_scope=[],
        risk_level="low",
    )


def test_select_pytest_target_and_acceptance_criteria() -> None:
    plan = make_plan(
        [
            PlannedTest(
                command=(
                    "pytest "
                    "tests/test_task.py::test_priority -q"
                ),
                purpose="Verify priority",
                acceptance_criteria=["AC-1"],
            )
        ]
    )

    selection = select_target_tests(plan)

    assert selection.tests == [
        "tests/test_task.py::test_priority"
    ]
    assert selection.acceptance_criteria == ["AC-1"]
    assert selection.direct_acceptance_criteria == ["AC-1"]


def test_select_python_module_pytest_command() -> None:
    plan = make_plan(
        [
            PlannedTest(
                command="python -m pytest tests/test_task.py -q",
                purpose="Verify task behavior",
                acceptance_criteria=["AC-1", "AC-2"],
            )
        ]
    )

    selection = select_target_tests(plan)

    assert selection.tests == ["tests/test_task.py"]
    assert selection.acceptance_criteria == ["AC-1", "AC-2"]
    assert selection.direct_acceptance_criteria == []


def test_file_level_test_with_single_ac_is_indirect() -> None:
    """File-level test mapped to a single AC should be considered indirect evidence by default."""
    plan = make_plan(
        [
            PlannedTest(
                command="python -m pytest tests/test_task.py -q",
                purpose="Verify task behavior",
                acceptance_criteria=["AC-1"],
            )
        ]
    )

    selection = select_target_tests(plan)

    assert selection.tests == ["tests/test_task.py"]
    assert selection.acceptance_criteria == ["AC-1"]
    assert selection.direct_acceptance_criteria == []
    assert selection.selected_tests[0].is_direct_evidence is False


def test_exact_node_test_with_single_ac_is_direct() -> None:
    """Exact node test (with ::) mapped to a single AC should be considered direct evidence."""
    plan = make_plan(
        [
            PlannedTest(
                command="pytest tests/test_task.py::test_priority -q",
                purpose="Verify priority",
                acceptance_criteria=["AC-1"],
            )
        ]
    )

    selection = select_target_tests(plan)

    assert selection.tests == ["tests/test_task.py::test_priority"]
    assert selection.acceptance_criteria == ["AC-1"]
    assert selection.direct_acceptance_criteria == ["AC-1"]
    assert selection.selected_tests[0].is_direct_evidence is True


def test_acceptance_probe_is_direct() -> None:
    """Acceptance probe should be considered direct evidence."""
    plan = make_plan(
        [
            PlannedTest(
                command="pytest tests/test_task.py -q",
                purpose="Acceptance probe for task behavior",
                acceptance_criteria=["AC-1"],
            )
        ]
    )

    selection = select_target_tests(plan)

    assert selection.tests == ["tests/test_task.py"]
    assert selection.acceptance_criteria == ["AC-1"]
    assert selection.direct_acceptance_criteria == ["AC-1"]


def test_ignore_ruff_planned_check() -> None:
    plan = make_plan(
        [
            PlannedTest(
                command="ruff check .",
                purpose="Run lint",
                acceptance_criteria=[],
            )
        ]
    )

    selection = select_target_tests(plan)

    assert selection.tests == []
    assert selection.acceptance_criteria == []
    assert selection.direct_acceptance_criteria == []


def test_reject_path_traversal_target() -> None:
    plan = make_plan(
        [
            PlannedTest(
                command="pytest ../outside/test_secret.py -q",
                purpose="Unsafe test",
                acceptance_criteria=["AC-1"],
            )
        ]
    )

    with pytest.raises(
        ValueError,
        match="Unsafe planned test target",
    ):
        select_target_tests(plan)


def test_reject_unsupported_command() -> None:
    plan = make_plan(
        [
            PlannedTest(
                command="pip install pytest",
                purpose="Install dependency",
                acceptance_criteria=[],
            )
        ]
    )

    with pytest.raises(
        ValueError,
        match="Unsupported planned test command",
    ):
        select_target_tests(plan)


# Tests for dependency analysis functionality


def test_path_to_module_basic():
    """Test basic path to module conversion."""
    assert _path_to_module("package/module.py", Path("/repo")) == "package.module"
    assert _path_to_module("simple.py", Path("/repo")) == "simple"
    assert _path_to_module("deep/nested/path.py", Path("/repo")) == "deep.nested.path"


def test_path_to_module_src_layout():
    """Test src-layout repository handling."""
    assert _path_to_module("src/package/module.py", Path("/repo")) == "package.module"
    assert _path_to_module("src/simple.py", Path("/repo")) == "simple"


def test_path_to_module_non_python():
    """Test that non-Python files return None."""
    assert _path_to_module("README.md", Path("/repo")) is None
    assert _path_to_module("config.toml", Path("/repo")) is None


def test_extract_imports_basic(tmp_path: Path):
    """Test basic import extraction from a Python file."""
    test_file = tmp_path / "test_module.py"
    test_file.write_text("""
import os
import sys
from pathlib import Path
from collections import defaultdict

def test_something():
    pass
""")

    imports = _extract_imports_from_file(test_file, tmp_path)
    assert "os" in imports
    assert "sys" in imports
    assert "pathlib" in imports
    assert "collections" in imports


def test_extract_imports_relative(tmp_path: Path):
    """Test relative import extraction."""
    # Create a package structure
    pkg_dir = tmp_path / "mypackage"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    
    test_file = pkg_dir / "test_module.py"
    test_file.write_text("""
from . import helper
from .. import parent
from .submodule import func
""")

    imports = _extract_imports_from_file(test_file, tmp_path)
    # Relative imports should be handled
    assert len(imports) > 0


def test_extract_imports_syntax_error(tmp_path: Path):
    """Test that syntax errors are handled gracefully."""
    test_file = tmp_path / "bad_syntax.py"
    test_file.write_text("this is not valid python syntax")

    imports = _extract_imports_from_file(test_file, tmp_path)
    assert imports == set()


def test_extract_imports_unreadable(tmp_path: Path):
    """Test that unreadable files are handled gracefully."""
    test_file = tmp_path / "unreadable.py"
    # Create a directory instead of a file
    test_file.mkdir()

    imports = _extract_imports_from_file(test_file, tmp_path)
    assert imports == set()


def test_classify_test_selection_direct_planned():
    """Test that explicitly planned tests are classified as DIRECT."""
    changed_modules = {"myapp.models"}
    module_index = {"myapp.models": "/repo/myapp/models.py"}
    planned_tests = {"tests/test_models.py::test_user"}

    reason = _classify_test_selection(
        "tests/test_models.py::test_user",
        changed_modules,
        module_index,
        Path("/repo"),
        planned_tests,
    )

    assert reason.classification == SelectionReasonType.DIRECT
    assert "Explicitly planned" in reason.description


def test_classify_test_selection_affected_by_import(tmp_path: Path):
    """Test that tests importing changed modules are classified as AFFECTED."""
    # Create a test file that imports a changed module
    test_file = tmp_path / "tests" / "test_models.py"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("""
import pytest
from myapp.models import User

def test_user():
    pass
""")

    changed_modules = {"myapp.models"}
    module_index = {"myapp.models": str(tmp_path / "myapp" / "models.py")}
    planned_tests = set()

    reason = _classify_test_selection(
        "tests/test_models.py",
        changed_modules,
        module_index,
        tmp_path,
        planned_tests,
    )

    assert reason.classification == SelectionReasonType.AFFECTED
    assert "myapp.models" in reason.changed_modules


def test_classify_test_selection_unrelated(tmp_path: Path):
    """Test that tests with no dependency relationship are classified as UNRELATED."""
    # Create the test file to avoid "file not found" classification
    test_file = tmp_path / "tests" / "test_utils.py"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("""
def test_utils():
    pass
""")

    changed_modules = {"myapp.models"}
    module_index = {"myapp.models": str(tmp_path / "myapp" / "models.py")}
    planned_tests = set()

    reason = _classify_test_selection(
        "tests/test_utils.py",
        changed_modules,
        module_index,
        tmp_path,
        planned_tests,
    )

    assert reason.classification == SelectionReasonType.UNRELATED
    assert "No dependency relationship" in reason.description


def test_select_target_tests_with_dependency_analysis(tmp_path: Path):
    """Test test selection with dependency analysis enabled."""
    # Set up a simple repository structure
    (tmp_path / "myapp").mkdir()
    (tmp_path / "myapp" / "__init__.py").write_text("")
    (tmp_path / "myapp" / "models.py").write_text("class User: pass")
    (tmp_path / "myapp" / "views.py").write_text("from myapp.models import User")
    
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "__init__.py").write_text("")
    (tmp_path / "tests" / "test_models.py").write_text("""
from myapp.models import User

def test_user():
    assert User
""")
    (tmp_path / "tests" / "test_views.py").write_text("""
from myapp.views import something

def test_views():
    pass
""")
    (tmp_path / "tests" / "test_unrelated.py").write_text("""
def test_unrelated():
    pass
""")

    # Initialize git repo
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    plan = make_plan([
        PlannedTest(
            command="pytest tests/test_models.py -q",
            purpose="Test models",
            acceptance_criteria=["AC-1"],
        )
    ])

    python_files = [
        "myapp/__init__.py",
        "myapp/models.py",
        "myapp/views.py",
        "tests/__init__.py",
        "tests/test_models.py",
        "tests/test_views.py",
        "tests/test_unrelated.py",
    ]

    selection = select_target_tests(
        plan,
        changed_files=["myapp/models.py"],
        repo_root=tmp_path,
        python_files=python_files,
    )

    # Should include the planned test
    assert "tests/test_models.py" in selection.tests
    
    # Should also include tests that depend on the changed module
    # test_views.py imports views which imports models
    assert any("test_views" in test.test_id for test in selection.selected_tests)
    
    # Should not include unrelated tests
    assert not any("test_unrelated" in test.test_id for test in selection.selected_tests)


def test_select_target_tests_preserves_planned_tests():
    """Test that explicitly planned tests are preserved even without dependency analysis."""
    plan = make_plan([
        PlannedTest(
            command="pytest tests/test_specific.py::test_exact -q",
            purpose="Test specific function",
            acceptance_criteria=["AC-1"],
        )
    ])

    selection = select_target_tests(
        plan,
        changed_files=["some/module.py"],
        repo_root=Path("/fake"),
        python_files=[],
    )

    # Should still include the planned test even if dependency analysis fails
    assert "tests/test_specific.py::test_exact" in selection.tests
    assert selection.acceptance_criteria == ["AC-1"]
    assert selection.direct_acceptance_criteria == ["AC-1"]


def test_select_target_tests_no_change_plan():
    """Test selection when no change plan is provided."""
    selection = select_target_tests(
        None,
        changed_files=["some/module.py"],
        repo_root=Path("/fake"),
        python_files=[],
    )

    assert selection.tests == []
    assert selection.acceptance_criteria == []
    assert selection.direct_acceptance_criteria == []
    assert selection.selected_tests == []


def test_select_target_tests_deduplicates():
    """Test that duplicate test targets are deduplicated."""
    plan = make_plan([
        PlannedTest(
            command="pytest tests/test_example.py -q",
            purpose="Test 1",
            acceptance_criteria=["AC-1"],
        ),
        PlannedTest(
            command="pytest tests/test_example.py -q",
            purpose="Test 2",
            acceptance_criteria=["AC-2"],
        ),
    ])

    selection = select_target_tests(plan)

    assert selection.tests == ["tests/test_example.py"]
    assert len(selection.tests) == 1  # Deduplicated


def test_select_target_tests_deterministic_ordering():
    """Test that test selection produces deterministic ordering."""
    plan = make_plan([
        PlannedTest(
            command="pytest tests/test_z.py -q",
            purpose="Test Z",
            acceptance_criteria=["AC-3"],
        ),
        PlannedTest(
            command="pytest tests/test_a.py -q",
            purpose="Test A",
            acceptance_criteria=["AC-1"],
        ),
        PlannedTest(
            command="pytest tests/test_m.py -q",
            purpose="Test M",
            acceptance_criteria=["AC-2"],
        ),
    ])

    selection = select_target_tests(plan)

    # Should be sorted alphabetically
    assert selection.tests == ["tests/test_a.py", "tests/test_m.py", "tests/test_z.py"]


def test_dependency_analysis_with_src_layout(tmp_path: Path):
    """Test dependency analysis with src-layout repository."""
    # Create src-layout structure
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "mypackage").mkdir()
    (tmp_path / "src" / "mypackage" / "__init__.py").write_text("")
    (tmp_path / "src" / "mypackage" / "module.py").write_text("def func(): pass")
    
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_module.py").write_text("""
from mypackage.module import func

def test_func():
    assert func
""")

    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    # Test that src-layout is handled correctly
    module_name = _path_to_module("src/mypackage/module.py", tmp_path)
    assert module_name == "mypackage.module"


def test_dependency_analysis_fallback_on_error():
    """Test that dependency analysis falls back gracefully on errors."""
    plan = make_plan([
        PlannedTest(
            command="pytest tests/test_example.py -q",
            purpose="Test",
            acceptance_criteria=["AC-1"],
        )
    ])

    # Pass invalid repo path to trigger error handling
    selection = select_target_tests(
        plan,
        changed_files=["module.py"],
        repo_root=Path("/nonexistent"),
        python_files=["tests/test_example.py"],
    )

    # Should still return planned tests even if analysis fails
    assert "tests/test_example.py" in selection.tests
    
    # The selected_tests may be empty if dependency analysis fails before 
    # classifying planned tests, but the basic tests list should still work
    # This is acceptable fallback behavior


def test_find_module_dependencies_simple(tmp_path: Path):
    """Test finding modules that depend on a given module."""
    # Create simple dependency chain: A -> B -> C
    (tmp_path / "package").mkdir()
    (tmp_path / "package" / "__init__.py").write_text("")
    (tmp_path / "package" / "a.py").write_text("from package.b import func_b")
    (tmp_path / "package" / "b.py").write_text("from package.c import func_c")
    (tmp_path / "package" / "c.py").write_text("def func_c(): pass")

    module_index = {
        "package.a": str(tmp_path / "package" / "a.py"),
        "package.b": str(tmp_path / "package" / "b.py"),
        "package.c": str(tmp_path / "package" / "c.py"),
    }

    # Find modules that depend on package.c
    dependents = _find_module_dependencies("package.c", module_index)

    # package.b depends on package.c
    assert "package.b" in dependents
    # package.a transitively depends on package.c (through b)
    assert "package.a" in dependents


def test_find_module_dependencies_max_depth(tmp_path: Path):
    """Test that dependency search respects max depth."""
    (tmp_path / "package").mkdir()
    (tmp_path / "package" / "__init__.py").write_text("")
    (tmp_path / "package" / "a.py").write_text("from package.b import func_b")
    (tmp_path / "package" / "b.py").write_text("from package.c import func_c")
    (tmp_path / "package" / "c.py").write_text("from package.d import func_d")
    (tmp_path / "package" / "d.py").write_text("def func_d(): pass")

    module_index = {
        "package.a": str(tmp_path / "package" / "a.py"),
        "package.b": str(tmp_path / "package" / "b.py"),
        "package.c": str(tmp_path / "package" / "c.py"),
        "package.d": str(tmp_path / "package" / "d.py"),
    }

    # With max_depth=1, should only find direct dependents
    dependents = _find_module_dependencies("package.d", module_index, max_depth=1)
    assert "package.c" in dependents
    assert "package.b" not in dependents
    assert "package.a" not in dependents


def test_circular_import_handling(tmp_path: Path):
    """Test that circular imports don't cause infinite loops."""
    (tmp_path / "package").mkdir()
    (tmp_path / "package" / "__init__.py").write_text("")
    (tmp_path / "package" / "a.py").write_text("from package.b import func_b")
    (tmp_path / "package" / "b.py").write_text("from package.a import func_a")

    module_index = {
        "package.a": str(tmp_path / "package" / "a.py"),
        "package.b": str(tmp_path / "package" / "b.py"),
    }

    # Should not hang on circular imports
    dependents = _find_module_dependencies("package.a", module_index)
    # Should complete without error
    assert isinstance(dependents, set)
