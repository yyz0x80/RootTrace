"""Tests for the constraint compiler and evaluator."""

import pytest

from patchpilot.issue.schema import TaskConstraint
from patchpilot.policy import (
    ConstraintCompiler,
    PolicyEvaluator,
    PolicySet,
    get_builtin_policies,
)
from patchpilot.policy.schema import (
    CompiledPathPolicy,
)


def test_builtin_policies_exist():
    """Test that built-in policies are properly defined."""
    policy_set = get_builtin_policies()

    assert isinstance(policy_set, PolicySet)
    assert len(policy_set.write_policies) > 0
    assert len(policy_set.read_policies) > 0
    assert len(policy_set.command_policies) > 0

    # Check that .env is protected
    env_protected = any(
        ".env" in policy.denied_paths
        for policy in policy_set.write_policies
    )
    assert env_protected


def test_compiler_with_explicit_write_scope():
    """Test compilation of explicit write scope constraints."""
    compiler = ConstraintCompiler()

    constraints = [
        TaskConstraint(
            id="C-1",
            description="Only modify benchmark/users.py and benchmark/user_service.py",
            kind="WRITE_SCOPE",
        ),
    ]

    result = compiler.compile(constraints)

    assert result.total_constraints == 1
    assert result.supported_constraints == 1
    assert result.policy_set.compilation_successful

    # Check that the policy was compiled correctly
    write_policies = result.policy_set.write_policies
    assert len(write_policies) > 0

    # Find the compiled policy
    compiled = next(
        (p for p in write_policies if p.id == "C-1"),
        None,
    )
    assert compiled is not None
    assert isinstance(compiled, CompiledPathPolicy)
    assert compiled.is_allowlist
    assert "benchmark/users.py" in compiled.allowed_paths
    assert "benchmark/user_service.py" in compiled.allowed_paths


def test_compiler_rejects_ambiguous_write_scope():
    """Test that compiler rejects ambiguous write scope constraints."""
    compiler = ConstraintCompiler()

    constraints = [
        TaskConstraint(
            id="C-1",
            description="Only modify user module related files",
            kind="WRITE_SCOPE",
        ),
    ]

    result = compiler.compile(constraints)

    assert result.total_constraints == 1
    assert result.failed_constraints == 1
    assert not result.policy_set.compilation_successful


def test_compiler_validates_paths():
    """Test that compiler validates and rejects invalid paths."""
    compiler = ConstraintCompiler()

    constraints = [
        TaskConstraint(
            id="C-1",
            description="Only modify /etc/passwd",
            kind="WRITE_SCOPE",
        ),
    ]

    result = compiler.compile(constraints)

    assert result.total_constraints == 1
    assert result.failed_constraints == 1
    assert not result.policy_set.compilation_successful


def test_compiler_rejects_path_traversal():
    """Test that compiler rejects path traversal attempts."""
    compiler = ConstraintCompiler()

    constraints = [
        TaskConstraint(
            id="C-1",
            description="Only modify ../etc/passwd",
            kind="WRITE_SCOPE",
        ),
    ]

    result = compiler.compile(constraints)

    assert result.total_constraints == 1
    assert result.failed_constraints == 1
    assert not result.policy_set.compilation_successful


def test_compiler_command_policy():
    """Test compilation of command policies."""
    compiler = ConstraintCompiler()

    constraints = [
        TaskConstraint(
            id="C-1",
            description="Do not run git push",
            kind="COMMAND",
        ),
    ]

    result = compiler.compile(constraints)

    assert result.total_constraints == 1
    assert result.supported_constraints == 1
    assert result.policy_set.compilation_successful

    # Command policies are added to the all_constraints list
    compiled = next(
        (p for p in result.policy_set.all_constraints if p.id == "C-1"),
        None,
    )
    assert compiled is not None
    # Check if it's a command policy
    from patchpilot.policy.schema import CompiledCommandPolicy
    assert isinstance(compiled, CompiledCommandPolicy)
    assert "git" in compiled.denied_commands or "push" in str(compiled.denied_commands).lower()


def test_evaluator_enforces_write_policy():
    """Test that policy evaluator enforces write policies."""
    compiler = ConstraintCompiler()

    constraints = [
        TaskConstraint(
            id="C-1",
            description="Only modify benchmark/users.py",
            kind="WRITE_SCOPE",
        ),
    ]

    result = compiler.compile(constraints)
    evaluator = PolicyEvaluator(result.policy_set)

    # The evaluator should check against the merged policies
    # Since we have an allowlist constraint, it should be enforced
    # But we also have builtin denylist policies that take precedence

    # Should deny writes to .env (builtin policy)
    with pytest.raises(PermissionError):
        evaluator.assert_write_allowed(".env")

    # Should deny writes to tests (builtin policy)
    with pytest.raises(PermissionError):
        evaluator.assert_write_allowed("tests/test_example.py")


def test_evaluator_enforces_builtin_policies():
    """Test that policy evaluator enforces built-in policies."""
    builtin_policies = get_builtin_policies()
    evaluator = PolicyEvaluator(builtin_policies)

    # Should deny writes to .env
    with pytest.raises(PermissionError):
        evaluator.assert_write_allowed(".env")

    # Should deny writes to .git
    with pytest.raises(PermissionError):
        evaluator.assert_write_allowed(".git/config")

    # Test immutability is repository-aware and enforced by ToolRegistry.
    evaluator.assert_write_allowed("tests/test_example.py")


def test_evaluator_enforces_command_policy():
    """Test that policy evaluator enforces command policies."""
    compiler = ConstraintCompiler()

    constraints = [
        TaskConstraint(
            id="C-1",
            description="Do not run sudo commands",
            kind="COMMAND",
        ),
    ]

    result = compiler.compile(constraints)
    evaluator = PolicyEvaluator(result.policy_set)

    # The evaluator should enforce both builtin and issue command policies
    # Builtin policies already deny sudo, so this should be denied
    with pytest.raises(PermissionError):
        evaluator.assert_command_allowed("sudo rm -rf /")


def test_policy_merging():
    """Test that policies are merged correctly with builtins."""
    compiler = ConstraintCompiler()

    constraints = [
        TaskConstraint(
            id="C-1",
            description="Only modify benchmark/users.py",
            kind="WRITE_SCOPE",
        ),
    ]

    result = compiler.compile(constraints)

    # Check that builtin policies are still present
    assert len(result.policy_set.write_policies) > 0

    # Check that .env is still protected after merging
    # The builtin policies should be preserved in the merged result
    env_protected = any(
        ".env" in policy.denied_paths or ".env" in str(policy.description).lower()
        for policy in result.policy_set.write_policies
    )
    assert env_protected, ".env protection should be preserved after merging"


def test_unsupported_constraint_kind():
    """Test that unsupported constraint kinds are marked as unsupported."""
    compiler = ConstraintCompiler()

    constraints = [
        TaskConstraint(
            id="C-1",
            description="Some other constraint",
            kind="OTHER",
        ),
    ]

    result = compiler.compile(constraints)

    assert result.total_constraints == 1
    assert result.unsupported_constraints == 1
    assert result.policy_set.compilation_successful  # Unsupported is not a failure


def test_network_policy_compilation():
    """Test compilation of network policies."""
    compiler = ConstraintCompiler()

    constraints = [
        TaskConstraint(
            id="C-1",
            description="No network access allowed",
            kind="NETWORK",
        ),
    ]

    result = compiler.compile(constraints)

    assert result.total_constraints == 1
    assert result.supported_constraints == 1

    # The compiled policy should be in the all_constraints list
    compiled = next(
        (p for p in result.policy_set.all_constraints if p.id == "C-1"),
        None,
    )
    assert compiled is not None
    # Check if it's a network policy
    from patchpilot.policy.schema import CompiledNetworkPolicy
    assert isinstance(compiled, CompiledNetworkPolicy)
    assert compiled.deny_all


def test_dependency_policy_compilation():
    """Test compilation of dependency policies."""
    compiler = ConstraintCompiler()

    constraints = [
        TaskConstraint(
            id="C-1",
            description="Do not install dependencies",
            kind="OTHER",
        ),
    ]

    result = compiler.compile(constraints)

    assert result.total_constraints == 1
    assert result.supported_constraints == 1

    # Dependency policies are added to the policy set
    # The compiled policy should be in the all_constraints list
    compiled = next(
        (p for p in result.policy_set.all_constraints if p.id == "C-1"),
        None,
    )
    assert compiled is not None
    # Check if it's a dependency policy
    from patchpilot.policy.schema import CompiledDependencyPolicy
    assert isinstance(compiled, CompiledDependencyPolicy)
    assert compiled.deny_installation
