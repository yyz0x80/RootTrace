"""Policy module for constraint compilation and evaluation.

This module provides the constraint compilation system that converts
TaskConstraint objects from normalized issues into executable policies
that enforce security boundaries during agent execution.

Key components:
- schema: Data models for compiled constraints and policy sets
- compiler: Constraint compiler that parses and compiles constraints
- evaluator: Policy evaluator that enforces compiled policies
- builtins: Built-in system constraints that cannot be overridden
"""

from patchpilot.policy.builtins import get_builtin_policies
from patchpilot.policy.compiler import ConstraintCompiler
from patchpilot.policy.evaluator import PolicyEvaluator
from patchpilot.policy.schema import (
    CompilationError,
    CompilationResult,
    CompiledCommandPolicy,
    CompiledConstraint,
    CompiledDependencyPolicy,
    CompiledNetworkPolicy,
    CompiledPathPolicy,
    ConstraintStatus,
    PolicySet,
)

__all__ = [
    "CompilationError",
    "CompilationResult",
    "CompiledCommandPolicy",
    "CompiledConstraint",
    "CompiledDependencyPolicy",
    "CompiledNetworkPolicy",
    "CompiledPathPolicy",
    "ConstraintCompiler",
    "ConstraintStatus",
    "PolicyEvaluator",
    "PolicySet",
    "get_builtin_policies",
]
