"""Data models for compiled constraints and policy sets.

This module defines the schema for compiled constraints that result from
the constraint compilation process. These models represent executable
policies that enforce security boundaries during agent execution.
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class ConstraintStatus(str, Enum):
    """Status of a constraint after compilation."""
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    ERROR = "error"


class CompiledConstraint(BaseModel):
    """Base model for a compiled constraint.

    Each compiled constraint represents a specific security boundary
    that has been successfully parsed and is ready for enforcement.
    """
    id: str
    description: str
    kind: Literal["READ_SCOPE", "WRITE_SCOPE", "COMMAND", "NETWORK", "OTHER"]
    status: ConstraintStatus = ConstraintStatus.SUPPORTED
    error_message: str | None = None


class CompiledPathPolicy(CompiledConstraint):
    """Compiled policy for file path access constraints.

    Defines which paths are allowed or denied for read/write operations.
    """
    kind: Literal["READ_SCOPE", "WRITE_SCOPE"] = Field(...)

    # Set of explicitly allowed paths (relative to repository root)
    allowed_paths: set[str] = Field(default_factory=set)

    # Set of explicitly denied paths (relative to repository root)
    denied_paths: set[str] = Field(default_factory=set)

    # Whether this is an allowlist (only allowed_paths permitted)
    # or a denylist (all paths except denied_paths permitted)
    is_allowlist: bool = False


class CompiledCommandPolicy(CompiledConstraint):
    """Compiled policy for command execution constraints.

    Defines which commands are allowed or denied during execution.
    """
    kind: Literal["COMMAND"] = Field(...)

    # Set of allowed command patterns (e.g., ["pytest", "git diff"])
    allowed_commands: set[str] = Field(default_factory=set)

    # Set of denied command patterns
    denied_commands: set[str] = Field(default_factory=set)

    # Whether this is an allowlist (only allowed_commands permitted)
    is_allowlist: bool = False


class CompiledNetworkPolicy(CompiledConstraint):
    """Compiled policy for network access constraints.

    Defines restrictions on network operations.
    """
    kind: Literal["NETWORK"] = Field(...)

    # Whether network access is completely denied
    deny_all: bool = False

    # Set of allowed domains (if deny_all is False)
    allowed_domains: set[str] = Field(default_factory=set)

    # Set of denied domains
    denied_domains: set[str] = Field(default_factory=set)


class CompiledDependencyPolicy(CompiledConstraint):
    """Compiled policy for dependency installation constraints.

    Defines restrictions on installing or modifying dependencies.
    """
    kind: Literal["OTHER"] = Field(...)

    # Whether dependency installation is completely denied
    deny_installation: bool = False

    # Whether modifying lockfiles is denied
    deny_lockfile_modification: bool = False

    # Set of allowed package names (if installation is permitted)
    allowed_packages: set[str] = Field(default_factory=set)


class PolicySet(BaseModel):
    """Complete set of compiled policies for a task.

    Represents the merged and final policies that will be enforced
    during agent execution. Policies are merged from system defaults,
    project configuration, and issue-specific constraints.
    """
    # Compiled path policies for read scope
    read_policies: list[CompiledPathPolicy] = Field(default_factory=list)

    # Compiled path policies for write scope
    write_policies: list[CompiledPathPolicy] = Field(default_factory=list)

    # Compiled command policies
    command_policies: list[CompiledCommandPolicy] = Field(default_factory=list)

    # Compiled network policies
    network_policies: list[CompiledNetworkPolicy] = Field(default_factory=list)

    # Compiled dependency policies
    dependency_policies: list[CompiledDependencyPolicy] = Field(default_factory=list)

    # All compiled constraints (including unsupported ones)
    all_constraints: list[CompiledConstraint] = Field(default_factory=list)

    # Whether compilation was successful (no required constraints failed)
    compilation_successful: bool = True

    # Error messages for failed required constraints
    compilation_errors: list[str] = Field(default_factory=list)


class CompilationError(Exception):
    """Exception raised when constraint compilation fails."""

    def __init__(self, message: str, constraint_id: str | None = None):
        self.message = message
        self.constraint_id = constraint_id
        super().__init__(message)


class CompilationResult(BaseModel):
    """Result of constraint compilation process.

    Contains the compiled policy set along with metadata about
    the compilation process.
    """
    policy_set: PolicySet
    total_constraints: int
    supported_constraints: int
    unsupported_constraints: int
    failed_constraints: int

    @property
    def success_rate(self) -> float:
        """Calculate the success rate of constraint compilation."""
        if self.total_constraints == 0:
            return 1.0
        return self.supported_constraints / self.total_constraints
