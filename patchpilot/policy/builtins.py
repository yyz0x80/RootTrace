"""Built-in system constraints that cannot be overridden.

This module defines the default security policies that are always enforced
regardless of issue-specific constraints. These built-in policies represent
the minimum security boundaries that cannot be weakened by model constraints.

Built-in constraints protect:
- .env files (secrets and credentials)
- .git directory internals
- Test files (read-only for target repository)
- CI/CD configuration files
- System commands (sudo, git push, etc.)
"""

from patchpilot.policy.schema import (
    CompiledCommandPolicy,
    CompiledDependencyPolicy,
    CompiledNetworkPolicy,
    CompiledPathPolicy,
    PolicySet,
)


def get_builtin_policies() -> PolicySet:
    """Get the built-in system policies that cannot be overridden.

    These policies represent the minimum security boundaries that are
    always enforced regardless of issue-specific constraints.

    Returns:
        PolicySet containing all built-in system constraints
    """
    return PolicySet(
        read_policies=_get_builtin_read_policies(),
        write_policies=_get_builtin_write_policies(),
        command_policies=_get_builtin_command_policies(),
        network_policies=_get_builtin_network_policies(),
        dependency_policies=_get_builtin_dependency_policies(),
    )


def _get_builtin_read_policies() -> list[CompiledPathPolicy]:
    """Get built-in read scope policies.

    Protects sensitive files from being read:
    - .env files (secrets)
    - .git directory internals

    Returns:
        List of CompiledPathPolicy for read restrictions
    """
    return [
        CompiledPathPolicy(
            id="builtin-read-1",
            description="System constraint: Cannot read .env files",
            kind="READ_SCOPE",
            allowed_paths=set(),
            denied_paths={".env"},
            is_allowlist=False,
        ),
        CompiledPathPolicy(
            id="builtin-read-2",
            description="System constraint: Cannot read .git directory internals",
            kind="READ_SCOPE",
            allowed_paths=set(),
            denied_paths={".git"},
            is_allowlist=False,
        ),
    ]


def _get_builtin_write_policies() -> list[CompiledPathPolicy]:
    """Get built-in write scope policies.

    Protects sensitive files and directories from modification:
    - .env files (secrets)
    - .git directory internals
    - CI/CD configuration files (.github/workflows)
    - Test files (test_*.py and tests/ directory) - Day 1 restriction

    Returns:
        List of CompiledPathPolicy for write restrictions
    """
    return [
        CompiledPathPolicy(
            id="builtin-write-1",
            description="System constraint: Cannot write .env files",
            kind="WRITE_SCOPE",
            allowed_paths=set(),
            denied_paths={".env"},
            is_allowlist=False,
        ),
        CompiledPathPolicy(
            id="builtin-write-2",
            description="System constraint: Cannot write .git directory internals",
            kind="WRITE_SCOPE",
            allowed_paths=set(),
            denied_paths={".git"},
            is_allowlist=False,
        ),
        CompiledPathPolicy(
            id="builtin-write-3",
            description="System constraint: Cannot modify CI/CD workflows",
            kind="WRITE_SCOPE",
            allowed_paths=set(),
            denied_paths={".github/workflows"},
            is_allowlist=False,
        ),
        CompiledPathPolicy(
            id="builtin-write-4",
            description="System constraint: Cannot modify test files (Day 1 restriction)",
            kind="WRITE_SCOPE",
            allowed_paths=set(),
            denied_paths={"tests"},
            is_allowlist=False,
        ),
    ]


def _get_builtin_command_policies() -> list[CompiledCommandPolicy]:
    """Get built-in command execution policies.

    Restricts dangerous commands:
    - sudo (privilege escalation)
    - git push (remote modification)
    - rm -rf (destructive operations)
    - Arbitrary shell access

    Returns:
        List of CompiledCommandPolicy for command restrictions
    """
    return [
        CompiledCommandPolicy(
            id="builtin-command-1",
            description="System constraint: Cannot run sudo commands",
            kind="COMMAND",
            allowed_commands=set(),
            denied_commands={"sudo"},
            is_allowlist=False,
        ),
        CompiledCommandPolicy(
            id="builtin-command-2",
            description="System constraint: Cannot run git push",
            kind="COMMAND",
            allowed_commands=set(),
            denied_commands={"git push"},
            is_allowlist=False,
        ),
        CompiledCommandPolicy(
            id="builtin-command-3",
            description="System constraint: Cannot run destructive commands (rm -rf)",
            kind="COMMAND",
            allowed_commands=set(),
            denied_commands={"rm -rf"},
            is_allowlist=False,
        ),
    ]


def _get_builtin_network_policies() -> list[CompiledNetworkPolicy]:
    """Get built-in network access policies.

    By default, allows network access but can be restricted by
    issue-specific constraints.

    Returns:
        List of CompiledNetworkPolicy for network restrictions
    """
    return [
        CompiledNetworkPolicy(
            id="builtin-network-1",
            description="System constraint: Network access allowed by default",
            kind="NETWORK",
            deny_all=False,
            allowed_domains=set(),
            denied_domains=set(),
        ),
    ]


def _get_builtin_dependency_policies() -> list[CompiledDependencyPolicy]:
    """Get built-in dependency policies.

    By default, dependency installation requires explicit approval.
    Lockfile modification is restricted.

    Returns:
        List of CompiledDependencyPolicy for dependency restrictions
    """
    return [
        CompiledDependencyPolicy(
            id="builtin-dep-1",
            description="System constraint: Dependency installation requires approval",
            kind="OTHER",
            deny_installation=False,  # Requires approval, not outright denial
            deny_lockfile_modification=True,
            allowed_packages=set(),
        ),
    ]
