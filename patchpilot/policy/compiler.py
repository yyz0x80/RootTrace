"""Constraint compiler for parsing and compiling TaskConstraint objects.

This module implements the constraint compiler that converts natural language
constraint descriptions from TaskConstraint objects into executable CompiledConstraint
objects. The compiler follows strict parsing rules and refuses to guess ambiguous
constraints.

Compilation principles:
- Only parse explicit, unambiguous constraints
- Reject vague descriptions that require guessing
- Apply path validation (no absolute paths, no .. traversal)
- Merge policies conservatively (policies can only become more strict)
"""

import re
from pathlib import Path

from patchpilot.issue.schema import TaskConstraint
from patchpilot.policy.builtins import get_builtin_policies
from patchpilot.policy.schema import (
    CompiledCommandPolicy,
    CompiledConstraint,
    CompiledDependencyPolicy,
    CompiledNetworkPolicy,
    CompiledPathPolicy,
    CompilationError,
    CompilationResult,
    ConstraintStatus,
    PolicySet,
)


class ConstraintCompiler:
    """Compiler for converting TaskConstraint objects into executable policies."""

    # Patterns for extracting explicit file paths from constraint descriptions
    # These patterns match explicit file path mentions like:
    # "Only modify benchmark/users.py and benchmark/user_service.py"
    # "Do not modify tests/test_example.py"
    EXPLICIT_PATH_PATTERN = re.compile(
        r'\b[\w./-]+\.(?:py|js|ts|json|yaml|yml|toml|txt|md|rst)\b'
    )

    # Patterns for command mentions
    COMMAND_PATTERN = re.compile(
        r'\b(?:pytest|python|ruff|git|npm|pip|cargo|go|java)\b'
    )

    # Keywords that indicate allowlist vs denylist semantics
    ALLOWLIST_KEYWORDS = {
        "only modify",
        "only access",
        "only run",
        "restrict to",
        "limit to",
    }

    DENYLIST_KEYWORDS = {
        "do not modify",
        "must not modify",
        "do not access",
        "must not access",
        "do not run",
        "must not run",
        "forbid",
        "prohibit",
    }

    # Keywords indicating network restrictions
    NETWORK_KEYWORDS = {
        "network",
        "download",
        "fetch",
        "request",
        "api call",
        "http",
        "https",
    }

    # Keywords indicating dependency restrictions
    DEPENDENCY_KEYWORDS = {
        "dependency",
        "package",
        "install",
        "pip install",
        "npm install",
        "lockfile",
        "requirements",
    }

    def __init__(self):
        """Initialize the constraint compiler."""
        self.builtin_policies = get_builtin_policies()

    def compile(
        self,
        constraints: list[TaskConstraint],
    ) -> CompilationResult:
        """Compile a list of TaskConstraint objects into a PolicySet.

        Args:
            constraints: List of TaskConstraint objects to compile

        Returns:
            CompilationResult containing the compiled PolicySet and metadata

        Raises:
            CompilationError: If a required constraint fails to compile
        """
        compiled_constraints: list[CompiledConstraint] = []
        compilation_errors: list[str] = []

        for constraint in constraints:
            try:
                compiled = self._compile_single_constraint(constraint)
                compiled_constraints.append(compiled)
            except CompilationError as e:
                # Mark constraint as failed
                failed_constraint = CompiledConstraint(
                    id=constraint.id,
                    description=constraint.description,
                    kind=constraint.kind,
                    status=ConstraintStatus.ERROR,
                    error_message=str(e),
                )
                compiled_constraints.append(failed_constraint)
                compilation_errors.append(
                    f"Constraint {constraint.id} failed: {e}"
                )

        # Merge compiled constraints with builtin policies
        policy_set = self._merge_policies(
            compiled_constraints,
            self.builtin_policies,
        )

        # Add all constraints to the policy set for tracking
        policy_set.all_constraints = compiled_constraints

        # Determine if compilation was successful
        # (no required constraints failed to compile)
        required_failed = [
            c for c in compiled_constraints
            if c.status == ConstraintStatus.ERROR
        ]
        policy_set.compilation_successful = len(required_failed) == 0
        policy_set.compilation_errors = compilation_errors

        # Calculate statistics
        total = len(compiled_constraints)
        supported = len([c for c in compiled_constraints if c.status == ConstraintStatus.SUPPORTED])
        unsupported = len([c for c in compiled_constraints if c.status == ConstraintStatus.UNSUPPORTED])
        failed = len([c for c in compiled_constraints if c.status == ConstraintStatus.ERROR])

        return CompilationResult(
            policy_set=policy_set,
            total_constraints=total,
            supported_constraints=supported,
            unsupported_constraints=unsupported,
            failed_constraints=failed,
        )

    def _compile_single_constraint(
        self,
        constraint: TaskConstraint,
    ) -> CompiledConstraint:
        """Compile a single TaskConstraint into a CompiledConstraint.

        Args:
            constraint: The TaskConstraint to compile

        Returns:
            CompiledConstraint

        Raises:
            CompilationError: If the constraint cannot be compiled
        """
        description = constraint.description.lower()

        if constraint.kind == "WRITE_SCOPE":
            return self._compile_write_scope(constraint)
        elif constraint.kind == "READ_SCOPE":
            return self._compile_read_scope(constraint)
        elif constraint.kind == "COMMAND":
            return self._compile_command_policy(constraint)
        elif constraint.kind == "NETWORK":
            return self._compile_network_policy(constraint)
        elif constraint.kind == "OTHER":
            # Try to detect dependency policies in OTHER constraints
            if any(keyword in description for keyword in self.DEPENDENCY_KEYWORDS):
                return self._compile_dependency_policy(constraint)
            else:
                # Mark as unsupported if we can't determine the type
                return CompiledConstraint(
                    id=constraint.id,
                    description=constraint.description,
                    kind=constraint.kind,
                    status=ConstraintStatus.UNSUPPORTED,
                    error_message="Cannot determine constraint type from description",
                )
        else:
            raise CompilationError(
                f"Unknown constraint kind: {constraint.kind}",
                constraint_id=constraint.id,
            )

    def _compile_write_scope(
        self,
        constraint: TaskConstraint,
    ) -> CompiledPathPolicy:
        """Compile a WRITE_SCOPE constraint into a CompiledPathPolicy.

        Parses explicit file paths from the constraint description.
        Rejects vague descriptions that require guessing.

        Args:
            constraint: The WRITE_SCOPE TaskConstraint to compile

        Returns:
            CompiledPathPolicy for write scope

        Raises:
            CompilationError: If paths cannot be reliably extracted
        """
        description = constraint.description.lower()

        # Determine if this is an allowlist or denylist
        is_allowlist = any(
            keyword in description
            for keyword in self.ALLOWLIST_KEYWORDS
        )

        # Extract explicit file paths
        paths = self._extract_explicit_paths(constraint.description)

        if not paths and is_allowlist:
            # Allowlist without explicit paths is ambiguous
            raise CompilationError(
                "Allowlist constraint lacks explicit file paths. "
                "Cannot reliably determine which files to allow.",
                constraint_id=constraint.id,
            )

        # Validate and normalize paths
        validated_paths = set()
        for path in paths:
            try:
                validated_path = self._validate_and_normalize_path(path)
                validated_paths.add(validated_path)
            except CompilationError as e:
                # Reject the entire constraint if any path is invalid
                raise CompilationError(
                    f"Invalid path '{path}': {e}",
                    constraint_id=constraint.id,
                )

        if is_allowlist:
            return CompiledPathPolicy(
                id=constraint.id,
                description=constraint.description,
                kind="WRITE_SCOPE",
                allowed_paths=validated_paths,
                denied_paths=set(),
                is_allowlist=True,
            )
        else:
            # Denylist semantics
            return CompiledPathPolicy(
                id=constraint.id,
                description=constraint.description,
                kind="WRITE_SCOPE",
                allowed_paths=set(),
                denied_paths=validated_paths,
                is_allowlist=False,
            )

    def _compile_read_scope(
        self,
        constraint: TaskConstraint,
    ) -> CompiledPathPolicy:
        """Compile a READ_SCOPE constraint into a CompiledPathPolicy.

        Args:
            constraint: The READ_SCOPE TaskConstraint to compile

        Returns:
            CompiledPathPolicy for read scope
        """
        # Read scope compilation follows the same logic as write scope
        return self._compile_write_scope(constraint).model_copy(
            update={"kind": "READ_SCOPE"}
        )

    def _compile_command_policy(
        self,
        constraint: TaskConstraint,
    ) -> CompiledCommandPolicy:
        """Compile a COMMAND constraint into a CompiledCommandPolicy.

        Args:
            constraint: The COMMAND TaskConstraint to compile

        Returns:
            CompiledCommandPolicy for command restrictions
        """
        description = constraint.description.lower()

        # Determine if this is an allowlist or denylist
        is_allowlist = any(
            keyword in description
            for keyword in self.ALLOWLIST_KEYWORDS
        )

        # Extract command patterns
        commands = self._extract_commands(constraint.description)

        if not commands and is_allowlist:
            raise CompilationError(
                "Allowlist constraint lacks explicit commands. "
                "Cannot reliably determine which commands to allow.",
                constraint_id=constraint.id,
            )

        if is_allowlist:
            return CompiledCommandPolicy(
                id=constraint.id,
                description=constraint.description,
                kind="COMMAND",
                allowed_commands=commands,
                denied_commands=set(),
                is_allowlist=True,
            )
        else:
            return CompiledCommandPolicy(
                id=constraint.id,
                description=constraint.description,
                kind="COMMAND",
                allowed_commands=set(),
                denied_commands=commands,
                is_allowlist=False,
            )

    def _compile_network_policy(
        self,
        constraint: TaskConstraint,
    ) -> CompiledNetworkPolicy:
        """Compile a NETWORK constraint into a CompiledNetworkPolicy.

        Args:
            constraint: The NETWORK TaskConstraint to compile

        Returns:
            CompiledNetworkPolicy for network restrictions
        """
        description = constraint.description.lower()

        # Check for complete network denial
        deny_all = any(
            keyword in description
            for keyword in ["no network", "deny network", "disable network"]
        )

        # Extract domain names if any
        domains = self._extract_domains(constraint.description)

        return CompiledNetworkPolicy(
            id=constraint.id,
            description=constraint.description,
            kind="NETWORK",
            deny_all=deny_all,
            allowed_domains=domains if not deny_all else set(),
            denied_domains=set(),
        )

    def _compile_dependency_policy(
        self,
        constraint: TaskConstraint,
    ) -> CompiledDependencyPolicy:
        """Compile a dependency constraint into a CompiledDependencyPolicy.

        Args:
            constraint: The dependency-related TaskConstraint to compile

        Returns:
            CompiledDependencyPolicy for dependency restrictions
        """
        description = constraint.description.lower()

        # Check for installation denial
        deny_installation = any(
            keyword in description
            for keyword in [
                "do not install",
                "must not install",
                "no installation",
                "forbid installation",
            ]
        )

        # Check for lockfile modification denial
        deny_lockfile = any(
            keyword in description
            for keyword in [
                "do not modify lockfile",
                "must not modify lockfile",
                "no lockfile changes",
            ]
        )

        # Extract allowed package names if installation is permitted
        allowed_packages = set()
        if not deny_installation:
            allowed_packages = self._extract_package_names(constraint.description)

        return CompiledDependencyPolicy(
            id=constraint.id,
            description=constraint.description,
            kind="OTHER",
            deny_installation=deny_installation,
            deny_lockfile_modification=deny_lockfile,
            allowed_packages=allowed_packages,
        )

    def _extract_explicit_paths(self, description: str) -> list[str]:
        """Extract explicit file paths from a constraint description.

        Only extracts clearly specified file paths. Does not guess.

        Args:
            description: The constraint description

        Returns:
            List of explicit file paths found
        """
        matches = self.EXPLICIT_PATH_PATTERN.findall(description)
        return list(set(matches))  # Deduplicate

    def _extract_commands(self, description: str) -> set[str]:
        """Extract command patterns from a constraint description.

        Args:
            description: The constraint description

        Returns:
            Set of command patterns found
        """
        matches = self.COMMAND_PATTERN.findall(description.lower())
        return set(matches)

    def _extract_domains(self, description: str) -> set[str]:
        """Extract domain names from a constraint description.

        Args:
            description: The constraint description

        Returns:
            Set of domain names found
        """
        # Simple pattern for domain names
        domain_pattern = re.compile(r'\b[\w.-]+\.(?:com|org|net|io|dev|app)\b')
        matches = domain_pattern.findall(description.lower())
        return set(matches)

    def _extract_package_names(self, description: str) -> set[str]:
        """Extract package names from a constraint description.

        Args:
            description: The constraint description

        Returns:
            Set of package names found
        """
        # Simple pattern for package names (alphanumeric with hyphens/underscores)
        package_pattern = re.compile(r'\b[\w-]+\b')
        matches = package_pattern.findall(description.lower())
        # Filter out common words that aren't package names
        excluded = {"install", "package", "dependency", "dependencies"}
        return set(matches) - excluded

    def _validate_and_normalize_path(self, path: str) -> str:
        """Validate and normalize a file path.

        Ensures the path is relative, doesn't contain .. traversal,
        and is not empty.

        Args:
            path: The path to validate

        Returns:
            Normalized path

        Raises:
            CompilationError: If the path is invalid
        """
        if not path or not path.strip():
            raise CompilationError("Empty path is not allowed")

        # Check for absolute path
        if Path(path).is_absolute():
            raise CompilationError(f"Absolute path '{path}' is not allowed")

        # Check for path traversal
        if ".." in path:
            raise CompilationError(f"Path traversal '..' in '{path}' is not allowed")

        # Normalize the path
        normalized = str(Path(path).as_posix())

        # Remove leading ./ if present
        if normalized.startswith("./"):
            normalized = normalized[2:]

        return normalized

    def _merge_policies(
        self,
        compiled_constraints: list[CompiledConstraint],
        builtin_policies: PolicySet,
    ) -> PolicySet:
        """Merge compiled constraints with builtin policies.

        Merging follows the principle that policies can only become
        more strict. Denied sets are unioned, allowed sets are intersected.

        Args:
            compiled_constraints: List of compiled constraints
            builtin_policies: Built-in system policies

        Returns:
            Merged PolicySet
        """
        merged = PolicySet()

        # Separate constraints by type
        write_policies = [
            c for c in compiled_constraints
            if isinstance(c, CompiledPathPolicy) and c.kind == "WRITE_SCOPE"
        ]
        read_policies = [
            c for c in compiled_constraints
            if isinstance(c, CompiledPathPolicy) and c.kind == "READ_SCOPE"
        ]
        command_policies = [
            c for c in compiled_constraints
            if isinstance(c, CompiledCommandPolicy)
        ]
        network_policies = [
            c for c in compiled_constraints
            if isinstance(c, CompiledNetworkPolicy)
        ]
        dependency_policies = [
            c for c in compiled_constraints
            if isinstance(c, CompiledDependencyPolicy)
        ]

        # Merge write policies with builtin
        merged.write_policies = self._merge_path_policies(
            write_policies,
            builtin_policies.write_policies,
        )

        # Merge read policies with builtin
        merged.read_policies = self._merge_path_policies(
            read_policies,
            builtin_policies.read_policies,
        )

        # Merge command policies with builtin
        merged.command_policies = self._merge_command_policies(
            command_policies,
            builtin_policies.command_policies,
        )

        # Merge network policies with builtin
        merged.network_policies = self._merge_network_policies(
            network_policies,
            builtin_policies.network_policies,
        )

        # Merge dependency policies with builtin
        merged.dependency_policies = self._merge_dependency_policies(
            dependency_policies,
            builtin_policies.dependency_policies,
        )

        return merged

    def _merge_path_policies(
        self,
        issue_policies: list[CompiledPathPolicy],
        builtin_policies: list[CompiledPathPolicy],
    ) -> list[CompiledPathPolicy]:
        """Merge path policies, ensuring they only become more strict.

        Args:
            issue_policies: Path policies from the issue
            builtin_policies: Built-in path policies

        Returns:
            Merged path policies
        """
        if not issue_policies:
            return builtin_policies

        # Start with builtin policies
        merged = list(builtin_policies)

        # Apply issue policies on top
        for issue_policy in issue_policies:
            if issue_policy.is_allowlist:
                # For allowlist policies, we need to ensure that:
                # 1. Builtin denylists are still enforced
                # 2. The issue allowlist is intersected with any existing allowlists

                # Create a new merged policy that combines denylist with allowlist
                # First, collect all denied paths from builtins
                all_denied = set()
                for builtin in merged:
                    all_denied |= builtin.denied_paths

                # Remove denied paths from the issue allowlist
                filtered_allowed = issue_policy.allowed_paths - all_denied

                # Add or update the merged policy
                existing = next(
                    (p for p in merged if p.id == issue_policy.id),
                    None,
                )
                if existing:
                    existing.allowed_paths = filtered_allowed
                    existing.denied_paths = all_denied
                    existing.is_allowlist = True
                else:
                    merged.append(
                        CompiledPathPolicy(
                            id=issue_policy.id,
                            description=issue_policy.description,
                            kind=issue_policy.kind,
                            allowed_paths=filtered_allowed,
                            denied_paths=all_denied,
                            is_allowlist=True,
                        )
                    )
            else:
                # Union of denied paths
                for builtin in merged:
                    builtin.denied_paths |= issue_policy.denied_paths

        return merged

    def _merge_command_policies(
        self,
        issue_policies: list[CompiledCommandPolicy],
        builtin_policies: list[CompiledCommandPolicy],
    ) -> list[CompiledCommandPolicy]:
        """Merge command policies, ensuring they only become more strict.

        Args:
            issue_policies: Command policies from the issue
            builtin_policies: Built-in command policies

        Returns:
            Merged command policies
        """
        if not issue_policies:
            return builtin_policies

        # Start with builtin policies
        merged = list(builtin_policies)

        # Apply issue policies on top
        for issue_policy in issue_policies:
            if issue_policy.is_allowlist:
                # Intersection of allowed commands
                for builtin in merged:
                    if builtin.is_allowlist:
                        builtin.allowed_commands &= issue_policy.allowed_commands
                    else:
                        # Builtin is denylist, convert to allowlist with intersection
                        builtin.is_allowlist = True
                        builtin.allowed_commands = (
                            builtin.allowed_commands & issue_policy.allowed_commands
                        )
                        builtin.denied_commands = set()
            else:
                # Union of denied commands
                for builtin in merged:
                    builtin.denied_commands |= issue_policy.denied_commands

        return merged

    def _merge_network_policies(
        self,
        issue_policies: list[CompiledNetworkPolicy],
        builtin_policies: list[CompiledNetworkPolicy],
    ) -> list[CompiledNetworkPolicy]:
        """Merge network policies, ensuring they only become more strict.

        Args:
            issue_policies: Network policies from the issue
            builtin_policies: Built-in network policies

        Returns:
            Merged network policies
        """
        if not issue_policies:
            return builtin_policies

        # Start with builtin policies
        merged = list(builtin_policies)

        # Apply issue policies on top
        for issue_policy in issue_policies:
            for builtin in merged:
                # If any policy denies all, result denies all
                builtin.deny_all = builtin.deny_all or issue_policy.deny_all
                # Union of denied domains
                builtin.denied_domains |= issue_policy.denied_domains
                # Intersection of allowed domains
                builtin.allowed_domains &= issue_policy.allowed_domains

        return merged

    def _merge_dependency_policies(
        self,
        issue_policies: list[CompiledDependencyPolicy],
        builtin_policies: list[CompiledDependencyPolicy],
    ) -> list[CompiledDependencyPolicy]:
        """Merge dependency policies, ensuring they only become more strict.

        Args:
            issue_policies: Dependency policies from the issue
            builtin_policies: Built-in dependency policies

        Returns:
            Merged dependency policies
        """
        if not issue_policies:
            return builtin_policies

        # Start with builtin policies
        merged = list(builtin_policies)

        # Apply issue policies on top
        for issue_policy in issue_policies:
            for builtin in merged:
                # If any policy denies installation, result denies installation
                builtin.deny_installation = (
                    builtin.deny_installation or issue_policy.deny_installation
                )
                # If any policy denies lockfile modification, result denies it
                builtin.deny_lockfile_modification = (
                    builtin.deny_lockfile_modification
                    or issue_policy.deny_lockfile_modification
                )
                # Intersection of allowed packages
                builtin.allowed_packages &= issue_policy.allowed_packages

        return merged
