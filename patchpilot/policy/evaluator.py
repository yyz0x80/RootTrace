"""Policy evaluator for enforcing compiled constraints.

This module implements the policy evaluator that checks whether specific
operations (file access, command execution, network access, etc.) are
allowed according to the compiled policy set.

The evaluator is called by the tool implementations to enforce security
boundaries during agent execution.
"""

from pathlib import Path

from patchpilot.policy.schema import PolicySet


class PolicyEvaluator:
    """Evaluator for enforcing compiled policies during agent execution."""

    def __init__(self, policy_set: PolicySet):
        """Initialize the policy evaluator with a compiled policy set.

        Args:
            policy_set: The compiled PolicySet to enforce
        """
        self.policy_set = policy_set

    def assert_read_allowed(self, relative_path: str) -> None:
        """Check if file read is allowed according to read policies.

        Args:
            relative_path: Relative path to the file to read

        Raises:
            PermissionError: If the read operation is not allowed
        """
        normalized_path = self._normalize_path(relative_path)

        for policy in self.policy_set.read_policies:
            if policy.is_allowlist:
                # Allowlist: path must be in allowed_paths
                if not self._path_matches_any(normalized_path, policy.allowed_paths):
                    raise PermissionError(
                        f"Read denied: '{relative_path}' is not in the allowed read scope. "
                        f"Policy: {policy.description}"
                    )
            else:
                # Denylist: path must not be in denied_paths
                if self._path_matches_any(normalized_path, policy.denied_paths):
                    raise PermissionError(
                        f"Read denied: '{relative_path}' is in the denied read scope. "
                        f"Policy: {policy.description}"
                    )

    def assert_write_allowed(self, relative_path: str) -> None:
        """Check if file write is allowed according to write policies.

        Args:
            relative_path: Relative path to the file to write

        Raises:
            PermissionError: If the write operation is not allowed
        """
        normalized_path = self._normalize_path(relative_path)

        for policy in self.policy_set.write_policies:
            if policy.is_allowlist:
                # Allowlist: path must be in allowed_paths
                if not self._path_matches_any(normalized_path, policy.allowed_paths):
                    raise PermissionError(
                        f"Write denied: '{relative_path}' is not in the allowed write scope. "
                        f"Policy: {policy.description}"
                    )
            else:
                # Denylist: path must not be in denied_paths
                if self._path_matches_any(normalized_path, policy.denied_paths):
                    raise PermissionError(
                        f"Write denied: '{relative_path}' is in the denied write scope. "
                        f"Policy: {policy.description}"
                    )

    def assert_command_allowed(self, command: str) -> None:
        """Check if command execution is allowed according to command policies.

        Args:
            command: The command string to execute

        Raises:
            PermissionError: If the command is not allowed
        """
        normalized_command = command.lower().strip()

        for policy in self.policy_set.command_policies:
            if policy.is_allowlist:
                # Allowlist: command must match an allowed pattern
                if not self._command_matches_any(normalized_command, policy.allowed_commands):
                    raise PermissionError(
                        f"Command denied: '{command}' is not in the allowed command list. "
                        f"Policy: {policy.description}"
                    )
            else:
                # Denylist: command must not match a denied pattern
                if self._command_matches_any(normalized_command, policy.denied_commands):
                    raise PermissionError(
                        f"Command denied: '{command}' is in the denied command list. "
                        f"Policy: {policy.description}"
                    )

    def assert_network_allowed(self, domain: str | None = None) -> None:
        """Check if network access is allowed according to network policies.

        Args:
            domain: Optional domain name being accessed

        Raises:
            PermissionError: If network access is not allowed
        """
        for policy in self.policy_set.network_policies:
            if policy.deny_all:
                raise PermissionError(
                    f"Network access denied: {policy.description}"
                )

            if domain:
                normalized_domain = domain.lower().strip()

                # Check if domain is in denied list
                if self._domain_matches(normalized_domain, policy.denied_domains):
                    raise PermissionError(
                        f"Network access denied: Domain '{domain}' is in the denied list. "
                        f"Policy: {policy.description}"
                    )

                # If allowlist is configured, check if domain is allowed
                if policy.allowed_domains and not self._domain_matches(normalized_domain, policy.allowed_domains):
                    raise PermissionError(
                        f"Network access denied: Domain '{domain}' is not in the allowed list. "
                        f"Policy: {policy.description}"
                    )

    def assert_dependency_installation_allowed(self, package_name: str) -> None:
        """Check if dependency installation is allowed according to dependency policies.

        Args:
            package_name: The name of the package to install

        Raises:
            PermissionError: If dependency installation is not allowed
        """
        for policy in self.policy_set.dependency_policies:
            if policy.deny_installation:
                raise PermissionError(
                    f"Dependency installation denied: {policy.description}"
                )

            # If allowlist is configured, check if package is allowed
            if policy.allowed_packages:
                normalized_package = package_name.lower().strip()
                if normalized_package not in policy.allowed_packages:
                    raise PermissionError(
                        f"Dependency installation denied: Package '{package_name}' is not in the allowed list. "
                        f"Policy: {policy.description}"
                    )

    def assert_lockfile_modification_allowed(self, lockfile_path: str) -> None:
        """Check if lockfile modification is allowed according to dependency policies.

        Args:
            lockfile_path: Path to the lockfile being modified

        Raises:
            PermissionError: If lockfile modification is not allowed
        """
        for policy in self.policy_set.dependency_policies:
            if policy.deny_lockfile_modification:
                raise PermissionError(
                    f"Lockfile modification denied: {policy.description}"
                )

    def _normalize_path(self, path: str) -> str:
        """Normalize a file path for consistent comparison.

        Args:
            path: The path to normalize

        Returns:
            Normalized path string
        """
        # Convert to POSIX-style path
        normalized = str(Path(path).as_posix())

        # Remove leading ./ if present
        normalized = normalized.removeprefix("./")

        # Remove trailing slashes
        normalized = normalized.rstrip("/")

        return normalized

    def _path_matches_any(self, path: str, patterns: set[str]) -> bool:
        """Check if a path matches any of the given patterns.

        Supports both exact matches and directory prefix matches.
        For example, "tests" pattern matches "tests/example.py" and "tests/subdir/file.py".
        Also supports prefix patterns like "test_" to match files starting with "test_".

        Args:
            path: The normalized path to check
            patterns: Set of path patterns to match against

        Returns:
            True if the path matches any pattern, False otherwise
        """
        for pattern in patterns:
            normalized_pattern = self._normalize_path(pattern)

            # Exact match
            if path == normalized_pattern:
                return True

            # Directory prefix match (pattern is a directory, path is inside it)
            if path.startswith(normalized_pattern + "/"):
                return True

            # File pattern match (pattern contains a file extension)
            if "." in normalized_pattern:
                # Check if this is a specific file pattern
                pattern_parts = normalized_pattern.split("/")
                if len(pattern_parts) > 1:
                    # Pattern includes directory, check exact match or prefix
                    if path.startswith(normalized_pattern):
                        return True
                else:
                    # Pattern is just a filename, check if path ends with it
                    if path.endswith("/" + normalized_pattern) or path == normalized_pattern:
                        return True

            # Prefix pattern match (e.g., "test_" matches "test_example.py")
            if not "/" in normalized_pattern and normalized_pattern.endswith("_"):
                # Check if filename starts with the pattern
                filename = path.split("/")[-1]
                if filename.startswith(normalized_pattern):
                    return True

        return False

    def _command_matches_any(self, command: str, patterns: set[str]) -> bool:
        """Check if a command matches any of the given patterns.

        Supports both exact matches and prefix matches.
        For example, "git" pattern matches "git status" and "git diff".

        Args:
            command: The normalized command to check
            patterns: Set of command patterns to match against

        Returns:
            True if the command matches any pattern, False otherwise
        """
        for pattern in patterns:
            normalized_pattern = pattern.lower().strip()

            # Exact match
            if command == normalized_pattern:
                return True

            # Prefix match (command starts with pattern)
            if command.startswith(normalized_pattern + " "):
                return True

        return False

    def _domain_matches(self, domain: str, patterns: set[str]) -> bool:
        """Check if a domain matches any of the given patterns.

        Supports both exact matches and subdomain matches.
        For example, "example.com" pattern matches "api.example.com".

        Args:
            domain: The normalized domain to check
            patterns: Set of domain patterns to match against

        Returns:
            True if the domain matches any pattern, False otherwise
        """
        for pattern in patterns:
            normalized_pattern = pattern.lower().strip()

            # Exact match
            if domain == normalized_pattern:
                return True

            # Subdomain match (domain ends with pattern)
            if domain.endswith("." + normalized_pattern):
                return True

        return False
