"""Docker sandbox execution environment for PatchPilot.

This module provides the DockerSandbox class for creating isolated
Docker containers to execute code operations safely with network
isolation, resource limits, and security restrictions.
"""

from patchpilot.sandbox.docker_runner import CommandResult, DockerSandbox

__all__ = ["CommandResult", "DockerSandbox"]
