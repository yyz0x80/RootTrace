"""Docker sandbox execution environment for isolated code operations.

This module provides the DockerSandbox class which creates and manages
isolated Docker containers for executing code operations safely. It enforces:

- Network isolation (no external network access by default)
- Resource limits (CPU, memory, process count)
- Security restrictions (read-only root filesystem, dropped capabilities)
- Workspace isolation (only mounted target repository)
- Non-root user execution
- Command timeout enforcement

The DockerSandbox is the primary execution environment for running
verification commands (pytest, ruff, git) in a controlled, isolated context.
"""

from __future__ import annotations

import os
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CommandResult:
    """Result of executing a command in the Docker sandbox.
    
    Attributes:
        command: The command string that was executed
        exit_code: The process exit code (124 indicates timeout)
        stdout: Standard output from the command
        stderr: Standard error from the command
        duration_seconds: Execution time in seconds
        timed_out: Whether the command timed out
    """
    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False


class DockerSandbox:
    """Manage an isolated Docker container for secure code execution.
    
    The DockerSandbox creates a container with:
    - Network isolation (no external access)
    - Resource limits (CPU, memory, process count)
    - Security hardening (read-only root, dropped capabilities)
    - Non-root user execution
    - Workspace mount at /workspace
    
    Example:
        with DockerSandbox(workspace=Path("/path/to/repo")) as sandbox:
            result = sandbox.run("pytest tests/ -q")
            if result.exit_code == 0:
                print("Tests passed")
    
    Args:
        workspace: Path to the repository workspace to mount
        image: Docker image name (default: patchpilot-sandbox:py312)
        cpus: CPU limit (default: 1.0)
        memory: Memory limit string (default: "512m")
    """
    
    def __init__(
        self,
        workspace: Path,
        image: str = "patchpilot-sandbox:py312",
        cpus: float = 1.0,
        memory: str = "512m",
    ):
        # Use absolute path but avoid resolving symlinks on macOS
        # Docker Desktop on macOS has better compatibility with /var/folders paths
        self.workspace = workspace.absolute()
        if not self.workspace.exists():
            raise RuntimeError(
                f"Workspace path does not exist: {self.workspace}. "
                f"Ensure the directory is created before starting the sandbox."
            )
        if not self.workspace.is_dir():
            raise RuntimeError(
                f"Workspace path is not a directory: {self.workspace}"
            )
        self.image = image
        self.cpus = cpus
        self.memory = memory
        self.container_name = f"patchpilot-{uuid.uuid4().hex[:12]}"

    def start(self) -> None:
        """Start the Docker container with security and resource limits."""
        # On macOS, use the actual filesystem path for Docker bind mounts
        # This avoids issues with resolved symlink paths that Docker may not recognize
        workspace_source = str(self.workspace)
        
        command = [
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            self.container_name,
            
            # Network isolation - prevent external network access
            "--network",
            "none",
            
            # Resource limits
            "--cpus",
            str(self.cpus),
            "--memory",
            self.memory,
            "--pids-limit",
            "128",
            
            # Security hardening
            "--security-opt",
            "no-new-privileges",
            "--cap-drop",
            "ALL",
            
            # Read-only root filesystem
            "--read-only",
            
            # Temporary directory for pytest and other tools
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=64m",
            
            # Mount only the target workspace
            "--mount",
            (
                f"type=bind,"
                f"source={workspace_source},"
                f"target=/workspace"
            ),
            
            "--workdir",
            "/workspace",
            
            # Run as non-root user using host UID/GID
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            
            "--env",
            "HOME=/tmp",
            
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            
            self.image,
            "sleep",
            "infinity",
        ]

        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            # Provide detailed error information for debugging
            stderr = e.stderr if e.stderr else "No error output"
            raise RuntimeError(
                f"Failed to start Docker container. "
                f"Exit code: {e.returncode}. "
                f"Error: {stderr}. "
                f"Command: {' '.join(command)}. "
                f"Workspace path: {workspace_source}"
            ) from e

    def run(
        self,
        command: str,
        timeout_seconds: int = 30,
    ) -> CommandResult:
        """Execute a command inside the Docker container.
        
        Args:
            command: Shell command to execute
            timeout_seconds: Maximum execution time in seconds
            
        Returns:
            CommandResult with execution details and output
        """
        started = time.monotonic()

        docker_command = [
            "docker",
            "exec",
            self.container_name,
            
            # Linux timeout command for enforcement
            "timeout",
            "-k",
            "2s",
            f"{timeout_seconds}s",
            
            "sh",
            "-lc",
            command,
        ]

        try:
            result = subprocess.run(
                docker_command,
                capture_output=True,
                text=True,
                timeout=timeout_seconds + 5,
                check=False,
            )

            duration = time.monotonic() - started

            return CommandResult(
                command=command,
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                duration_seconds=duration,
                timed_out=result.returncode == 124,
            )

        except subprocess.TimeoutExpired as exc:
            duration = time.monotonic() - started

            return CommandResult(
                command=command,
                exit_code=124,
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
                duration_seconds=duration,
                timed_out=True,
            )

    def stop(self) -> None:
        """Stop and remove the Docker container."""
        subprocess.run(
            [
                "docker",
                "rm",
                "-f",
                self.container_name,
            ],
            capture_output=True,
            text=True,
            check=False,
        )

    def __enter__(self):
        """Context manager entry - start the container."""
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        """Context manager exit - ensure container is stopped."""
        self.stop()
