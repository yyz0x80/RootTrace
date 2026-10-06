"""Pinned Docker environments for disposable pytest verification.

Only an explicitly selected image (or an instance mapping) is accepted.  Image
inspection is local; Dockerfiles from target repositories are never executed.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

_IMAGE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@-]{0,255}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_SWE_INSTANCE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*__[A-Za-z0-9_.-]+-[0-9]+$")
_PINNED_REQUIREMENT = re.compile(
    r"^[A-Za-z0-9_.-]+==[A-Za-z0-9_.!+-]+ --hash=sha256:[0-9a-f]{64}$"
)


class EnvironmentPreparationError(RuntimeError):
    """A test environment could not be prepared safely."""


@dataclass(frozen=True)
class DockerEnvironment:
    image: str
    digest: str
    platform: str
    python_version: str
    cache_key: str
    reference: str | None = None
    pulled: bool = False


def swebench_image_reference(instance_id: str) -> str:
    """Derive only the official Docker Hub instance reference."""
    if not _SWE_INSTANCE.fullmatch(instance_id) or len(instance_id) > 200:
        raise EnvironmentPreparationError("invalid SWE-bench instance ID")
    key = instance_id.lower().replace("__", "_1776_")
    return f"swebench/sweb.eval.x86_64.{key}:latest"


class DockerEnvironmentPreparer:
    """Resolve a local image and optionally install hash-pinned wheels offline."""

    def __init__(
        self,
        image: str | None = None,
        *,
        instance_id: str | None = None,
        image_map: Path | None = None,
        platform_name: str | None = None,
        requirements: Path | None = None,
        index_url: str | None = None,
        timeout_seconds: int = 180,
        wheels_parent: Path | None = None,
        swebench_image: bool = False,
        pull_missing: bool = False,
    ) -> None:
        self.image = image
        self.instance_id = instance_id
        self.image_map = image_map
        machine = platform.machine().lower().replace("x86_64", "amd64").replace("aarch64", "arm64")
        self.platform_name = platform_name
        self._default_platform = f"linux/{machine}"
        self.requirements = requirements
        self.index_url = index_url
        self.timeout_seconds = timeout_seconds
        self.wheels_parent = wheels_parent
        self.swebench_image = swebench_image
        self.pull_missing = pull_missing
        self._deadline: float | None = None

    def _run(self, argv: list[str], *, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
        remaining = (self._deadline - time.monotonic()) if self._deadline else self.timeout_seconds
        if remaining <= 0:
            raise EnvironmentPreparationError("environment preparation timed out")
        try:
            result = subprocess.run(
                argv, capture_output=True, text=True,
                timeout=min(timeout or self.timeout_seconds, remaining), check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise EnvironmentPreparationError(f"environment command failed: {type(exc).__name__}") from exc
        if result.returncode:
            detail = result.stderr.strip()[:300]
            detail = re.sub(r"https?://\S+", "<url>", detail)
            detail = re.sub(r"(?<![A-Za-z0-9])/(?:[^\s:]+)", "<path>", detail)
            raise EnvironmentPreparationError(
                f"environment command failed ({argv[0]} {argv[1]}): "
                f"{detail or f'exit status {result.returncode}'}"
            )
        return result

    def _select(self, base_commit: str) -> str:
        if self.image_map is not None:
            if not self.instance_id:
                raise EnvironmentPreparationError("instance_id is required for image mapping")
            try:
                mapping = json.loads(self.image_map.read_text(encoding="utf-8"))
                entry = mapping[self.instance_id]
                image = entry["image"]
                if entry["base_commit"] != base_commit:
                    raise EnvironmentPreparationError("instance image base commit mismatch")
                if self.platform_name is not None and entry["platform"] != self.platform_name:
                    raise EnvironmentPreparationError("instance image platform mismatch")
                self.platform_name = entry["platform"]
            except (OSError, KeyError, TypeError, ValueError) as exc:
                raise EnvironmentPreparationError("instance image mapping is missing or invalid") from exc
        elif self.swebench_image:
            if self.image is not None or self.instance_id is None:
                raise EnvironmentPreparationError("automatic SWE-bench image selection is invalid")
            image = swebench_image_reference(self.instance_id)
            if self.platform_name is not None and self.platform_name != "linux/amd64":
                raise EnvironmentPreparationError("SWE-bench image requires linux/amd64")
            self.platform_name = "linux/amd64"
        else:
            image = self.image
        if not isinstance(image, str) or not _IMAGE.fullmatch(image):
            raise EnvironmentPreparationError("no valid verification image selected")
        if self.platform_name is None:
            self.platform_name = self._default_platform
        if self.platform_name not in {"linux/amd64", "linux/arm64"}:
            raise EnvironmentPreparationError("unsupported verification image platform")
        return image

    def _check_pytest(self, image_id: str) -> None:
        """Fail preparation when the selected environment cannot start pytest."""
        try:
            self._run([
                "docker", "run", "--rm", "--pull", "never", "--network", "none",
                "--read-only", "--cap-drop", "ALL", "--security-opt",
                "no-new-privileges", "--pids-limit", "64", "--memory", "512m",
                "--cpus", "1", "--platform", self.platform_name,
                "--env", "PYTHONPATH=/roottrace-deps", "--entrypoint", "python",
                image_id, "-m", "pytest", "--version",
            ], timeout=30)
        except EnvironmentPreparationError as exc:
            if "No module named pytest" in str(exc):
                raise EnvironmentPreparationError(
                    "verification image lacks pytest in its default Python"
                ) from exc
            raise

    def _check_swebench_commit(self, image_id: str, base_commit: str) -> None:
        """Prove the official image contains the target revision without edits."""
        try:
            self._run([
                "docker", "run", "--rm", "--pull", "never", "--network", "none",
                "--read-only", "--cap-drop", "ALL", "--security-opt",
                "no-new-privileges", "--pids-limit", "64", "--memory", "512m",
                "--cpus", "1", "--platform", self.platform_name,
                "--entrypoint", "git", image_id, "-C", "/testbed",
                "merge-base", "--is-ancestor", base_commit, "HEAD",
            ], timeout=30)
        except EnvironmentPreparationError as exc:
            raise EnvironmentPreparationError(
                "SWE-bench image cannot verify the target base commit"
            ) from exc

    def prepare(self, base_commit: str) -> DockerEnvironment:
        self._deadline = time.monotonic() + self.timeout_seconds
        image = self._select(base_commit)
        pulled = False
        try:
            inspected = self._run(["docker", "image", "inspect", image])
        except EnvironmentPreparationError as exc:
            missing = "No such image" in str(exc) or "No such object" in str(exc)
            if not (self.swebench_image and self.pull_missing and missing):
                raise
            self._run([
                "docker", "pull", "--quiet", "--platform", self.platform_name, image,
            ])
            pulled = True
            inspected = self._run(["docker", "image", "inspect", image])
        try:
            metadata = json.loads(inspected.stdout)[0]
            repo_digests = metadata.get("RepoDigests") or []
            digest = next((item.rsplit("@", 1)[-1] for item in repo_digests if "@" in item), None)
            if digest is None:
                # Local image ID is content-addressed even when no repository digest exists.
                digest = metadata["Id"]
            if not _DIGEST.fullmatch(digest):
                raise ValueError("image has no content digest")
            image_id = metadata["Id"]
            if not _DIGEST.fullmatch(image_id):
                raise ValueError("image has no content ID")
            actual_platform = f"{metadata['Os']}/{metadata['Architecture']}"
            if actual_platform != self.platform_name:
                raise EnvironmentPreparationError("verification image platform mismatch")
            labels = metadata.get("Config", {}).get("Labels") or {}
            if self.image_map is not None and labels.get("roottrace.base_commit") != base_commit:
                raise EnvironmentPreparationError("verification image commit label mismatch")
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise EnvironmentPreparationError("invalid Docker image inspection result") from exc
        if self.swebench_image:
            self._check_swebench_commit(image_id, base_commit)
        python = self._run([
            "docker", "run", "--rm", "--network", "none", "--read-only",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--pids-limit", "64", "--memory", "512m", "--cpus", "1",
            "--platform", self.platform_name, "--entrypoint", "python",
            image_id, "--version",
        ], timeout=30).stdout.strip()
        if not python.startswith("Python 3."):
            raise EnvironmentPreparationError("image lacks a supported Python runtime")
        try:
            dependency_bytes = self.requirements.read_bytes() if self.requirements else b""
        except OSError as exc:
            raise EnvironmentPreparationError("requirements file is unavailable") from exc
        if dependency_bytes:
            lines = dependency_bytes.decode("utf-8").splitlines()
            if not lines or any(not _PINNED_REQUIREMENT.fullmatch(line) for line in lines):
                raise EnvironmentPreparationError("requirements must contain only hash-pinned package versions")
        key = hashlib.sha256(b"\0".join([
            digest.encode(), self.platform_name.encode(), python.encode(),
            dependency_bytes, b"pip-hashes-no-deps-target-v1",
        ])).hexdigest()
        if self.requirements is None:
            self._check_pytest(image_id)
            return DockerEnvironment(image_id, digest, actual_platform, python, key, image, pulled)
        if self.index_url is None or not self.index_url.startswith("https://"):
            raise EnvironmentPreparationError("hash-pinned dependency download requires an HTTPS index")
        cached = f"roottrace-env:{key}"
        found = subprocess.run(
            ["docker", "image", "inspect", cached], capture_output=True,
            text=True, timeout=30, check=False,
        )
        if found.returncode == 0:
            try:
                cached_data = json.loads(found.stdout)[0]
                cached_id = cached_data["Id"]
                if not _DIGEST.fullmatch(cached_id):
                    raise ValueError("invalid cached image ID")
                if (cached_data.get("Config", {}).get("Labels") or {}).get(
                    "roottrace.cache_key"
                ) != key:
                    raise ValueError("cache ownership label mismatch")
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                raise EnvironmentPreparationError("invalid cached Docker image") from exc
            self._check_pytest(cached_id)
            return DockerEnvironment(cached_id, digest, actual_platform, python, key, image, pulled)
        with tempfile.TemporaryDirectory(
            prefix="roottrace-wheels-", dir=self.wheels_parent,
        ) as directory:
            wheels = Path(directory)
            staged_requirements = wheels / "requirements.txt"
            shutil.copyfile(self.requirements, staged_requirements)
            self._run([
                "python", "-m", "pip", "download", "--isolated", "--require-hashes",
                "--only-binary", ":all:", "--no-deps", "--index-url", self.index_url,
                "-r", str(staged_requirements), "-d", str(wheels),
            ])
            name = f"roottrace-install-{uuid.uuid4().hex}"
            try:
                self._run([
                    "docker", "run", "--name", name, "--network", "none",
                    "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                    "--pids-limit", "64", "--memory", "1g", "--cpus", "1",
                    "--platform", self.platform_name,
                    "-v", f"{wheels}:/wheels:ro",
                    "-v", f"{staged_requirements}:/requirements:ro",
                    "--user", "0:0", "--entrypoint", "python",
                    image_id, "-m", "pip", "install", "--no-index",
                    "--find-links", "/wheels", "--no-deps", "--target", "/roottrace-deps",
                    "-r", "/requirements",
                ])
                committed = self._run([
                    "docker", "commit", "--change", f"LABEL roottrace.cache_key={key}",
                    name, cached,
                ]).stdout.strip()
            finally:
                subprocess.run(["docker", "rm", "-f", name], capture_output=True,
                               timeout=30, check=False)
        if not _DIGEST.fullmatch(committed):
            raise EnvironmentPreparationError("Docker commit returned no content ID")
        self._check_pytest(committed)
        return DockerEnvironment(committed, digest, actual_platform, python, key, image, pulled)

    def remove_cached_environment(self, cache_key: str) -> bool:
        """Remove only a RootTrace-labeled cache tag; never remove base images."""
        if not re.fullmatch(r"[0-9a-f]{64}", cache_key):
            raise ValueError("invalid environment cache key")
        tag = f"roottrace-env:{cache_key}"
        self._deadline = None
        try:
            inspected = self._run(["docker", "image", "inspect", tag], timeout=30)
        except EnvironmentPreparationError:
            return False
        try:
            labels = json.loads(inspected.stdout)[0]["Config"]["Labels"] or {}
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            raise EnvironmentPreparationError("invalid cached Docker image") from exc
        if labels.get("roottrace.cache_key") != cache_key:
            raise EnvironmentPreparationError("cache ownership label mismatch")
        self._run(["docker", "image", "rm", tag], timeout=30)
        return True
