"""Pinned Docker environments for disposable pytest verification.

Images are selected by an explicit reference, instance mapping, or bounded
SWE-bench instance ID. Dockerfiles from target repositories are never executed.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
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
_TESTBED_PYTHON = "/opt/miniconda3/envs/testbed/bin/python"
_PYTEST_BOOTSTRAP_WHEELS = {
    "pytest8": ("pytest", "8.3.5", "c69214aa47deac29fad6c2a4f590b9c4a9fdb16a403176fe154b79c0b4d4d820"),
    "pytest9": ("pytest", "9.0.3", "2c5efc453d45394fdd706ade797c0a81091eccd1d6e4bccfcd476e2b8e0ab5d9"),
    "pluggy": ("pluggy", "1.5.0", "44e1ad92c8ca002de6377e165f3e0f1be63266ab4d554740532335b9d75ea669"),
    "iniconfig": ("iniconfig", "2.1.0", "9deba5723312380e77435581c6bf4935c94cbfab9b1ed33ef8d238ea168eb760"),
    "packaging": ("packaging", "25.0", "29572ef2b1f17581046b3a2227d5c611fb25ec70ca1ba8554b24b0e69331a484"),
    "exceptiongroup": ("exceptiongroup", "1.2.2", "3111b9d131c238bec2f8f516e123e14ba243563fb135d3fe885990585aa7795b"),
    "tomli": ("tomli", "2.2.1", "cb55c73c5f4408779d0cf3eef9f762b9c9f147a77de7b258bef0a5628adc85cc"),
    "pygments": ("pygments", "2.19.2", "86540386c03d588bb81d44bc3928634ff26449851e99741617ecb9037ee5ec0b"),
}


def _pytest_bootstrap_requirements(python_version: str) -> bytes:
    match = re.fullmatch(r"Python (3)\.(\d+)\.\d+", python_version)
    if match is None:
        raise EnvironmentPreparationError("cannot select pytest wheels for unknown Python version")
    minor = int(match.group(2))
    if not 8 <= minor <= 13:
        raise EnvironmentPreparationError("automatic pytest setup supports Python 3.8 through 3.13")
    names = ["pytest8" if minor < 10 else "pytest9", "pluggy", "iniconfig", "packaging"]
    if minor < 11:
        names.extend(["exceptiongroup", "tomli"])
    if minor >= 10:
        names.append("pygments")
    lines = []
    for package in names:
        name, version, digest = _PYTEST_BOOTSTRAP_WHEELS[package]
        lines.append(f"{name}=={version} --hash=sha256:{digest}")
    return ("\n".join(lines) + "\n").encode()


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
    python_executable: str = "python"
    pytest_bootstrapped: bool = False
    cache_hit: bool = False
    base_digest: str | None = None


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
        bootstrap_pytest: bool = False,
        prefer_testbed_python: bool = False,
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
        self.bootstrap_pytest = bootstrap_pytest
        self.prefer_testbed_python = prefer_testbed_python
        self.python_executable = "python"
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
            detail = result.stderr.strip()
            if "Traceback (most recent call last)" in detail:
                # Python tracebacks end with the actionable exception message.
                detail = detail.splitlines()[-1]
            detail = detail[:300]
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
                "--tmpfs", "/tmp:rw,nosuid,nodev,size=128m",
                "--env", "HOME=/tmp", "--env", "PYTHONDONTWRITEBYTECODE=1",
                "--env", "PYTHONPATH=/roottrace-deps", "--entrypoint", self.python_executable,
                image_id, "-m", "pytest", "--version",
            ], timeout=30)
        except EnvironmentPreparationError as exc:
            if "No module named pytest" in str(exc):
                raise EnvironmentPreparationError(
                    "verification image lacks pytest in its selected Python"
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
        if self.prefer_testbed_python:
            try:
                testbed = self._run([
                    "docker", "run", "--rm", "--pull", "never", "--network", "none",
                    "--read-only", "--cap-drop", "ALL", "--security-opt",
                    "no-new-privileges", "--pids-limit", "64", "--memory", "512m",
                    "--cpus", "1", "--platform", self.platform_name,
                    "--entrypoint", _TESTBED_PYTHON, image_id, "--version",
                ], timeout=30).stdout.strip()
            except EnvironmentPreparationError:
                testbed = ""
            if testbed.startswith("Python 3."):
                self.python_executable = _TESTBED_PYTHON
        python = self._run([
            "docker", "run", "--rm", "--network", "none", "--read-only",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--pids-limit", "64", "--memory", "512m", "--cpus", "1",
            "--platform", self.platform_name, "--entrypoint", self.python_executable,
            image_id, "--version",
        ], timeout=30).stdout.strip()
        if not python.startswith("Python 3."):
            raise EnvironmentPreparationError("image lacks a supported Python runtime")
        bootstrapped = False
        if self.requirements is None:
            try:
                self._check_pytest(image_id)
            except EnvironmentPreparationError as exc:
                if not self.bootstrap_pytest or "lacks pytest" not in str(exc):
                    raise
                dependency_bytes = _pytest_bootstrap_requirements(python)
                bootstrapped = True
            else:
                dependency_bytes = b""
        else:
            try:
                dependency_bytes = self.requirements.read_bytes()
            except OSError as exc:
                raise EnvironmentPreparationError("requirements file is unavailable") from exc
        if dependency_bytes:
            lines = dependency_bytes.decode("utf-8").splitlines()
            if not lines or any(not _PINNED_REQUIREMENT.fullmatch(line) for line in lines):
                raise EnvironmentPreparationError("requirements must contain only hash-pinned package versions")
        key = hashlib.sha256(b"\0".join([
            digest.encode(), self.platform_name.encode(), python.encode(),
            self.python_executable.encode(),
            dependency_bytes, b"pip-hashes-no-deps-target-v1",
        ])).hexdigest()
        if not dependency_bytes:
            return DockerEnvironment(
                image_id, digest, actual_platform, python, key, image, pulled,
                self.python_executable, False, False, digest,
            )
        index_url = self.index_url or ("https://pypi.org/simple" if bootstrapped else None)
        if index_url is None or not index_url.startswith("https://"):
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
            return DockerEnvironment(cached_id, cached_id, actual_platform, python, key, image, pulled,
                                     self.python_executable, bootstrapped, True, digest)
        with tempfile.TemporaryDirectory(
            prefix="roottrace-wheels-", dir=self.wheels_parent,
        ) as directory:
            wheels = Path(directory)
            staged_requirements = wheels / "requirements.txt"
            staged_requirements.write_bytes(dependency_bytes)
            download_argv = [
                "python", "-m", "pip", "download", "--isolated", "--require-hashes",
                "--only-binary", ":all:", "--no-deps",
            ]
            if bootstrapped:
                download_argv.extend([
                    "--platform", "any", "--implementation", "py", "--abi", "none",
                    "--python-version", python.split()[1].rsplit(".", 1)[0],
                ])
            download_argv.extend([
                "--index-url", index_url, "-r", str(staged_requirements),
                "-d", str(wheels),
            ])
            self._run(download_argv)
            name = f"roottrace-install-{uuid.uuid4().hex}"
            try:
                self._run([
                    "docker", "run", "--name", name, "--network", "none",
                    "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                    "--pids-limit", "64", "--memory", "1g", "--cpus", "1",
                    "--platform", self.platform_name,
                    "-v", f"{wheels}:/wheels:ro",
                    "--user", "0:0", "--entrypoint", self.python_executable,
                    image_id, "-m", "pip", "install", "--no-index",
                    "--find-links", "/wheels", "--no-deps", "--target", "/roottrace-deps",
                    "-r", "/wheels/requirements.txt",
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
        return DockerEnvironment(committed, committed, actual_platform, python, key, image, pulled,
                                 self.python_executable, bootstrapped, False, digest)

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
