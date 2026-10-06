"""Docker verification policy tests without requiring a local daemon."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from roottrace.runtime.docker import (
    DockerEnvironment,
    DockerEnvironmentPreparer,
    EnvironmentPreparationError,
    swebench_image_reference,
)
from roottrace.runtime.sandbox import (
    PytestExecutionClassification,
    RuntimeVerificationSandbox,
)
from roottrace.runtime.workspace import capture_repository_fingerprint


def _image_metadata() -> str:
    return json.dumps([{
        "Id": "sha256:" + "a" * 64,
        "RepoDigests": ["example@sha256:" + "b" * 64],
        "Os": "linux", "Architecture": "amd64",
        "Config": {"Labels": {"roottrace.base_commit": "c" * 40}},
    }])


def test_official_swebench_image_name_is_bounded() -> None:
    assert swebench_image_reference("django__django-14672") == (
        "swebench/sweb.eval.x86_64.django_1776_django-14672:latest"
    )
    for case_id in ("../bad", "x__y-1;echo", "x__y-latest", "x__y-1:other"):
        with pytest.raises(EnvironmentPreparationError, match="instance ID"):
            swebench_image_reference(case_id)


@pytest.mark.parametrize(("local", "pull", "expected_pulls"), [
    (True, True, 0),
    (False, True, 1),
])
def test_swebench_auto_resolves_and_preflights_image(
    monkeypatch, local: bool, pull: bool, expected_pulls: int,
) -> None:
    calls: list[list[str]] = []
    image = swebench_image_reference("django__django-14672")
    inspections = 0

    def fake_run(argv, **kwargs):
        nonlocal inspections
        calls.append(argv)
        if argv[:3] == ["docker", "image", "inspect"]:
            inspections += 1
            if not local and inspections == 1:
                return subprocess.CompletedProcess(argv, 1, "", "No such image")
            return subprocess.CompletedProcess(argv, 0, _image_metadata(), "")
        if argv[:2] == ["docker", "pull"]:
            assert argv[-1] == image
            assert argv[argv.index("--platform") + 1] == "linux/amd64"
            return subprocess.CompletedProcess(argv, 0, "", "")
        if "--entrypoint" in argv and argv[argv.index("--entrypoint") + 1] == "git":
            assert "--is-ancestor" in argv
            assert argv[-2] == "c" * 40
            return subprocess.CompletedProcess(argv, 0, "", "")
        if argv[-1] == "--version" and "pytest" not in argv:
            return subprocess.CompletedProcess(argv, 0, "Python 3.8.20\n", "")
        if argv[-3:] == ["-m", "pytest", "--version"]:
            return subprocess.CompletedProcess(argv, 0, "pytest 8.3.5\n", "")
        raise AssertionError(argv)

    monkeypatch.setattr("roottrace.runtime.docker.subprocess.run", fake_run)
    preparer = DockerEnvironmentPreparer(
        instance_id="django__django-14672", swebench_image=True,
        pull_missing=pull,
    )
    environment = preparer.prepare("c" * 40)
    assert environment.reference == image
    assert environment.platform == "linux/amd64"
    assert environment.pulled is (expected_pulls == 1)
    assert sum(argv[:2] == ["docker", "pull"] for argv in calls) == expected_pulls


def test_missing_swebench_image_without_pull_is_explicit(monkeypatch) -> None:
    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 1, "", "No such image")

    monkeypatch.setattr("roottrace.runtime.docker.subprocess.run", fake_run)
    preparer = DockerEnvironmentPreparer(
        instance_id="django__django-14672", swebench_image=True,
        pull_missing=False,
    )
    with pytest.raises(EnvironmentPreparationError, match="No such image"):
        preparer.prepare("c" * 40)


def test_official_image_pull_failure_is_bounded_and_never_runs_image(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        if argv[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(argv, 1, "", "No such image")
        assert argv[:2] == ["docker", "pull"]
        return subprocess.CompletedProcess(argv, 1, "", "pull access denied")

    monkeypatch.setattr("roottrace.runtime.docker.subprocess.run", fake_run)
    preparer = DockerEnvironmentPreparer(
        instance_id="django__django-14672", swebench_image=True,
        pull_missing=True,
    )
    with pytest.raises(EnvironmentPreparationError, match="pull access denied"):
        preparer.prepare("c" * 40)
    assert len(calls) == 2
    assert calls[-1][-1] == swebench_image_reference("django__django-14672")


@pytest.mark.parametrize(("git_ok", "pytest_ok", "reason"), [
    (False, True, "target base commit"),
    (True, False, "lacks pytest"),
])
def test_swebench_preflight_failure_is_specific(
    monkeypatch, git_ok: bool, pytest_ok: bool, reason: str,
) -> None:
    def fake_run(argv, **kwargs):
        if argv[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(argv, 0, _image_metadata(), "")
        if argv[argv.index("--entrypoint") + 1] == "git":
            return subprocess.CompletedProcess(argv, 0 if git_ok else 1, "", "bad commit")
        if argv[-3:] == ["-m", "pytest", "--version"]:
            return subprocess.CompletedProcess(
                argv, 0 if pytest_ok else 1, "", "No module named pytest",
            )
        return subprocess.CompletedProcess(argv, 0, "Python 3.8.20\n", "")

    monkeypatch.setattr("roottrace.runtime.docker.subprocess.run", fake_run)
    preparer = DockerEnvironmentPreparer(
        instance_id="django__django-14672", swebench_image=True,
    )
    with pytest.raises(EnvironmentPreparationError, match=reason):
        preparer.prepare("c" * 40)


def test_instance_image_is_checked_and_cached(monkeypatch, tmp_path: Path) -> None:
    mapping = tmp_path / "images.json"
    mapping.write_text(json.dumps({"case-1": {
        "image": "example@sha256:" + "b" * 64,
        "base_commit": "c" * 40, "platform": "linux/amd64",
    }}), encoding="utf-8")
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        output = _image_metadata() if argv[:3] == ["docker", "image", "inspect"] else "Python 3.11.8\n"
        return subprocess.CompletedProcess(argv, 0, output, "")

    monkeypatch.setattr("roottrace.runtime.docker.subprocess.run", fake_run)
    preparer = DockerEnvironmentPreparer(instance_id="case-1", image_map=mapping,
                                         platform_name="linux/amd64")
    first = preparer.prepare("c" * 40)
    second = preparer.prepare("c" * 40)
    assert first == second
    assert first.digest == "sha256:" + "b" * 64
    assert "--network" in calls[1] and "none" in calls[1]
    with pytest.raises(EnvironmentPreparationError, match="base commit mismatch"):
        preparer.prepare("d" * 40)


def test_dependency_change_invalidates_environment_key(monkeypatch, tmp_path: Path) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("pytest==8.0 --hash=sha256:" + "a" * 64, encoding="utf-8")

    cached_tags: list[str] = []

    def fake_run(argv, **kwargs):
        if argv[:3] == ["docker", "image", "inspect"]:
            if argv[-1].startswith("roottrace-env:"):
                cached_tags.append(argv[-1])
                return subprocess.CompletedProcess(argv, 1, "", "not cached")
            return subprocess.CompletedProcess(argv, 0, _image_metadata(), "")
        if argv[:2] == ["docker", "run"] and "--version" in argv:
            return subprocess.CompletedProcess(argv, 0, "Python 3.11.8\n", "")
        if argv[:3] == ["python", "-m", "pip"]:
            return subprocess.CompletedProcess(argv, 1, "", "download unavailable")
        return subprocess.CompletedProcess(argv, 1, "", "missing")

    monkeypatch.setattr("roottrace.runtime.docker.subprocess.run", fake_run)
    preparer = DockerEnvironmentPreparer("example:latest", platform_name="linux/amd64",
                                         requirements=requirements, index_url="https://example.invalid/simple")
    with pytest.raises(EnvironmentPreparationError, match="download unavailable"):
        preparer.prepare("c" * 40)
    requirements.write_text("pytest==8.1 --hash=sha256:" + "b" * 64, encoding="utf-8")
    with pytest.raises(EnvironmentPreparationError, match="download unavailable"):
        preparer.prepare("c" * 40)
    assert len(cached_tags) == 2 and cached_tags[0] != cached_tags[1]


def test_compatible_cases_reuse_owned_dependency_image(monkeypatch, tmp_path: Path) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("pytest==8.0 --hash=sha256:" + "a" * 64, encoding="utf-8")
    cache_tags: list[str] = []

    def fake_run(argv, **kwargs):
        if argv[:3] == ["docker", "image", "inspect"]:
            if argv[-1].startswith("roottrace-env:"):
                cache_tags.append(argv[-1])
                key = argv[-1].split(":", 1)[1]
                cached = [{"Id": "sha256:" + "d" * 64,
                           "Config": {"Labels": {"roottrace.cache_key": key}}}]
                return subprocess.CompletedProcess(argv, 0, json.dumps(cached), "")
            return subprocess.CompletedProcess(argv, 0, _image_metadata(), "")
        if argv[:2] == ["docker", "run"] and "--version" in argv:
            return subprocess.CompletedProcess(argv, 0, "Python 3.11.8\n", "")
        raise AssertionError("cache hit must not download or install dependencies")

    monkeypatch.setattr("roottrace.runtime.docker.subprocess.run", fake_run)
    preparer = DockerEnvironmentPreparer("example:latest", platform_name="linux/amd64",
                                         requirements=requirements, index_url="https://example.invalid/simple")
    first = preparer.prepare("c" * 40)
    second = preparer.prepare("d" * 40)
    assert first == second
    assert first.image == "sha256:" + "d" * 64
    assert cache_tags[0] == cache_tags[1]


def test_cache_cleanup_refuses_unowned_image(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        metadata = [{"Config": {"Labels": {"owner": "user"}}}]
        return subprocess.CompletedProcess(argv, 0, json.dumps(metadata), "")

    monkeypatch.setattr("roottrace.runtime.docker.subprocess.run", fake_run)
    preparer = DockerEnvironmentPreparer("example:latest")
    with pytest.raises(EnvironmentPreparationError, match="ownership"):
        preparer.remove_cached_environment("a" * 64)
    assert len(calls) == 1
    assert calls[0][:3] == ["docker", "image", "inspect"]


def test_docker_uses_disposable_copy_and_junit(monkeypatch, git_repo, tmp_path: Path) -> None:
    original = subprocess.run
    docker_commands: list[list[str]] = []

    def fake_run(argv, **kwargs):
        if argv[0] != "docker":
            return original(argv, **kwargs)
        docker_commands.append(argv)
        assert "--network" in argv and "none" in argv
        assert "--read-only" in argv and "--cap-drop" in argv
        assert not any("docker.sock" in token for token in argv)
        mount = argv[argv.index("--mount") + 1]
        work = Path(mount.split("src=", 1)[1].split(",dst=", 1)[0])
        junit = work / ".roottrace" / "pytest-results.xml"
        junit.write_text('<testsuite tests="1" failures="0" errors="0" skipped="0">'
                         '<testcase name="test_add"/></testsuite>', encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, "1 passed", "")

    monkeypatch.setattr("roottrace.runtime.sandbox.subprocess.run", fake_run)
    environment = DockerEnvironment("example:latest", "sha256:" + "b" * 64,
                                    "linux/amd64", "Python 3.11", "key")
    before = capture_repository_fingerprint(git_repo.repo)
    with RuntimeVerificationSandbox(git_repo.repo, base_commit=git_repo.base_sha,
                                    work_dir=tmp_path, docker_environment=environment) as sandbox:
        with pytest.raises(ValueError, match="does not exist"):
            sandbox.run(["python", "-m", "pytest", "tests/missing.py"])
        assert not docker_commands
        result = sandbox.run(["python", "-m", "pytest", "tests/test_calc.py"])
        assert result.classification is PytestExecutionClassification.PASSED
        assert sandbox.head_sha == git_repo.base_sha
    assert before == capture_repository_fingerprint(git_repo.repo)
    assert not sandbox.work_root.exists()


def test_unavailable_environment_does_not_run_tests(git_repo, tmp_path: Path) -> None:
    with RuntimeVerificationSandbox(git_repo.repo, work_dir=tmp_path,
                                    unavailable_reason="image missing") as sandbox:
        with pytest.raises(ValueError, match="does not exist"):
            sandbox.run(["python", "-m", "pytest", "tests/missing.py"])
        with pytest.raises(RuntimeError, match="image missing"):
            sandbox.run(["python", "-m", "pytest", "tests/test_calc.py"])
