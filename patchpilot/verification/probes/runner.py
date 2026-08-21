"""Execute declarative acceptance probes in the Docker sandbox."""

from __future__ import annotations

import base64
import json
import shlex
import time
from typing import TYPE_CHECKING

from patchpilot.planning.schema import AcceptanceProbeSpec
from patchpilot.verification.probes.schema import ProbeExecutionResult, StepResult

if TYPE_CHECKING:
    from patchpilot.sandbox.docker_runner import DockerSandbox


_PROBE_HARNESS = r"""
import base64
import importlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile


def import_target_module(module_name):
    try:
        return importlib.import_module(module_name)
    except ImportError as direct_error:
        package_init = os.path.join(os.getcwd(), "__init__.py")
        if not os.path.isfile(package_init):
            raise direct_error
        package_name = "_patchpilot_target"
        package_spec = importlib.util.spec_from_file_location(
            package_name,
            package_init,
            submodule_search_locations=[os.getcwd()],
        )
        if package_spec is None or package_spec.loader is None:
            raise direct_error
        package = importlib.util.module_from_spec(package_spec)
        sys.modules[package_name] = package
        package_spec.loader.exec_module(package)
        return importlib.import_module(f"{package_name}.{module_name}")


def resolve_target(spec):
    module = import_target_module(spec["module"])
    parts = spec["target"].split(".")
    target = getattr(module, parts[0])
    if isinstance(target, type) and len(parts) > 1:
        target = target(
            *spec["constructor_args"],
            **spec["constructor_kwargs"],
        )
        parts = parts[1:]
    else:
        parts = parts[1:]
    for part in parts:
        target = getattr(target, part)
    return target


def read_attribute(value, path):
    for part in path.split("."):
        value = getattr(value, part)
    return value


probe_root = tempfile.mkdtemp(prefix="patchpilot-probe-")
shutil.copytree(
    "/workspace",
    probe_root,
    dirs_exist_ok=True,
    ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"),
)
os.chdir(probe_root)
sys.path.insert(0, probe_root)

spec = json.loads(base64.urlsafe_b64decode(sys.argv[1]).decode("utf-8"))
assertion = spec["assertion"]
try:
    result = resolve_target(spec)(
        *spec["arguments"],
        **spec["keyword_arguments"],
    )
except Exception as error:
    if assertion == "raises" and type(error).__name__ == spec["exception"]:
        print(json.dumps({"passed": True, "actual": type(error).__name__}))
        raise SystemExit(0)
    print(json.dumps({"passed": False, "actual": type(error).__name__, "error": str(error)}))
    raise SystemExit(1)

if assertion == "raises":
    passed = False
    actual = "no_exception"
elif assertion == "equals":
    actual = result
    passed = actual == spec["expected"]
elif assertion == "attribute_equals":
    actual = read_attribute(result, spec["attribute"])
    passed = actual == spec["expected"]
elif assertion == "truthy":
    actual = bool(result)
    passed = actual is True
else:
    actual = bool(result)
    passed = actual is False

print(json.dumps({"passed": passed, "actual": actual}, default=repr))
raise SystemExit(0 if passed else 1)
"""


class ProbeRunner:
    """Run immutable declarative probes against the mounted workspace."""

    def __init__(self, sandbox: DockerSandbox) -> None:
        self.sandbox = sandbox

    def run_probe(self, spec: AcceptanceProbeSpec) -> ProbeExecutionResult:
        """Execute one probe without writing a probe file into the repository."""
        started = time.monotonic()
        payload = base64.urlsafe_b64encode(
            spec.model_dump_json().encode("utf-8")
        ).decode("ascii")
        command = (
            f"python -c {shlex.quote(_PROBE_HARNESS)} "
            f"{shlex.quote(payload)}"
        )
        result = self.sandbox.run(command, timeout_seconds=30)
        output = result.stdout.strip() or result.stderr.strip()
        actual = ""
        error = None
        if output:
            try:
                parsed = json.loads(output.splitlines()[-1])
                actual = repr(parsed.get("actual"))
                error = parsed.get("error")
            except json.JSONDecodeError:
                error = output[:2_000]
        passed = result.exit_code == 0 and not result.timed_out
        step_result = StepResult(
            step_index=0,
            description=f"Evaluate {spec.target}",
            passed=passed,
            expected_outcome=spec.assertion,
            actual_outcome=actual or ("timeout" if result.timed_out else "failed"),
            error=None if passed else error or output[:2_000],
        )
        return ProbeExecutionResult(
            probe_id=spec.probe_id,
            passed=passed,
            step_results=[step_result],
            execution_time_seconds=time.monotonic() - started,
            output=output[:2_000],
            error=None if passed else step_result.error,
            artifacts=[],
        )

    def run_baseline_probe(
        self,
        spec: AcceptanceProbeSpec,
    ) -> ProbeExecutionResult:
        """Run a frozen probe before source changes."""
        return self.run_probe(spec)

    def run_post_patch_probe(
        self,
        spec: AcceptanceProbeSpec,
    ) -> ProbeExecutionResult:
        """Run the same frozen probe after source changes."""
        return self.run_probe(spec)
