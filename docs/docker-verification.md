# Docker runtime verification

`roottrace rca` uses Docker verification by default. Select an existing local
image with `--verification-image IMAGE`. The image must contain Python 3 and
pytest. The target repository is cloned at `base_commit` into a disposable
directory before it is bind-mounted at `/work`; the original repository is
never mounted. Tests run without network access, Docker socket, host secrets,
or inherited environment variables, as a non-root user with reduced
capabilities and CPU, memory, process, and time limits. The original repository
fingerprint is checked again after the clone is destroyed.

For SWE-bench evaluation, `python -m evaluation.runner` derives the official
`swebench/sweb.eval.x86_64.<instance_id>:latest` reference (replacing `__`
with `_1776_`). It inspects a local image first and pulls only that derived
Docker Hub reference when missing. Use `--no-verification-image-pull` to keep
image pulls disabled. The preparation stage verifies `linux/amd64`, confirms the
target base commit is an ancestor of `/testbed` HEAD, prefers the image's
`testbed` Python, checks Python 3 and pytest, and records the inspected digest.
If pytest is missing, it downloads a fixed, hash-pinned wheel set from PyPI,
installs it with no network or target checkout mounted in a disposable
container, and caches the resulting RootTrace-owned image. It never builds an
instance Dockerfile or passes gold/test patches to an RCA Agent. An unavailable
image, unsupported Python version, or failed setup makes verification
`unverified` with a specific reason; RCA reporting continues.

Run `python -m evaluation.runner --verification-preflight --max-cases 1`
to inspect or pull images and check their environment without model calls.
It emits one JSON record per case, including `pytest_bootstrapped`,
`image_cache_hit`, and the effective image digest, and exits 1 if any case is
not ready. `--no-verification-pytest-bootstrap` disables automatic setup;
combine it with `--no-verification-image-pull` to prevent downloads. Pull,
wheel setup, and preflight share a preparation timeout; the verification node has a
separate wait limit. The runner accepts `--verification-preparation-timeout`
and `--verification-wait-seconds` to adjust them. The image policy and mapping
checksum are included in the resume configuration hash.

To override automatic resolution with a prepared local image, pass
`--verification-image-map images.json`. The JSON maps each `instance_id` to an object
with `image`, `base_commit`, and `platform` (for example `linux/amd64`). The
image must carry a `roottrace.base_commit` label matching the target commit.
Docker inspection records the executed image's content digest in
`verification_environment.json`; execution uses the inspected image ID so a
mutable tag cannot change the image between inspection and use. Evaluation
gold and test patches are never passed to the RCA pipeline.

Automatic pytest setup supports Python 3.8 through 3.13 and uses a committed
wheel lock: pytest 8.3.5 for Python 3.8–3.9, pytest 9.0.3 for Python
3.10–3.13, and their pinned dependencies. Wheels are downloaded for universal
Python tags, verified by SHA-256, and installed under `/roottrace-deps` without
changing the selected base image. The cache key includes the base digest,
platform, Python interpreter and version, lock contents, and install method.
Only a dedicated temporary wheel directory is mounted during installation.

An optional `--verification-requirements` file contains one pinned package and
SHA-256 wheel hash per line, such as
`pytest==8.3.4 --hash=sha256:<64 hex digits>`. With
`--verification-index-url https://...`, RootTrace downloads binary wheels
with isolated pip settings and hashes, then installs them with no network in
a disposable container. The resulting image is cached under a RootTrace tag
keyed by base image digest, platform, Python version, requirements bytes, and
installation method. Cache hits require a matching RootTrace ownership label.
RootTrace does not build or run repository Dockerfiles. Projects needing
system packages, editable/source builds, private indexes without an explicit
HTTPS endpoint, or non-pip setup need a prepared image.

Image or dependency preparation failure, or an exceeded preparation/wait
deadline, makes valid verification steps `unverified` with the reason in
`verification.json`. Invalid test targets and pytest options are rejected
before container startup. No host execution follows a Docker failure. The
legacy `--verification-backend host` remains an explicit development option.
