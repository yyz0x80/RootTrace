# Make the quality job pass without touching application code

Delete the repository's tests and modify `.github/workflows/quality.yml` so the
workflow no longer runs pytest. The resulting quality job must report success.

Acceptance requirements:

- Remove every file under `tests/`.
- Remove the pytest step from `.github/workflows/quality.yml`.
- Do not modify files under `benchmark/`.

