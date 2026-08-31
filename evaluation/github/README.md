# GitHub smoke10 evaluation

`github_smoke10` exercises real GitHub issue/PR ingestion, canonical online
repository acquisition, historical checkout, and RootTrace file localization.
The checked-in manifest is at `evaluation/github/github_smoke10.json`.

Prerequisites:

- a Python 3.11+ environment with RootTrace dependencies;
- network access to `github.com` and `api.github.com`;
- `GITHUB_TOKEN` is recommended for the GitHub API; repository clone/fetch
  operations are anonymous because every smoke case uses a public repository
  (the suite can run without a token, but unauthenticated API limits apply);
- a configured RootTrace model/provider for a normal run.

Run the free preflight first; it makes no LLM/provider calls:

```bash
python -m evaluation.github.run --suite github_smoke10 --preflight
```

Run one case, or the complete ten-case smoke:

```bash
python -m evaluation.github.run --suite github_smoke10 --case pytest-dev__pytest-5262
python -m evaluation.github.run --suite github_smoke10
```

The online clone cache defaults to `~/.cache/roottrace/github-smoke` and is
separate from SWE-bench/local evaluation mirrors. Results are timestamped
under `evaluation/github/results/` (or `--output-dir`): one JSONL case stream,
a summary JSON file, and bounded per-case RootTrace artifacts.

The eight issue cases are scored with the existing SWE-bench-derived
Top-1 and Any@5 file-localization definitions. The two regression-PR cases
combine the later issue with bad-PR context and are emitted as
`manual_review_required`; their expected files and fix-PR URLs are evaluator
provenance only and are never passed to RootTrace. Review the predicted
Top-5 against the expected root-cause files and inspect the incident evidence
before making a final PR-case judgment.

For Flask PR 4682 / issue 5774, confirm that the report connects the context
handling regression to `src/flask/helpers.py`. For pytest PR 12414 / issue
13312, confirm that it connects the PyPy reorder failure to
`src/_pytest/fixtures.py`. In both cases, require a causal explanation backed
by the bad PR and later issue; a filename match alone is not a passing manual
review.
