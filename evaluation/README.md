# PatchPilot evaluation set

The task set follows the Day 5 evaluation rules:

- every task points at a fixed, clean Git commit;
- the agent receives only the target repository and `issue.md`;
- tests inside a target repository are read-only;
- hidden tests stay under `evaluation/tasks/` and are run only by the evaluator;
- expected non-success outcomes are scored as correct outcomes, not as failed fixes.

`tasks/index.json` is the machine-readable task catalog. Each task directory
contains a manifest, its issue, and (where applicable) evaluator-only hidden
tests. Paths in manifests are relative to this `evaluation/` directory.

The initial set contains ten tasks: two single-file bugs, two cross-file bugs,
two small features, one repair-loop task, one ambiguous request, one unsafe
request, and one environment failure.

## Materializing a target repository

Fixtures are stored without nested `.git` directories so the evaluation set
can be versioned normally. Create a clean checkout with:

```bash
python evaluation/materialize_fixture.py \
  --source evaluation/fixtures/day5_python_repo \
  --destination /tmp/patchpilot-task-repo \
  --expected-commit a3df5b5f8aadf0015070e07ad21c22f744de3230
```

The command fails if the generated commit differs from the manifest. Each run
must use a new destination. Hidden scoring should use a second checkout: apply
the produced `patch.diff` there and run the manifest's score commands with
`{task_dir}` replaced by the absolute evaluator-only task directory.

The ambiguous and unsafe tasks intentionally stop during `prepare`. The current
CLI does not write `run_summary.json` for prepare-time exits, so their manifests
include an `expected_signal` for the evaluation runner to match deterministically.

## Aggregate metrics

Each evaluation run writes the following automatic metrics to
`evaluation/results/<timestamp>/aggregate.json`:

- expected outcome match, verified task, verifier pass, acceptance criteria
  coverage, regression pass, retry recovery, and unsafe action block rates;
- average execute duration;
- average LLM call count and prompt, completion, and total token usage.

Rate entries include their numerator and denominator. Average entries include
the number of available and missing observations. A metric value is `null` when
it cannot be calculated from exact artifacts; missing data is never treated as
zero. Token totals come from provider response metadata and are not estimated.
Cost is intentionally not calculated.
