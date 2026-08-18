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

The ambiguous and unsafe PatchPilot tasks intentionally stop during `prepare`.
Their outcome is read from `prepare_summary.json`; console text is retained only
for diagnostics and is not used for scoring.

## Running variants

Run the raw-issue baseline with:

```bash
python evaluation/runner.py \
  --tasks evaluation/tasks \
  --variant baseline \
  --model <model> \
  --max-rounds 16 \
  --max-repairs 0
```

The baseline sends the raw issue directly to one Agent Loop and then runs the
deterministic Verifier once. It keeps the same workspace protections, command
allowlist, Docker isolation, and read-only tests as PatchPilot.

Use `--variant patchpilot` for the full
`prepare -> approve -> execute -> verify -> repair -> evidence` workflow.
Every subprocess writes `stdout.log` and `stderr.log` beside its artifacts.
Each task's `score.json` keeps `actual_status`,
`verification_report_present`, and `patch_generated` as separate fields. A
missing report or patch therefore remains diagnosable without replacing the
terminal status recorded by `prepare_summary.json` or `run_summary.json`.

## Aggregate metrics

Each evaluation run writes the following automatic metrics to
`evaluation/runs/<timestamp>/aggregate.json`:

- expected outcome match, verified task, verifier pass, acceptance criteria
  coverage, regression pass, retry recovery, and unsafe action block rates;
- average execute duration;
- average LLM call count and prompt, completion, and total token usage.

Rate entries include their numerator and denominator. Average entries include
the number of available and missing observations. A metric value is `null` when
it cannot be calculated from exact artifacts; missing data is never treated as
zero. Token totals come from provider response metadata and are not estimated.
Cost is intentionally not calculated.
