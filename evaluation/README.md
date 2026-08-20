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
  --expected-commit e32138dad45ca3652677aa9aaef4417975047d0e
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

## Scoring semantics (schema version 2.0)

PatchPilot evaluation uses **separate deterministic dimensions** for scoring:

### Functional correctness
Measures whether the generated patch actually works correctly:

- **Value**: 1.0 if the patch applies successfully and all hidden tests pass, 0.0 otherwise
- **Applies to**: Tasks that reach the execute phase and have hidden tests configured
- **Key principle**: Reporting VERIFIED never awards partial functional credit when hidden tests fail
- **For prepare-only tasks**: Functional correctness is not applicable (set to 0.0)
- **For tasks without hidden tests**: Functional correctness depends on patch applicability only

### Outcome accuracy
Measures whether the agent correctly predicted its final status:

- **Value**: 1.0 if actual_status equals expected_final_status, 0.0 otherwise
- **Applies to**: All tasks
- **Purpose**: Separate from functional correctness; measures if the agent described its outcome correctly
- **For prepare-only tasks**: This is the primary scoring dimension (e.g., BLOCKED, NEEDS_CLARIFICATION)

### Key scoring rules

1. **VERIFIED + passing hidden tests**: functional_correctness = 1.0, outcome_accuracy = 1.0
2. **VERIFIED + failing hidden tests**: functional_correctness = 0.0, outcome_accuracy = 1.0 (false VERIFIED)
3. **Wrong status + passing hidden tests**: functional_correctness = 0.0, outcome_accuracy = 0.0
4. **Missing patch**: functional_correctness = 0.0
5. **Patch application failure**: functional_correctness = 0.0
6. **Prepare-only with expected BLOCKED**: functional_correctness = 0.0 (N/A), outcome_accuracy = 1.0
7. **Prepare-only with expected NEEDS_CLARIFICATION**: functional_correctness = 0.0 (N/A), outcome_accuracy = 1.0
8. **Execute task with no score commands**: hidden_tests_applicable = false, functional_correctness based on patch applicability

### Score artifact schema (score.json)

Each task's `score.json` contains:

```json
{
  "schema_version": "2.0",
  "task_id": "string",
  "category": "string",
  "expected_status": "string",
  "actual_status": "string",
  "phase_reached": "prepare|execute",
  "functional_correctness": 0.0 | 1.0,
  "outcome_accuracy": 0.0 | 1.0,
  "hidden_tests_passed": boolean,
  "hidden_tests_applicable": boolean,
  "verification_report_present": boolean,
  "patch_generated": boolean,
  "patch_applied": boolean,
  "details": {}
}
```

### Field definitions

- `schema_version`: "2.0" for the new separated scoring schema
- `functional_correctness`: 1.0 only when patch applies and all hidden tests pass
- `outcome_accuracy`: 1.0 when actual_status matches expected_status
- `hidden_tests_passed`: True if all configured hidden tests passed
- `hidden_tests_applicable`: True if hidden tests were configured and run (not the same as passed)
- `patch_applied`: True if patch was successfully applied to the scoring repository copy
- `actual_status`, `expected_status`: Terminal status values from execution
- `verification_report_present`: True if verification_report.json exists
- `patch_generated`: True if patch.diff was generated

### Legacy schema compatibility

Schema version 1.0 (legacy) used a single `score` field that mixed functional and outcome correctness:
- `score`: 1.0 for perfect, 0.5 for status match but failed hidden tests, 0.0 otherwise
- `outcome_matched`: Boolean for status accuracy

The aggregation layer handles both schemas for backward compatibility but produces v2.0 aggregates.

## Aggregate metrics

Each evaluation run writes the following automatic metrics to
`evaluation/runs/<timestamp>/aggregate.json`:

### New separated metrics (schema 2.0)

- `functional_correctness_rate`: Rate of tasks with passing hidden tests (functional correctness)
- `outcome_accuracy_rate`: Rate of tasks where actual_status matches expected_status
- `false_verified_rate`: Rate of tasks that reported VERIFIED but failed functional checks
- `patch_applicability_rate`: Rate of tasks where patches applied successfully

### Legacy metrics (for backward compatibility)

- expected outcome match, verified task, verifier pass, acceptance criteria
  coverage, regression pass, retry recovery, and unsafe action block rates;
- average execute duration;
- average LLM call count and prompt, completion, and total token usage.

Rate entries include their numerator and denominator. Average entries include
the number of available and missing observations. A metric value is `null` when
it cannot be calculated from exact artifacts; missing data is never treated as
zero. Token totals come from provider response metadata and are not estimated.
Cost is intentionally not calculated.

### Top-level aggregate fields

- `average_functional_correctness`: Average functional correctness across all completed tasks
- `average_outcome_accuracy`: Average outcome accuracy across all completed tasks
- `schema_version`: "2.0" for new scoring schema

The top-level average prioritizes functional correctness for patch-required tasks
and outcome correctness for prepare-only/non-patch tasks, avoiding silent blending
of hidden-test failure into half credit.
