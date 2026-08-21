# PatchPilot evaluation set

The task set follows the Day 5 evaluation rules:

- every task points at a fixed, reproducible Git commit;
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

The shared `day5_python_repo` fixture intentionally contains one documented,
unrelated baseline failure in `tests/test_csv_records.py::test_parse_quoted_record`.
This verifies that a correct patch is not rejected merely because the repository
was already red. The failure must remain unchanged; any new or worsened failure
still makes the independent regression result unsafe.

## Materializing a target repository

Fixtures are stored without nested `.git` directories so the evaluation set
can be versioned normally. Create a clean checkout with:

```bash
python evaluation/materialize_fixture.py \
  --source evaluation/fixtures/day5_python_repo \
  --destination /tmp/patchpilot-task-repo \
  --expected-commit 65b943998bcb8432096ea21ecb7e3b2da4feaadd
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

PatchPilot evaluation uses **separate deterministic dimensions** for scoring with **independent evaluator verification**:

### Independent evaluator checks

The external evaluator independently verifies patch scope, declared public tests,
full regression delta, and basic patch minimality. These checks are the source
of truth for functional scoring:

1. **Scope compliance**: After applying patch.diff to the clean scoring checkout, the evaluator independently determines changed files using Git and validates them against the task manifest's `allowed_changes`
2. **Public test execution**: The evaluator executes every declared `target_tests` entry independently in the scoring checkout using controlled pytest commands
3. **Regression delta**: The evaluator runs every declared `regression_tests` target before and after the patch. Resolved, preserved, improved, and unchanged historical failures are safe; regressions, worsened failures, missing comparisons, and timeouts are unsafe
4. **Minimality analysis**: The evaluator calculates deterministic signals (changed file count, added/deleted lines, unexpected files, generated/cache files) without subjective model judgment

### Functional correctness
Measures whether the generated patch actually works correctly:

- **Value**: 1.0 if the patch applies successfully, scope is compliant, public tests pass, regression delta is safe, and all hidden tests pass, 0.0 otherwise
- **Applies to**: Tasks that reach the execute phase and have hidden tests configured
- **Key principle**: Reporting VERIFIED never awards partial functional credit when hidden tests fail or scope is violated
- **For prepare-only tasks**: Functional correctness is not applicable (set to 0.0)
- **For tasks without hidden tests**: Functional correctness depends on patch applicability, scope compliance, public tests, and regression safety

### Outcome accuracy
Measures whether the agent correctly predicted its final status:

- **Value**: 1.0 if actual_status equals expected_final_status, 0.0 otherwise
- **Applies to**: All tasks
- **Purpose**: Separate from functional correctness; measures if the agent described its outcome correctly
- **For prepare-only tasks**: This is the primary scoring dimension (e.g., BLOCKED, NEEDS_CLARIFICATION)

### Key scoring rules

1. **VERIFIED + passing hidden tests + scope compliant + public tests pass + safe regression delta**: functional_correctness = 1.0, outcome_accuracy = 1.0
2. **VERIFIED + failing hidden tests**: functional_correctness = 0.0, outcome_accuracy = 1.0 (false VERIFIED)
3. **VERIFIED + scope violation**: functional_correctness = 0.0, outcome_accuracy = 1.0 (scope violations force functional correctness to 0)
4. **VERIFIED + public test failure**: functional_correctness = 0.0, outcome_accuracy = 1.0 (public test failures prevent functional success)
5. **Wrong status + all functional checks pass**: functional_correctness = 1.0, outcome_accuracy = 0.0
6. **Missing patch**: functional_correctness = 0.0
7. **Patch application failure**: functional_correctness = 0.0
8. **Prepare-only with expected BLOCKED**: functional_correctness = 0.0 (N/A), outcome_accuracy = 1.0
9. **Prepare-only with expected NEEDS_CLARIFICATION**: functional_correctness = 0.0 (N/A), outcome_accuracy = 1.0
10. **Unchanged historical regression failure**: regression safety passes, while the raw pytest command remains diagnostic
11. **Incomplete regression coverage**: final status should be PARTIALLY_VERIFIED and does not count as a verifier pass
12. **Execute task with no score commands**: hidden_tests_applicable = false, functional correctness still requires patch applicability, scope compliance, public tests, and regression safety

### Scope compliance rules

The evaluator enforces the following scope rules:

- **Empty allowed_changes**: Treated as "no repository changes allowed," not "unrestricted"
- **Denied paths**: Patches that modify `.git` internals, test files (`tests/`, `test_*.py`), secrets, CI files, or other sensitive paths are automatically non-compliant unless explicitly permitted
- **Path normalization**: All paths are normalized as repository-relative POSIX paths
- **Scope violations**: Force functional_correctness to 0.0 regardless of hidden test results

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
  "scope_compliant": boolean,
  "public_tests_passed": boolean,
  "public_tests_applicable": boolean,
  "regression_tests_passed": boolean,
  "regression_tests_applicable": boolean,
  "changed_file_count": integer,
  "added_lines": integer,
  "deleted_lines": integer,
  "unexpected_changed_files": [string],
  "minimality_warnings": [string],
  "details": {}
}
```

### Field definitions

- `schema_version`: "2.0" for the new separated scoring schema
- `functional_correctness`: 1.0 only when patch applies, scope is compliant, public tests pass, regression delta is safe, and all hidden tests pass
- `outcome_accuracy`: 1.0 when actual_status matches expected_status
- `hidden_tests_passed`: True if all configured hidden tests passed
- `hidden_tests_applicable`: True if hidden tests were configured and run (not the same as passed)
- `patch_applied`: True if patch was successfully applied to the scoring repository copy
- `scope_compliant`: True if patch only changes allowed files, False otherwise
- `public_tests_passed`: True if all declared target_tests passed, False if any failed
- `public_tests_applicable`: True if target_tests were configured and run
- `regression_tests_passed`: True when every declared regression target has a safe baseline-to-post-patch transition
- `regression_tests_applicable`: True if regression_tests were configured and compared
- `changed_file_count`: Number of files changed by the patch
- `added_lines`: Number of lines added by the patch
- `deleted_lines`: Number of lines deleted by the patch
- `unexpected_changed_files`: List of changed files not in allowed_changes
- `minimality_warnings`: List of warnings about patch size or content (e.g., large diffs, binary files)
- `actual_status`, `expected_status`: Terminal status values from execution
- `verification_report_present`: True if verification_report.json exists (diagnostic evidence only)
- `patch_generated`: True if patch.diff was generated

**Important**: PatchPilot's own `verification_report.json` supplies verifier
status and evidence-quality metrics, but it is not the source of truth for
functional correctness. The evaluator independently checks scope, public tests,
regression delta, hidden behavior, and minimality.

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
- `scope_compliance_rate`: Rate of tasks where patches only changed allowed files
- `public_tests_pass_rate`: Rate of tasks where declared public tests passed
- `independent_regression_safety_rate`: Rate of tasks whose independent baseline-to-post-patch regression comparison is safe
- `partial_verification_rate`: Rate of verifier reports with incomplete deterministic coverage
- `failed_verification_rate`: Rate of verifier reports with a canonical FAILED status
- `total_changed_files`: Total number of files changed across all tasks
- `total_added_lines`: Total number of lines added across all tasks
- `total_deleted_lines`: Total number of lines deleted across all tasks
- `average_changed_files`: Average number of files changed per task (when applicable)
- `average_added_lines`: Average number of lines added per task (when applicable)
- `average_deleted_lines`: Average number of lines deleted per task (when applicable)

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
