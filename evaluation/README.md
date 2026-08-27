# RootTrace RCA evaluation

This directory contains the reproducible evaluation harness for RootTrace's
evidence-grounded root-cause analysis (RCA) pipeline. It evaluates source-file
localization on SWE-bench Verified cases; it is not a patch-generation or
patch-application benchmark.

For each case, the harness:

1. validates public case metadata and a pinned `base_commit`;
2. creates a disposable shallow checkout containing only that base commit;
3. sends the incident and repository to RootTrace without exposing gold data;
4. reads the RCA report's ranked `top_k_locations`; and
5. parses the gold patch after the RCA run to derive non-test source files for
   deterministic localization metrics.

The original repository mirrors are never modified. Gold patches and test
metadata are evaluator-only and never cross into the RootTrace run.

## Input data

The benchmark data is kept outside this repository. By default the runner
expects the following layout under `~/Datasets/roottrace-swebench`:

```text
roottrace-swebench/
├── public/verified_public.jsonl
├── gold/verified_gold.jsonl
├── manifests/smoke3.json
└── repos/
    ├── <owner>__<name>.git
    └── ...
```

The public JSONL contains `instance_id`, `repo`, `base_commit`, and
`problem_statement` (plus optional provenance fields). The gold JSONL is read
only after each RootTrace invocation. A manifest contains only the case ID,
repository, and base commit.

## Run the smoke3 evaluation

The existing smoke3 run is:

```bash
python -m evaluation.runner
```

The runner defaults to the smoke3 manifest, the data-root above, and the
`three_specialists_retrieval_off` variant. A configured model can be supplied
for a real run:

```bash
python -m evaluation.runner \
  --data-root ~/Datasets/roottrace-swebench \
  --model <model>
```

Useful controls include:

```text
--manifest PATH       manifest (default: <data-root>/manifests/smoke3.json)
--variant NAME        ablation variant
--max-cases N         run only the first N manifest cases
--dry-run             validate inputs and mirrors without invoking RootTrace
--resume              reuse completed results for the same configuration
--output-dir PATH     output directory (default: output/rca-eval-<variant>)
--history-corpus PATH optional historical RCA JSONL for retrieval variants
--history-index PATH  optional persisted historical index
```

Before a development-subset run, validate its inputs without model calls:

```bash
python -m evaluation.preflight \
  --data-root ~/Datasets/roottrace-swebench \
  --manifest ~/Datasets/roottrace-swebench/manifests/dev50.json
```

## Ablation variants

All variants use the same manifest, prompts, budgets, context limits, and
concurrency. Only the evidence-agent subset and retrieval mode vary:

- `deterministic_baseline`: deterministic context ranking, no model calls;
- `lead_code`: Lead planner plus Code Specialist;
- `three_specialists_retrieval_off`: Issue/CI, Code, and Git History
  Specialists, without historical retrieval (the default);
- `three_specialists_retrieval_on`: the three Specialists with clustered
  historical retrieval when a corpus is supplied; and
- `clustering_off`: the three Specialists with flat historical retrieval.

Historical retrieval is optional. Target instance IDs are excluded by the
leakage guard, and retrieved cases are only hints—they cannot replace current
repository evidence.

## Outputs

Each variant writes deterministic aggregate reports and per-case artifacts:

```text
<output-dir>/
├── variant.json
├── metrics.json
├── report.md
└── cases/<instance_id>/
    ├── root_trace_input.json
    ├── result.json
    └── roottrace/          # RootTrace RCA artifacts
```

The reports include Top-1 File Accuracy, Any/All File Recall@3 and @5, Mean
Gold File Recall@5, coverage, invalid-output rate, latency P50/P95, model-call
counts, and exact/null token-usage observations. These are file-localization
metrics, not natural-language RCA accuracy. Missing usage remains `null` and
is never treated as zero.

The fixed development manifest can be selected reproducibly with
`python -m evaluation.select_manifest`; see `--help` for the source
dataset, smoke3 exclusion, seed, and output options.
