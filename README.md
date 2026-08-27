# RootTrace

RootTrace is an evidence-grounded multi-agent Root Cause Analysis (RCA) system for GitHub Issues in existing Python repositories.

It answers: **what failed, where the likely root cause is, why it happened, and what evidence supports the conclusion**.

## Architecture

```text
Issue / CI / Stack Trace / PR Context
                 ↓
             Lead Agent
                 ↓
   ┌─────────────┼─────────────┐
   ↓             ↓             ↓
Issue/CI       Code        Git History
Specialist   Specialist     Specialist
   └─────────────┼─────────────┘
                 ↓
           EvidenceGraph
                 ↑
      Historical RCA Retrieval
                 ↓
         Ranked Hypotheses
                 ↓
       Runtime/Test Verifier
       (ephemeral sandbox)
                 ↓
          Final RCA Report
```

## Core Features

* Parallel evidence gathering across Issue/CI, code, and Git history.
* Structured `EvidenceGraph` with provenance and contradiction tracking.
* Falsifiable root-cause hypotheses with runtime/test verification.
* Read-only analysis of the original repository; tests run only in a disposable sandbox.
* Optional historical RCA retrieval with TF-IDF, MiniBatchKMeans, and Top-K similarity.
* JSON/Markdown reports, execution traces, timing, and provider usage.

## Quick Start

Create an environment, install RootTrace, and configure an OpenAI-compatible
model provider. The repository includes an environment template for the
default Zhipu endpoint:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
cp .env.example .env
```

Edit `.env` and replace `your_api_key` with a valid key. Set
`ROOTTRACE_BASE_URL` and `ROOTTRACE_MODEL` to match the provider and model you
intend to use. The target repository must also have its own test dependencies
installed because runtime verification runs its tests in an ephemeral copy.

Run an analysis with a local Issue-style Markdown or JSON file:

```bash
roottrace rca \
  --repo /path/to/repo \
  --issue /path/to/issue.md \
  --model glm-4.7-flash \
  --output-dir output/roottrace-demo
```

`--model` can be omitted when `ROOTTRACE_MODEL` is set in `.env`. Optional
evidence files can be supplied with `--stack-trace`, `--ci-log`, and
`--pr-diff`.

The command writes the RCA report and supporting artifacts to the output
directory, including `rca_report.md`, `rca_report.json`, and
`evidence_graph.json`.

## Evaluation

RootTrace derives file-localization metrics from SWE-bench Verified gold patches, including **Top-1 File Accuracy** and **Recall@K**. The evaluation workflow uses a fixed 50-case development subset before any full 500-case run.

## Scope

RootTrace diagnoses and recommends a fix scope, but does **not** edit code, generate/apply patches, commit, push, merge, or open PRs in RCA mode.
