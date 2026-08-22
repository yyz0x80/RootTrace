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

```bash
patchpilot rca --repo /path/to/repo --issue issue.md --model <model> --output-dir output/roottrace-demo
```

Optional inputs: `--stack-trace`, `--ci-log`, `--pr-diff`.

## Evaluation

RootTrace derives file-localization metrics from SWE-bench Verified gold patches, including **Top-1 File Accuracy** and **Recall@K**. The MVP uses a fixed 50-case development subset before any full 500-case run.

## Scope

RootTrace diagnoses and recommends a fix scope, but does **not** edit code, generate/apply patches, commit, push, merge, or open PRs in RCA mode.

The internal Python package may remain `patchpilot` during the MVP for compatibility; the user-facing project name is **RootTrace**.
