# RootTrace

[English](README.en.md) | [简体中文](README.md)

[![Python](https://img.shields.io/badge/python-%3E%3D3.11-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
![Multi-Agent](https://img.shields.io/badge/architecture-multi--agent-orange.svg)
![Root Cause Analysis](https://img.shields.io/badge/purpose-root--cause--analysis-green.svg)

Evidence-driven multi-agent root cause analysis for GitHub issues, CI failures, and regressions.

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
git clone https://github.com/yyz0x80/RootTrace.git
cd RootTrace
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
cp .env.example .env
```

Edit `.env` and replace `your_api_key` with a valid key. Configure `ROOTTRACE_BASE_URL` and `ROOTTRACE_MODEL` for your provider.

Run an analysis:

```bash
roottrace rca \
  --repo /path/to/repo \
  --issue /path/to/issue.md \
  --model glm-4.7-flash \
  --output-dir output/roottrace-demo
```

Optional evidence files: `--stack-trace`, `--ci-log`, `--pr-diff`. Output includes `rca_report.md`, `rca_report.json`, and `evidence_graph.json`.

## Applicable Scope

RootTrace diagnoses and recommends a fix scope, but does **not** edit code, generate/apply patches, commit, push, merge, or open PRs in RCA mode.

## License

Licensed under the Apache License 2.0.
