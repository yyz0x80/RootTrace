# RootTrace Architecture

RootTrace performs evidence-grounded root cause analysis on local Python
repositories. The analyzed repository is always read-only; runtime tests run
only in a disposable sandbox copy.

## Runtime flow

```text
Issue / CI / stack trace / PR context
                  ↓
        deterministic context builder
                  ↓
              Lead planner
                  ↓
     Issue/CI + Code + Git specialists
                  ↓
            EvidenceGraph
                  ↓
       falsifiable hypotheses
                  ↓
        runtime/test verifier
                  ↓
             RCA report
```

The three evidence specialists run concurrently and communicate only through
typed findings merged into the `EvidenceGraph`. Historical retrieval is
optional shared infrastructure, not an additional agent. The verifier runs
after hypotheses exist and returns `supported`, `rejected`, or `unverified`.

## Package boundaries

- `roottrace.agents` owns Lead planning/synthesis and the three evidence
  Specialists; it composes capabilities but does not implement retrieval,
  repository policy, or provider transport.
- `roottrace.incident` owns input normalization and deterministic, bounded
  context construction.
- `roottrace.evidence` owns evidence, hypothesis, finding, and graph contracts
  plus deterministic graph aggregation.
- `roottrace.history` owns historical-case import, TF-IDF, clustering, and
  retrieval without depending on Agents or orchestration.
- `roottrace.tools` owns typed, bounded, read-only repository tools and their
  registry.
- `roottrace.runtime` owns path/workspace policy, target fingerprinting, and
  disposable sandbox execution. It has no dependency on Agents or orchestration.
- `roottrace.verification` maps sandbox results to supported, rejected, or
  unverified hypothesis outcomes.
- `roottrace.llm` owns model configuration, provider retries/translation, and
  exact/null usage accounting.
- `roottrace.reporting` owns the final report contract and Markdown rendering.
- `roottrace.orchestrator` is the composition root; `roottrace.cli` owns the
  product command and complete persisted pipeline.
- `roottrace.tracing` and `roottrace.artifacts` remain focused cross-cutting
  single-file modules.
- `evaluation` owns reproducible SWE-bench-derived localization evaluation and
  never exposes gold data to RootTrace.

Low-level capabilities do not depend on orchestration: in particular,
`runtime`, `history`, and `llm` never import `agents` or `orchestrator`, and
the `roottrace` package never imports `evaluation`.

## Safety invariants

- Original repositories cannot be edited, renamed, deleted, or switched to a
  different revision.
- Repository paths are relative and cannot traverse outside the workspace.
- Secrets and `.git` internals are not exposed through RCA tools.
- Git inspection uses validated argument lists and `shell=False`.
- Runtime commands use a parsed allowlist inside a disposable copy.
- Persisted locations are repository-relative and factual conclusions cite
  evidence IDs.

## Outputs

Each run emits the normalized incident, investigation plan, per-specialist
findings, evidence graph, hypotheses, verification results, JSON/Markdown RCA
report, execution trace, and run summary. Artifacts record timing, model ID,
usage, uncertainty, and partial failures without estimating missing tokens.
