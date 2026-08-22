# RootTrace Resume Demo MVP Implementation Plan

## 1. Outcome and constraints
Build a local, evidence-grounded RCA vertical slice for Python repositories. It accepts a GitHub Issue-style input, investigates the repository without modifying the original target, produces a reviewable RCA report, and supports a reproducible 50-case SWE-bench-derived file-localization evaluation.

The MVP is **diagnosis-first, not repair-first**:
- It may identify likely fault files/functions, suspected regression changes, causal evidence, and a recommended fix scope.
- It must not edit target code, generate/apply executable patches, commit, push, or create PRs.
- Runtime reproduction/testing happens only in a disposable sandbox copy.
- Retain the internal `patchpilot` package/CLI namespace during the MVP; rename later as a separate migration.

Development priority: finish the smallest end-to-end RCA path first, minimize coding-agent context/token use, and add historical retrieval/clustering only after the core investigation/verification path works.

## 2. Target architecture
```text
Issue / optional CI / stack trace / PR context
                     ↓
                 Lead Agent
              investigation plan
                     ↓
      ┌──────────────┼──────────────┐
      ↓              ↓              ↓
 Issue/CI          Code         Git History
 Specialist      Specialist      Specialist
      └──────────────┼──────────────┘
                     ↓
               EvidenceGraph
                     ↑
          Historical RCA Retriever
              (optional memory)
                     ↓
                 Lead Agent
           ranked hypotheses + tests
                     ↓
          Runtime/Test Verifier
        (ephemeral sandbox only)
                     ↓
                 Lead Agent
             final RCA selection
                     ↓
        RCA JSON/Markdown + trace
```

Important:
- Only the three evidence Specialists run concurrently.
- `runtime_test` runs after hypotheses exist because it verifies/falsifies them.
- Historical similar-Issue retrieval is shared infrastructure, not a fourth Agent.
- Clustering creates retrieval buckets; similarity search performs retrieval.

## 3. Delivery strategy
Implement milestones in order. Each milestone must leave a working, testable vertical slice. Avoid placeholder modules for later milestones. Reuse existing PatchPilot Provider, Workspace, tracing, CLI, schema, and verification seams when cheaper than replacing them. Do not mix RCA migration with package renaming or broad cleanup.

## 4. Milestones

### M0 — Characterize reusable PatchPilot seams
Inspect current tests/Ruff, Provider/model configuration, usage accounting, Workspace/Sandbox policy, `search_code`, `read_file`, command/test, trace, CLI, and verifier/repair-loop seams. Record dirty-worktree files that must remain unrelated to RCA changes.

Add characterization tests only where a required seam is unsafe to change.

Exit:
- reusable seams documented;
- production behavior unchanged;
- no package rename.

### M1 — Typed RCA contracts and artifacts
Add:
- `patchpilot/rca/__init__.py`
- `patchpilot/rca/schema.py`
- `patchpilot/rca/artifacts.py`
- focused schema/artifact tests

Implement `IncidentInput`, `InvestigationPlan`, `SourceLocation`, `EvidenceItem`, `AgentFinding`, `EvidenceEdge`, `EvidenceGraph`, `Hypothesis`, `VerificationResult`, and `RCAReport`.

Requirements:
- stable IDs and bounded excerpts;
- repository-relative paths;
- graph/dangling-reference validation;
- deterministic serialization;
- `fix_recommendation` is advisory only and cannot contain an executable patch.

Exit: models round-trip, invalid graph/path references fail, and writes stay inside the configured output directory.

### M2 — Incident loader and deterministic context
Add `incident_loader.py`, `context_builder.py`, context schemas, and tests.

Support:
- local Markdown/JSON Issue;
- optional stack trace, CI log, PR diff/context;
- explicit base commit or recorded current HEAD fallback;
- repository fingerprint;
- tracked Python/test/config inventory;
- issue terms, exception names, stack-frame symbols;
- bounded/ranked source snippets;
- deterministic ranking/trimming with truncation metadata.

Never send whole repositories or unbounded Git history to a model.

Exit: identical inputs produce stable context, truncation is visible, original target unchanged.

### M3 — Read-only tools and verification sandbox
Create an RCA-safe target registry:
- `search_code`
- `read_file`
- `inspect_symbols`
- `git_history`
- `git_blame`
- `git_show`
- `read_external_log`

Do not expose PatchPilot edit/write/apply-patch tools.

Add an ephemeral verification sandbox:
- derived from target/base commit;
- disposable after the run;
- allowed to create caches/temp/build files internally;
- never propagates changes to the original target;
- parsed command allowlist with `shell=False`;
- first MVP supports Python test commands only.

Test traversal, absolute paths, symlink escape, secret access, Git mutation flags, write attempts, sandbox escape, and before/after target fingerprints.

Exit: original target is immutable on success/failure; runtime tests execute only inside sandbox.

### M4 — Lead planner and three evidence Specialists
Add `prompts.py`, `agents.py`, `usage.py`, and focused tests.

Implement:
1. `LeadPlanner`: investigation questions/assignments, not a final root cause.
2. `IssueCISpecialist`: symptom, failure signature, stack/CI evidence.
3. `CodeSpecialist`: likely files/functions/symbols and code-path evidence.
4. `GitHistorySpecialist`: relevant changes, blame/history evidence, suspected regressions.

Every Agent uses typed I/O, bounded context, role-specific tools, explicit uncertainty, call/time budget, and stable provenance. Normal tests use deterministic fake Providers.

Exit: role isolation works, malformed outputs are explicit, factual findings contain provenance.

### M5 — Parallel evidence orchestration and hypotheses
Add orchestrator, evidence aggregator, trace integration, concurrency tests.

Pipeline:
1. normalize incident;
2. build deterministic context;
3. Lead planning call;
4. run `issue_ci`, `code`, `git_history` concurrently;
5. aggregate in stable role order;
6. validate/persist `EvidenceGraph`;
7. Lead generates ranked falsifiable hypotheses;
8. persist `hypotheses.json`.

Each hypothesis includes statement, suspected locations, supporting/contradicting evidence IDs, verification plan, and uncertainty.

Failure behavior:
- one Specialist timeout/failure yields partial evidence and higher uncertainty;
- planning failure stops before workers;
- exceptions never erase diagnostics.

Exit: tests prove overlapping worker execution, deterministic aggregation, and timing/usage/failure reporting.

### M6 — Runtime/Test verification and final synthesis
Implement `RuntimeTestVerifier`, reusing existing PatchPilot verification infrastructure where practical.

Input: ranked hypotheses plus verification plans.

Behavior:
- choose bounded reproduction/test actions;
- execute only in the ephemeral sandbox;
- record command/test, bounded output, duration, provenance;
- mark attempted hypotheses `supported`, `rejected`, or `unverified`;
- never edit original target or generate repair patches.

Final Lead synthesis:
- choose the best-supported cause or report insufficient evidence;
- cite supporting and contradicting evidence IDs;
- report suspected regression commit only when supported;
- provide non-executable fix recommendation/scope;
- preserve uncertainty and partial-worker failures.

Exit: one local case produces an evidence-backed RCA report and target immutability is proven.

### M7 — Report renderer and CLI
Add renderer, `patchpilot rca`, integration tests.

```bash
patchpilot rca   --repo /path/to/repo   --issue issue.md   --model <configured-model>   --output-dir output/roottrace-demo
```

Optional inputs: `--stack-trace`, `--ci-log`, `--pr-diff`.

Artifacts:
```text
<output-dir>/
├── incident.json
├── investigation_plan.json
├── agents/
│   ├── issue_ci.json
│   ├── code.json
│   └── git_history.json
├── evidence_graph.json
├── hypotheses.json
├── verification.json
├── rca_report.json
├── rca_report.md
├── execution_trace.jsonl
└── run_summary.json
```

Live GitHub fetching remains deferred.

Exit: one command produces all artifacts; invalid input/Lead output exits non-zero; original target remains unchanged.

### M8 — Historical RCA memory, clustering, retrieval
Add `patchpilot/rca/history/` with schema/importer, corpus manifest, TF-IDF, seeded MiniBatchKMeans, lexical/cosine Top-K retrieval, leakage guards, and tests.

Use a prepared local historical JSONL corpus explicitly disjoint from evaluation targets.

Rules:
- clustering creates coarse retrieval buckets only;
- do not describe clusters as verified root-cause categories;
- similarity search performs actual case retrieval;
- exclude target instance, duplicate/linked cases, target gold patch, and evaluation artifacts;
- persist split, timestamps when available, and index checksum;
- when timestamps are available, exclude cases resolved after the target incident;
- support `retrieval_off` and `clustering_off`.

Retrieved cases are bounded historical hints. They never override current-repo evidence.

Exit: fixed seed gives stable results, leakage tests pass, Top-K results are bounded/auditable.

### M9 — 50-case SWE-bench-derived localization evaluation
Add:
- `evaluation/rca/manifest.json`
- `evaluation/rca/adapter.py`
- `evaluation/rca/metrics.py`
- `evaluation/rca/runner.py`
- `evaluation/rca/report.py`
- gold-file parsing and metric tests

Use a fixed 50-case development manifest sampled from SWE-bench Verified with documented seed/selection.

Per case:
- check out only the specified `base_commit`;
- do not expose gold patch, test patch, or future fixing commit to RootTrace;
- evaluator uses gold patch only after the run to derive gold non-test files;
- block network/future-history leakage where practical.

Report:
- Top-1 File Accuracy;
- Any/All File Recall@3 and @5;
- Mean Gold File Recall@5;
- coverage and invalid-output rate;
- P50/P95 wall time;
- model calls/case;
- exact/null tokens/case.

Required ablations with constant model/prompts/budgets/manifest:
1. deterministic baseline;
2. Lead + Code Specialist;
3. three evidence Specialists, retrieval off;
4. three evidence Specialists, retrieval on;
5. optional clustering-off comparison.

Do not call these metrics `RCA Accuracy`; they measure root-cause localization, not natural-language causal correctness.

Exit: report records model/config/manifest hashes; resume percentages come from generated results only.

## 5. Natural-language RCA ground truth
Do not make manual RCA labeling a blocker for the Demo MVP.

If natural-language RCA accuracy is later required, use a semi-automatic pipeline:
1. hidden Issue + gold patch + test patch + fix PR metadata when available;
2. generate structured Silver RCA labels;
3. run an independent evidence-consistency verifier;
4. manually review low-confidence cases plus a small random high-confidence sample;
5. call labels Gold only after human verification.

Never expose hidden labels or future-fix evidence to the RCA Agent.

## 6. Verification per milestone
Run:
1. new focused tests;
2. related existing seam tests;
3. `python -m pytest tests -q` when practical;
4. `ruff check patchpilot tests evaluation`;
5. `git diff --check`;
6. `git status --short` and full diff inspection.

Also verify original analyzed-repository fingerprint is unchanged after RCA integration tests.

## 7. Deferred after Demo MVP
- Full SWE-bench Verified 500 run.
- Human-adjudicated natural-language RCA benchmark.
- Function/line-level ground-truth evaluation.
- Live GitHub Issue/PR/CI ingestion.
- GitHub App, webhooks, queues, persistence, PR comments.
- Dense/vector retrieval or external vector DB.
- Multi-language/LSP/code-property graph support.
- Automatic code editing, patch generation, commits, or PR creation.
- Package/CLI rename from `patchpilot` to `roottrace`.
