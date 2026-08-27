# AGENTS.md
## 1. Project Identity
RootTrace is an evidence-grounded, multi-agent Root Cause Analysis (RCA) system for GitHub Issues and regression-related PR context in existing Python repositories.

RootTrace answers: **what failed, where the likely root cause is, why it happened, and what evidence supports that conclusion**. RCA mode may recommend a fix scope, but it must not edit, patch, commit, push, or open a PR against the analyzed repository.

RootTrace is not a general chatbot, autonomous repair agent, pull-request
reviewer, hosted GitHub App, or production incident-management platform.

## 2. Supported RCA Workflow

The local RCA workflow must:
1. Accept local GitHub Issue-style Markdown/JSON and a local Python repo.
2. Build deterministic, bounded context before model calls.
3. Use one Lead Agent to plan the investigation.
4. Run three evidence-gathering Specialists concurrently:
   - `issue_ci`: issue text, stack traces, CI excerpts, failure signatures.
   - `code`: repository structure, symbols, references, likely fault locations.
   - `git_history`: log, blame, show, diffs, suspected regression changes.
5. Store findings in a typed, auditable `EvidenceGraph`.
6. Let the Lead generate falsifiable root-cause hypotheses from gathered evidence.
7. Use a `runtime_test` Verifier after hypotheses exist to reproduce/falsify them in an ephemeral sandbox derived from the target repo.
8. Let the Lead select supported causes, retain uncertainty, and produce an RCA report.
9. Optionally retrieve similar historical RCA cases through shared memory/retrieval; retrieval is infrastructure, not a fourth evidence Agent.
10. Emit JSON/Markdown reports, per-agent artifacts, trace, timing, and provider usage.
11. Evaluate file localization on a reproducible 50-case development subset
    before any full SWE-bench Verified 500-case run.

The original analyzed repository is always read-only. Runtime verification may write only inside a disposable sandbox copy destroyed after the run.

## 3. Repository Structure and Boundaries
```text
RootTrace/
├── roottrace/      # RootTrace package; RCA code under roottrace/rca/
├── tests/          # RootTrace tests; writable
├── evaluation/     # RCA evaluation harness/manifests
├── docs/           # design and evaluation documentation
├── pyproject.toml
└── AGENTS.md
```
Rules:
- Modify only RootTrace unless the user explicitly names another workspace.
- Treat analyzed repos, demo repos, and benchmark workspaces as inputs.
- Never modify the original analyzed repository.
- RootTrace implementation/tests are writable during development.
- Preserve unrelated user changes in a dirty worktree.
- Do not package datasets, outputs, caches, secrets, or benchmark workspaces.

## 4. Supported Scope
In scope:
- Python repositories and local Issue Markdown/JSON.
- Optional stack trace, CI log, and PR diff/context.
- One Lead, three concurrent evidence Specialists, one runtime/test Verifier.
- Deterministic context construction and bounded model reasoning.
- File- and symbol-level localization.
- Evidence provenance, competing hypotheses, contradictions, uncertainty.
- Read-only Git inspection.
- Prepared local historical-case JSONL.
- TF-IDF + seeded MiniBatchKMeans retrieval buckets plus lexical/cosine Top-K retrieval.
- JSON, Markdown, JSONL trace, timing, and exact/null token-usage artifacts.
- Reproducible 50-case SWE-bench-derived file-localization evaluation.

Out of scope unless explicitly requested:
- Editing/patching the analyzed repository or generating executable patches.
- Automatic commits, pushes, merges, PR comments, or PR creation.
- Hosted GitHub App, webhooks, queues, databases, or Web UI.
- Live GitHub/API ingestion.
- Multi-language support, mandatory LSP, full code-property graphs, or fine-tuning.
- Claiming unmeasured RCA accuracy, latency, token, or retrieval improvements.

## 5. Architecture
```text
CLI / Evaluation Adapter
          ↓
RCA Orchestrator
    ├── Deterministic Context Builder
    ├── Lead Agent
    ├── Issue/CI + Code + Git Specialists   # parallel
    ├── Evidence Graph Aggregator
    ├── Historical RCA Retriever            # shared memory, optional
    ├── Runtime/Test Verifier                # after hypotheses
    └── RCA Report Renderer
          ↓
Provider + RCA Tool Registry
          ↓
Workspace Policy
    ├── Original Repository                 # read-only
    └── Ephemeral Verification Sandbox      # disposable
```

### Provider
Provider owns model calls, translation, retries, structured-output parsing, and usage accounting. It must not inspect repos, execute commands, or contain RCA workflow logic. Raw SDK objects must not escape it. Concurrent workers must use thread-safe accounting; aggregate immutable usage snapshots after completion.

### Orchestrator
The orchestrator owns ordering, concurrency, timeouts, failure isolation, artifacts, and final status. It must not bypass Tool Registry or Workspace Policy.

Preferred flow:
1. Normalize incident and build deterministic context.
2. Lead plans investigation; do not commit to a root cause before evidence.
3. Run `issue_ci`, `code`, and `git_history` concurrently.
4. Validate/merge findings into `EvidenceGraph`.
5. Lead generates ranked, falsifiable hypotheses and verification plans.
6. `runtime_test` verifies selected hypotheses in the disposable sandbox.
7. Lead performs final evidence-based selection and uncertainty reporting.
8. Render artifacts deterministically.

Do not create unrestricted coding loops. Deterministic code should collect, rank, trim, and validate evidence before LLM calls.

### Agents
Each Agent has one role, bounded context, typed input/output, budgets, stable IDs, allowed tools, and explicit uncertainty.
- Specialists never communicate directly.
- Shared state is the `EvidenceGraph`, not chat history.
- Every factual finding cites evidence/provenance.
- Worker failure is explicit and increases uncertainty.
- Runtime verification returns `supported`, `rejected`, or `unverified`.

### Tools
Original-repository tools are scoped, typed, bounded, auditable, and read-only:
- `search_code`, `read_file`, `inspect_symbols`
- `git_history`, `git_blame`, `git_show`
- `read_external_log`

Never expose `edit_file`, `write_file`, patch tools, unrestricted shell, `git commit`, `git push`, branch-changing commands, or direct `.git/` access to RCA Agents.

Runtime verification uses a separate sandbox registry with parsed allowlisted commands such as `pytest`, `python -m pytest`, or project-specific test commands. Sandbox writes are disposable and never propagate to the original repo.

### Historical retrieval
- TF-IDF + seeded MiniBatchKMeans creates coarse retrieval buckets only.
- Lexical/cosine similarity retrieves Top-K historical cases.
- Cluster IDs are not ground-truth root-cause categories.
- Retrieved cases provide prior evidence/localization hints, never final answers.
- Exclude evaluation targets, their gold patches, duplicates, and linked cases.
- Persist split, instance ID, source timestamp, and index checksum.
- When timestamps are available, only use cases resolved before the target incident.

## 6. Core Data Contracts
Use Pydantic for persisted/public models:
- `IncidentInput`: ID, repo, base commit, problem, optional logs/diff, provenance.
- `InvestigationPlan`: questions, assignments, budgets; no premature final cause.
- `SourceLocation`: repo-relative path, optional symbol/line range.
- `EvidenceItem`: stable ID, agent, kind, observation, location, provenance, excerpt.
- `Hypothesis`: statement, locations, supporting/contradicting evidence IDs, verification plan, confidence, disposition.
- `AgentFinding`: status, ranked locations, evidence, uncertainty, timing, usage.
- `EvidenceEdge`: source, target, relation (`supports`, `contradicts`, `caused_by`).
- `EvidenceGraph`: incident, findings, hypotheses, evidence, edges.
- `VerificationResult`: hypothesis ID, command/test, outcome, evidence IDs, status.
- `RCAReport`: ranked causes, Top-K locations, causal chain, verification, suspected regression change, fix recommendation/scope, uncertainty, timing, usage.

Rules:
- Every factual RCA claim references evidence IDs.
- Every evidence item has reproducible provenance.
- Confidence ranks evidence; it never replaces evidence.
- Missing/conflicting evidence yields `uncertain` or `insufficient_evidence`.
- Fix recommendations are advisory text/locations only, never generated edits.
- Persist repo-relative paths; never leak host/temp paths.

## 7. Security and Workspace Guarantees
Workspace Policy is authoritative; prompts are not a security boundary.

For the original analyzed repository always deny:
- File creation/edit/rename/delete and index/worktree mutation.
- Absolute paths, traversal, symlink escapes, outside-workspace access.
- `.env`, credentials, API keys, SSH keys, tokens, system secrets.
- `sudo`, downloads, destructive commands, `shell=True`, commit/push/reset/checkout.

Git tools use validated argument lists. Commands use `subprocess.run`, `shell=False`, repo-scoped `cwd`, bounded captured output, and finite timeouts.

Capture target fingerprint/Git status before and after each run. A successful RCA run must leave the original target unchanged. Runtime tests execute only in the disposable verification sandbox, where caches/temp files are allowed and discarded. Never print or serialize complete secrets.

## 8. Token, Context, and Concurrency Policy
- Send ranked snippets, never whole repos or unbounded Git history.
- Deduplicate shared snippets; cap candidates, excerpts, history, and Top-K cases.
- Run at most three concurrent evidence Specialists.
- Prefer deterministic tools over extra model turns.
- Record exact usage when returned; otherwise record `null`.
- Record wall-clock, model-call, and verification durations separately.
- Typical target: no more than 7 model calls per incident; exceed only with measured quality/cost justification.

## 9. Evaluation Rules
SWE-bench Verified is a patch-resolution benchmark. RootTrace derives localization metrics from its gold patch; these are not native SWE-bench RCA metrics.

Required metrics:
- `top_1_file_accuracy`: first predicted source file is a gold non-test patch file.
- `any_file_recall_at_k`: at least one gold source file appears in Top-K.
- `all_file_recall_at_k`: all gold source files appear in Top-K.
- `mean_gold_file_recall_at_k`.
- Coverage, invalid-output rate, P50/P95 latency, calls/case, tokens/case.

Use a checked-in fixed-seed 50-case development manifest with documented selection. Do not tune on the full 500 and report it as untouched test data.

Required ablations:
1. deterministic baseline without LLM Agents;
2. Lead + Code Specialist;
3. three evidence Specialists, historical retrieval off;
4. three evidence Specialists, retrieval on;
5. optional clustering-off comparison.

Hold model, prompts, budgets, concurrency, and manifest constant between comparable runs. Never call file localization `RCA Accuracy`. Natural-language root-cause correctness needs separate Silver/Gold adjudication and is deferred unless implemented.

## 10. Development and Testing
Before editing:
1. Read this file and the relevant design or evaluation documentation.
2. Inspect relevant code/tests and preserve unrelated changes.
3. Prefer a working vertical slice over placeholder abstractions.

During implementation:
- Add RCA code under `roottrace/rca/`.
- Reuse Provider, Workspace, trace, and config seams where practical.
- Keep deterministic collection separate from LLM reasoning.
- Type public APIs; use Pydantic for persisted schemas and `pathlib.Path`.
- Never hide worker failures, validation errors, or missing usage.

Tests must cover schema integrity, role isolation, true concurrency, deterministic aggregation, timeout/partial failure, path/security attacks, target immutability, sandbox isolation, retrieval leakage, seeded clustering, localization metrics, usage aggregation, CLI artifacts, and non-zero failure exits.

Run focused tests first, then when practical:
```bash
python -m pytest tests -q
ruff check roottrace tests evaluation
git diff --check
git status --short
```
Never weaken tests to hide defects.

## 11. Packaging, Artifacts, and Git
The distribution is `roottrace`; package discovery includes `roottrace*`
only, and the console script is `roottrace = roottrace.cli:main`. Exclude
tests, datasets, benchmark repositories, outputs, caches, and model artifacts
from package discovery.

```text
<output-dir>/
├── incident.json
├── investigation_plan.json
├── agents/{issue_ci,code,git_history}.json
├── evidence_graph.json
├── hypotheses.json
├── verification.json
├── rca_report.json
├── rca_report.md
├── execution_trace.jsonl
└── run_summary.json
```

Artifacts must be bounded, secret-free, deterministic where order is irrelevant, and explicit about model ID, base commit, config, timing, usage, and partial failures.

Keep Git changes focused. Do not rewrite history, force-push, auto-push, or
commit generated data. Prefer `feat(rca):`, `fix(rca):`, `test(rca):`,
`docs(rca):`, and `refactor(rca):`.

## 12. Completion Criteria
A RootTrace run is complete only when its vertical behavior works, the
original target is unchanged, sandbox writes are isolated, persisted outputs
validate, conclusions are evidence-linked or explicitly uncertain, relevant
tests and Ruff checks pass, usage and failures are honest, and the diff
contains no unrelated or generated data.
