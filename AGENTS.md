# AGENTS.md

## 1. Project Overview

PatchPilot is an **Issue-to-Patch Code Agent** for existing Python repositories.

Its goal is to:

1. Accept a local issue description or GitHub Issue.
2. Inspect the target repository.
3. Locate relevant implementation and test files.
4. Make the smallest valid source-code change.
5. Run deterministic verification such as Ruff and Pytest.
6. Produce a reviewable Git diff and verification result.

PatchPilot is not a general-purpose chatbot or a greenfield vibe-coding tool.

---

## 2. Repository Boundaries

The active development repository is the current `PatchPilot` directory.

Typical structure:

```text
PatchPilot/
├── patchpilot/       # PatchPilot application source
├── tests/            # Tests for PatchPilot itself
├── demo_repo/        # Repository used as an Agent execution target
├── pyproject.toml
└── AGENTS.md
```

Rules:

* Modify only files inside the current `PatchPilot` repository.
* Do not access or modify parent directories.
* Do not import code directly from external reference repositories.
* `demo_repo/` is a target repository used for demonstrations and evaluation. It is not part of the `patchpilot` Python package.
* Do not include `demo_repo/` or `tests/` in the distributed Python package.
* Never modify another Git repository unless the current task explicitly names it as the target workspace.

---

## 3. Current Scope

The initial version supports:

* Python repositories
* Local Markdown issue files
* OpenAI-compatible model providers
* Structured tool calling
* Repository-scoped file access
* Source-code search
* Bounded file reading
* Exact text replacement
* Restricted command execution
* Git diff output
* Ruff and Pytest verification

Out of scope unless explicitly requested:

* Web UI
* VS Code extension
* Multi-agent conversation
* Vector databases
* Long-term memory
* Automatic Git push or merge
* Arbitrary shell access
* Multi-language repository support
* Greenfield project generation
* Large-scale SWE-bench integration

Do not expand the project scope without an explicit task.

---

## 4. Core Architecture

Maintain clear separation between the following modules:

```text
CLI
  ↓
Workflow / Agent Loop
  ↓
Provider + Tool Registry
  ↓
Workspace Policy
  ↓
Target Repository
```

### Provider

The Provider is responsible for:

* Calling an OpenAI-compatible API
* Converting internal messages into provider requests
* Converting provider responses into internal models
* Parsing tool-call arguments
* Handling provider-specific errors and retries

The Provider must not:

* Read or write repository files
* Execute shell commands
* Contain workflow logic
* Return raw SDK objects to other modules

### Agent Loop

The Agent Loop is responsible for:

* Maintaining the message history
* Calling the Provider
* Detecting tool requests
* Dispatching tools through the Tool Registry
* Adding tool results back to the conversation
* Enforcing maximum-round limits
* Returning the final Agent response

The Agent Loop must not:

* Implement provider HTTP logic
* Perform direct file-system operations
* Run shell commands directly
* Bypass the Tool Registry or Workspace Policy

### Tools

Each tool must:

* Have one clear responsibility
* Define an explicit JSON-compatible input schema
* Return a structured success or failure result
* Validate all inputs
* Enforce output-size and execution-time limits
* Operate only inside the configured target workspace

Initial tools:

* `search_code`
* `read_file`
* `edit_file`
* `run_command`

### Workspace Policy

The Workspace Policy is the authoritative security boundary.

It must:

* Resolve all paths relative to the target repository root
* Reject absolute paths
* Reject `..` path traversal
* Reject files outside the target repository
* Reject sensitive files and directories
* Apply separate read and write policies

Prompt instructions are not a substitute for programmatic enforcement.

---

## 5. Security Rules

Never weaken these rules without an explicit security-related task.

### Always deny

* Reading or writing `.env` files
* Reading credentials, API keys, SSH keys, or system secrets
* Accessing `.git/` internals directly
* Accessing files outside the target workspace
* Running commands with `sudo`
* Running `git push`
* Running destructive commands such as `rm -rf`
* Running arbitrary network download commands
* Using `shell=True`
* Disabling tests to make a patch appear successful

### Require explicit approval or configuration

* Installing dependencies
* Deleting files
* Modifying dependency lockfiles
* Modifying CI/CD configuration
* Modifying files outside the approved change plan
* Running commands that require network access

### Day 1 write restrictions

For the initial minimal implementation:

* Source files may be modified.
* Target repository tests must be treated as read-only.
* Files matching `test_*.py` or paths under `tests/` must not be modified by the coding Agent.

PatchPilot's own tests under the root `tests/` directory may be modified when developing PatchPilot itself.

---

## 6. Command Execution Policy

Use `subprocess.run` with:

* An argument list rather than a shell string
* `shell=False`
* A repository-scoped working directory
* Captured standard output and error
* A finite timeout
* A bounded output size

Initially allowed commands:

```text
python -m pytest
pytest
ruff check
git diff
git status
```

Command validation must inspect parsed arguments, not only string prefixes.

Do not add a general unrestricted shell tool.

---

## 7. File Editing Policy

Prefer the smallest possible change.

The initial `edit_file` tool should use exact text replacement:

1. Read the current file.
2. Verify that `old_text` exists exactly once.
3. Reject zero or multiple matches.
4. Apply the replacement.
5. Return a unified diff.

Do not rewrite an entire file when a local replacement is sufficient.

Preserve:

* Existing public interfaces
* Existing formatting style
* Existing type annotations
* Existing behavior unrelated to the issue

Do not modify tests merely to make failing code pass.

---

## 8. Agent Behavior

The coding Agent must follow this sequence:

1. Inspect the repository before editing.
2. Search for relevant symbols and files.
3. Read the implementation and related tests.
4. Form a minimal change hypothesis.
5. Modify source code.
6. Run the narrowest relevant verification.
7. Run broader verification when practical.
8. Report success only when deterministic checks pass.
9. Report a concrete blocker when the task cannot be completed.

The Agent must not:

* Guess file contents
* Claim tests passed without running them
* Invent tool results
* Continue indefinitely
* Expand an issue into unrelated refactoring
* Change tests to hide an implementation defect

---

## 9. Development Workflow for AI Coding Tools

Before making changes:

1. Read this file completely.
2. Inspect the relevant existing files.
3. Summarize the intended change in a few sentences.
4. Identify the files that need modification.
5. Avoid editing unrelated files.

During implementation:

* Work on one focused task at a time.
* Preserve current module boundaries.
* Prefer extending existing abstractions over introducing parallel ones.
* Avoid broad refactoring unless required by the task.
* Do not generate placeholder implementations presented as complete.
* Do not silently ignore exceptions.
* Use clear error types and actionable messages.
* Keep functions small and single-purpose.

After implementation:

1. Run the relevant focused tests.
2. Run the broader test suite when practical.
3. Run Ruff on changed Python files or the project.
4. Inspect `git diff`.
5. Report:

   * Files changed
   * Main design decisions
   * Commands executed
   * Test results
   * Remaining limitations

Do not claim completion if required tests fail.

---

## 10. Testing Requirements

Every behavior change should include or update tests.

Tests should cover:

* Successful behavior
* Invalid input
* Security boundaries
* Failure handling
* Path traversal attempts
* Command allowlist enforcement
* Provider tool-call parsing
* Agent-loop stopping conditions

Use:

```bash
python -m pytest tests -q
ruff check patchpilot tests
```

For a focused change, run the relevant test file first:

```bash
python -m pytest tests/test_workspace.py -q
```

Do not delete or weaken an existing test without explaining why the expected behavior has legitimately changed.

---

## 11. Python Standards

* Python 3.11 or later
* Type annotations for public functions and methods
* `pathlib.Path` for file paths
* Dataclasses or Pydantic models for structured internal data
* Explicit exceptions for expected failure conditions
* UTF-8 text handling
* No mutable default arguments
* No hidden global state
* No hard-coded API keys
* No provider-specific SDK objects outside the Provider module
* No `print` calls in reusable library code unless they are part of an explicit CLI or tracing interface

Prefer standard-library solutions when they are clear and sufficient.

---

## 12. Configuration and Secrets

Configuration may come from:

* Environment variables
* CLI arguments
* Explicit configuration files

Secrets must come from environment variables.

Never:

* Commit `.env`
* Hard-code an API key
* Print complete API keys
* Include secrets in traces, exceptions, fixtures, or examples

Use `.env.example` only for variable names and safe placeholder values.

---

## 13. Packaging Rules

The installable package is `patchpilot`.

`pyproject.toml` must explicitly include only `patchpilot*` and exclude:

* `demo_repo*`
* `tests*`
* Build artifacts

Do not turn `demo_repo/` into an installable Python package.

The following command must remain valid:

```bash
python -m pip install -e ".[dev]"
```

---

## 14. Git Rules

* Keep commits focused and descriptive.
* Do not rewrite Git history.
* Do not force-push.
* Do not push automatically.
* Do not commit generated caches, secrets, or build output.
* Do not mix unrelated refactoring with feature work.

Recommended commit prefixes:

```text
chore:
feat:
fix:
test:
docs:
refactor:
```

Always inspect:

```bash
git status
git diff
```

before reporting completion.

---

## 15. Definition of Done

A task is complete only when:

* The requested behavior is implemented.
* Security boundaries remain enforced.
* Relevant tests pass.
* Ruff reports no new violations.
* The package remains installable.
* The diff contains no unrelated changes.
* No secrets or generated files are included.
* The final report clearly states what was changed and verified.

For Agent-executed repository tasks, success additionally requires:

* The target repository tests were not modified.
* Verification commands were actually executed.
* The resulting Git diff is available for human review.
* The Agent did not access files outside the target workspace.

---

## 16. Instruction Priority

When instructions conflict, follow this order:

1. Security and workspace restrictions in this file
2. Explicit user task
3. Existing project architecture and tests
4. Local implementation convenience

Do not bypass a security restriction merely because a task prompt requests it. Report the conflict instead.
