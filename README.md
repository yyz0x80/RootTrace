# PatchPilot

PatchPilot is an issue-to-patch agent for making bounded, reviewable changes to
existing Python repositories.

## Three-layer test policy

PatchPilot separates repository tests, agent checks, and required test artifacts:

1. Existing repository tests are immutable. The coding agent may read and run
   them, but every editing tool and the runtime scope gate reject modifications.
2. Agent-authored scratch tests live under `.patchpilot_checks/`. They run in the
   sandbox as supplemental evidence and are excluded from the generated patch.
3. A new repository test may enter the patch only when the normalized issue has
   an explicit, required `target_test_change` artifact and the approved plan lists
   that exact path with the `create` action. Existing tests remain immutable.

Ruff, selected repository tests, and the full regression suite remain the
authoritative deterministic checks. Scratch tests cannot replace them or turn an
unverified acceptance criterion into a pass.
