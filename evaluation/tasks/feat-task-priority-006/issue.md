# Add task priorities and filtering

The task list needs a small, backwards-compatible priority feature.

Acceptance requirements:

- Add a `priority: str` field to `Task` with default value `"normal"`.
- `TaskList.add` accepts an optional `priority` argument with the same default.
- Only `"low"`, `"normal"`, and `"high"` are valid; invalid values raise
  `ValueError` and must not append a task.
- Add `TaskList.by_priority(priority: str) -> list[Task]`, preserving insertion
  order among matches and applying the same validation.
- Existing calls that pass only a title continue to work.
- Keep tests read-only and change only `benchmark/tasks.py`.

