# Add description field to Task

The Task dataclass currently only has a title field. We need to add a description field to support more detailed task information.

Acceptance requirements:

- Add a `description: str` field to the Task dataclass
- Update TaskService.create_task to accept an optional description parameter
- When description is not provided, default to empty string
- Update tests to verify the new description field works correctly
