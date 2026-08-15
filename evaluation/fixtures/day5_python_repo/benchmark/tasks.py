"""Task collection."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    title: str


class TaskList:
    def __init__(self) -> None:
        self._tasks: list[Task] = []

    def add(self, title: str) -> Task:
        task = Task(title=title)
        self._tasks.append(task)
        return task

    def all(self) -> list[Task]:
        return list(self._tasks)

