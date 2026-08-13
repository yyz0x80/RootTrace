from .task import Task


class TaskService:
    def __init__(self):
        self.tasks: list[Task] = []

    def create_task(self, title: str) -> Task:
        task = Task(title=title)
        self.tasks.append(task)
        return task

    def list_tasks(self) -> list[Task]:
        return self.tasks