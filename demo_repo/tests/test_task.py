from ..task_service import TaskService


def test_create_task():
    service = TaskService()

    task = service.create_task("learn agents")

    assert task.title == "learn agents"


def test_list_tasks():
    service = TaskService()

    service.create_task("task 1")
    service.create_task("task 2")

    assert len(service.list_tasks()) == 2