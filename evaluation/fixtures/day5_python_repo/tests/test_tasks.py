from benchmark.tasks import TaskList


def test_task_list_adds_and_returns_tasks() -> None:
    tasks = TaskList()
    task = tasks.add("ship release")

    assert task.title == "ship release"
    assert tasks.all() == [task]

