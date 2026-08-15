import pytest
from benchmark.tasks import TaskList


def test_priority_defaults_and_filtering() -> None:
    tasks = TaskList()
    normal = tasks.add("normal")
    first = tasks.add("urgent one", priority="high")
    second = tasks.add("urgent two", priority="high")
    assert normal.priority == "normal"
    assert tasks.by_priority("high") == [first, second]


def test_invalid_priority_does_not_append() -> None:
    tasks = TaskList()
    with pytest.raises(ValueError):
        tasks.add("invalid", priority="urgent")
    assert tasks.all() == []


def test_filter_rejects_invalid_priority() -> None:
    with pytest.raises(ValueError):
        TaskList().by_priority("urgent")
