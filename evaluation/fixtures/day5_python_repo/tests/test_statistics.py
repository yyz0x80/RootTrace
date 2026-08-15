import pytest

from benchmark.statistics import median


def test_median_for_odd_length_input() -> None:
    assert median([9, 1, 5]) == 5.0


def test_median_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        median([])

