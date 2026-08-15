from benchmark.statistics import median


def test_even_length_median() -> None:
    assert median([1, 10, 2, 8]) == 5.0


def test_input_is_not_mutated() -> None:
    values = [4, 1, 3, 2]
    assert median(values) == 2.5
    assert values == [4, 1, 3, 2]

