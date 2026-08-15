import pytest

from benchmark.booleans import parse_bool


def test_parse_bool_is_case_insensitive() -> None:
    assert parse_bool("TRUE") is True
    assert parse_bool("No") is False


def test_parse_bool_rejects_unknown_values() -> None:
    with pytest.raises(ValueError):
        parse_bool("maybe")

