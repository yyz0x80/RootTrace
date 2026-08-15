import pytest
from benchmark.booleans import parse_bool


@pytest.mark.parametrize(
    ("value", "expected"),
    [("  YES\n", True), ("\tfalse ", False), ("\u2003TrUe\u2003", True)],
)
def test_surrounding_whitespace_is_ignored(value: str, expected: bool) -> None:
    assert parse_bool(value) is expected


def test_whitespace_only_value_is_rejected() -> None:
    with pytest.raises(ValueError):
        parse_bool(" \t\n")
