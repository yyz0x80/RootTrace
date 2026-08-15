from csv_demo import parse_record


def test_parse_simple_record() -> None:
    assert parse_record("Ada,Lovelace,36") == ["Ada", "Lovelace", "36"]


def test_parse_quoted_comma() -> None:
    assert parse_record('"Lovelace, Ada",36') == ["Lovelace, Ada", "36"]

