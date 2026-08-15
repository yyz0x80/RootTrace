from benchmark.csv_records import parse_record


def test_parse_simple_record() -> None:
    assert parse_record("Ada, Lovelace, 36") == ["Ada", "Lovelace", "36"]

