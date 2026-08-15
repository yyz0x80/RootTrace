from csv_demo import parse_record


def test_escaped_quote() -> None:
    assert parse_record('"Ada ""Countess"" Lovelace",36') == [
        'Ada "Countess" Lovelace',
        "36",
    ]


def test_quoted_space_is_preserved_but_unquoted_space_is_trimmed() -> None:
    assert parse_record('  Ada  ," Lovelace ", 36 ') == [
        "Ada",
        " Lovelace ",
        "36",
    ]

