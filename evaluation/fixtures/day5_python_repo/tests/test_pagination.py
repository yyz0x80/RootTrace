from benchmark.pagination import paginate


def test_paginate_returns_requested_page() -> None:
    assert paginate(["a", "b", "c"], page=2, page_size=2) == {
        "items": ["c"],
        "page": 2,
    }

