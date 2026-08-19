from benchmark.pagination import paginate


def test_paginate_returns_requested_page() -> None:
    result = paginate(["a", "b", "c"], page=2, page_size=2)
    assert result["items"] == ["c"]
    assert result["page"] == 2

