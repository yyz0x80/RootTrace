import pytest
from benchmark.pagination import paginate


def test_pagination_metadata_and_out_of_range_page() -> None:
    result = paginate(["a", "b", "c", "d", "e"], page=4, page_size=2)
    assert result == {
        "items": [],
        "page": 4,
        "page_size": 2,
        "total_items": 5,
        "total_pages": 3,
    }


def test_empty_collection_has_zero_pages() -> None:
    assert paginate([], page=1, page_size=10)["total_pages"] == 0


@pytest.mark.parametrize(("page", "page_size"), [(0, 1), (1, 0)])
def test_invalid_page_inputs_remain_rejected(page: int, page_size: int) -> None:
    with pytest.raises(ValueError):
        paginate([], page=page, page_size=page_size)
