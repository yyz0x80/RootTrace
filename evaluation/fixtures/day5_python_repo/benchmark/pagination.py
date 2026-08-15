"""Pagination helpers."""


def paginate(items: list[str], page: int, page_size: int) -> dict[str, object]:
    """Return one 1-based page of items."""
    if page < 1 or page_size < 1:
        raise ValueError("page and page_size must be positive")
    start = (page - 1) * page_size
    return {"items": items[start : start + page_size], "page": page}

