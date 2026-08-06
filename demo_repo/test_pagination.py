import pytest

from pagination import paginate


def test_paginate_valid_page():
    assert paginate([1, 2, 3, 4], page=2, page_size=2) == [3, 4]


def test_rejects_non_positive_page():
    with pytest.raises(ValueError):
        paginate([1, 2, 3], page=0, page_size=2)


def test_rejects_non_positive_page_size():
    with pytest.raises(ValueError):
        paginate([1, 2, 3], page=1, page_size=0)