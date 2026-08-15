import pytest

from benchmark.inventory import StockItem
from benchmark.inventory_service import fulfill


def test_fulfill_reduces_quantity() -> None:
    item = StockItem("SKU-1", 5)
    assert fulfill(item, 2) == 3


def test_fulfill_rejects_insufficient_stock() -> None:
    with pytest.raises(ValueError):
        fulfill(StockItem("SKU-1", 2), 3)

