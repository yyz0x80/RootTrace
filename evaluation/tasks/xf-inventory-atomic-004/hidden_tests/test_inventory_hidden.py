import pytest
from benchmark.inventory import StockItem
from benchmark.inventory_service import fulfill


@pytest.mark.parametrize("amount", [0, -1, 4])
def test_invalid_removal_is_atomic(amount: int) -> None:
    item = StockItem("SKU-1", 3)
    with pytest.raises(ValueError):
        item.remove(amount)
    assert item.quantity == 3


def test_exact_fulfillment_reaches_zero() -> None:
    item = StockItem("SKU-1", 3)
    assert fulfill(item, 3) == 0
