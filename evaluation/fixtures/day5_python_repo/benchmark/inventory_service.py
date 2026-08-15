"""Inventory operations."""

from benchmark.inventory import StockItem


def fulfill(item: StockItem, amount: int) -> int:
    """Remove stock and return the remaining quantity."""
    if item.quantity < amount:
        raise ValueError("insufficient stock")
    item.remove(amount)
    return item.quantity

