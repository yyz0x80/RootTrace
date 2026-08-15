"""Inventory domain model."""

from dataclasses import dataclass


@dataclass
class StockItem:
    sku: str
    quantity: int

    def remove(self, amount: int) -> None:
        if amount < 0:
            raise ValueError("amount must be positive")
        self.quantity -= amount

