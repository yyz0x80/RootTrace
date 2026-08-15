# Keep inventory removal validation atomic

Validation is split between `fulfill` and `StockItem.remove`, and direct model
calls can make stock negative or accept a zero amount.

Acceptance requirements:

- `StockItem.remove` must reject amounts less than or equal to zero.
- `StockItem.remove` must reject amounts greater than the available quantity.
- A rejected removal must not change `quantity`.
- `fulfill` must delegate validation and removal to `StockItem.remove` and return
  the remaining quantity.
- Removing exactly the available quantity must succeed and leave zero.
- Keep tests read-only and change only `benchmark/inventory.py` and
  `benchmark/inventory_service.py`.

