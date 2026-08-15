"""Small statistics helpers."""


def median(values: list[float]) -> float:
    """Return the median of a non-empty list."""
    if not values:
        raise ValueError("median requires at least one value")
    ordered = sorted(values)
    return float(ordered[(len(ordered) - 1) // 2])

