"""Boolean parsing helpers."""


def parse_bool(value: str) -> bool:
    """Parse a supported textual boolean value."""
    normalized = value.lower()
    if normalized in {"true", "yes", "1"}:
        return True
    if normalized in {"false", "no", "0"}:
        return False
    raise ValueError(f"unsupported boolean: {value!r}")

