"""CSV parsing fixture."""


def parse_record(line: str) -> list[str]:
    """Parse one CSV record into fields."""
    return [field.strip() for field in line.split(",")]

