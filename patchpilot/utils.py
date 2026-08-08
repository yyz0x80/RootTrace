"""Utility functions for PatchPilot."""

from pathlib import Path


def save_json(path: str, json_text: str) -> None:
    """Save JSON text to a file, creating parent directories if needed.

    Args:
        path: Path to the output file.
        json_text: JSON content to write.
    """
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json_text, encoding="utf-8")
