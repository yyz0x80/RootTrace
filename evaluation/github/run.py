"""Command-line entry point for the online ``github_smoke10`` suite."""

from __future__ import annotations

import sys

from .runner import build_parser, run_from_args


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and execute the smoke runner."""
    return run_from_args(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["build_parser", "main", "run_from_args"]
