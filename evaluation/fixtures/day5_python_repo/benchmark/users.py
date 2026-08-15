"""User domain model."""

from dataclasses import dataclass


@dataclass(frozen=True)
class User:
    email: str


def normalize_email(email: str) -> str:
    """Return the canonical form used for user identity."""
    return email.strip().lower()

