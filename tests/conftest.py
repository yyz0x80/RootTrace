"""Pytest configuration and shared fixtures."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def sandbox_mock() -> MagicMock:
    """Create a mock DockerSandbox for testing."""
    return MagicMock()
