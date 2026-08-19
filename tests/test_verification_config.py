"""Tests for verification timeout configuration."""

from __future__ import annotations

import pytest

from patchpilot.verification.config import VerificationTimeouts


def test_default_timeouts() -> None:
    """Test that VerificationTimeouts has correct default values."""
    timeouts = VerificationTimeouts()
    assert timeouts.ruff == 30
    assert timeouts.target_tests == 60
    assert timeouts.regression_tests == 300
    assert timeouts.specialized == 60


def test_custom_timeouts() -> None:
    """Test that VerificationTimeouts accepts custom values."""
    timeouts = VerificationTimeouts(
        ruff=15,
        target_tests=90,
        regression_tests=600,
        specialized=120,
    )
    assert timeouts.ruff == 15
    assert timeouts.target_tests == 90
    assert timeouts.regression_tests == 600
    assert timeouts.specialized == 120


def test_timeout_validation_positive() -> None:
    """Test that timeout validation rejects non-positive values."""
    with pytest.raises(ValueError, match="must be positive"):
        VerificationTimeouts(ruff=0)

    with pytest.raises(ValueError, match="must be positive"):
        VerificationTimeouts(target_tests=-1)

    with pytest.raises(ValueError, match="must be positive"):
        VerificationTimeouts(regression_tests=-10)


def test_timeout_validation_integer() -> None:
    """Test that timeout validation rejects non-integer values."""
    with pytest.raises(TypeError, match="must be an integer"):
        VerificationTimeouts(ruff=30.5)

    with pytest.raises(TypeError, match="must be an integer"):
        VerificationTimeouts(target_tests="60")


def test_timeout_validation_maximums() -> None:
    """Test that timeout validation enforces maximum values."""
    with pytest.raises(ValueError, match="must be at most 120"):
        VerificationTimeouts(ruff=121)

    with pytest.raises(ValueError, match="must be at most 300"):
        VerificationTimeouts(target_tests=301)

    with pytest.raises(ValueError, match="must be at most 3600"):
        VerificationTimeouts(regression_tests=3601)

    with pytest.raises(ValueError, match="must be at most 300"):
        VerificationTimeouts(specialized=301)


def test_from_dict_with_all_values() -> None:
    """Test creating VerificationTimeouts from dictionary with all values."""
    data = {
        "ruff": 20,
        "target_tests": 90,
        "regression_tests": 600,
        "specialized": 120,
    }
    timeouts = VerificationTimeouts.from_dict(data)
    assert timeouts.ruff == 20
    assert timeouts.target_tests == 90
    assert timeouts.regression_tests == 600
    assert timeouts.specialized == 120


def test_from_dict_with_partial_values() -> None:
    """Test creating VerificationTimeouts from dictionary with partial values."""
    data = {"ruff": 20, "regression_tests": 600}
    timeouts = VerificationTimeouts.from_dict(data)
    assert timeouts.ruff == 20
    assert timeouts.target_tests == 60  # default
    assert timeouts.regression_tests == 600
    assert timeouts.specialized == 60  # default


def test_from_dict_with_empty_dict() -> None:
    """Test creating VerificationTimeouts from empty dictionary uses defaults."""
    timeouts = VerificationTimeouts.from_dict({})
    assert timeouts.ruff == 30
    assert timeouts.target_tests == 60
    assert timeouts.regression_tests == 300
    assert timeouts.specialized == 60


def test_from_dict_validation() -> None:
    """Test that from_dict validates timeout values."""
    with pytest.raises(ValueError, match="must be positive"):
        VerificationTimeouts.from_dict({"ruff": 0})

    with pytest.raises(ValueError, match="must be at most 120"):
        VerificationTimeouts.from_dict({"ruff": 200})


def test_to_dict() -> None:
    """Test converting VerificationTimeouts to dictionary."""
    timeouts = VerificationTimeouts(
        ruff=15,
        target_tests=90,
        regression_tests=600,
        specialized=120,
    )
    data = timeouts.to_dict()
    assert data == {
        "ruff": 15,
        "target_tests": 90,
        "regression_tests": 600,
        "specialized": 120,
    }


def test_frozen_dataclass() -> None:
    """Test that VerificationTimeouts is frozen (immutable)."""
    from dataclasses import FrozenInstanceError

    timeouts = VerificationTimeouts()
    with pytest.raises(FrozenInstanceError):
        timeouts.ruff = 45


def test_boundary_values() -> None:
    """Test that boundary values are accepted."""
    # Maximum allowed values
    timeouts = VerificationTimeouts(
        ruff=120,
        target_tests=300,
        regression_tests=3600,
        specialized=300,
    )
    assert timeouts.ruff == 120
    assert timeouts.target_tests == 300
    assert timeouts.regression_tests == 3600
    assert timeouts.specialized == 300

    # Minimum allowed values (1)
    timeouts = VerificationTimeouts(
        ruff=1,
        target_tests=1,
        regression_tests=1,
        specialized=1,
    )
    assert timeouts.ruff == 1
    assert timeouts.target_tests == 1
    assert timeouts.regression_tests == 1
    assert timeouts.specialized == 1
