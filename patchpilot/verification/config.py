"""Verification timeout and strategy configuration.

This module provides configuration for verification timeouts and tiered
verification strategies. It allows users to customize verification behavior
based on their repository's characteristics and confidence requirements.

The configuration supports:
- VerificationTimeouts: Time budgets for different verification levels
- VerificationStrategy: Tiered verification policy (strict, balanced, focused)
- VerificationTier: Test classification (REQUIRED, AFFECTED, OPTIONAL)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class VerificationTimeouts:
    """Configuration for verification timeout budgets.

    Attributes:
        ruff: Timeout in seconds for Ruff linting checks (default: 30)
        target_tests: Timeout in seconds for targeted pytest runs (default: 60)
        regression_tests: Timeout in seconds for full regression test suites (default: 300)
        specialized: Timeout in seconds for specialized verification checks (default: 60)

    All timeouts must be positive integers and bounded to reasonable maximums
    to prevent misconfiguration.
    """

    ruff: int = 30
    target_tests: int = 60
    regression_tests: int = 300
    specialized: int = 60

    # Maximum allowed timeout values to prevent misconfiguration
    MAX_RUFF_TIMEOUT = 120
    MAX_TARGET_TESTS_TIMEOUT = 300
    MAX_REGRESSION_TESTS_TIMEOUT = 3600  # 1 hour
    MAX_SPECIALIZED_TIMEOUT = 300

    def __post_init__(self) -> None:
        """Validate timeout values after initialization.

        Raises:
            ValueError: If any timeout value is invalid
        """
        self._validate_timeout("ruff", self.ruff, self.MAX_RUFF_TIMEOUT)
        self._validate_timeout(
            "target_tests", self.target_tests, self.MAX_TARGET_TESTS_TIMEOUT
        )
        self._validate_timeout(
            "regression_tests",
            self.regression_tests,
            self.MAX_REGRESSION_TESTS_TIMEOUT,
        )
        self._validate_timeout(
            "specialized", self.specialized, self.MAX_SPECIALIZED_TIMEOUT
        )

    def _validate_timeout(
        self, name: str, value: int, max_value: int
    ) -> None:
        """Validate a single timeout value.

        Args:
            name: Name of the timeout parameter for error messages
            value: Timeout value in seconds
            max_value: Maximum allowed timeout value

        Raises:
            ValueError: If value is not a positive integer or exceeds max_value
        """
        if not isinstance(value, int):
            raise TypeError(
                f"Timeout '{name}' must be an integer, got {type(value).__name__}"
            )
        if value <= 0:
            raise ValueError(
                f"Timeout '{name}' must be positive, got {value}"
            )
        if value > max_value:
            raise ValueError(
                f"Timeout '{name}' must be at most {max_value} seconds, got {value}"
            )

    @classmethod
    def from_dict(cls, data: dict[str, int]) -> VerificationTimeouts:
        """Create VerificationTimeouts from a dictionary.

        Args:
            data: Dictionary with timeout values (keys: ruff, target_tests,
                  regression_tests, specialized)

        Returns:
            VerificationTimeouts instance with values from dictionary

        Raises:
            ValueError: If any value is invalid
        """
        return cls(
            ruff=data.get("ruff", cls().ruff),
            target_tests=data.get("target_tests", cls().target_tests),
            regression_tests=data.get(
                "regression_tests", cls().regression_tests
            ),
            specialized=data.get("specialized", cls().specialized),
        )

    def to_dict(self) -> dict[str, int]:
        """Convert VerificationTimeouts to a dictionary.

        Returns:
            Dictionary with all timeout values
        """
        return {
            "ruff": self.ruff,
            "target_tests": self.target_tests,
            "regression_tests": self.regression_tests,
            "specialized": self.specialized,
        }


class VerificationTier(str, Enum):
    """Classification of verification test tiers.

    Tiers represent the confidence level and relationship of tests to the changes:
    - REQUIRED: Direct target tests, explicit acceptance tests, exact pytest node IDs
    - AFFECTED: Tests connected to changed files through deterministic dependency analysis
    - OPTIONAL: Remaining full-suite regression tests
    """

    REQUIRED = "required"
    AFFECTED = "affected"
    OPTIONAL = "optional"


class VerificationStrategy(str, Enum):
    """Tiered verification policy strategies.

    Strategies define how verification results across tiers determine overall success:
    - strict: Any new post-patch failure blocks VERIFIED
    - balanced: Required, affected, and deterministic regression checks must pass
    - focused: REQUIRED tests must pass; missing broader tests prevent full-confidence VERIFIED
    """

    STRICT = "strict"
    BALANCED = "balanced"
    FOCUSED = "focused"

    @classmethod
    def from_string(cls, value: str) -> VerificationStrategy:
        """Convert string to VerificationStrategy with validation.

        Args:
            value: String representation of the strategy

        Returns:
            VerificationStrategy enum value

        Raises:
            ValueError: If the string is not a valid strategy
        """
        try:
            return cls(value.lower())
        except ValueError:
            valid_strategies = [s.value for s in cls]
            raise ValueError(
                f"Invalid verification strategy '{value}'. "
                f"Valid strategies are: {', '.join(valid_strategies)}"
            )
