"""Model configuration module for unified model management.

This module provides functionality to load and resolve model configurations
from a configuration file, allowing users to specify models by name with
automatic resolution of base_url, api_key, and provider type.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    """Configuration for a specific model.

    Attributes:
        name: Model identifier name
        provider: Provider type ("openai" or "anthropic")
        base_url: API base URL for the provider
        api_key: API authentication key
        model_id: Actual model identifier to send to the API
    """

    name: str
    provider: str
    base_url: str | None
    api_key: str
    model_id: str

    def to_dict(self) -> dict[str, Any]:
        """Convert model config to dictionary."""
        return {
            "name": self.name,
            "provider": self.provider,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "model_id": self.model_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        """Create model config from dictionary."""
        return cls(
            name=data["name"],
            provider=data["provider"],
            base_url=data.get("base_url"),
            api_key=data["api_key"],
            model_id=data["model_id"],
        )


class ModelConfigManager:
    """Manager for loading and resolving model configurations."""

    DEFAULT_CONFIG_PATH = Path.home() / ".patchpilot" / "models.json"
    FALLBACK_CONFIG_PATH = Path(".patchpilot") / "models.json"

    def __init__(self, config_path: Path | None = None) -> None:
        """Initialize the model configuration manager.

        Args:
            config_path: Optional path to model configuration file.
                        If not provided, searches default locations.
        """
        self._config_path = config_path
        self._configs: dict[str, ModelConfig] = {}
        self._load_configs()

    def _load_configs(self) -> None:
        """Load model configurations from file."""
        config_path = self._find_config_file()
        if config_path and config_path.exists():
            try:
                with open(config_path, encoding="utf-8") as f:
                    data = json.load(f)
                    for model_data in data.get("models", []):
                        config = ModelConfig.from_dict(model_data)
                        self._configs[config.name] = config
            except (json.JSONDecodeError, OSError, KeyError) as e:
                # Log warning but continue with empty configs
                print(f"Warning: Failed to load model config from {config_path}: {e}")

    def _find_config_file(self) -> Path | None:
        """Find the model configuration file.

        Search order:
        1. Explicitly provided path
        2. ~/.patchpilot/models.json
        3. .patchpilot/models.json (project local)
        """
        if self._config_path:
            return self._config_path

        if self.DEFAULT_CONFIG_PATH.exists():
            return self.DEFAULT_CONFIG_PATH

        if self.FALLBACK_CONFIG_PATH.exists():
            return self.FALLBACK_CONFIG_PATH

        return None

    def get_config(self, model_name: str) -> ModelConfig | None:
        """Get configuration for a specific model name.

        Args:
            model_name: The model name to look up

        Returns:
            ModelConfig if found, None otherwise
        """
        return self._configs.get(model_name)

    def list_models(self) -> list[str]:
        """List all available model names.

        Returns:
            List of model names
        """
        return list(self._configs.keys())

    def resolve_with_env_fallback(
        self,
        model_name: str | None,
        provider_type: str = "openai",
    ) -> tuple[str, str | None, str]:
        """Resolve model configuration with environment variable fallback.

        This method tries to get configuration from the model config file first,
        then falls back to environment variables if not found.

        Args:
            model_name: Optional model name from config file
            provider_type: Provider type ("openai" or "anthropic")

        Returns:
            Tuple of (model_id, base_url, api_key)

        Raises:
            ValueError: If configuration cannot be resolved
        """
        # Try to get from model config file
        if model_name:
            config = self.get_config(model_name)
            if config:
                return (config.model_id, config.base_url, config.api_key)

        # Fallback to environment variables (use existing variable names for compatibility)
        if provider_type == "openai":
            # Support both ZHIPU_API_KEY (existing) and OPENAI_API_KEY (standard)
            api_key = os.getenv("ZHIPU_API_KEY") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("ZHIPU_API_KEY or OPENAI_API_KEY environment variable is not set")
            base_url = os.getenv("PATCHPILOT_BASE_URL") or os.getenv("OPENAI_BASE_URL")
            model_id = model_name or os.getenv("PATCHPILOT_MODEL") or "gpt-4o"
            return (model_id, base_url, api_key)

        elif provider_type == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
            # Anthropic doesn't use custom base_url typically
            model_id = model_name or os.getenv("PATCHPILOT_MODEL") or "claude-sonnet-4-20250514"
            return (model_id, None, api_key)

        else:
            raise ValueError(f"Unsupported provider type: {provider_type}")
