"""Tests for model configuration management."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from roottrace.llm.config import ModelConfig, ModelConfigManager


def test_model_config_to_dict() -> None:
    """Test converting ModelConfig to dictionary."""
    config = ModelConfig(
        name="test-model",
        provider="openai",
        base_url="https://api.test.com",
        api_key="test-key",
        model_id="gpt-4",
    )
    data = config.to_dict()
    assert data["name"] == "test-model"
    assert data["provider"] == "openai"
    assert data["base_url"] == "https://api.test.com"
    assert data["api_key"] == "test-key"
    assert data["model_id"] == "gpt-4"


def test_model_config_from_dict() -> None:
    """Test creating ModelConfig from dictionary."""
    data = {
        "name": "test-model",
        "provider": "openai",
        "base_url": "https://api.test.com",
        "api_key": "test-key",
        "model_id": "gpt-4",
    }
    config = ModelConfig.from_dict(data)
    assert config.name == "test-model"
    assert config.provider == "openai"
    assert config.base_url == "https://api.test.com"
    assert config.api_key == "test-key"
    assert config.model_id == "gpt-4"


def test_model_config_from_dict_without_base_url() -> None:
    """Test creating ModelConfig from dictionary without base_url."""
    data = {
        "name": "test-model",
        "provider": "anthropic",
        "api_key": "test-key",
        "model_id": "claude-3",
    }
    config = ModelConfig.from_dict(data)
    assert config.name == "test-model"
    assert config.provider == "anthropic"
    assert config.base_url is None
    assert config.api_key == "test-key"
    assert config.model_id == "claude-3"


def test_model_config_manager_loads_from_file(tmp_path: Path) -> None:
    """Test loading model configurations from file."""
    config_file = tmp_path / "models.json"
    config_data = {
        "models": [
            {
                "name": "gpt4-custom",
                "provider": "openai",
                "base_url": "https://api.custom.com",
                "api_key": "custom-key",
                "model_id": "gpt-4",
            },
            {
                "name": "claude-custom",
                "provider": "anthropic",
                "api_key": "anthropic-key",
                "model_id": "claude-3-opus",
            },
        ]
    }
    config_file.write_text(json.dumps(config_data), encoding="utf-8")

    manager = ModelConfigManager(config_path=config_file)
    assert manager.list_models() == ["gpt4-custom", "claude-custom"]

    gpt_config = manager.get_config("gpt4-custom")
    assert gpt_config is not None
    assert gpt_config.provider == "openai"
    assert gpt_config.base_url == "https://api.custom.com"
    assert gpt_config.api_key == "custom-key"
    assert gpt_config.model_id == "gpt-4"

    claude_config = manager.get_config("claude-custom")
    assert claude_config is not None
    assert claude_config.provider == "anthropic"
    assert claude_config.base_url is None
    assert claude_config.api_key == "anthropic-key"
    assert claude_config.model_id == "claude-3-opus"


def test_model_config_manager_handles_missing_file() -> None:
    """Test that manager handles missing config file gracefully."""
    manager = ModelConfigManager(config_path=Path("/nonexistent/path/models.json"))
    assert manager.list_models() == []
    assert manager.get_config("any-model") is None


def test_model_config_manager_handles_invalid_json(tmp_path: Path) -> None:
    """Test that manager handles invalid JSON gracefully."""
    config_file = tmp_path / "models.json"
    config_file.write_text("invalid json", encoding="utf-8")

    # Should not raise exception, just print warning
    manager = ModelConfigManager(config_path=config_file)
    assert manager.list_models() == []


def test_resolve_with_env_fallback_uses_config() -> None:
    """Test that resolve_with_env_fallback uses config when available."""
    config_data = {
        "models": [
            {
                "name": "gpt4-configured",
                "provider": "openai",
                "base_url": "https://api.configured.com",
                "api_key": "configured-key",
                "model_id": "gpt-4-turbo",
            }
        ]
    }

    # Use actual file for more realistic test
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f)
        config_path = Path(f.name)

    try:
        manager = ModelConfigManager(config_path=config_path)
        model_id, base_url, api_key = manager.resolve_with_env_fallback("gpt4-configured", "openai")

        assert model_id == "gpt-4-turbo"
        assert base_url == "https://api.configured.com"
        assert api_key == "configured-key"
    finally:
        config_path.unlink()


def test_resolve_with_env_fallback_uses_env_when_config_not_found() -> None:
    """Test that resolve_with_env_fallback falls back to env vars when model not in config."""
    manager = ModelConfigManager()

    with patch.dict("os.environ", {"ZHIPU_API_KEY": "env-key", "ROOTTRACE_BASE_URL": "https://env.com"}, clear=False):
        model_id, base_url, api_key = manager.resolve_with_env_fallback("unknown-model", "openai")

        assert model_id == "unknown-model"
        assert base_url == "https://env.com"
        assert api_key == "env-key"


def test_resolve_with_env_fallback_raises_error_when_no_api_key() -> None:
    """Test that resolve_with_env_fallback raises error when API key is missing."""
    manager = ModelConfigManager()

    with patch.dict("os.environ", {}, clear=True):
        try:
            manager.resolve_with_env_fallback("unknown-model", "openai")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "ZHIPU_API_KEY" in str(e) or "OPENAI_API_KEY" in str(e)


def test_resolve_with_env_fallback_uses_default_model() -> None:
    """Test that resolve_with_env_fallback uses default model when none provided."""
    manager = ModelConfigManager()

    # Clear ROOTTRACE_MODEL to test the default behavior
    with patch.dict("os.environ", {"ZHIPU_API_KEY": "env-key", "ROOTTRACE_BASE_URL": "https://env.com"}, clear=True):
        # Add back the required variables
        os.environ["ZHIPU_API_KEY"] = "env-key"
        os.environ["ROOTTRACE_BASE_URL"] = "https://env.com"

        model_id, base_url, api_key = manager.resolve_with_env_fallback(None, "openai")

        assert model_id == "gpt-4o"  # Default model when ROOTTRACE_MODEL is not set
        assert base_url == "https://env.com"
        assert api_key == "env-key"
