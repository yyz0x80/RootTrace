"""Ablation variant config tests (M9-C)."""

from __future__ import annotations

import json

import pytest

from evaluation.rca.variants import (
    AblationConfig,
    AblationVariant,
    variant_settings,
)
from patchpilot.rca.schema import AgentRole


def test_all_variants_parse_and_map_to_pipeline_settings() -> None:
    settings = {
        AblationVariant.DETERMINISTIC_BASELINE: (
            (),
            "off",
            True,
        ),
        AblationVariant.LEAD_CODE: (
            (AgentRole.CODE,),
            "off",
            False,
        ),
        AblationVariant.THREE_SPECIALISTS_RETRIEVAL_OFF: (
            (AgentRole.ISSUE_CI, AgentRole.CODE, AgentRole.GIT_HISTORY),
            "off",
            False,
        ),
        AblationVariant.THREE_SPECIALISTS_RETRIEVAL_ON: (
            (AgentRole.ISSUE_CI, AgentRole.CODE, AgentRole.GIT_HISTORY),
            "clustered",
            False,
        ),
        AblationVariant.CLUSTERING_OFF: (
            (AgentRole.ISSUE_CI, AgentRole.CODE, AgentRole.GIT_HISTORY),
            "flat",
            False,
        ),
    }
    for variant, (roles, mode, deterministic) in settings.items():
        parsed = variant_settings(variant)
        assert parsed.variant is variant
        assert parsed.enabled_roles == roles
        assert parsed.retrieval_mode == mode
        assert parsed.deterministic is deterministic
        assert parsed.label


def test_ablation_config_hash_is_stable_and_sensitive_to_variant() -> None:
    first = AblationConfig(variant="lead_code", model="m")
    second = AblationConfig(variant="lead_code", model="m")
    different = AblationConfig(variant="three_specialists_retrieval_off", model="m")
    assert first.config_hash() == second.config_hash()
    assert first.config_hash() != different.config_hash()


def test_ablation_config_round_trips_through_json(tmp_path) -> None:
    config = AblationConfig(
        variant="three_specialists_retrieval_on",
        model="glm-4-flash",
        budgets={"timeout_seconds": 180, "max_llm_calls": 10},
        worker_concurrency=3,
        retrieval_top_k=5,
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config.model_dump(mode="json")), encoding="utf-8")
    loaded = AblationConfig.model_validate_json(path.read_text(encoding="utf-8"))
    assert loaded == config
    assert loaded.config_hash() == config.config_hash()


def test_ablation_config_rejects_unknown_variant_and_bad_budgets() -> None:
    with pytest.raises(ValueError):
        AblationConfig(variant="unknown_variant")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        AblationConfig(variant="lead_code", worker_concurrency=9)
    with pytest.raises(ValueError):
        AblationConfig(variant="lead_code", retrieval_top_k=99)


def test_shared_settings_identical_across_variants() -> None:
    shared = {
        "model": "m",
        "budgets": {"timeout_seconds": 180},
        "max_snippets": 20,
        "window_lines": 5,
        "max_candidates": 100,
        "worker_concurrency": 3,
    }
    configs = [
        AblationConfig(variant=variant, **shared) for variant in AblationVariant
    ]
    for config in configs:
        assert config.model == "m"
        assert config.budgets.timeout_seconds == 180
        assert config.max_snippets == 20
        assert config.window_lines == 5
        assert config.max_candidates == 100
        assert config.worker_concurrency == 3
