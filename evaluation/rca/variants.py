"""Ablation variant configuration for the benchmark workflow (M9-C).

Variants are switched by configuration only; production code is never edited
between runs. All variants share the same manifest, model, prompts, budgets,
context limits, and concurrency settings; only the evidence-agent subset and
historical-retrieval mode differ.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from patchpilot.rca.schema import AgentRole, PlanBudgets

SCHEMA_VERSION = "1.0"
RetrievalMode = Literal["off", "clustered", "flat"]


class AblationVariant(str, Enum):
    """The five supported benchmark ablation variants."""

    DETERMINISTIC_BASELINE = "deterministic_baseline"
    LEAD_CODE = "lead_code"
    THREE_SPECIALISTS_RETRIEVAL_OFF = "three_specialists_retrieval_off"
    THREE_SPECIALISTS_RETRIEVAL_ON = "three_specialists_retrieval_on"
    CLUSTERING_OFF = "clustering_off"


@dataclass(frozen=True)
class VariantSettings:
    """How one variant maps onto the RCA pipeline."""

    variant: AblationVariant
    enabled_roles: tuple[AgentRole, ...]
    retrieval_mode: RetrievalMode
    deterministic: bool
    label: str


_ALL_ROLES = (AgentRole.ISSUE_CI, AgentRole.CODE, AgentRole.GIT_HISTORY)

VARIANT_SETTINGS: dict[AblationVariant, VariantSettings] = {
    AblationVariant.DETERMINISTIC_BASELINE: VariantSettings(
        variant=AblationVariant.DETERMINISTIC_BASELINE,
        enabled_roles=(),
        retrieval_mode="off",
        deterministic=True,
        label="Deterministic baseline (no LLM agents)",
    ),
    AblationVariant.LEAD_CODE: VariantSettings(
        variant=AblationVariant.LEAD_CODE,
        enabled_roles=(AgentRole.CODE,),
        retrieval_mode="off",
        deterministic=False,
        label="Lead planner + code specialist",
    ),
    AblationVariant.THREE_SPECIALISTS_RETRIEVAL_OFF: VariantSettings(
        variant=AblationVariant.THREE_SPECIALISTS_RETRIEVAL_OFF,
        enabled_roles=_ALL_ROLES,
        retrieval_mode="off",
        deterministic=False,
        label="Three evidence specialists, retrieval off",
    ),
    AblationVariant.THREE_SPECIALISTS_RETRIEVAL_ON: VariantSettings(
        variant=AblationVariant.THREE_SPECIALISTS_RETRIEVAL_ON,
        enabled_roles=_ALL_ROLES,
        retrieval_mode="clustered",
        deterministic=False,
        label="Three evidence specialists, retrieval on",
    ),
    AblationVariant.CLUSTERING_OFF: VariantSettings(
        variant=AblationVariant.CLUSTERING_OFF,
        enabled_roles=_ALL_ROLES,
        retrieval_mode="flat",
        deterministic=False,
        label="Three evidence specialists, clustering off (flat retrieval)",
    ),
}


def variant_settings(variant: AblationVariant | str) -> VariantSettings:
    """Resolve settings for one ablation variant."""
    return VARIANT_SETTINGS[AblationVariant(variant)]


class AblationConfig(BaseModel):
    """Shared benchmark settings plus one variant; hashed for reproducibility.

    The shared fields (model, budgets, context limits, concurrency) are
    constant across variants; the variant field is the only behavioral switch.
    """

    schema_version: str = SCHEMA_VERSION
    variant: AblationVariant
    model: str | None = None
    manifest_name: str = ""
    manifest_sha256: str = ""
    budgets: PlanBudgets = Field(default_factory=PlanBudgets)
    max_snippets: int = Field(default=20, ge=1, le=100)
    window_lines: int = Field(default=5, ge=1, le=20)
    max_candidates: int = Field(default=100, ge=1, le=1_000)
    worker_concurrency: int = Field(default=3, ge=1, le=3)
    retrieval_top_k: int = Field(default=5, ge=0, le=10)
    history_corpus: str | None = Field(default=None, max_length=500)
    history_index: str | None = Field(default=None, max_length=500)
    history_excluded_ids: list[str] = Field(default_factory=list, max_length=1_000)
    max_cases: int | None = Field(default=None, ge=1)

    @field_validator("variant")
    @classmethod
    def _known_variant(cls, value: AblationVariant) -> AblationVariant:
        if value not in VARIANT_SETTINGS:
            raise ValueError(f"unknown ablation variant: {value}")
        return value

    def config_hash(self) -> str:
        """Deterministic content hash of the full effective configuration."""
        payload = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
