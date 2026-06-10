"""Verify-layer settings: budgets, thresholds, models, aggregator constants.

Loaded from ``configs/mvp.yaml`` (override path with ``VERIFY_CONFIG``);
every leaf can also be overridden programmatically. Aggregator constants are
deliberately config-level, not code-level — they get tuned on the trap set in
week 6 (MVP plan §4.7).
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict

import yaml
from pydantic import BaseModel, Field

from verify.schema import Budget

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "mvp.yaml"


class ModelSettings(BaseModel):
    """Which model plays which role in the verify layer."""

    # Frontier roles (planner-grade quality required)
    claimify_model: str = "openai:gpt-4.1"
    skeptic_model: str = "openai:gpt-4.1"
    composer_model: str = "openai:gpt-4.1"
    # Mid-tier roles
    notetaker_model: str = "openai:gpt-4.1-mini"
    stance_model: str = "openai:gpt-4.1-mini"
    max_tokens: int = 8192
    composer_max_tokens: int = 16000


class StanceSettings(BaseModel):
    """Stance classifier backend selection (MVP plan §4.5)."""

    backend: str = "llm"  # "llm" | "minicheck"
    minicheck_url: str | None = None  # vLLM OpenAI-compatible endpoint
    minicheck_model: str = "bespokelabs/Bespoke-MiniCheck-7B"
    # Borderline band where MiniCheck defers to the LLM for refutes/qualifies
    borderline_band: float = 0.15


class AggregateSettings(BaseModel):
    """Constants of the sigma formula v1 (MVP plan §4.7)."""

    lambda_: float = Field(default=1.5, alias="lambda")
    beta: float = 1.2
    prior: Dict[str, float] = Field(
        default_factory=lambda: {
            "factual": 0.0,
            "statistical": 0.0,
            "causal": -0.4,
            "trend": -0.2,
            "predictive": -0.6,
            "normative": -0.6,
            "definitional": 0.0,
        }
    )
    tier_w: Dict[str, float] = Field(
        default_factory=lambda: {"A": 1.0, "B": 0.75, "C": 0.45, "D": 0.15}
    )
    type_w: Dict[str, float] = Field(
        default_factory=lambda: {
            "primary": 1.0,
            "secondary": 0.7,
            "opinion": 0.35,
            "dataset": 1.0,
        }
    )
    tau_days: float = 365.0  # freshness half-life scale; domain configs override
    single_origin_cap: float = 0.90
    contrib_threshold: float = 0.15
    bootstrap_iterations: int = 200
    # flags
    not_enough_evidence_u: float = 0.5
    stale_age_days: float = 540.0

    model_config = {"populate_by_name": True}


class Thresholds(BaseModel):
    """Quality gates inside the pipeline."""

    tau_extract: float = 0.6  # claim<->finding grounding below this => extraction_unstable
    tau_drift: float = 0.55  # report sentence<->claim grounding below this => drift violation
    max_repairs: int = 2


class VerifySettings(BaseModel):
    """Top-level settings bundle for the verify phase."""

    budget: Budget = Field(default_factory=Budget)
    models: ModelSettings = Field(default_factory=ModelSettings)
    stance: StanceSettings = Field(default_factory=StanceSettings)
    aggregate: AggregateSettings = Field(default_factory=AggregateSettings)
    thresholds: Thresholds = Field(default_factory=Thresholds)
    tiers_file: str = str(REPO_ROOT / "configs" / "domains" / "tiers.yaml")
    domain_tau_days: Dict[str, float] = Field(
        default_factory=lambda: {
            "ai_market": 120.0,
            "nutrition": 1095.0,
            "finance": 120.0,
        }
    )

    @classmethod
    def load(cls, path: str | None = None) -> "VerifySettings":
        """Load settings from YAML; missing file -> defaults."""
        cfg_path = Path(path or os.environ.get("VERIFY_CONFIG", DEFAULT_CONFIG_PATH))
        if not cfg_path.exists():
            return cls()
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        return cls(**data.get("verify", data))


@lru_cache(maxsize=4)
def get_settings(path: str | None = None) -> VerifySettings:
    """Return cached settings shared across verify nodes."""
    return VerifySettings.load(path)
